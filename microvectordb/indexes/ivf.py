"""
microvectordb.indexes.ivf
-------------------------
Optimize edilmiş Inverted File Index (IVF) — Saf NumPy + OpenBLAS.

v0.3.0 İyileştirmeleri:
1. Gerçek K-Means++ başlangıç merkezi seçimi (daha dengeli kümeler, daha yüksek recall).
2. Vektörize centroid güncellemesi (np.bincount + np.add.at -> 5-10x daha hızlı eğitim).
3. Bitişik (contiguous) küme bellek düzeni (_packed_vectors): sorgu anında
   fancy-indexing scatter/gather kopyası olmadan doğrudan BLAS SGEMV!
4. Tüm metrikler desteklenir: 'euclidean', 'cosine' (Spherical K-Means), 'dot_product'.
5. v0.2.0'daki `nprobe >= n_clusters` argpartition çökme hatası giderildi.
6. Opsiyonel boolean filtre maskesi ile filtre-farkında küme araması.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from microvectordb.metrics.kernels import (
    compute_inv_norms,
    compute_sq_norms,
    normalize_vectors,
    top_k_indices,
    validate_metric,
)
from microvectordb.types import MetricType


class IVFIndex:
    """
    Inverted File Index — K-Means++ kümeleme ve bitişik blok BLAS araması.

    Args:
        n_clusters: Toplam küme sayısı (varsayılan: 64).
        nprobe: Aramada ziyaret edilecek küme sayısı (varsayılan: 4).
        metric: 'euclidean' (varsayılan, v0.2.0 uyumlu), 'cosine' veya 'dot_product'.
    """

    def __init__(
        self,
        n_clusters: int = 64,
        nprobe: int = 4,
        metric: MetricType = "euclidean",
    ):
        validate_metric(metric)
        self.n_clusters = max(1, int(n_clusters))
        self.nprobe = max(1, int(nprobe))
        self.metric: MetricType = metric

        self.centroids: Optional[np.ndarray] = None  # (C, D) float32
        self._centroid_sq_norms: Optional[np.ndarray] = None  # (C,) float32
        self.inverted_lists: Dict[int, List[int]] = {}
        self._trained = False
        self._dimension: int = 0

        # Bitişik küme düzeni (zero-copy BLAS blokları)
        self._packed_vectors: Optional[np.ndarray] = None
        self._packed_ids: Optional[np.ndarray] = None
        self._packed_sq_norms: Optional[np.ndarray] = None
        self._packed_inv_norms: Optional[np.ndarray] = None
        self._cluster_starts: Optional[np.ndarray] = None
        self._cluster_ends: Optional[np.ndarray] = None
        self._total_items: int = 0
        # Artımlı eklemeler için hafif tampon (küme -> (ids, vecs, sq, inv))
        self._extra_ids: Dict[int, List[int]] = {}
        self._extra_vecs: Dict[int, List[np.ndarray]] = {}
        self._extra_count: int = 0

    @property
    def is_trained(self) -> bool:
        return self._trained

    def _prepare_for_clustering(self, vectors: np.ndarray) -> np.ndarray:
        v32 = np.ascontiguousarray(vectors, dtype=np.float32)
        if self.metric == "cosine":
            return normalize_vectors(v32)
        return v32

    def _kmeans_plus_plus_init(
        self,
        sample: np.ndarray,
        k: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Gerçek K-Means++ başlangıç merkezi seçimi (deterministik seed ile)."""
        n, d = sample.shape
        centroids = np.empty((k, d), dtype=np.float32)
        first = int(rng.integers(n))
        centroids[0] = sample[first]

        if self.metric == "cosine":
            # Kosinüs mesafesi: 1 - dot(x, c)
            min_dists = np.maximum(np.float32(0.0), np.float32(1.0) - np.dot(sample, centroids[0]))
            for c in range(1, k):
                total = float(min_dists.sum())
                if total <= 1e-12:
                    next_idx = int(rng.integers(n))
                else:
                    probs = min_dists.astype(np.float64) / total
                    probs /= probs.sum()
                    next_idx = int(rng.choice(n, p=probs))
                centroids[c] = sample[next_idx]
                d_new = np.maximum(np.float32(0.0), np.float32(1.0) - np.dot(sample, centroids[c]))
                np.minimum(min_dists, d_new, out=min_dists)
        else:
            sample_sq = compute_sq_norms(sample)
            c0_sq = float(np.dot(centroids[0], centroids[0]))
            min_dists = np.maximum(
                np.float32(0.0), sample_sq + np.float32(c0_sq) - np.float32(2.0) * np.dot(sample, centroids[0])
            )
            for c in range(1, k):
                total = float(min_dists.sum())
                if total <= 1e-12:
                    next_idx = int(rng.integers(n))
                else:
                    probs = min_dists.astype(np.float64) / total
                    probs /= probs.sum()
                    next_idx = int(rng.choice(n, p=probs))
                centroids[c] = sample[next_idx]
                cc_sq = float(np.dot(centroids[c], centroids[c]))
                d_new = np.maximum(
                    np.float32(0.0), sample_sq + np.float32(cc_sq) - np.float32(2.0) * np.dot(sample, centroids[c])
                )
                np.minimum(min_dists, d_new, out=min_dists)
        return centroids

    def _assign_to_centroids(
        self,
        vectors: np.ndarray,
        vectors_sq: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Her vektörü en yakın küme merkezine atar (BLAS GEMM, bloklu bellek dostu)."""
        assert self.centroids is not None
        n = vectors.shape[0]
        c_cnt = self.centroids.shape[0]
        assignments = np.empty(n, dtype=np.int32)
        chunk = 8192

        if self.metric == "cosine":
            # Hem centroids hem vectors normalize (veya en yüksek iç çarpım)
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                sims = np.dot(vectors[start:end], self.centroids.T)
                assignments[start:end] = np.argmax(sims, axis=1)
            return assignments

        c_sq = self._centroid_sq_norms
        if c_sq is None:
            c_sq = compute_sq_norms(self.centroids)
        c_sq_row = c_sq.reshape(1, c_cnt)

        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            # ||v - c||^2 için argmin yalnızca (||c||^2 - 2 v.c) gerektirir (||v||^2 sabittir!)
            scores = np.dot(vectors[start:end], self.centroids.T)
            scores *= np.float32(-2.0)
            scores += c_sq_row
            assignments[start:end] = np.argmin(scores, axis=1)
        return assignments

    def _pack_from_assignments(
        self,
        raw_vectors: np.ndarray,
        assignments: np.ndarray,
    ) -> None:
        """
        Vektörleri küme sırasına göre tek bir bitişik (N, D) diziye paketler.
        """
        n, d = raw_vectors.shape
        c_cnt = self.n_clusters
        order = np.argsort(assignments, kind="stable").astype(np.int64)
        counts = np.bincount(assignments, minlength=c_cnt)

        starts = np.zeros(c_cnt, dtype=np.int64)
        if c_cnt > 1:
            np.cumsum(counts[:-1], out=starts[1:])
        ends = starts + counts

        v32 = np.ascontiguousarray(raw_vectors[order], dtype=np.float32)
        sq, inv = compute_inv_norms(v32)

        self._packed_vectors = v32
        self._packed_ids = order
        self._packed_sq_norms = sq
        self._packed_inv_norms = inv
        self._cluster_starts = starts
        self._cluster_ends = ends
        self._total_items = n
        self._extra_ids.clear()
        self._extra_vecs.clear()
        self._extra_count = 0

        # Geriye dönük uyumluluk: inverted_lists sözlüğünü de doldur
        self.inverted_lists = {}
        for cid in range(c_cnt):
            s = int(starts[cid])
            e = int(ends[cid])
            if e > s:
                self.inverted_lists[cid] = order[s:e].tolist()

    def train(
        self,
        vectors: np.ndarray,
        n_iter: int = 20,
        sample_size: Optional[int] = None,
    ) -> None:
        """
        K-Means++ ile küme merkezlerini öğrenir ve bitişik ters indeksi kurar.
        """
        v_raw = np.ascontiguousarray(vectors, dtype=np.float32)
        if v_raw.ndim != 2 or v_raw.shape[0] == 0:
            raise ValueError("Eğitim vektörleri boş olmayan 2B dizi olmalıdır.")

        n, d = v_raw.shape
        self._dimension = d
        actual_clusters = min(self.n_clusters, n)
        if actual_clusters < self.n_clusters:
            self.n_clusters = actual_clusters

        rng = np.random.default_rng(42)
        v_work = self._prepare_for_clustering(v_raw)

        # Büyük veri kümelerinde K-Means eğitimini alt-örneklem üzerinde hızlandır
        max_train = sample_size or max(actual_clusters * 256, 32768)
        if n > max_train:
            train_idx = rng.choice(n, max_train, replace=False)
            train_data = np.ascontiguousarray(v_work[train_idx])
        else:
            train_data = v_work

        # K-Means++ başlangıç örneklemi (init hızını yüksek tutmak için sınırlı alt-örnek)
        init_cap = min(len(train_data), max(actual_clusters * 32, 4096))
        if len(train_data) > init_cap:
            init_sample = train_data[rng.choice(len(train_data), init_cap, replace=False)]
        else:
            init_sample = train_data

        self.centroids = self._kmeans_plus_plus_init(init_sample, actual_clusters, rng)
        self._centroid_sq_norms = compute_sq_norms(self.centroids)

        # Vektörize K-Means iterasyonları
        for _ in range(n_iter):
            assignments = self._assign_to_centroids(train_data)
            counts = np.bincount(assignments, minlength=actual_clusters)
            sums = np.zeros((actual_clusters, d), dtype=np.float32)
            np.add.at(sums, assignments, train_data)

            non_empty = counts > 0
            new_centroids = np.empty_like(self.centroids)
            new_centroids[non_empty] = sums[non_empty] / counts[non_empty, None].astype(np.float32)

            if not np.all(non_empty):
                empty_ids = np.flatnonzero(~non_empty)
                for cid in empty_ids:
                    new_centroids[cid] = train_data[int(rng.integers(len(train_data)))]

            if self.metric == "cosine":
                new_centroids = normalize_vectors(new_centroids)

            shift = float(np.linalg.norm(new_centroids - self.centroids))
            self.centroids = new_centroids
            self._centroid_sq_norms = compute_sq_norms(self.centroids)
            if shift < 1e-5:
                break

        # Tüm veri setini nihai merkezlere ata ve bitişik olarak paketle
        full_assignments = self._assign_to_centroids(v_work)
        self._pack_from_assignments(v_raw, full_assignments)
        self._trained = True

    def add_vectors(self, vectors: np.ndarray, start_idx: int = 0) -> None:
        """
        Eğitilmiş indekse yeni vektörleri artımlı olarak ekler.
        """
        if not self._trained or self.centroids is None:
            raise RuntimeError("İndeks henüz eğitilmedi. Önce train() çağırın.")

        v_raw = np.ascontiguousarray(vectors, dtype=np.float32)
        if v_raw.ndim == 1:
            v_raw = v_raw.reshape(1, -1)
        if v_raw.shape[0] == 0:
            return

        v_work = self._prepare_for_clustering(v_raw)
        assignments = self._assign_to_centroids(v_work)

        for i, cid_val in enumerate(assignments):
            cid = int(cid_val)
            gid = start_idx + i
            self.inverted_lists.setdefault(cid, []).append(gid)
            self._extra_ids.setdefault(cid, []).append(gid)
            self._extra_vecs.setdefault(cid, []).append(v_raw[i])

        self._extra_count += v_raw.shape[0]
        self._total_items += v_raw.shape[0]

    def rebuild(self, vectors: np.ndarray) -> None:
        """Mevcut merkezleri koruyarak bitişik ters indeksi sıfırdan yeniden paketler."""
        if not self._trained or self.centroids is None:
            raise RuntimeError("İndeks henüz eğitilmedi.")
        v_raw = np.ascontiguousarray(vectors, dtype=np.float32)
        if v_raw.shape[0] == 0:
            self.inverted_lists = {}
            self._packed_vectors = np.empty((0, self._dimension), dtype=np.float32)
            self._packed_ids = np.empty(0, dtype=np.int64)
            self._packed_sq_norms = np.empty(0, dtype=np.float32)
            self._packed_inv_norms = np.empty(0, dtype=np.float32)
            self._cluster_starts = np.zeros(self.n_clusters, dtype=np.int64)
            self._cluster_ends = np.zeros(self.n_clusters, dtype=np.int64)
            self._total_items = 0
            self._extra_ids.clear()
            self._extra_vecs.clear()
            self._extra_count = 0
            return

        v_work = self._prepare_for_clustering(v_raw)
        assignments = self._assign_to_centroids(v_work)
        self._pack_from_assignments(v_raw, assignments)

    def _nearest_clusters(self, query_f32: np.ndarray, nprobe: Optional[int] = None) -> np.ndarray:
        """Sorguya en yakın nprobe küme ID'sini döndürür (kth taşma hatası olmadan)."""
        assert self.centroids is not None
        probe = min(nprobe or self.nprobe, len(self.centroids))
        if probe >= len(self.centroids):
            return np.arange(len(self.centroids), dtype=np.int64)

        if self.metric == "cosine":
            qn = normalize_vectors(query_f32.reshape(1, -1)).ravel()
            cdists = np.float32(1.0) - np.dot(self.centroids, qn)
        else:
            assert self._centroid_sq_norms is not None
            # argpartition için ||q||^2 sabit olduğundan ihmal edilebilir
            cdists = self._centroid_sq_norms - np.float32(2.0) * np.dot(self.centroids, query_f32)

        return np.argpartition(cdists, probe - 1)[:probe]

    def _score_block(
        self,
        q_f32: np.ndarray,
        q_norm: Optional[np.ndarray],
        q_sq: np.float32,
        block: np.ndarray,
        block_sq: Optional[np.ndarray],
        block_inv: Optional[np.ndarray],
    ) -> np.ndarray:
        """Tek bir bitişik küme bloğu için BLAS mesafesi hesaplar."""
        if self.metric == "cosine":
            assert q_norm is not None
            sims = np.dot(block, q_norm)
            if block_inv is not None:
                sims *= block_inv
            else:
                _, inv = compute_inv_norms(block)
                sims *= inv
            np.clip(sims, np.float32(-1.0), np.float32(1.0), out=sims)
            return (np.float32(1.0) - sims).astype(np.float32, copy=False)

        if self.metric == "dot_product":
            return (-np.dot(block, q_f32)).astype(np.float32, copy=False)

        # euclidean
        sq_b = block_sq if block_sq is not None else compute_sq_norms(block)
        sq_d = np.dot(block, q_f32)
        sq_d *= np.float32(-2.0)
        sq_d += q_sq
        sq_d += sq_b
        np.maximum(sq_d, np.float32(0.0), out=sq_d)
        return np.sqrt(sq_d, out=sq_d)

    def search(
        self,
        query: np.ndarray,
        vectors: Optional[np.ndarray] = None,
        k: int = 10,
        nprobe: Optional[int] = None,
        allowed_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        En yakın k komşuyu bitişik küme blokları üzerinden BLAS ile bulur.
        """
        if not self._trained or self.centroids is None:
            raise RuntimeError("İndeks henüz eğitilmedi. Önce train() çağırın.")
        if k <= 0 or self._total_items == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        q_f32 = np.asarray(query, dtype=np.float32).ravel()
        q_norm = normalize_vectors(q_f32.reshape(1, -1)).ravel() if self.metric == "cosine" else None
        q_sq = np.float32(np.dot(q_f32, q_f32))

        clusters = self._nearest_clusters(q_f32, nprobe=nprobe)

        assert self._packed_vectors is not None
        assert self._packed_ids is not None
        assert self._cluster_starts is not None
        assert self._cluster_ends is not None

        cand_id_parts: List[np.ndarray] = []
        cand_dist_parts: List[np.ndarray] = []

        for cid_val in clusters:
            cid = int(cid_val)
            s = int(self._cluster_starts[cid])
            e = int(self._cluster_ends[cid])
            if e > s:
                ids_slice = self._packed_ids[s:e]
                if allowed_mask is not None:
                    keep = allowed_mask[ids_slice]
                    if keep.any():
                        sub_ids = ids_slice[keep]
                        sub_vecs = self._packed_vectors[s:e][keep]
                        sub_sq = self._packed_sq_norms[s:e][keep] if self._packed_sq_norms is not None else None
                        sub_inv = self._packed_inv_norms[s:e][keep] if self._packed_inv_norms is not None else None
                        dists = self._score_block(q_f32, q_norm, q_sq, sub_vecs, sub_sq, sub_inv)
                        cand_id_parts.append(sub_ids)
                        cand_dist_parts.append(dists)
                else:
                    block = self._packed_vectors[s:e]
                    b_sq = self._packed_sq_norms[s:e] if self._packed_sq_norms is not None else None
                    b_inv = self._packed_inv_norms[s:e] if self._packed_inv_norms is not None else None
                    dists = self._score_block(q_f32, q_norm, q_sq, block, b_sq, b_inv)
                    cand_id_parts.append(ids_slice)
                    cand_dist_parts.append(dists)

            # Artımlı eklenmiş vektörler varsa dahil et
            if cid in self._extra_ids:
                ex_ids = np.asarray(self._extra_ids[cid], dtype=np.int64)
                ex_vecs = np.asarray(self._extra_vecs[cid], dtype=np.float32)
                if allowed_mask is not None:
                    keep = allowed_mask[ex_ids]
                    if keep.any():
                        ex_ids = ex_ids[keep]
                        ex_vecs = ex_vecs[keep]
                    else:
                        continue
                ex_dists = self._score_block(q_f32, q_norm, q_sq, ex_vecs, None, None)
                cand_id_parts.append(ex_ids)
                cand_dist_parts.append(ex_dists)

        if not cand_id_parts:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        if len(cand_id_parts) == 1:
            all_ids = cand_id_parts[0]
            all_dists = cand_dist_parts[0]
        else:
            all_ids = np.concatenate(cand_id_parts)
            all_dists = np.concatenate(cand_dist_parts)

        top_idx = top_k_indices(all_dists, min(k, len(all_dists)))
        return all_ids[top_idx], all_dists[top_idx]

    def search_batch(
        self,
        queries: np.ndarray,
        vectors: Optional[np.ndarray] = None,
        k: int = 10,
        nprobe: Optional[int] = None,
        allowed_mask: Optional[np.ndarray] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Birden fazla sorguyu toplu olarak arar."""
        q_arr = np.asarray(queries, dtype=np.float32)
        if q_arr.ndim == 1:
            q_arr = q_arr.reshape(1, -1)

        all_indices: List[np.ndarray] = []
        all_dists: List[np.ndarray] = []
        for q_idx in range(q_arr.shape[0]):
            idx, dist = self.search(
                q_arr[q_idx], vectors=vectors, k=k, nprobe=nprobe, allowed_mask=allowed_mask
            )
            all_indices.append(idx)
            all_dists.append(dist)
        return all_indices, all_dists

    def __repr__(self) -> str:
        status = "trained" if self._trained else "untrained"
        return (
            f"<IVFIndex(n_clusters={self.n_clusters}, nprobe={self.nprobe}, "
            f"metric='{self.metric}', status={status}, items={self._total_items})>"
        )
