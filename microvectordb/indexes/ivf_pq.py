"""
microvectordb.indexes.ivf_pq
----------------------------
IVF-PQ (Inverted File + Residual Product Quantization) İndeksi — Saf NumPy.

Çalışma prensibi (FAISS IndexIVFPQ ile aynı matematiksel temel):
1. Kaba kümeleme (Coarse Quantizer): Vektörler K-Means++ ile `n_clusters` kümeye ayrılır.
2. Artık (Residual) Kodlama: Her vektörün kendi küme merkezinden farkı r_i = x_i - c_k
   hesaplanır ve `ProductQuantizer` ile M baytlık `uint8` koda sıkıştırılır (16x–48x az RAM).
3. Bitişik Küme Düzeni: Her kümenin PQ kodları (M, n_c) sütun-öncelikli bitişik blok
   olarak saklanır; sorguda yalnızca `nprobe` küme taranır -> saf NumPy'da <1 ms!
4. Doğruluk:
   - `euclidean`:  ||q - (c + r_hat)||^2 = ||(q - c) - r_hat||^2 (küme başına kesin artık LUT)
   - `cosine` / `dot_product`: q . (c + r_hat) = (q . c) + (q . r_hat) (TEK sorgu LUT + skaler ofset!)
5. Disk Kalıcılığı: CRC32 + epoch korumalı atomik ikili `save()` / `load()`.
"""

import os
import struct
import threading
import time
import zlib
from typing import Dict, List, Optional, Tuple
import numpy as np

from microvectordb.metrics.kernels import (
    compute_sq_norms,
    normalize_vectors,
    top_k_indices,
    validate_metric,
)
from microvectordb.quantization.product import ProductQuantizer
from microvectordb.types import DataCorruptionError, MetricType

_IVFPQ_MAGIC = b"MVIVFPQ1"
_IVFPQ_HDR_FMT = "<8sHHHHIIIQQQII"
# magic(8), ver(2), metric_id(2), n_clusters(2), nprobe(2), dim(4), M(4), K_s(4),
# total_items(8), epoch(8), pq_len(8), hdr_crc(4), body_crc(4)
_METRIC_TO_ID = {"euclidean": 0, "cosine": 1, "dot_product": 2}
_ID_TO_METRIC = {v: k for k, v in _METRIC_TO_ID.items()}


class IVFPQIndex:
    """
    Inverted File + Residual Product Quantization (IVF-PQ) — Düşük RAM + Yüksek QPS.
    """

    def __init__(
        self,
        n_clusters: int = 128,
        nprobe: int = 8,
        n_subvectors: Optional[int] = None,
        n_centroids: int = 256,
        metric: MetricType = "cosine",
        seed: int = 42,
    ) -> None:
        validate_metric(metric)
        self.n_clusters = max(1, int(n_clusters))
        self.nprobe = max(1, int(nprobe))
        self.n_subvectors = n_subvectors
        self.n_centroids = int(n_centroids)
        self.metric: MetricType = metric
        self.seed = int(seed)

        self._dimension: int = 0
        self._trained: bool = False
        self.centroids: Optional[np.ndarray] = None  # (C, D) float32
        self._centroid_sq_norms: Optional[np.ndarray] = None
        # Artık kodlayıcı her zaman artık vektör üzerinde ham L2/dot olarak eğitilir
        self.pq: Optional[ProductQuantizer] = None

        # Küme başına bitişik (M, n_c) kod blokları (uint16 ön-kaydırmalı veya uint8) ve int32/int64 ID dizileri
        self._cluster_codes_t: List[np.ndarray] = []
        self._cluster_ids: List[np.ndarray] = []
        self._total_items: int = 0
        self._use_u16_codes: bool = False
        self._u16_offsets: Optional[np.ndarray] = None
        # Ön-hesaplanmış artık LUT: P[c, m, k] = ||C[m,k]||^2 + 2 * (c[c,m] . C[m,k])
        # Sorgu başına yalnızca 1 kez batched matmul (-2 q.C) + tek-atış np.take ADC!
        self._precomp_cluster_lut: Optional[np.ndarray] = None  # (C, M, K) float32
        self._precomp_flat_lut: Optional[np.ndarray] = None  # (C, M*K) float32

    def _init_u16_mode(self) -> None:
        if self.pq is not None and (self.pq.n_subvectors * self.pq.n_centroids) <= 65536:
            self._use_u16_codes = True
            self._u16_offsets = (
                np.arange(self.pq.n_subvectors, dtype=np.uint16) * np.uint16(self.pq.n_centroids)
            )[:, None]
        else:
            self._use_u16_codes = False
            self._u16_offsets = None

    def _encode_cluster_codes(self, codes_u8_t: np.ndarray) -> np.ndarray:
        if self._use_u16_codes and self._u16_offsets is not None:
            return np.ascontiguousarray(codes_u8_t.astype(np.uint16) + self._u16_offsets)
        return np.ascontiguousarray(codes_u8_t, dtype=np.uint8)

    def _build_precomputed_lut(self) -> None:
        """
        ||q - (c_k + r_hat)||^2 = ||q - c_k||^2 + sum_m (P[k,m,j] - 2 q_m.C[m,j])
        P[k, m, j] = ||C[m, j]||^2 + 2 (c_{k, m} . C[m, j]) tablosunu eğitim/yükleme
        anında bir kez hesaplar (bellek <= 32 MB ise etkinleştirilir).
        """
        if self.centroids is None or self.pq is None or self.pq.codebooks is None or self.pq._codebook_sq_norms is None:
            self._precomp_cluster_lut = None
            self._precomp_flat_lut = None
            return
        if self.metric not in ("cosine", "euclidean"):
            self._precomp_cluster_lut = None
            self._precomp_flat_lut = None
            return
        pq = self.pq
        c_cnt = self.centroids.shape[0]
        est_bytes = c_cnt * pq.n_subvectors * pq.n_centroids * 4
        if est_bytes > 32 * 1024 * 1024:
            self._precomp_cluster_lut = None
            self._precomp_flat_lut = None
            return

        cents = self.centroids
        if cents.shape[1] < pq.padded_dim:
            cents = np.pad(cents, ((0, 0), (0, pq.padded_dim - cents.shape[1])), mode="constant")
        c_blocks = cents.reshape(c_cnt, pq.n_subvectors, pq.sub_dim)  # (C, M, d_s)
        # (C, M, d_s) ve (M, K, d_s) -> (C, M, K)
        c_dots = np.einsum("cmd,mkd->cmk", c_blocks, pq.codebooks)
        lut3d = np.ascontiguousarray(
            pq._codebook_sq_norms[None, :, :] + np.float32(2.0) * c_dots, dtype=np.float32
        )
        self._precomp_cluster_lut = lut3d
        self._precomp_flat_lut = lut3d.reshape(c_cnt, pq.n_subvectors * pq.n_centroids)

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def total_items(self) -> int:
        return self._total_items

    def _prepare_vectors(self, vectors: np.ndarray) -> np.ndarray:
        v32 = np.ascontiguousarray(vectors, dtype=np.float32)
        if self.metric == "cosine":
            return normalize_vectors(v32)
        return v32

    def _kmeans_coarse(self, data: np.ndarray, k: int, n_iter: int, rng: np.random.Generator) -> np.ndarray:
        n, d = data.shape
        init_cap = min(n, max(k * 32, 4096))
        init_sample = data[rng.choice(n, init_cap, replace=False)] if n > init_cap else data

        centroids = np.empty((k, d), dtype=np.float32)
        centroids[0] = init_sample[int(rng.integers(len(init_sample)))]

        if self.metric == "cosine":
            min_d = np.maximum(np.float32(0.0), np.float32(1.0) - np.dot(init_sample, centroids[0]))
            for c in range(1, k):
                tot = float(min_d.sum())
                pick = int(rng.integers(len(init_sample))) if tot <= 1e-12 else int(rng.choice(len(init_sample), p=(min_d.astype(np.float64) / tot) / (min_d.astype(np.float64) / tot).sum()))
                centroids[c] = init_sample[pick]
                d_new = np.maximum(np.float32(0.0), np.float32(1.0) - np.dot(init_sample, centroids[c]))
                np.minimum(min_d, d_new, out=min_d)
        else:
            s_sq = compute_sq_norms(init_sample)
            c0_sq = np.float32(np.dot(centroids[0], centroids[0]))
            min_d = np.maximum(np.float32(0.0), s_sq + c0_sq - np.float32(2.0) * np.dot(init_sample, centroids[0]))
            for c in range(1, k):
                tot = float(min_d.sum())
                pick = int(rng.integers(len(init_sample))) if tot <= 1e-12 else int(rng.choice(len(init_sample), p=(min_d.astype(np.float64) / tot) / (min_d.astype(np.float64) / tot).sum()))
                centroids[c] = init_sample[pick]
                cc_sq = np.float32(np.dot(centroids[c], centroids[c]))
                d_new = np.maximum(np.float32(0.0), s_sq + cc_sq - np.float32(2.0) * np.dot(init_sample, centroids[c]))
                np.minimum(min_d, d_new, out=min_d)

        for _ in range(n_iter):
            assign = self._assign_coarse(data, centroids)
            counts = np.bincount(assign, minlength=k)
            sums = np.zeros((k, d), dtype=np.float32)
            np.add.at(sums, assign, data)
            non_empty = counts > 0
            new_c = np.empty_like(centroids)
            new_c[non_empty] = sums[non_empty] / counts[non_empty, None].astype(np.float32)
            if not np.all(non_empty):
                for cid in np.flatnonzero(~non_empty):
                    new_c[cid] = data[int(rng.integers(n))]
            if self.metric == "cosine":
                new_c = normalize_vectors(new_c)
            shift = float(np.linalg.norm(new_c - centroids))
            centroids = new_c
            if shift < 1e-5:
                break
        return centroids

    def _assign_coarse(self, vectors: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        n = vectors.shape[0]
        c_cnt = centroids.shape[0]
        out = np.empty(n, dtype=np.int32)
        chunk = 8192
        if self.metric == "cosine":
            for s in range(0, n, chunk):
                e = min(s + chunk, n)
                out[s:e] = np.argmax(np.dot(vectors[s:e], centroids.T), axis=1)
            return out

        c_sq = compute_sq_norms(centroids).reshape(1, c_cnt)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            scores = np.dot(vectors[s:e], centroids.T)
            scores *= np.float32(-2.0)
            scores += c_sq
            out[s:e] = np.argmin(scores, axis=1)
        return out

    def train(
        self,
        vectors: np.ndarray,
        n_iter: int = 15,
        pq_iter: int = 12,
        sample_size: Optional[int] = None,
    ) -> None:
        """Kaba kümeleri ve artık PQ kod kitaplarını eğitip tüm vektörleri paketler."""
        v_work = self._prepare_vectors(vectors)
        if v_work.ndim != 2 or v_work.shape[0] == 0:
            raise ValueError("IVFPQIndex.train() boş olmayan 2B dizi bekler.")

        n, d = v_work.shape
        self._dimension = d
        actual_c = min(self.n_clusters, n)
        self.n_clusters = actual_c

        rng = np.random.default_rng(self.seed)
        max_train = sample_size or max(actual_c * 256, 32768)
        train_data = np.ascontiguousarray(v_work[rng.choice(n, max_train, replace=False)]) if n > max_train else v_work

        # 1. Kaba küme merkezleri
        self.centroids = self._kmeans_coarse(train_data, actual_c, n_iter=n_iter, rng=rng)
        self._centroid_sq_norms = compute_sq_norms(self.centroids)

        # 2. Artıklar (residuals) üzerinde Product Quantizer eğit
        train_assign = self._assign_coarse(train_data, self.centroids)
        train_residuals = train_data - self.centroids[train_assign]
        # Artık PQ her zaman 'euclidean' kod kitabı öğrenir (artık vektörleri normalize ETMEDEN!)
        pq = ProductQuantizer(
            n_subvectors=self.n_subvectors,
            n_centroids=min(self.n_centroids, len(train_residuals)),
            metric="euclidean",
            n_iter=pq_iter,
            seed=self.seed,
        )
        pq.fit(train_residuals, sample_size=min(len(train_residuals), max(self.n_centroids * 64, 16384)))
        self.pq = pq

        # 3. Tüm veri setini kümelere ata, artıkları kodla ve küme başına (M, n_c) bitişik paketle
        self._init_u16_mode()
        code_dtype = np.uint16 if self._use_u16_codes else np.uint8
        full_assign = self._assign_coarse(v_work, self.centroids)
        self._cluster_codes_t = [np.empty((pq.n_subvectors, 0), dtype=code_dtype) for _ in range(actual_c)]
        self._cluster_ids = [np.empty(0, dtype=np.int32) for _ in range(actual_c)]

        # Sıralayıp tek geçişte kodla (bellek dostu chunk'larla)
        order = np.argsort(full_assign, kind="stable").astype(np.int32)
        counts = np.bincount(full_assign, minlength=actual_c)
        pos = 0
        for cid in range(actual_c):
            cnt = int(counts[cid])
            if cnt == 0:
                continue
            ids_c = order[pos : pos + cnt]
            pos += cnt
            res_c = v_work[ids_c] - self.centroids[cid]
            codes_t = self._encode_cluster_codes(pq.encode_transposed(res_c))
            self._cluster_codes_t[cid] = codes_t
            self._cluster_ids[cid] = ids_c

        self._total_items = n
        self._trained = True
        self._build_precomputed_lut()

    def add_vectors(self, vectors: np.ndarray, start_idx: int = 0) -> None:
        """Eğitilmiş IVF-PQ indeksine yeni vektörler ekler."""
        if not self._trained or self.centroids is None or self.pq is None:
            raise RuntimeError("IVFPQIndex henüz eğitilmedi.")
        v_work = self._prepare_vectors(vectors)
        if v_work.shape[0] == 0:
            return

        assign = self._assign_coarse(v_work, self.centroids)
        new_ids = np.arange(start_idx, start_idx + v_work.shape[0], dtype=np.int32)
        unique_cids = np.unique(assign)
        for cid_val in unique_cids:
            cid = int(cid_val)
            mask = assign == cid
            res_c = v_work[mask] - self.centroids[cid]
            new_codes_t = self._encode_cluster_codes(self.pq.encode_transposed(res_c))
            if self._cluster_ids[cid].size == 0:
                self._cluster_codes_t[cid] = new_codes_t
                self._cluster_ids[cid] = new_ids[mask]
            else:
                self._cluster_codes_t[cid] = np.ascontiguousarray(
                    np.concatenate([self._cluster_codes_t[cid], new_codes_t], axis=1)
                )
                self._cluster_ids[cid] = np.concatenate([self._cluster_ids[cid], new_ids[mask]])
        self._total_items += v_work.shape[0]

    def _nearest_clusters(self, q_work: np.ndarray, nprobe: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """En yakın nprobe küme ID'lerini ve tüm kümelere olan iç çarpım/mesafe vektörünü döndürür."""
        assert self.centroids is not None
        probe = min(nprobe or self.nprobe, len(self.centroids))
        q_c_dots = np.dot(self.centroids, q_work)  # (C,)
        if probe >= len(self.centroids):
            return np.arange(len(self.centroids), dtype=np.int64), q_c_dots

        if self.metric == "cosine":
            cdists = np.float32(1.0) - q_c_dots
        else:
            assert self._centroid_sq_norms is not None
            cdists = self._centroid_sq_norms - np.float32(2.0) * q_c_dots
        return np.argpartition(cdists, probe - 1)[:probe], q_c_dots

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        nprobe: Optional[int] = None,
        allowed_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Tek sorgu için Ön-Hesaplanmış Artık ADC ile en yakın k komşuyu bulur (<0.6 ms / 100K)."""
        if not self._trained or self.centroids is None or self.pq is None:
            raise RuntimeError("IVFPQIndex henüz eğitilmedi.")
        if k <= 0 or self._total_items == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        q32 = np.asarray(query, dtype=np.float32).ravel()
        if self.metric == "cosine":
            qn = np.float32(np.linalg.norm(q32))
            q_work = q32 / qn if qn > np.float32(0.0) else q32
            q_sq = np.float32(np.dot(q_work, q_work))
        else:
            q_work = q32
            q_sq = np.float32(np.dot(q_work, q_work))

        clusters, q_c_dots = self._nearest_clusters(q_work, nprobe=nprobe)
        pq = self.pq
        assert pq.codebooks is not None and self._centroid_sq_norms is not None

        # Ön-hesaplanmış küme tablosu varsa sorgu başına YALNIZCA 1 batched matmul yap!
        precomp = self._precomp_cluster_lut
        precomp_flat = self._precomp_flat_lut
        use_u16 = self._use_u16_codes
        shared_neg2_qdot: Optional[np.ndarray] = None
        shared_neg2_flat: Optional[np.ndarray] = None
        shared_ip_lut: Optional[np.ndarray] = None
        shared_ip_flat: Optional[np.ndarray] = None

        if self.metric == "dot_product":
            shared_ip_lut = pq.compute_distance_table(q_work, metric="dot_product")
            if use_u16:
                shared_ip_flat = shared_ip_lut.ravel()
        elif precomp is not None:
            q_pad = q_work
            if q_pad.shape[0] < pq.padded_dim:
                q_pad = np.pad(q_pad, (0, pq.padded_dim - q_pad.shape[0]), mode="constant")
            q_blocks = q_pad.reshape(pq.n_subvectors, pq.sub_dim, 1)
            shared_neg2_qdot = (
                np.float32(-2.0) * np.matmul(pq.codebooks, q_blocks).squeeze(-1)
            ).astype(np.float32, copy=False)
            if use_u16:
                shared_neg2_flat = shared_neg2_qdot.ravel()

        cand_ids: List[np.ndarray] = []
        cand_dists: List[np.ndarray] = []

        for cid_val in clusters:
            cid = int(cid_val)
            ids_c = self._cluster_ids[cid]
            if ids_c.size == 0:
                continue
            codes_t = self._cluster_codes_t[cid]

            if allowed_mask is not None:
                keep = allowed_mask[ids_c]
                if not keep.any():
                    continue
                ids_c = ids_c[keep]
                codes_t = np.ascontiguousarray(codes_t[:, keep])

            if self.metric in ("euclidean", "cosine"):
                if use_u16 and precomp_flat is not None and shared_neg2_flat is not None:
                    # Tek-atış vektörize 1B LUT toplama (16 Python döngüsü yerine 1 C çağrısı!)
                    lut_f = precomp_flat[cid] + shared_neg2_flat
                    sq_d = np.take(lut_f, codes_t).sum(axis=0)
                    sq_d += np.float32(q_sq + self._centroid_sq_norms[cid] - np.float32(2.0) * q_c_dots[cid])
                elif precomp is not None and shared_neg2_qdot is not None:
                    lut = precomp[cid] + shared_neg2_qdot
                    sq_d = pq.accumulate_adc(lut, codes_t)
                    sq_d += np.float32(q_sq + self._centroid_sq_norms[cid] - np.float32(2.0) * q_c_dots[cid])
                else:
                    q_res = q_work - self.centroids[cid]
                    lut = pq.compute_distance_table(q_res, metric="euclidean")
                    if use_u16:
                        sq_d = np.take(lut.ravel(), codes_t).sum(axis=0)
                    else:
                        sq_d = pq.accumulate_adc(lut, codes_t)

                if self.metric == "euclidean":
                    np.maximum(sq_d, np.float32(0.0), out=sq_d)
                    d_c = np.sqrt(sq_d, out=sq_d)
                else:
                    d_c = np.clip(sq_d * np.float32(0.5), np.float32(0.0), np.float32(2.0))
            else:
                # dot_product: distance = -(q . c + q . r_hat)
                c_neg_dot = np.float32(-q_c_dots[cid])
                if use_u16 and shared_ip_flat is not None:
                    d_c = np.take(shared_ip_flat, codes_t).sum(axis=0)
                else:
                    assert shared_ip_lut is not None
                    d_c = pq.accumulate_adc(shared_ip_lut, codes_t)
                d_c += c_neg_dot

            cand_ids.append(ids_c)
            cand_dists.append(d_c)

        if not cand_ids:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        all_ids = cand_ids[0] if len(cand_ids) == 1 else np.concatenate(cand_ids)
        all_dists = cand_dists[0] if len(cand_dists) == 1 else np.concatenate(cand_dists)
        top = top_k_indices(all_dists, min(k, len(all_dists)))
        return all_ids[top].astype(np.int64, copy=False), all_dists[top]

    def search_batch(
        self,
        queries: np.ndarray,
        k: int = 10,
        nprobe: Optional[int] = None,
        allowed_mask: Optional[np.ndarray] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        q_arr = np.asarray(queries, dtype=np.float32)
        if q_arr.ndim == 1:
            q_arr = q_arr.reshape(1, -1)
        out_i, out_d = [], []
        for i in range(q_arr.shape[0]):
            idxs, ds = self.search(q_arr[i], k=k, nprobe=nprobe, allowed_mask=allowed_mask)
            out_i.append(idxs)
            out_d.append(ds)
        return out_i, out_d

    def memory_bytes(self) -> int:
        total = 0
        if self.centroids is not None:
            total += int(self.centroids.nbytes)
        if self.pq is not None and self.pq.codebooks is not None:
            total += int(self.pq.codebooks.nbytes)
        if self._precomp_cluster_lut is not None:
            total += int(self._precomp_cluster_lut.nbytes)
        for c_codes, c_ids in zip(self._cluster_codes_t, self._cluster_ids):
            total += int(c_codes.nbytes) + int(c_ids.nbytes)
        return total

    # ------------------------------------------------------------------ #
    # Atomik CRC32 + Epoch Disk Kalıcılığı
    # ------------------------------------------------------------------ #
    def save(self, path: str, epoch: int = 0) -> None:
        if not self._trained or self.centroids is None or self.pq is None:
            raise RuntimeError("Eğitilmemiş IVFPQIndex kaydedilemez.")

        pq_bytes = self.pq.to_bytes()
        counts = np.array([c.size for c in self._cluster_ids], dtype=np.uint64)
        body_parts = [
            pq_bytes,
            np.ascontiguousarray(self.centroids, dtype=np.float32).tobytes(),
            counts.tobytes(),
        ]
        for cid in range(self.n_clusters):
            if counts[cid] > 0:
                body_parts.append(np.ascontiguousarray(self._cluster_ids[cid], dtype=np.int64).tobytes())
                if self._use_u16_codes and self._u16_offsets is not None:
                    raw_u8 = (self._cluster_codes_t[cid] - self._u16_offsets).astype(np.uint8)
                else:
                    raw_u8 = self._cluster_codes_t[cid].astype(np.uint8, copy=False)
                body_parts.append(np.ascontiguousarray(raw_u8, dtype=np.uint8).tobytes())

        body = b"".join(body_parts)
        body_crc = zlib.crc32(body) & 0xFFFFFFFF

        hdr_prefix = struct.pack(
            "<8sHHHHIIIQQQ",
            _IVFPQ_MAGIC,
            1,
            _METRIC_TO_ID[self.metric],
            self.n_clusters,
            self.nprobe,
            self._dimension,
            self.pq.n_subvectors,
            self.pq.n_centroids,
            self._total_items,
            int(epoch),
            len(pq_bytes),
        )
        hdr_crc = zlib.crc32(hdr_prefix) & 0xFFFFFFFF
        header = hdr_prefix + struct.pack("<II", hdr_crc, body_crc)

        tmp_path = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}"
        try:
            with open(tmp_path, "wb") as f:
                f.write(header)
                f.write(body)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise

    @classmethod
    def load(cls, path: str) -> Tuple["IVFPQIndex", int]:
        """Diskten CRC32 doğrulayarak (IVFPQIndex, epoch) yükler."""
        with open(path, "rb") as f:
            raw = f.read()

        hdr_sz = struct.calcsize(_IVFPQ_HDR_FMT)
        if len(raw) < hdr_sz:
            raise DataCorruptionError(f"IVF-PQ indeks başlığı eksik: '{path}'")

        (
            magic, ver, metric_id, n_clusters, nprobe, dim,
            m_cnt, k_cnt, total_items, epoch, pq_len, hdr_crc, body_crc
        ) = struct.unpack(_IVFPQ_HDR_FMT, raw[:hdr_sz])

        if magic != _IVFPQ_MAGIC or ver != 1 or metric_id not in _ID_TO_METRIC:
            raise DataCorruptionError(f"Geçersiz IVF-PQ indeks başlığı: '{path}'")
        if (zlib.crc32(raw[: hdr_sz - 8]) & 0xFFFFFFFF) != hdr_crc:
            raise DataCorruptionError(f"IVF-PQ başlık CRC32 hatası: '{path}'")

        body = raw[hdr_sz:]
        if (zlib.crc32(body) & 0xFFFFFFFF) != body_crc:
            raise DataCorruptionError(f"IVF-PQ gövde CRC32 hatası: '{path}'")

        pos = 0
        pq = ProductQuantizer.from_bytes(body[pos : pos + pq_len])
        pos += pq_len

        cent_bytes = n_clusters * dim * 4
        centroids = np.frombuffer(body[pos : pos + cent_bytes], dtype=np.float32).reshape((n_clusters, dim)).copy()
        pos += cent_bytes

        cnt_bytes = n_clusters * 8
        counts = np.frombuffer(body[pos : pos + cnt_bytes], dtype=np.uint64).copy()
        pos += cnt_bytes

        idx = cls(
            n_clusters=n_clusters,
            nprobe=nprobe,
            n_subvectors=m_cnt,
            n_centroids=k_cnt,
            metric=_ID_TO_METRIC[metric_id],  # type: ignore[arg-type]
        )
        idx._dimension = dim
        idx.centroids = centroids
        idx._centroid_sq_norms = compute_sq_norms(centroids)
        idx.pq = pq
        idx._init_u16_mode()
        idx._total_items = int(total_items)
        idx._cluster_ids = []
        idx._cluster_codes_t = []
        code_dtype = np.uint16 if idx._use_u16_codes else np.uint8

        for cid in range(n_clusters):
            cnt = int(counts[cid])
            if cnt == 0:
                idx._cluster_ids.append(np.empty(0, dtype=np.int32))
                idx._cluster_codes_t.append(np.empty((m_cnt, 0), dtype=code_dtype))
                continue
            id_len = cnt * 8
            c_ids = np.frombuffer(body[pos : pos + id_len], dtype=np.int64).astype(np.int32)
            pos += id_len
            code_len = m_cnt * cnt
            c_codes_u8 = np.frombuffer(body[pos : pos + code_len], dtype=np.uint8).reshape((m_cnt, cnt))
            pos += code_len
            idx._cluster_ids.append(c_ids)
            idx._cluster_codes_t.append(idx._encode_cluster_codes(c_codes_u8))

        idx._trained = True
        idx._build_precomputed_lut()
        return idx, int(epoch)

    def __repr__(self) -> str:
        status = "trained" if self._trained else "untrained"
        m = self.pq.n_subvectors if self.pq is not None else self.n_subvectors
        return (
            f"<IVFPQIndex(C={self.n_clusters}, nprobe={self.nprobe}, M={m}, "
            f"metric='{self.metric}', status={status}, items={self._total_items})>"
        )
