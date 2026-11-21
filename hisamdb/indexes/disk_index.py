"""
hisamdb.indexes.disk_index
--------------------------------
Out-of-Core / Disk Tabanlı Arama ve İndeksleme Motoru.

İki mod sunar:
1. `DiskStreamingScanner`: RAM'den büyük `mmap` vektör dosyalarını sabit bellek
   bütçesiyle (ör. 4096 satırlık bloklar ~12 MB) tarayıp kesin (exact) Top-K üretir.
2. `DiskIndex` (Bölümlenmiş Disk İndeksi — Disk-Resident IVF):
   - RAM'de yalnızca kaba küme merkezleri (~128 KB) ve hafif küme ofset tablosu tutulur.
   - Küme sırasına dizilmiş tam vektörler diskteki `.mvdbdisk` dosyasında `np.memmap`
     üzerinden okunur.
   - Sorguda yalnızca en yakın `nprobe` kümenin (~%5) bitişik disk sayfaları okunur ->
     RAM tüketimini >%90 azaltırken yüksek QPS ve yüksek Recall@10 sağlar.
"""

import os
import struct
import threading
import time
import zlib
from typing import List, Optional, Tuple
import numpy as np

from hisamdb.metrics.kernels import (
    compute_distances,
    compute_inv_norms,
    compute_sq_norms,
    normalize_vectors,
    top_k_indices,
    validate_metric,
)
from hisamdb.storage.binary_format import load_vectors_file, write_vectors_atomic
from hisamdb.types import DataCorruptionError, MetricType

_DISK_META_MAGIC = b"MVDBDSK1"
_DISK_META_HDR_FMT = "<8sHHHHIIQQII"
# magic(8), ver(2), metric_id(2), n_clusters(2), nprobe(2), dim(4), dtype_code(4),
# total_items(8), epoch(8), hdr_crc(4), body_crc(4)
_METRIC_TO_ID = {"euclidean": 0, "cosine": 1, "dot_product": 2}
_ID_TO_METRIC = {v: k for k, v in _METRIC_TO_ID.items()}


class DiskStreamingScanner:
    """
    Sabit RAM bütçeli, bloklu (out-of-core) exact tarayıcı.
    10 GB'lık bir `.mvdbvec` dosyasını bile birkaç MB çalışma belleğiyle tarar.
    """

    def __init__(self, block_rows: int = 4096) -> None:
        self.block_rows = max(256, int(block_rows))

    def search(
        self,
        queries: np.ndarray,
        vectors_mmap: np.ndarray,
        metric: MetricType = "cosine",
        k: int = 10,
        inv_norms: Optional[np.ndarray] = None,
        sq_norms: Optional[np.ndarray] = None,
        candidate_indices: Optional[np.ndarray] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Bloklar halinde tarayıp her sorgu için (global_indices, distances, similarities) döndürür.
        """
        q_arr = np.asarray(queries, dtype=np.float32)
        if q_arr.ndim == 1:
            q_arr = q_arr.reshape(1, -1)
        q_cnt = q_arr.shape[0]
        total = vectors_mmap.shape[0] if candidate_indices is None else len(candidate_indices)

        if k <= 0 or total == 0:
            return (
                [np.empty(0, dtype=np.int64) for _ in range(q_cnt)],
                [np.empty(0, dtype=np.float32) for _ in range(q_cnt)],
                [np.empty(0, dtype=np.float32) for _ in range(q_cnt)],
            )

        best_ids: List[Optional[np.ndarray]] = [None] * q_cnt
        best_dists: List[Optional[np.ndarray]] = [None] * q_cnt
        best_sims: List[Optional[np.ndarray]] = [None] * q_cnt

        step = self.block_rows
        for start in range(0, total, step):
            end = min(start + step, total)
            if candidate_indices is None:
                g_ids = np.arange(start, end, dtype=np.int64)
                block = vectors_mmap[start:end]
                b_inv = inv_norms[start:end] if inv_norms is not None else None
                b_sq = sq_norms[start:end] if sq_norms is not None else None
            else:
                g_ids = candidate_indices[start:end]
                block = vectors_mmap[g_ids]
                b_inv = inv_norms[g_ids] if inv_norms is not None else None
                b_sq = sq_norms[g_ids] if sq_norms is not None else None

            dists, sims = compute_distances(
                q_arr,
                block,
                metric=metric,
                db_inv_norms=b_inv,
                db_sq_norms=b_sq,
            )

            k_block = min(k, len(g_ids))
            for q_i in range(q_cnt):
                loc = top_k_indices(dists[q_i], k_block)
                cur_ids = g_ids[loc]
                cur_d = dists[q_i, loc]
                cur_s = sims[q_i, loc]

                if best_ids[q_i] is None:
                    best_ids[q_i] = cur_ids
                    best_dists[q_i] = cur_d
                    best_sims[q_i] = cur_s
                else:
                    m_ids = np.concatenate([best_ids[q_i], cur_ids])  # type: ignore[list-item]
                    m_d = np.concatenate([best_dists[q_i], cur_d])  # type: ignore[list-item]
                    m_s = np.concatenate([best_sims[q_i], cur_s])  # type: ignore[list-item]
                    top = top_k_indices(m_d, min(k, len(m_d)))
                    best_ids[q_i] = m_ids[top]
                    best_dists[q_i] = m_d[top]
                    best_sims[q_i] = m_s[top]

        return (
            [x if x is not None else np.empty(0, dtype=np.int64) for x in best_ids],
            [x if x is not None else np.empty(0, dtype=np.float32) for x in best_dists],
            [x if x is not None else np.empty(0, dtype=np.float32) for x in best_sims],
        )


class DiskIndex:
    """
    Disk Tabanlı Bölümlenmiş Vektör İndeksi (Out-of-Core Partitioned MMap IVF).

    Vektör matrisi küme sırasına göre diskte (`.mvdbdisk`) saklanır ve `mmap` ile
    yalnızca seçilen `nprobe` kümelerin ardışık sayfaları okunur.
    """

    def __init__(
        self,
        n_clusters: int = 128,
        nprobe: int = 8,
        metric: MetricType = "cosine",
        storage_dtype: str = "float32",
        seed: int = 42,
    ) -> None:
        validate_metric(metric)
        self.n_clusters = max(1, int(n_clusters))
        self.nprobe = max(1, int(nprobe))
        self.metric: MetricType = metric
        self.storage_dtype = np.dtype(np.float16 if storage_dtype == "float16" else np.float32)
        self.seed = int(seed)

        self._dimension: int = 0
        self._total_items: int = 0
        self._trained: bool = False

        # RAM'de tutulan hafif meta veriler
        self.centroids: Optional[np.ndarray] = None
        self._centroid_sq_norms: Optional[np.ndarray] = None
        self._cluster_starts: Optional[np.ndarray] = None
        self._cluster_ends: Optional[np.ndarray] = None
        self._packed_ids: Optional[np.ndarray] = None
        self._packed_sq_norms: Optional[np.ndarray] = None
        self._packed_inv_norms: Optional[np.ndarray] = None

        # Diskteki bitişik küme vektörleri (np.memmap veya RAM fallback)
        self._disk_vectors: Optional[np.ndarray] = None
        self._disk_vec_path: Optional[str] = None

    @property
    def is_trained(self) -> bool:
        return self._trained

    def close(self) -> None:
        """Açık memmap dosya tutamacını Windows-güvenli biçimde kapatır."""
        if isinstance(self._disk_vectors, np.memmap):
            mm = getattr(self._disk_vectors, "_mmap", None)
            if mm is not None:
                try:
                    mm.close()
                except Exception:
                    pass
        self._disk_vectors = None

    def _assign_coarse(self, vectors: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        n = vectors.shape[0]
        c_cnt = centroids.shape[0]
        out = np.empty(n, dtype=np.int32)
        chunk = 8192
        if self.metric == "cosine":
            for s in range(0, n, chunk):
                e = min(s + chunk, n)
                v = vectors[s:e].astype(np.float32, copy=False)
                out[s:e] = np.argmax(np.dot(v, centroids.T), axis=1)
            return out

        c_sq = compute_sq_norms(centroids).reshape(1, c_cnt)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            v = vectors[s:e].astype(np.float32, copy=False)
            scores = np.dot(v, centroids.T)
            scores *= np.float32(-2.0)
            scores += c_sq
            out[s:e] = np.argmin(scores, axis=1)
        return out

    def build(
        self,
        vectors: np.ndarray,
        disk_vec_path: Optional[str] = None,
        meta_path: Optional[str] = None,
        epoch: int = 0,
        n_iter: int = 12,
    ) -> None:
        """
        Kaba kümeleri eğitir, vektörleri küme sırasına dizip diske yazar ve `mmap` ile bağlar.
        `disk_vec_path=None` (in-memory mod) ise bellek içi bitişik düzenle çalışır.
        """
        self.close()
        n, d = vectors.shape
        if n == 0:
            raise ValueError("DiskIndex.build() boş diziyle çağrılamaz.")

        self._dimension = d
        actual_c = min(self.n_clusters, n)
        self.n_clusters = actual_c
        rng = np.random.default_rng(self.seed)

        # Sınırlı alt-örneklem üzerinde eğitim (RAM dostu)
        max_train = min(n, max(actual_c * 256, 32768))
        if n > max_train:
            t_idx = np.sort(rng.choice(n, max_train, replace=False))
            sample = np.ascontiguousarray(vectors[t_idx], dtype=np.float32)
        else:
            sample = np.ascontiguousarray(vectors, dtype=np.float32)
        if self.metric == "cosine":
            sample = normalize_vectors(sample)

        # K-Means++ başlangıç + vektörize iterasyon
        init_cap = min(len(sample), max(actual_c * 32, 4096))
        init_s = sample[rng.choice(len(sample), init_cap, replace=False)] if len(sample) > init_cap else sample
        centroids = np.empty((actual_c, d), dtype=np.float32)
        centroids[0] = init_s[int(rng.integers(len(init_s)))]
        if self.metric == "cosine":
            min_d = np.maximum(np.float32(0.0), np.float32(1.0) - np.dot(init_s, centroids[0]))
            for c in range(1, actual_c):
                tot = float(min_d.sum())
                pick = int(rng.integers(len(init_s))) if tot <= 1e-12 else int(rng.choice(len(init_s), p=(min_d.astype(np.float64) / tot) / (min_d.astype(np.float64) / tot).sum()))
                centroids[c] = init_s[pick]
                np.minimum(min_d, np.maximum(np.float32(0.0), np.float32(1.0) - np.dot(init_s, centroids[c])), out=min_d)
        else:
            s_sq = compute_sq_norms(init_s)
            c0_sq = np.float32(np.dot(centroids[0], centroids[0]))
            min_d = np.maximum(np.float32(0.0), s_sq + c0_sq - np.float32(2.0) * np.dot(init_s, centroids[0]))
            for c in range(1, actual_c):
                tot = float(min_d.sum())
                pick = int(rng.integers(len(init_s))) if tot <= 1e-12 else int(rng.choice(len(init_s), p=(min_d.astype(np.float64) / tot) / (min_d.astype(np.float64) / tot).sum()))
                centroids[c] = init_s[pick]
                cc_sq = np.float32(np.dot(centroids[c], centroids[c]))
                np.minimum(min_d, np.maximum(np.float32(0.0), s_sq + cc_sq - np.float32(2.0) * np.dot(init_s, centroids[c])), out=min_d)

        for _ in range(n_iter):
            assign = self._assign_coarse(sample, centroids)
            counts = np.bincount(assign, minlength=actual_c)
            sums = np.zeros((actual_c, d), dtype=np.float32)
            np.add.at(sums, assign, sample)
            nz = counts > 0
            new_c = np.empty_like(centroids)
            new_c[nz] = sums[nz] / counts[nz, None].astype(np.float32)
            if not np.all(nz):
                for cid in np.flatnonzero(~nz):
                    new_c[cid] = sample[int(rng.integers(len(sample)))]
            if self.metric == "cosine":
                new_c = normalize_vectors(new_c)
            if float(np.linalg.norm(new_c - centroids)) < 1e-5:
                centroids = new_c
                break
            centroids = new_c

        # Tüm vektörleri kümelere ata ve küme sırasına göre diz
        if self.metric == "cosine":
            # Normları hesaplamadan argmax(v . c) işareti pozitifse normalize ile aynı küme; kesinlik için normalize blokla
            full_assign = np.empty(n, dtype=np.int32)
            for s in range(0, n, 8192):
                e = min(s + 8192, n)
                blk = normalize_vectors(vectors[s:e].astype(np.float32, copy=False))
                full_assign[s:e] = np.argmax(np.dot(blk, centroids.T), axis=1)
        else:
            full_assign = self._assign_coarse(vectors, centroids)

        order = np.argsort(full_assign, kind="stable").astype(np.int64)
        counts = np.bincount(full_assign, minlength=actual_c)
        starts = np.zeros(actual_c, dtype=np.int64)
        if actual_c > 1:
            np.cumsum(counts[:-1], out=starts[1:])
        ends = starts + counts

        reordered = np.ascontiguousarray(vectors[order], dtype=self.storage_dtype)
        sq, inv = compute_inv_norms(reordered)

        self.centroids = centroids
        self._centroid_sq_norms = compute_sq_norms(centroids)
        self._cluster_starts = starts
        self._cluster_ends = ends
        self._packed_ids = order
        self._packed_sq_norms = sq
        self._packed_inv_norms = inv
        self._total_items = n
        self._trained = True

        if disk_vec_path is not None:
            write_vectors_atomic(disk_vec_path, reordered, compute_data_crc=True)
            del reordered
            self._disk_vec_path = disk_vec_path
            self._disk_vectors = load_vectors_file(disk_vec_path, use_mmap=True, verify_crc=False)
            if meta_path is not None:
                self.save_meta(meta_path, epoch=epoch)
        else:
            self._disk_vectors = reordered

    def save_meta(self, meta_path: str, epoch: int = 0) -> None:
        if not self._trained or self.centroids is None:
            raise RuntimeError("Eğitilmemiş DiskIndex kaydedilemez.")
        assert self._cluster_starts is not None and self._cluster_ends is not None
        assert self._packed_ids is not None and self._packed_sq_norms is not None and self._packed_inv_norms is not None

        dtype_code = 1 if self.storage_dtype == np.float16 else 0
        body = b"".join([
            np.ascontiguousarray(self.centroids, dtype=np.float32).tobytes(),
            np.ascontiguousarray(self._cluster_starts, dtype=np.int64).tobytes(),
            np.ascontiguousarray(self._cluster_ends, dtype=np.int64).tobytes(),
            np.ascontiguousarray(self._packed_ids, dtype=np.int64).tobytes(),
            np.ascontiguousarray(self._packed_sq_norms, dtype=np.float32).tobytes(),
            np.ascontiguousarray(self._packed_inv_norms, dtype=np.float32).tobytes(),
        ])
        body_crc = zlib.crc32(body) & 0xFFFFFFFF
        prefix = struct.pack(
            "<8sHHHHIIQQ",
            _DISK_META_MAGIC,
            1,
            _METRIC_TO_ID[self.metric],
            self.n_clusters,
            self.nprobe,
            self._dimension,
            dtype_code,
            self._total_items,
            int(epoch),
        )
        hdr_crc = zlib.crc32(prefix) & 0xFFFFFFFF
        header = prefix + struct.pack("<II", hdr_crc, body_crc)

        tmp = f"{meta_path}.tmp.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}"
        try:
            with open(tmp, "wb") as f:
                f.write(header)
                f.write(body)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, meta_path)
        except Exception:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            raise

    @classmethod
    def load(cls, disk_vec_path: str, meta_path: str) -> Tuple["DiskIndex", int]:
        """Diskteki `.mvdbdisk` (mmap) ve meta dosyasını CRC32 doğrulayarak açar."""
        with open(meta_path, "rb") as f:
            raw = f.read()
        hdr_sz = struct.calcsize(_DISK_META_HDR_FMT)
        if len(raw) < hdr_sz:
            raise DataCorruptionError(f"DiskIndex meta başlığı eksik: '{meta_path}'")

        magic, ver, metric_id, n_clusters, nprobe, dim, dtype_code, total_items, epoch, hdr_crc, body_crc = (
            struct.unpack(_DISK_META_HDR_FMT, raw[:hdr_sz])
        )
        if magic != _DISK_META_MAGIC or ver != 1 or metric_id not in _ID_TO_METRIC:
            raise DataCorruptionError(f"Geçersiz DiskIndex meta başlığı: '{meta_path}'")
        if (zlib.crc32(raw[: hdr_sz - 8]) & 0xFFFFFFFF) != hdr_crc:
            raise DataCorruptionError(f"DiskIndex başlık CRC32 hatası: '{meta_path}'")

        body = raw[hdr_sz:]
        if (zlib.crc32(body) & 0xFFFFFFFF) != body_crc:
            raise DataCorruptionError(f"DiskIndex gövde CRC32 hatası: '{meta_path}'")

        pos = 0
        c_bytes = n_clusters * dim * 4
        centroids = np.frombuffer(body[pos : pos + c_bytes], dtype=np.float32).reshape((n_clusters, dim)).copy()
        pos += c_bytes
        s_bytes = n_clusters * 8
        starts = np.frombuffer(body[pos : pos + s_bytes], dtype=np.int64).copy()
        pos += s_bytes
        ends = np.frombuffer(body[pos : pos + s_bytes], dtype=np.int64).copy()
        pos += s_bytes
        id_bytes = total_items * 8
        packed_ids = np.frombuffer(body[pos : pos + id_bytes], dtype=np.int64).copy()
        pos += id_bytes
        n_bytes = total_items * 4
        sq_norms = np.frombuffer(body[pos : pos + n_bytes], dtype=np.float32).copy()
        pos += n_bytes
        inv_norms = np.frombuffer(body[pos : pos + n_bytes], dtype=np.float32).copy()

        mm = load_vectors_file(disk_vec_path, use_mmap=True, verify_crc=False)
        if mm.shape != (total_items, dim):
            if isinstance(mm, np.memmap) and getattr(mm, "_mmap", None) is not None:
                mm._mmap.close()
            raise DataCorruptionError("DiskIndex vektör dosyası boyut uyumsuzluğu.")

        idx = cls(
            n_clusters=n_clusters,
            nprobe=nprobe,
            metric=_ID_TO_METRIC[metric_id],  # type: ignore[arg-type]
            storage_dtype="float16" if dtype_code == 1 else "float32",
        )
        idx._dimension = dim
        idx._total_items = int(total_items)
        idx.centroids = centroids
        idx._centroid_sq_norms = compute_sq_norms(centroids)
        idx._cluster_starts = starts
        idx._cluster_ends = ends
        idx._packed_ids = packed_ids
        idx._packed_sq_norms = sq_norms
        idx._packed_inv_norms = inv_norms
        idx._disk_vec_path = disk_vec_path
        idx._disk_vectors = mm
        idx._trained = True
        return idx, int(epoch)

    def _nearest_clusters(self, q_f32: np.ndarray, nprobe: Optional[int] = None) -> np.ndarray:
        assert self.centroids is not None
        probe = min(nprobe or self.nprobe, len(self.centroids))
        if probe >= len(self.centroids):
            return np.arange(len(self.centroids), dtype=np.int64)
        if self.metric == "cosine":
            qn = normalize_vectors(q_f32.reshape(1, -1)).ravel()
            cdists = np.float32(1.0) - np.dot(self.centroids, qn)
        else:
            assert self._centroid_sq_norms is not None
            cdists = self._centroid_sq_norms - np.float32(2.0) * np.dot(self.centroids, q_f32)
        return np.argpartition(cdists, probe - 1)[:probe]

    def search_batch(
        self,
        queries: np.ndarray,
        k: int = 10,
        nprobe: Optional[int] = None,
        allowed_mask: Optional[np.ndarray] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        if not self._trained or self._disk_vectors is None:
            raise RuntimeError("DiskIndex henüz oluşturulmadı.")
        assert self._cluster_starts is not None and self._cluster_ends is not None
        assert self._packed_ids is not None and self._packed_sq_norms is not None and self._packed_inv_norms is not None

        q_arr = np.asarray(queries, dtype=np.float32)
        if q_arr.ndim == 1:
            q_arr = q_arr.reshape(1, -1)
        q_cnt = q_arr.shape[0]
        out_i, out_d, out_s = [], [], []

        for q_idx in range(q_cnt):
            qv = q_arr[q_idx]
            clusters = self._nearest_clusters(qv, nprobe=nprobe)
            c_ids_parts, c_d_parts, c_s_parts = [], [], []

            for cid_val in clusters:
                cid = int(cid_val)
                s = int(self._cluster_starts[cid])
                e = int(self._cluster_ends[cid])
                if e <= s:
                    continue
                ids_slice = self._packed_ids[s:e]
                if allowed_mask is not None:
                    keep = allowed_mask[ids_slice]
                    if not keep.any():
                        continue
                    ids_slice = ids_slice[keep]
                    block = np.ascontiguousarray(self._disk_vectors[s:e][keep])
                    b_inv = self._packed_inv_norms[s:e][keep] if self.metric == "cosine" else None
                    b_sq = self._packed_sq_norms[s:e][keep] if self.metric == "euclidean" else None
                else:
                    # Yalnızca bu kümenin ardışık disk sayfalarına dokunulur (mmap slice)!
                    block = self._disk_vectors[s:e]
                    b_inv = self._packed_inv_norms[s:e] if self.metric == "cosine" else None
                    b_sq = self._packed_sq_norms[s:e] if self.metric == "euclidean" else None

                d_mat, s_mat = compute_distances(
                    qv.reshape(1, -1), block, metric=self.metric, db_inv_norms=b_inv, db_sq_norms=b_sq
                )
                c_ids_parts.append(ids_slice)
                c_d_parts.append(d_mat[0])
                c_s_parts.append(s_mat[0])

            if not c_ids_parts:
                out_i.append(np.empty(0, dtype=np.int64))
                out_d.append(np.empty(0, dtype=np.float32))
                out_s.append(np.empty(0, dtype=np.float32))
                continue

            all_i = c_ids_parts[0] if len(c_ids_parts) == 1 else np.concatenate(c_ids_parts)
            all_d = c_d_parts[0] if len(c_d_parts) == 1 else np.concatenate(c_d_parts)
            all_s = c_s_parts[0] if len(c_s_parts) == 1 else np.concatenate(c_s_parts)
            top = top_k_indices(all_d, min(k, len(all_d)))
            out_i.append(all_i[top])
            out_d.append(all_d[top])
            out_s.append(all_s[top])

        return out_i, out_d, out_s

    def ram_bytes(self) -> int:
        """DiskIndex'in RAM'de tuttuğu bayt miktarı (mmap disk sayfaları hariç!)."""
        total = 0
        for arr in (
            self.centroids,
            self._cluster_starts,
            self._cluster_ends,
            self._packed_ids,
            self._packed_sq_norms,
            self._packed_inv_norms,
        ):
            if arr is not None:
                total += int(arr.nbytes)
        if self._disk_vectors is not None and not isinstance(self._disk_vectors, np.memmap):
            total += int(self._disk_vectors.nbytes)
        return total

    def __repr__(self) -> str:
        mode = "mmap-disk" if isinstance(self._disk_vectors, np.memmap) else "ram"
        return (
            f"<DiskIndex(C={self.n_clusters}, nprobe={self.nprobe}, metric='{self.metric}', "
            f"mode={mode}, items={self._total_items}, ram={self.ram_bytes()/(1024*1024):.2f}MB)>"
        )
