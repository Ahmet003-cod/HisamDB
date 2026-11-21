"""
hisamdb.indexes.hnsw
--------------------------
Hierarchical Navigable Small World (HNSW) grafik tabanlı ANN indeksi.
Opsiyonel `hnswlib` kütüphanesi üzerinden yüksek hızlı O(log N) arama sunar.
Çekirdek kütüphane `hnswlib` olmadan da çalışmaya devam eder (lazy import).
"""

import json
import os
import threading
import time
import zlib
from typing import Callable, List, Optional, Set, Tuple
import numpy as np

from hisamdb.concurrency.rwlock import RWLock
from hisamdb.metrics.kernels import validate_metric
from hisamdb.types import DataCorruptionError, MetricType

_METRIC_TO_HNSW_SPACE = {
    "cosine": "cosine",
    "dot_product": "ip",
    "euclidean": "l2",
}


def is_hnsw_available() -> bool:
    """Sistemde hnswlib'in kurulu olup olmadığını kontrol eder."""
    try:
        import hnswlib  # noqa: F401
        return True
    except ImportError:
        return False


class HNSWIndex:
    """
    HNSW (Hierarchical Navigable Small World) ANN İndeksi.

    Parametreler:
        dimension: Vektör boyutu.
        metric: 'cosine', 'dot_product' veya 'euclidean'.
        M: Grafik düğüm başına maksimum bağlantı sayısı (varsayılan: 16).
        ef_construction: İndeks oluşturma kalitesi (varsayılan: 128).
        ef_search: Arama kalitesi / hız dengesi (varsayılan: 64).
        initial_capacity: Başlangıç kapasitesi (otomatik büyür).
    """

    def __init__(
        self,
        dimension: int,
        metric: MetricType = "cosine",
        M: int = 16,
        ef_construction: int = 128,
        ef_search: int = 64,
        initial_capacity: int = 1024,
        _skip_init: bool = False,
    ) -> None:
        try:
            import hnswlib
        except ImportError as exc:
            raise ImportError(
                "HNSW indeksi için opsiyonel 'hnswlib' paketi gereklidir: "
                "pip install hisamdb[hnsw] (veya pip install hnswlib)"
            ) from exc

        validate_metric(metric)
        self.dimension = int(dimension)
        self.metric: MetricType = metric
        self.M = int(M)
        self.ef_construction = int(ef_construction)
        self.ef_search = int(ef_search)
        self._space = _METRIC_TO_HNSW_SPACE[metric]
        self._capacity = max(64, int(initial_capacity))
        self._active_count = 0
        self._ef_lock = RWLock()

        self._index = hnswlib.Index(space=self._space, dim=self.dimension)
        if not _skip_init:
            self._index.init_index(
                max_elements=self._capacity,
                ef_construction=self.ef_construction,
                M=self.M,
                allow_replace_deleted=True,
            )
            self._index.set_ef(self.ef_search)

    @property
    def count(self) -> int:
        return self._active_count

    def set_ef(self, ef_search: int) -> None:
        self.ef_search = int(ef_search)
        self._index.set_ef(self.ef_search)

    def _ensure_capacity(self, needed: int) -> None:
        if needed <= self._capacity:
            return
        new_cap = max(needed, self._capacity * 2)
        self._index.resize_index(new_cap)
        self._capacity = new_cap

    def add_items(self, vectors: np.ndarray, labels: np.ndarray) -> None:
        """Vektörleri belirtilen tamsayı etiketlerle HNSW grafiğine ekler."""
        v32 = np.ascontiguousarray(vectors, dtype=np.float32)
        if v32.ndim == 1:
            v32 = v32.reshape(1, -1)
        lbl = np.asarray(labels, dtype=np.int64).ravel()
        if v32.shape[0] == 0:
            return
        self._ensure_capacity(self._index.get_current_count() + v32.shape[0])
        self._index.add_items(v32, lbl)
        self._active_count = self._index.get_current_count()

    def update_item(self, vector: np.ndarray, label: int) -> None:
        """Var olan etiketteki vektörü günceller."""
        v32 = np.ascontiguousarray(vector, dtype=np.float32).reshape(1, -1)
        self._index.add_items(v32, np.array([label], dtype=np.int64))

    def mark_deleted(self, label: int) -> None:
        try:
            self._index.mark_deleted(int(label))
            self._active_count = max(0, self._active_count - 1)
        except RuntimeError:
            pass

    def _convert_distances(self, raw_dists: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """hnswlib ham mesafelerini MicroVectorDB (distances, similarities) formatına dönüştürür."""
        d = raw_dists.astype(np.float32, copy=False)
        if self.metric == "cosine":
            # hnswlib cosine mesafesi = 1 - cos_sim
            sims = np.clip(np.float32(1.0) - d, np.float32(-1.0), np.float32(1.0))
            dists = np.float32(1.0) - sims
            return dists, sims
        if self.metric == "dot_product":
            # hnswlib ip mesafesi = 1 - dot
            sims = np.float32(1.0) - d
            dists = -sims
            return dists, sims
        # euclidean: hnswlib l2 karesel mesafe döndürür
        dists = np.sqrt(np.maximum(d, np.float32(0.0)))
        sims = np.float32(1.0) / (np.float32(1.0) + dists)
        return dists, sims

    def search(
        self,
        queries: np.ndarray,
        k: int = 10,
        ef_search: Optional[int] = None,
        filter_fn: Optional[Callable[[int], bool]] = None,
        num_threads: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Çoklu veya tekil sorgu araması (RWLock korumalı thread-safe `ef_search` yönetimi).
        Dönüş: (labels (Q, K), distances (Q, K), similarities (Q, K))
        """
        q32 = np.ascontiguousarray(queries, dtype=np.float32)
        if q32.ndim == 1:
            q32 = q32.reshape(1, -1)

        if k <= 0 or self._active_count == 0:
            q_cnt = q32.shape[0]
            return (
                np.empty((q_cnt, 0), dtype=np.int64),
                np.empty((q_cnt, 0), dtype=np.float32),
                np.empty((q_cnt, 0), dtype=np.float32),
            )

        actual_k = min(k, self._active_count)
        prev_ef = self.ef_search
        target_ef = max(ef_search or self.ef_search, actual_k)
        use_threads = 1 if filter_fn is not None else int(num_threads)

        if target_ef == prev_ef:
            with self._ef_lock.read_lock():
                if filter_fn is not None:
                    labels, raw_dists = self._index.knn_query(q32, k=actual_k, filter=filter_fn, num_threads=1)
                else:
                    labels, raw_dists = self._index.knn_query(q32, k=actual_k, num_threads=use_threads)
        else:
            with self._ef_lock.write_lock():
                self._index.set_ef(target_ef)
                try:
                    if filter_fn is not None:
                        labels, raw_dists = self._index.knn_query(q32, k=actual_k, filter=filter_fn, num_threads=1)
                    else:
                        labels, raw_dists = self._index.knn_query(q32, k=actual_k, num_threads=use_threads)
                finally:
                    self._index.set_ef(prev_ef)

        dists, sims = self._convert_distances(raw_dists)
        return labels.astype(np.int64, copy=False), dists, sims

    def search_post_filtered(
        self,
        queries: np.ndarray,
        allowed_sorted: np.ndarray,
        k: int = 10,
        oversample_k: Optional[int] = None,
        ef_search: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Yüksek seçicilikli filtrelerde (s >= %8) C++ içinde sıfır-GIL oversampling
        yapıp sonucu O(k_over * log M) `np.searchsorted` ile sıfır Python set
        tahsis ederek filtreler; k'dan az sonuç çıkarsa otomatik graf-içi callback'e düşer.
        """
        q32 = np.ascontiguousarray(queries, dtype=np.float32)
        if q32.ndim == 1:
            q32 = q32.reshape(1, -1)
        q_cnt = q32.shape[0]
        n_allowed = len(allowed_sorted)
        target_k = min(k, n_allowed)
        if target_k <= 0 or self._active_count == 0:
            return (
                [np.empty(0, dtype=np.int64) for _ in range(q_cnt)],
                [np.empty(0, dtype=np.float32) for _ in range(q_cnt)],
                [np.empty(0, dtype=np.float32) for _ in range(q_cnt)],
            )

        k_over = min(self._active_count, oversample_k or max(target_k * 4, 64))
        ef_use = max(ef_search or self.ef_search, k_over)
        labels, dists, sims = self.search(q32, k=k_over, ef_search=ef_use, filter_fn=None)

        out_l: List[np.ndarray] = []
        out_d: List[np.ndarray] = []
        out_s: List[np.ndarray] = []
        allowed_set: Optional[Set[int]] = None

        for qi in range(q_cnt):
            row_l = labels[qi]
            pos = np.searchsorted(allowed_sorted, row_l)
            in_bounds = pos < n_allowed
            safe_pos = np.where(in_bounds, pos, 0)
            keep_mask = in_bounds & (allowed_sorted[safe_pos] == row_l)
            kept_idx = np.flatnonzero(keep_mask)

            if len(kept_idx) >= target_k or k_over >= self._active_count:
                kept_idx = kept_idx[:target_k]
                out_l.append(row_l[kept_idx])
                out_d.append(dists[qi, kept_idx])
                out_s.append(sims[qi, kept_idx])
            else:
                # Korelasyonlu filtrede oversample yetmediyse yalnızca o an tembel set kurup callback'e düş
                if allowed_set is None:
                    allowed_set = set(allowed_sorted.tolist())
                fb_l, fb_d, fb_s = self.search(
                    q32[qi : qi + 1],
                    k=target_k,
                    ef_search=max(ef_search or self.ef_search, target_k * 2),
                    filter_fn=allowed_set.__contains__,
                )
                out_l.append(fb_l[0])
                out_d.append(fb_d[0])
                out_s.append(fb_s[0])
        return out_l, out_d, out_s

    def save(self, path: str) -> None:
        self._index.save_index(path)

    def load(self, path: str, max_elements: int) -> None:
        self._capacity = max(64, max_elements)
        self._index.load_index(path, max_elements=self._capacity)
        self._index.set_ef(self.ef_search)
        self._active_count = self._index.get_current_count()

    @staticmethod
    def _file_crc32(path: str) -> int:
        crc = 0
        with open(path, "rb") as f:
            while True:
                chunk = f.read(4 * 1024 * 1024)
                if not chunk:
                    break
                crc = zlib.crc32(chunk, crc)
        return crc & 0xFFFFFFFF

    def save_with_meta(self, bin_path: str, meta_path: str, epoch: int = 0) -> None:
        """
        HNSW grafiğini ve CRC32+epoch doğrulamalı yan manifest dosyasını atomik kaydeder.
        """
        suffix = f".tmp.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}"
        tmp_bin = bin_path + suffix
        tmp_meta = meta_path + suffix
        try:
            self._index.save_index(tmp_bin)
            bin_crc = self._file_crc32(tmp_bin)
            os.replace(tmp_bin, bin_path)

            manifest = {
                "format_version": 1,
                "epoch": int(epoch),
                "dimension": self.dimension,
                "metric": self.metric,
                "M": self.M,
                "ef_construction": self.ef_construction,
                "ef_search": self.ef_search,
                "count": self._active_count,
                "capacity": self._capacity,
                "bin_crc32": bin_crc,
            }
            with open(tmp_meta, "w", encoding="utf-8") as f:
                json.dump(manifest, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_meta, meta_path)
        except Exception:
            for p in (tmp_bin, tmp_meta):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            raise

    @classmethod
    def load_with_meta(
        cls,
        bin_path: str,
        meta_path: str,
        verify_crc: bool = True,
    ) -> Tuple["HNSWIndex", int]:
        """CRC32 ve manifest doğrulamasıyla diskten HNSW grafiğini anında yükler."""
        if not os.path.exists(bin_path) or not os.path.exists(meta_path):
            raise FileNotFoundError("HNSW indeks veya manifest dosyası bulunamadı.")
        with open(meta_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        if verify_crc:
            actual_crc = cls._file_crc32(bin_path)
            if actual_crc != manifest.get("bin_crc32"):
                raise DataCorruptionError(f"HNSW indeks CRC32 uyuşmazlığı: '{bin_path}'")

        idx = cls(
            dimension=int(manifest["dimension"]),
            metric=manifest["metric"],
            M=int(manifest.get("M", 16)),
            ef_construction=int(manifest.get("ef_construction", 128)),
            ef_search=int(manifest.get("ef_search", 64)),
            initial_capacity=max(64, int(manifest.get("capacity", manifest.get("count", 64)))),
            _skip_init=True,
        )
        idx.load(bin_path, max_elements=max(64, int(manifest.get("capacity", manifest.get("count", 64)))))
        return idx, int(manifest.get("epoch", 0))

    def __repr__(self) -> str:
        return (
            f"<HNSWIndex(dim={self.dimension}, metric='{self.metric}', "
            f"M={self.M}, ef_construction={self.ef_construction}, "
            f"ef_search={self.ef_search}, count={self._active_count})>"
        )
