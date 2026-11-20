"""
microvectordb.indexes.hnsw
--------------------------
Hierarchical Navigable Small World (HNSW) grafik tabanlı ANN indeksi.
Opsiyonel `hnswlib` kütüphanesi üzerinden yüksek hızlı O(log N) arama sunar.
Çekirdek kütüphane `hnswlib` olmadan da çalışmaya devam eder (lazy import).
"""

from typing import Callable, Optional, Tuple
import numpy as np

from microvectordb.metrics.kernels import validate_metric
from microvectordb.types import MetricType

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
    ) -> None:
        try:
            import hnswlib
        except ImportError as exc:
            raise ImportError(
                "HNSW indeksi için opsiyonel 'hnswlib' paketi gereklidir: "
                "pip install microvectordb[hnsw] (veya pip install hnswlib)"
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

        self._index = hnswlib.Index(space=self._space, dim=self.dimension)
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Çoklu veya tekil sorgu araması.
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
        if target_ef != self.ef_search:
            self._index.set_ef(target_ef)

        try:
            if filter_fn is not None:
                labels, raw_dists = self._index.knn_query(q32, k=actual_k, filter=filter_fn)
            else:
                labels, raw_dists = self._index.knn_query(q32, k=actual_k)
        finally:
            if target_ef != prev_ef:
                self._index.set_ef(prev_ef)

        dists, sims = self._convert_distances(raw_dists)
        return labels.astype(np.int64, copy=False), dists, sims

    def save(self, path: str) -> None:
        self._index.save_index(path)

    def load(self, path: str, max_elements: int) -> None:
        self._capacity = max(64, max_elements)
        self._index.load_index(path, max_elements=self._capacity)
        self._index.set_ef(self.ef_search)
        self._active_count = self._index.get_current_count()

    def __repr__(self) -> str:
        return (
            f"<HNSWIndex(dim={self.dimension}, metric='{self.metric}', "
            f"M={self.M}, ef_construction={self.ef_construction}, "
            f"ef_search={self.ef_search}, count={self._active_count})>"
        )
