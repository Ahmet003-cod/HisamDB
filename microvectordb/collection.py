"""
microvectordb.collection
------------------------
Koleksiyon sınıfı: Vektör depolama (VectorBuffer), ters metadata indeksi
(MetadataIndex), indeks motorları (Flat / IVF / IVF-PQ / HNSW / DiskIndex),
kuantizasyon (Float16 / INT8 / Binary / PQ), maliyet tabanlı Query Optimizer,
epoch/CRC32 doğrulamalı indeks disk kalıcılığı, RWLock eşzamanlılık güvenliği
ve delta kalıcılık yönetiminin ana merkezi.

v0.2.0 ve v0.3.0 API'leri ile %100 geriye dönük uyumludur.
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from microvectordb.concurrency import RWLock
from microvectordb.filtering import MetadataIndex, matches_filter
from microvectordb.indexes import (
    DiskIndex,
    DiskStreamingScanner,
    HNSWIndex,
    IVFIndex,
    IVFPQIndex,
    QueryOptimizer,
    is_hnsw_available,
)
from microvectordb.metrics import (
    compute_distances,
    top_k_indices,
    validate_metric,
    validate_vectors,
)
from microvectordb.quantization import (
    BinaryQuantizer,
    ProductQuantizer,
    ScalarQuantizer,
)
from microvectordb.storage import StorageEngine, VectorBuffer
from microvectordb.types import (
    BatchVectors,
    DataCorruptionError,
    DimensionMismatchError,
    IndexType,
    Metadata,
    MetricType,
    QuantizationType,
    QueryPlan,
    QueryResult,
    SearchResultItem,
    StorageDType,
    VectorDBError,
    VectorLike,
)

_VALID_INDEX_TYPES = ("flat", "ivf", "ivf_pq", "hnsw", "disk", "auto")
_VALID_QUANTIZATIONS = ("none", "float16", "int8", "binary", "pq")
_VALID_STORAGE_DTYPES = ("float32", "float16")
_AUTO_ANN_THRESHOLD = 20_000


class Collection:
    """
    Yüksek performanslı, düşük bellek tüketimli, thread-safe vektör koleksiyonu.
    """

    def __init__(
        self,
        name: str,
        storage: StorageEngine,
        dimension: Optional[int] = None,
        metric: MetricType = "cosine",
        created_at: Optional[float] = None,
        index_type: IndexType = "flat",
        quantization: QuantizationType = "none",
        storage_dtype: StorageDType = "float32",
        use_mmap: bool = False,
        keep_raw_vectors: bool = True,
        index_params: Optional[Dict[str, Any]] = None,
        data_epoch: int = 0,
    ):
        validate_metric(metric)
        if index_type not in _VALID_INDEX_TYPES:
            raise ValueError(f"Geçersiz index_type: '{index_type}'. Desteklenenler: {_VALID_INDEX_TYPES}")
        if quantization not in _VALID_QUANTIZATIONS:
            raise ValueError(f"Geçersiz quantization: '{quantization}'. Desteklenenler: {_VALID_QUANTIZATIONS}")
        if storage_dtype not in _VALID_STORAGE_DTYPES:
            raise ValueError(f"Geçersiz storage_dtype: '{storage_dtype}'. Desteklenenler: {_VALID_STORAGE_DTYPES}")

        self.name = name
        self._storage = storage
        self.dimension = dimension
        self.metric: MetricType = metric
        self.created_at = created_at or time.time()
        self.index_type: IndexType = index_type
        self.quantization: QuantizationType = quantization
        self.storage_dtype: StorageDType = "float16" if quantization == "float16" else storage_dtype
        self.use_mmap: bool = bool(use_mmap)
        self.keep_raw_vectors: bool = bool(keep_raw_vectors)
        self.index_params: Dict[str, Any] = dict(index_params or {})
        self._data_epoch: int = int(data_epoch)
        self._index_dirty: bool = False
        self._index_loaded_from_disk: bool = False

        self._lock = RWLock()
        self._storage.ensure_collection_storage(self.name)

        # Dinamik bitişik vektör tamponu (sq_norms + inv_norms dahil)
        buf_dtype = np.float16 if self.storage_dtype == "float16" else np.float32
        self._vbuf = VectorBuffer(dimension=self.dimension, dtype=buf_dtype)

        # Kayıt metadata & doküman yapıları
        self._ids: List[str] = []
        self._id_to_idx: Dict[str, int] = {}
        self._documents: List[Optional[str]] = []
        self._metadatas: List[Optional[Metadata]] = []
        self._meta_index = MetadataIndex()

        # Geriye dönük uyumluluk bayrağı (v0.2.0 testleri için)
        self._norm_cache_valid: bool = False

        # Opsiyonel ANN / Disk indeksleri
        self._ivf_index: Optional[IVFIndex] = None
        self._ivf_pq_index: Optional[IVFPQIndex] = None
        self._hnsw_index: Optional[HNSWIndex] = None
        self._disk_index: Optional[DiskIndex] = None
        self._disk_scanner = DiskStreamingScanner(
            block_rows=int(self.index_params.get("stream_block_rows", 4096))
        )

        # Opsiyonel INT8 / Binary / PQ kuantizasyon durumları
        self._scalar_quantizer: Optional[ScalarQuantizer] = None
        self._quantized_int8: Optional[np.ndarray] = None
        self._sq_scaled_sq: Optional[np.ndarray] = None
        self._sq_recon_inv: Optional[np.ndarray] = None

        self._binary_quantizer: Optional[BinaryQuantizer] = None
        self._quantized_binary: Optional[np.ndarray] = None

        self._pq_quantizer: Optional[ProductQuantizer] = None
        self._quantized_pq_t: Optional[np.ndarray] = None  # (M, N) bitişik uint8

        # Diskten mevcut verileri ve (varsa) güncel indeksleri yükle
        self._load_from_storage()

    # ------------------------------------------------------------------ #
    # Geriye Dönük Uyumluluk Özellikleri (_vectors, _vectors_normalized)
    # ------------------------------------------------------------------ #
    @property
    def _vectors(self) -> np.ndarray:
        """Sıfır-kopya aktif vektör matrisi görünümü (N, D)."""
        return self._vbuf.view

    @_vectors.setter
    def _vectors(self, value: np.ndarray) -> None:
        self._vbuf.load_array(np.asarray(value, dtype=self._vbuf.dtype), is_mmap=False)

    @property
    def _vectors_normalized(self) -> Optional[np.ndarray]:
        if not self._norm_cache_valid or self._vbuf.size == 0:
            return None
        v32 = self._vbuf.view.astype(np.float32, copy=False)
        return (v32 * self._vbuf.inv_norms[:, None]).astype(np.float32, copy=False)

    @_vectors_normalized.setter
    def _vectors_normalized(self, value: Optional[np.ndarray]) -> None:
        self._norm_cache_valid = value is not None

    def _invalidate_norm_cache(self) -> None:
        self._norm_cache_valid = False

    def _get_normalized_vectors(self) -> Optional[np.ndarray]:
        if self._vbuf.size == 0:
            return None
        self._norm_cache_valid = True
        return self._vectors_normalized

    # ------------------------------------------------------------------ #
    # Disk Yükleme, Çökme Kurtarma ve Anında İndeks Yükleme
    # ------------------------------------------------------------------ #
    def _load_from_storage(self) -> None:
        """Diskteki verileri yükler; varsa epoch/CRC32 uyumlu indeksleri anında açar."""
        items = self._storage.load_all_items(self.name)
        if not items:
            self._vbuf = VectorBuffer(
                dimension=self.dimension,
                dtype=np.float16 if self.storage_dtype == "float16" else np.float32,
            )
            return

        # DiskIndex veya mmap modunda vektör dosyasını bellek-eşlemeli açarak RAM koru
        load_as_mmap = self.use_mmap or self.index_type == "disk"
        vecs = self._storage.load_vectors(
            self.name, self.dimension, use_mmap=load_as_mmap, verify_crc=not load_as_mmap
        )
        n_items = len(items)
        n_vecs = vecs.shape[0]

        if n_vecs > n_items:
            if isinstance(vecs, np.memmap):
                mm = getattr(vecs, "_mmap", None)
                if mm is not None:
                    mm.close()
                self._storage.truncate_vectors(self.name, n_items)
                vecs = self._storage.load_vectors(
                    self.name, self.dimension, use_mmap=load_as_mmap, verify_crc=False
                )
            else:
                self._storage.truncate_vectors(self.name, n_items)
                vecs = np.ascontiguousarray(vecs[:n_items])
        elif n_vecs < n_items:
            raise DataCorruptionError(
                f"Koleksiyon '{self.name}' veri tutarsızlığı: "
                f"SQLite kayıt sayısı ({n_items}) > vektör sayısı ({n_vecs})."
            )

        if self.dimension is None and vecs.ndim == 2 and vecs.shape[1] > 0:
            self.dimension = int(vecs.shape[1])

        self._vbuf.load_array(vecs, is_mmap=isinstance(vecs, np.memmap))
        self._ids = [item["id"] for item in items]
        self._id_to_idx = {item["id"]: idx for idx, item in enumerate(items)}
        self._documents = [item["document"] for item in items]
        self._metadatas = [item["metadata"] for item in items]
        self._meta_index.rebuild(self._metadatas)

        if len(self._ids) > 0:
            self._rebuild_quantization_locked()
            if not self._try_load_persisted_index_locked():
                self._rebuild_ann_index_locked()

    def _try_load_persisted_index_locked(self) -> bool:
        """
        Diskte kayıtlı, epoch + boyut + metrik + CRC32 eşleşen bir indeks varsa
        yeniden inşa etmeden milisaniyeler içinde yükler.
        """
        if not self._storage.is_persistent:
            return False
        eff = self._effective_index_type()
        n = len(self._ids)
        if eff == "flat" or n == 0 or self.dimension is None:
            return False

        try:
            if eff == "hnsw" and is_hnsw_available():
                bin_p = self._storage.get_index_file_path(self.name, ".hnsw.bin")
                meta_p = self._storage.get_index_file_path(self.name, ".hnsw.meta.json")
                if bin_p and meta_p and os.path.exists(bin_p) and os.path.exists(meta_p):
                    idx, saved_epoch = HNSWIndex.load_with_meta(bin_p, meta_p, verify_crc=True)
                    if (
                        saved_epoch == self._data_epoch
                        and idx.count == n
                        and idx.dimension == self.dimension
                        and idx.metric == self.metric
                    ):
                        self._hnsw_index = idx
                        self._index_loaded_from_disk = True
                        self._index_dirty = False
                        return True
            elif eff == "ivf":
                ivf_p = self._storage.get_index_file_path(self.name, ".ivf.mvdbidx")
                if ivf_p and os.path.exists(ivf_p):
                    idx, saved_epoch = IVFIndex.load(ivf_p)
                    if (
                        saved_epoch == self._data_epoch
                        and idx._total_items == n
                        and idx._dimension == self.dimension
                        and idx.metric == self.metric
                    ):
                        self._ivf_index = idx
                        self._index_loaded_from_disk = True
                        self._index_dirty = False
                        return True
            elif eff == "ivf_pq":
                pq_p = self._storage.get_index_file_path(self.name, ".ivfpq.mvdbidx")
                if pq_p and os.path.exists(pq_p):
                    idx, saved_epoch = IVFPQIndex.load(pq_p)
                    if (
                        saved_epoch == self._data_epoch
                        and idx.total_items == n
                        and idx._dimension == self.dimension
                        and idx.metric == self.metric
                    ):
                        self._ivf_pq_index = idx
                        self._index_loaded_from_disk = True
                        self._index_dirty = False
                        if not self.keep_raw_vectors:
                            self._vbuf = VectorBuffer(dimension=self.dimension, dtype=self._vbuf.dtype)
                        return True
            elif eff == "disk":
                d_vec = self._storage.get_index_file_path(self.name, ".mvdbdisk")
                d_meta = self._storage.get_index_file_path(self.name, ".disk.meta")
                if d_vec and d_meta and os.path.exists(d_vec) and os.path.exists(d_meta):
                    idx, saved_epoch = DiskIndex.load(d_vec, d_meta)
                    if (
                        saved_epoch == self._data_epoch
                        and idx._total_items == n
                        and idx._dimension == self.dimension
                        and idx.metric == self.metric
                    ):
                        self._disk_index = idx
                        self._index_loaded_from_disk = True
                        self._index_dirty = False
                        return True
        except Exception:
            # Bozuk/uyumsuz indeks dosyası varsa güvenli biçimde yeniden oluşturmaya düş
            return False
        return False

    def _save_ann_index_locked(self) -> None:
        """Aktif ANN/Disk indeksini güncel epoch ve CRC32 ile diske atomik kaydeder."""
        if not self._storage.is_persistent or len(self._ids) == 0:
            return
        if self._hnsw_index is not None:
            bin_p = self._storage.get_index_file_path(self.name, ".hnsw.bin")
            meta_p = self._storage.get_index_file_path(self.name, ".hnsw.meta.json")
            if bin_p and meta_p:
                self._hnsw_index.save_with_meta(bin_p, meta_p, epoch=self._data_epoch)
                self._index_dirty = False
        elif self._ivf_index is not None and self._ivf_index.is_trained:
            ivf_p = self._storage.get_index_file_path(self.name, ".ivf.mvdbidx")
            if ivf_p:
                self._ivf_index.save(ivf_p, epoch=self._data_epoch)
                self._index_dirty = False
        elif self._ivf_pq_index is not None and self._ivf_pq_index.is_trained:
            pq_p = self._storage.get_index_file_path(self.name, ".ivfpq.mvdbidx")
            if pq_p:
                self._ivf_pq_index.save(pq_p, epoch=self._data_epoch)
                self._index_dirty = False
        elif self._disk_index is not None and self._disk_index.is_trained:
            d_meta = self._storage.get_index_file_path(self.name, ".disk.meta")
            if d_meta:
                self._disk_index.save_meta(d_meta, epoch=self._data_epoch)
                self._index_dirty = False

    def _save_meta_locked(self) -> None:
        self._storage.save_collection_meta(
            self.name,
            self.dimension,
            self.metric,
            self.created_at,
            index_type=self.index_type,
            quantization=self.quantization,
            storage_dtype=self.storage_dtype,
            data_epoch=self._data_epoch,
            index_params=self.index_params,
        )

    # ------------------------------------------------------------------ #
    # Kuantizasyon & ANN İndeks Yönetimi (write_lock altında çağrılır)
    # ------------------------------------------------------------------ #
    def _rebuild_quantization_locked(self) -> None:
        if self.quantization in ("none", "float16") or self._vbuf.size == 0:
            self._scalar_quantizer = None
            self._quantized_int8 = None
            self._sq_scaled_sq = None
            self._sq_recon_inv = None
            self._binary_quantizer = None
            self._quantized_binary = None
            self._pq_quantizer = None
            self._quantized_pq_t = None
            return

        v32 = self._vbuf.view.astype(np.float32, copy=False)
        if self.quantization == "int8":
            sq = ScalarQuantizer(percentile=self.index_params.get("sq_percentile"))
            sq.fit(v32)
            enc = sq.encode(v32)
            scaled_sq, recon_inv = sq.precompute_db_norms(enc)
            self._scalar_quantizer = sq
            self._quantized_int8 = enc
            self._sq_scaled_sq = scaled_sq
            self._sq_recon_inv = recon_inv
            if not self.keep_raw_vectors:
                self._vbuf = VectorBuffer(dimension=self.dimension, dtype=self._vbuf.dtype)
        elif self.quantization == "binary":
            bq = BinaryQuantizer(use_mean_threshold=self.index_params.get("binary_mean_threshold", True))
            bq.fit(v32)
            self._binary_quantizer = bq
            self._quantized_binary = bq.encode(v32)
            if not self.keep_raw_vectors:
                self._vbuf = VectorBuffer(dimension=self.dimension, dtype=self._vbuf.dtype)
        elif self.quantization == "pq":
            pq = ProductQuantizer(
                n_subvectors=self.index_params.get("pq_subvectors"),
                n_centroids=int(self.index_params.get("pq_centroids", 256)),
                metric=self.metric,
                n_iter=int(self.index_params.get("pq_iter", 15)),
            ).fit(v32)
            self._pq_quantizer = pq
            self._quantized_pq_t = pq.encode_transposed(v32)
            if not self.keep_raw_vectors:
                self._vbuf = VectorBuffer(dimension=self.dimension, dtype=self._vbuf.dtype)

    def _append_quantization_locked(self, new_vecs_f32: np.ndarray) -> None:
        if self.quantization in ("none", "float16"):
            return
        if self.quantization == "int8":
            if self._scalar_quantizer is None or not self._scalar_quantizer.is_fitted or self._quantized_int8 is None:
                self._rebuild_quantization_locked()
                return
            enc_new = self._scalar_quantizer.encode(new_vecs_f32)
            s_sq, r_inv = self._scalar_quantizer.precompute_db_norms(enc_new)
            self._quantized_int8 = np.vstack([self._quantized_int8, enc_new])
            assert self._sq_scaled_sq is not None and self._sq_recon_inv is not None
            self._sq_scaled_sq = np.concatenate([self._sq_scaled_sq, s_sq])
            self._sq_recon_inv = np.concatenate([self._sq_recon_inv, r_inv])
            if not self.keep_raw_vectors:
                self._vbuf = VectorBuffer(dimension=self.dimension, dtype=self._vbuf.dtype)
        elif self.quantization == "binary":
            if self._binary_quantizer is None or not self._binary_quantizer.is_fitted or self._quantized_binary is None:
                self._rebuild_quantization_locked()
                return
            enc_new = self._binary_quantizer.encode(new_vecs_f32)
            self._quantized_binary = np.vstack([self._quantized_binary, enc_new])
            if not self.keep_raw_vectors:
                self._vbuf = VectorBuffer(dimension=self.dimension, dtype=self._vbuf.dtype)
        elif self.quantization == "pq":
            if self._pq_quantizer is None or not self._pq_quantizer.is_fitted or self._quantized_pq_t is None:
                self._rebuild_quantization_locked()
                return
            enc_new_t = self._pq_quantizer.encode_transposed(new_vecs_f32)
            self._quantized_pq_t = np.ascontiguousarray(
                np.concatenate([self._quantized_pq_t, enc_new_t], axis=1)
            )
            if not self.keep_raw_vectors:
                self._vbuf = VectorBuffer(dimension=self.dimension, dtype=self._vbuf.dtype)

    def _effective_index_type(self) -> str:
        if self.index_type != "auto":
            return self.index_type
        if len(self._ids) < _AUTO_ANN_THRESHOLD:
            return "flat"
        return "hnsw" if is_hnsw_available() else "ivf"

    def _release_ann_indexes_locked(self) -> None:
        if self._disk_index is not None:
            self._disk_index.close()
        self._ivf_index = None
        self._ivf_pq_index = None
        self._hnsw_index = None
        self._disk_index = None

    def _rebuild_ann_index_locked(self, source_vectors: Optional[np.ndarray] = None) -> None:
        self._release_ann_indexes_locked()
        eff = self._effective_index_type()
        n = len(self._ids)
        if n == 0 or self.dimension is None or eff == "flat":
            return

        if source_vectors is not None:
            v_src = source_vectors
        elif self._vbuf.size == n:
            v_src = self._vbuf.view
        elif self._storage.is_persistent:
            v_src = self._storage.load_vectors(self.name, self.dimension, use_mmap=False, verify_crc=False)
        else:
            return

        if eff == "ivf":
            v32 = v_src.astype(np.float32, copy=False)
            default_clusters = max(1, min(int(np.sqrt(n)), 1024))
            n_clusters = int(self.index_params.get("n_clusters", default_clusters))
            nprobe = int(self.index_params.get("nprobe", max(1, min(n_clusters // 8, 16))))
            ivf = IVFIndex(n_clusters=min(n_clusters, n), nprobe=nprobe, metric=self.metric)
            ivf.train(v32, n_iter=int(self.index_params.get("n_iter", 15)))
            self._ivf_index = ivf
            self._index_dirty = True
        elif eff == "ivf_pq":
            v32 = v_src.astype(np.float32, copy=False)
            default_clusters = max(1, min(int(np.sqrt(n)), 512))
            n_clusters = int(self.index_params.get("n_clusters", default_clusters))
            nprobe = int(self.index_params.get("nprobe", max(1, min(n_clusters // 8, 16))))
            ivfpq = IVFPQIndex(
                n_clusters=min(n_clusters, n),
                nprobe=nprobe,
                n_subvectors=self.index_params.get("pq_subvectors"),
                n_centroids=int(self.index_params.get("pq_centroids", 256)),
                metric=self.metric,
            )
            ivfpq.train(
                v32,
                n_iter=int(self.index_params.get("n_iter", 12)),
                pq_iter=int(self.index_params.get("pq_iter", 12)),
            )
            self._ivf_pq_index = ivfpq
            self._index_dirty = True
            if not self.keep_raw_vectors:
                self._vbuf = VectorBuffer(dimension=self.dimension, dtype=self._vbuf.dtype)
        elif eff == "hnsw":
            v32 = v_src.astype(np.float32, copy=False)
            hnsw = HNSWIndex(
                dimension=self.dimension,
                metric=self.metric,
                M=int(self.index_params.get("M", 16)),
                ef_construction=int(self.index_params.get("ef_construction", 128)),
                ef_search=int(self.index_params.get("ef_search", 64)),
                initial_capacity=max(1024, n),
            )
            hnsw.add_items(v32, np.arange(n, dtype=np.int64))
            self._hnsw_index = hnsw
            self._index_dirty = True
        elif eff == "disk":
            default_clusters = max(1, min(int(np.sqrt(n)), 512))
            n_clusters = int(self.index_params.get("n_clusters", default_clusters))
            nprobe = int(self.index_params.get("nprobe", max(1, min(n_clusters // 8, 16))))
            d_vec = self._storage.get_index_file_path(self.name, ".mvdbdisk")
            d_meta = self._storage.get_index_file_path(self.name, ".disk.meta")
            d_idx = DiskIndex(
                n_clusters=min(n_clusters, n),
                nprobe=nprobe,
                metric=self.metric,
                storage_dtype=self.storage_dtype,
            )
            d_idx.build(
                v_src,
                disk_vec_path=d_vec,
                meta_path=d_meta,
                epoch=self._data_epoch,
                n_iter=int(self.index_params.get("n_iter", 12)),
            )
            self._disk_index = d_idx
            self._index_dirty = False

    def _append_ann_index_locked(self, new_vecs_f32: np.ndarray, start_idx: int) -> None:
        eff = self._effective_index_type()
        if eff == "flat":
            return

        if eff == "hnsw":
            if self._hnsw_index is None:
                self._rebuild_ann_index_locked()
            else:
                labels = np.arange(start_idx, start_idx + new_vecs_f32.shape[0], dtype=np.int64)
                self._hnsw_index.add_items(new_vecs_f32, labels)
                self._index_dirty = True
        elif eff == "ivf":
            if self._ivf_index is None or not self._ivf_index.is_trained:
                if len(self._ids) >= int(self.index_params.get("min_train_size", 64)):
                    self._rebuild_ann_index_locked()
            else:
                self._ivf_index.add_vectors(new_vecs_f32, start_idx=start_idx)
                self._index_dirty = True
        elif eff == "ivf_pq":
            if self._ivf_pq_index is None or not self._ivf_pq_index.is_trained:
                if len(self._ids) >= int(self.index_params.get("min_train_size", 64)):
                    src = new_vecs_f32 if start_idx == 0 else None
                    self._rebuild_ann_index_locked(source_vectors=src)
            else:
                self._ivf_pq_index.add_vectors(new_vecs_f32, start_idx=start_idx)
                self._index_dirty = True
        elif eff == "disk":
            # DiskIndex bölümlenmiş mmap dosyası kullanır; artımlı eklemede yeniden bölümlenir
            if len(self._ids) >= int(self.index_params.get("min_train_size", 32)):
                self._rebuild_ann_index_locked()

    def build_index(
        self,
        index_type: Optional[IndexType] = None,
        save: bool = True,
        **params: Any,
    ) -> None:
        """Koleksiyon üzerinde indeksi (yeniden) kurar ve kalıcı modda diske kaydeder."""
        with self._lock.write_lock():
            if index_type is not None:
                if index_type not in _VALID_INDEX_TYPES:
                    raise ValueError(f"Geçersiz index_type: '{index_type}'")
                self.index_type = index_type
            if params:
                self.index_params.update(params)
            self._rebuild_ann_index_locked()
            self._save_meta_locked()
            if save:
                self._save_ann_index_locked()

    # ------------------------------------------------------------------ #
    # CRUD: Ekleme & Güncelleme (Delta Persistence + Rollback Güvenli)
    # ------------------------------------------------------------------ #
    def add(
        self,
        ids: List[str],
        embeddings: Optional[BatchVectors] = None,
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Metadata]] = None,
        *,
        vectors: Optional[BatchVectors] = None,
        metadata: Optional[List[Metadata]] = None,
    ) -> None:
        """
        Koleksiyona yeni vektörler ekler (O(M) artımlı bellek ve disk yazımı).
        Hata anında RAM durumu otomatik geri alınır (rollback).
        """
        raw_vecs = embeddings if embeddings is not None else vectors
        raw_metas = metadatas if metadatas is not None else metadata

        if len(ids) == 0:
            if raw_vecs is not None:
                arr_check = np.asarray(raw_vecs)
                if arr_check.size != 0:
                    raise ValueError(
                        f"ID sayısı (0) ile embedding sayısı ({arr_check.shape[0]}) eşit olmalıdır."
                    )
            return

        if raw_vecs is None:
            raise ValueError("'embeddings' veya 'vectors' parametresi zorunludur.")

        arr = validate_vectors(raw_vecs, dtype=np.float32)
        n_items = len(ids)
        if arr.shape[0] != n_items:
            raise ValueError(
                f"ID sayısı ({n_items}) ile embedding sayısı ({arr.shape[0]}) eşit olmalıdır."
            )
        if documents is not None and len(documents) != n_items:
            raise ValueError("Document sayısı ID sayısıyla eşit olmalıdır.")
        if raw_metas is not None and len(raw_metas) != n_items:
            raise ValueError("Metadata sayısı ID sayısıyla eşit olmalıdır.")

        seen_in_batch = set()
        for item_id in ids:
            if not isinstance(item_id, str) or not item_id:
                raise ValueError("Tüm ID'ler boş olmayan metin (str) olmalıdır.")
            if item_id in seen_in_batch:
                raise VectorDBError(f"Aynı add() çağrısında tekrarlanan ID tespit edildi: '{item_id}'.")
            seen_in_batch.add(item_id)

        with self._lock.write_lock():
            if self.dimension is None:
                new_dim = int(arr.shape[1])
            elif arr.shape[1] != self.dimension:
                raise DimensionMismatchError(
                    f"Vektör boyutu uyuşmazlığı: Beklenen {self.dimension}, gelen {arr.shape[1]}"
                )
            else:
                new_dim = self.dimension

            for item_id in ids:
                if item_id in self._id_to_idx:
                    raise VectorDBError(
                        f"ID '{item_id}' zaten bu koleksiyonda mevcut. Güncellemek için 'upsert' kullanın."
                    )

            prev_size = len(self._ids)
            prev_dim = self.dimension
            prev_epoch = self._data_epoch
            self.dimension = new_dim
            self._data_epoch += 1

            drop_raw_mode = (
                not self.keep_raw_vectors
                and (self.quantization in ("int8", "binary", "pq") or self.index_type == "ivf_pq")
            )
            keep_in_vbuf = not drop_raw_mode

            # 1. RAM güncelle
            start_idx = prev_size
            end_idx = prev_size + n_items
            if keep_in_vbuf:
                start_idx, end_idx = self._vbuf.append(arr)
                disk_slice = self._vbuf.view[start_idx:end_idx]
            else:
                disk_slice = (
                    np.clip(arr, np.float32(-65504.0), np.float32(65504.0)).astype(np.float16)
                    if self.storage_dtype == "float16"
                    else arr
                )

            new_docs = [documents[i] if documents is not None else None for i in range(n_items)]
            new_metas = [raw_metas[i] if raw_metas is not None else None for i in range(n_items)]

            for i, item_id in enumerate(ids):
                self._ids.append(item_id)
                self._id_to_idx[item_id] = start_idx + i
                self._documents.append(new_docs[i])
                self._metadatas.append(new_metas[i])

            # 2. Disk delta kalıcılığı (hata anında rollback!)
            try:
                if self._storage.is_persistent:
                    if prev_dim is None:
                        self._save_meta_locked()
                    else:
                        self._storage.update_collection_epoch(self.name, self._data_epoch)
                    self._storage.append_vectors_delta(
                        self.name,
                        disk_slice,
                        expected_prev_count=prev_size,
                    )
                    delta_items = [
                        (ids[i], start_idx + i, new_docs[i], new_metas[i])
                        for i in range(n_items)
                    ]
                    self._storage.save_items_batch(self.name, delta_items)
            except Exception:
                if keep_in_vbuf:
                    self._vbuf.truncate(prev_size)
                del self._ids[prev_size:]
                del self._documents[prev_size:]
                del self._metadatas[prev_size:]
                for item_id in ids:
                    self._id_to_idx.pop(item_id, None)
                self.dimension = prev_dim
                self._data_epoch = prev_epoch
                if self._storage.is_persistent:
                    try:
                        self._storage.truncate_vectors(self.name, prev_size)
                        self._storage.update_collection_epoch(self.name, prev_epoch)
                    except Exception:
                        pass
                raise

            # 3. İndeksleri ve kuantizasyonu güncelle
            self._meta_index.add_batch(new_metas)
            if not keep_in_vbuf and prev_size == 0 and self.quantization in ("int8", "binary", "pq"):
                if self.quantization == "int8":
                    sq = ScalarQuantizer(percentile=self.index_params.get("sq_percentile")).fit(arr)
                    enc = sq.encode(arr)
                    s_sq, r_inv = sq.precompute_db_norms(enc)
                    self._scalar_quantizer = sq
                    self._quantized_int8 = enc
                    self._sq_scaled_sq = s_sq
                    self._sq_recon_inv = r_inv
                elif self.quantization == "binary":
                    bq = BinaryQuantizer(use_mean_threshold=self.index_params.get("binary_mean_threshold", True)).fit(arr)
                    self._binary_quantizer = bq
                    self._quantized_binary = bq.encode(arr)
                elif self.quantization == "pq":
                    pq = ProductQuantizer(
                        n_subvectors=self.index_params.get("pq_subvectors"),
                        n_centroids=int(self.index_params.get("pq_centroids", 256)),
                        metric=self.metric,
                        n_iter=int(self.index_params.get("pq_iter", 15)),
                    ).fit(arr)
                    self._pq_quantizer = pq
                    self._quantized_pq_t = pq.encode_transposed(arr)
            else:
                self._append_quantization_locked(arr)
            self._append_ann_index_locked(arr, start_idx=start_idx)
            self._invalidate_norm_cache()

    def upsert(
        self,
        ids: List[str],
        embeddings: Optional[BatchVectors] = None,
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Metadata]] = None,
        *,
        vectors: Optional[BatchVectors] = None,
        metadata: Optional[List[Metadata]] = None,
    ) -> None:
        """Vektörleri ekler veya varsa yerinde O(1) günceller (Add or Update)."""
        raw_vecs = embeddings if embeddings is not None else vectors
        raw_metas = metadatas if metadatas is not None else metadata

        if len(ids) == 0:
            return
        if raw_vecs is None:
            raise ValueError("'embeddings' veya 'vectors' parametresi zorunludur.")

        arr = validate_vectors(raw_vecs, dtype=np.float32)
        n_items = len(ids)
        if arr.shape[0] != n_items:
            raise ValueError(
                f"ID sayısı ({n_items}) ile embedding sayısı ({arr.shape[0]}) eşit olmalıdır."
            )
        if documents is not None and len(documents) != n_items:
            raise ValueError("Document sayısı ID sayısıyla eşit olmalıdır.")
        if raw_metas is not None and len(raw_metas) != n_items:
            raise ValueError("Metadata sayısı ID sayısıyla eşit olmalıdır.")

        dedup_order: Dict[str, int] = {}
        for i, item_id in enumerate(ids):
            if not isinstance(item_id, str) or not item_id:
                raise ValueError("Tüm ID'ler boş olmayan metin (str) olmalıdır.")
            dedup_order[item_id] = i

        with self._lock.write_lock():
            if self.dimension is None:
                self.dimension = int(arr.shape[1])
            elif arr.shape[1] != self.dimension:
                raise DimensionMismatchError(
                    f"Vektör boyutu uyuşmazlığı: Beklenen {self.dimension}, gelen {arr.shape[1]}"
                )

            self._data_epoch += 1
            to_add_ids: List[str] = []
            to_add_vecs: List[np.ndarray] = []
            to_add_docs: List[Optional[str]] = []
            to_add_metas: List[Optional[Metadata]] = []
            updated_rows: List[Tuple[str, int, Optional[str], Optional[Metadata], np.ndarray]] = []

            for item_id, i in dedup_order.items():
                doc = documents[i] if documents is not None else None
                meta = raw_metas[i] if raw_metas is not None else None
                vec = arr[i]

                if item_id in self._id_to_idx:
                    idx = self._id_to_idx[item_id]
                    if self._vbuf.size == len(self._ids):
                        self._vbuf.update_row(idx, vec)
                    if documents is not None:
                        self._documents[idx] = doc
                    if raw_metas is not None:
                        self._metadatas[idx] = meta
                        self._meta_index.update(idx, meta)
                    updated_rows.append(
                        (item_id, idx, self._documents[idx], self._metadatas[idx], vec)
                    )
                    if self._hnsw_index is not None:
                        self._hnsw_index.update_item(vec, idx)
                        self._index_dirty = True
                else:
                    to_add_ids.append(item_id)
                    to_add_vecs.append(vec)
                    to_add_docs.append(doc)
                    to_add_metas.append(meta)

            prev_size = len(self._ids)
            new_block: Optional[np.ndarray] = None
            start_idx = prev_size
            end_idx = prev_size

            if to_add_ids:
                new_block = np.ascontiguousarray(np.stack(to_add_vecs, axis=0), dtype=np.float32)
                if self._vbuf.size == prev_size:
                    start_idx, end_idx = self._vbuf.append(new_block)
                else:
                    end_idx = prev_size + len(to_add_ids)
                for i, item_id in enumerate(to_add_ids):
                    self._ids.append(item_id)
                    self._id_to_idx[item_id] = start_idx + i
                    self._documents.append(to_add_docs[i])
                    self._metadatas.append(to_add_metas[i])
                self._meta_index.add_batch(to_add_metas)

            if self._storage.is_persistent:
                self._save_meta_locked()
                for _, idx, _, _, vec in updated_rows:
                    v_disk = self._vbuf.view[idx] if self._vbuf.size == len(self._ids) else vec
                    self._storage.update_vector_row(self.name, idx, v_disk)
                if to_add_ids and new_block is not None:
                    v_append = self._vbuf.view[start_idx:end_idx] if self._vbuf.size == len(self._ids) else new_block
                    self._storage.append_vectors_delta(
                        self.name, v_append, expected_prev_count=prev_size
                    )
                delta_items = [
                    (s_id, idx, d, m) for s_id, idx, d, m, _ in updated_rows
                ] + [
                    (to_add_ids[i], start_idx + i, to_add_docs[i], to_add_metas[i])
                    for i in range(len(to_add_ids))
                ]
                self._storage.save_items_batch(self.name, delta_items)

            if updated_rows:
                if self.quantization in ("int8", "binary", "pq"):
                    self._rebuild_quantization_locked()
                if self._ivf_index is not None:
                    self._ivf_index.rebuild(self._vbuf.view.astype(np.float32, copy=False))
                    self._index_dirty = True
                if self._ivf_pq_index is not None or self._disk_index is not None:
                    self._rebuild_ann_index_locked()
            elif new_block is not None:
                self._append_quantization_locked(new_block)
                self._append_ann_index_locked(new_block, start_idx=start_idx)

            self._invalidate_norm_cache()

    # ------------------------------------------------------------------ #
    # Metadata Filtreleme & Sorgu Planlayıcı (explain)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _matches_filter(item_meta: Optional[Metadata], where: Optional[Dict[str, Any]]) -> bool:
        return matches_filter(item_meta, where)

    def _plan_query_locked(
        self,
        k: int,
        where: Optional[Dict[str, Any]],
        candidate_indices: Optional[np.ndarray],
        ef_search: Optional[int] = None,
        nprobe: Optional[int] = None,
        strategy: str = "auto",
        filter_strategy: str = "auto",
    ) -> QueryPlan:
        total = len(self._ids)
        eff = self._effective_index_type()
        ivf_clusters = 0
        ivf_np = 8
        if self._ivf_index is not None and self._ivf_index.is_trained:
            ivf_clusters = self._ivf_index.n_clusters
            ivf_np = self._ivf_index.nprobe
        elif self._ivf_pq_index is not None and self._ivf_pq_index.is_trained:
            ivf_clusters = self._ivf_pq_index.n_clusters
            ivf_np = self._ivf_pq_index.nprobe
        elif self._disk_index is not None and self._disk_index.is_trained:
            ivf_clusters = self._disk_index.n_clusters
            ivf_np = self._disk_index.nprobe

        return QueryOptimizer.plan(
            total_vectors=total,
            dimension=self.dimension or 1,
            k=k,
            effective_index=eff,
            quantization=self.quantization,
            is_mmap=self._vbuf.is_mmap,
            has_filter=bool(where),
            candidate_count=len(candidate_indices) if candidate_indices is not None else total,
            has_raw_vectors=(self._vbuf.size == total and total > 0),
            hnsw_available=(self._hnsw_index is not None),
            ivf_ready=(self._ivf_index is not None and self._ivf_index.is_trained),
            ivf_n_clusters=ivf_clusters,
            ivf_default_nprobe=ivf_np,
            ivf_pq_ready=(self._ivf_pq_index is not None and self._ivf_pq_index.is_trained),
            disk_index_ready=(self._disk_index is not None and self._disk_index.is_trained),
            default_ef_search=self._hnsw_index.ef_search if self._hnsw_index is not None else 64,
            requested_ef_search=ef_search,
            requested_nprobe=nprobe,
            strategy=strategy,
            filter_strategy=filter_strategy,
        )

    def explain(
        self,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        ef_search: Optional[int] = None,
        nprobe: Optional[int] = None,
        strategy: str = "auto",
        filter_strategy: str = "auto",
    ) -> QueryPlan:
        """
        Sorguyu çalıştırmadan Query Optimizer'ın seçeceği yürütme planını döndürür.
        """
        with self._lock.read_lock():
            cand = self._meta_index.query_indices(where) if where else None
            return self._plan_query_locked(
                k=n_results,
                where=where,
                candidate_indices=cand,
                ef_search=ef_search,
                nprobe=nprobe,
                strategy=strategy,
                filter_strategy=filter_strategy,
            )

    # ------------------------------------------------------------------ #
    # Sorgu / Arama Motoru
    # ------------------------------------------------------------------ #
    def _build_query_result(
        self,
        n_queries: int,
        index_lists: List[np.ndarray],
        dist_lists: List[np.ndarray],
        sim_lists: List[np.ndarray],
        score_threshold: Optional[float] = None,
    ) -> QueryResult:
        out_ids: List[List[str]] = []
        out_dists: List[List[float]] = []
        out_sims: List[List[float]] = []
        out_docs: List[List[Optional[str]]] = []
        out_metas: List[List[Optional[Metadata]]] = []

        ids_ref = self._ids
        docs_ref = self._documents
        metas_ref = self._metadatas

        for q_idx in range(n_queries):
            idxs = index_lists[q_idx]
            ds = dist_lists[q_idx]
            ss = sim_lists[q_idx]

            if score_threshold is not None and len(ss) > 0:
                keep = ss >= score_threshold
                idxs = idxs[keep]
                ds = ds[keep]
                ss = ss[keep]

            out_ids.append([ids_ref[int(i)] for i in idxs])
            out_dists.append([float(x) for x in ds])
            out_sims.append([float(x) for x in ss])
            out_docs.append([docs_ref[int(i)] for i in idxs])
            out_metas.append([metas_ref[int(i)] for i in idxs])

        return QueryResult(
            ids=out_ids,
            distances=out_dists,
            similarities=out_sims,
            documents=out_docs,
            metadatas=out_metas,
        )

    def _distances_to_similarities(self, dists: np.ndarray) -> np.ndarray:
        if self.metric == "cosine":
            return np.float32(1.0) - dists
        if self.metric == "dot_product":
            return -dists
        return np.float32(1.0) / (np.float32(1.0) + dists)

    def query(
        self,
        query_embeddings: Optional[Union[VectorLike, BatchVectors]] = None,
        query_embedding: Optional[Union[VectorLike, BatchVectors]] = None,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
        ef_search: Optional[int] = None,
        nprobe: Optional[int] = None,
        strategy: str = "auto",
        filter_strategy: str = "auto",
    ) -> QueryResult:
        """
        Query Optimizer güdümlü yüksek hızlı benzerlik araması.
        """
        raw = query_embeddings if query_embeddings is not None else query_embedding
        if raw is None:
            raise ValueError(
                "Arama için 'query_embeddings' veya 'query_embedding' parametresi gereklidir."
            )

        q_arr = validate_vectors(raw, dtype=np.float32)
        n_queries = q_arr.shape[0]

        if n_results <= 0:
            return QueryResult(
                ids=[[] for _ in range(n_queries)],
                distances=[[] for _ in range(n_queries)],
                similarities=[[] for _ in range(n_queries)],
                documents=[[] for _ in range(n_queries)],
                metadatas=[[] for _ in range(n_queries)],
            )

        with self._lock.read_lock():
            total = len(self._ids)
            if total == 0:
                return QueryResult(
                    ids=[[] for _ in range(n_queries)],
                    distances=[[] for _ in range(n_queries)],
                    similarities=[[] for _ in range(n_queries)],
                    documents=[[] for _ in range(n_queries)],
                    metadatas=[[] for _ in range(n_queries)],
                )

            if self.dimension is not None and q_arr.shape[1] != self.dimension:
                raise DimensionMismatchError(
                    f"Sorgu vektör boyutu uyuşmazlığı: Beklenen {self.dimension}, gelen {q_arr.shape[1]}"
                )

            if self.metric == "cosine":
                self._norm_cache_valid = True

            candidate_indices: Optional[np.ndarray] = None
            if where:
                candidate_indices = self._meta_index.query_indices(where)

            plan = self._plan_query_locked(
                k=n_results,
                where=where,
                candidate_indices=candidate_indices,
                ef_search=ef_search,
                nprobe=nprobe,
                strategy=strategy,
                filter_strategy=filter_strategy,
            )

            if plan.execution_path == "empty":
                return QueryResult(
                    ids=[[] for _ in range(n_queries)],
                    distances=[[] for _ in range(n_queries)],
                    similarities=[[] for _ in range(n_queries)],
                    documents=[[] for _ in range(n_queries)],
                    metadatas=[[] for _ in range(n_queries)],
                )

            # --- YOL 1: HNSW (Filtresiz / Post-Filter Oversample / Callback) ---
            if plan.execution_path == "hnsw" and self._hnsw_index is not None:
                labels, dists, sims = self._hnsw_index.search(
                    q_arr, k=min(n_results, total), ef_search=plan.adaptive_ef_search
                )
                return self._build_query_result(
                    n_queries,
                    [labels[i] for i in range(n_queries)],
                    [dists[i] for i in range(n_queries)],
                    [sims[i] for i in range(n_queries)],
                    score_threshold=score_threshold,
                )

            if plan.execution_path == "hnsw_post_filter" and self._hnsw_index is not None and candidate_indices is not None:
                idx_l, dist_l, sim_l = self._hnsw_index.search_post_filtered(
                    q_arr,
                    allowed_sorted=candidate_indices,
                    k=min(n_results, len(candidate_indices)),
                    oversample_k=plan.oversample_k,
                    ef_search=plan.adaptive_ef_search,
                )
                return self._build_query_result(
                    n_queries, idx_l, dist_l, sim_l, score_threshold=score_threshold
                )

            if plan.execution_path == "hnsw_filter_cb" and self._hnsw_index is not None and candidate_indices is not None:
                allowed_set = set(candidate_indices.tolist())
                labels, dists, sims = self._hnsw_index.search(
                    q_arr,
                    k=min(n_results, len(allowed_set)),
                    ef_search=plan.adaptive_ef_search,
                    filter_fn=allowed_set.__contains__,
                )
                return self._build_query_result(
                    n_queries,
                    [labels[i] for i in range(n_queries)],
                    [dists[i] for i in range(n_queries)],
                    [sims[i] for i in range(n_queries)],
                    score_threshold=score_threshold,
                )

            # --- YOL 2: DiskIndex (Bölümlenmiş Out-of-Core MMap IVF) ---
            if plan.execution_path == "disk_partitioned" and self._disk_index is not None:
                allowed_mask = None
                if candidate_indices is not None:
                    allowed_mask = np.zeros(total, dtype=bool)
                    allowed_mask[candidate_indices] = True
                idx_l, dist_l, sim_l = self._disk_index.search_batch(
                    q_arr,
                    k=min(n_results, total),
                    nprobe=plan.adaptive_nprobe,
                    allowed_mask=allowed_mask,
                )
                return self._build_query_result(
                    n_queries, idx_l, dist_l, sim_l, score_threshold=score_threshold
                )

            # --- YOL 3: IVF-PQ (Residual Product Quantization) ---
            if plan.execution_path == "ivf_pq" and self._ivf_pq_index is not None:
                allowed_mask = None
                if candidate_indices is not None:
                    allowed_mask = np.zeros(total, dtype=bool)
                    allowed_mask[candidate_indices] = True
                idx_l, dist_l = self._ivf_pq_index.search_batch(
                    q_arr,
                    k=min(n_results, total),
                    nprobe=plan.adaptive_nprobe,
                    allowed_mask=allowed_mask,
                )
                sim_l = [self._distances_to_similarities(d) for d in dist_l]
                return self._build_query_result(
                    n_queries, idx_l, dist_l, sim_l, score_threshold=score_threshold
                )

            # --- YOL 4: IVF (Packed BLAS) ---
            if plan.execution_path == "ivf" and self._ivf_index is not None:
                allowed_mask = None
                if candidate_indices is not None:
                    allowed_mask = np.zeros(total, dtype=bool)
                    allowed_mask[candidate_indices] = True
                idx_l, dist_l = self._ivf_index.search_batch(
                    q_arr,
                    k=min(n_results, total),
                    nprobe=plan.adaptive_nprobe,
                    allowed_mask=allowed_mask,
                )
                sim_l = [self._distances_to_similarities(d) for d in dist_l]
                return self._build_query_result(
                    n_queries, idx_l, dist_l, sim_l, score_threshold=score_threshold
                )

            # --- YOL 5: Kuantize Flat (PQ / INT8 / Binary) ---
            if plan.execution_path == "quantized_pq" and self._pq_quantizer is not None and self._quantized_pq_t is not None:
                codes_t = (
                    np.ascontiguousarray(self._quantized_pq_t[:, candidate_indices])
                    if candidate_indices is not None
                    else self._quantized_pq_t
                )
                dists, sims = self._pq_quantizer.compute_batch_distances(q_arr, codes_t, metric=self.metric)
                k = min(n_results, codes_t.shape[1])
                idx_l, dist_l, sim_l = [], [], []
                for q_idx in range(n_queries):
                    loc = top_k_indices(dists[q_idx], k)
                    g_top = candidate_indices[loc] if candidate_indices is not None else loc
                    idx_l.append(g_top)
                    dist_l.append(dists[q_idx, loc])
                    sim_l.append(sims[q_idx, loc])
                return self._build_query_result(
                    n_queries, idx_l, dist_l, sim_l, score_threshold=score_threshold
                )

            if plan.execution_path == "quantized_int8" and self._scalar_quantizer is not None and self._quantized_int8 is not None:
                if candidate_indices is not None:
                    q_db = self._quantized_int8[candidate_indices]
                    s_sq = self._sq_scaled_sq[candidate_indices] if self._sq_scaled_sq is not None else None
                    r_inv = self._sq_recon_inv[candidate_indices] if self._sq_recon_inv is not None else None
                else:
                    q_db = self._quantized_int8
                    s_sq = self._sq_scaled_sq
                    r_inv = self._sq_recon_inv

                dists, sims = self._scalar_quantizer.compute_batch_distances(
                    q_arr, q_db, metric=self.metric, precomputed_scaled_sq=s_sq, precomputed_recon_inv=r_inv
                )
                k = min(n_results, q_db.shape[0])
                idx_l, dist_l, sim_l = [], [], []
                for q_idx in range(n_queries):
                    loc = top_k_indices(dists[q_idx], k)
                    g_top = candidate_indices[loc] if candidate_indices is not None else loc
                    idx_l.append(g_top)
                    dist_l.append(dists[q_idx, loc])
                    sim_l.append(sims[q_idx, loc])
                return self._build_query_result(
                    n_queries, idx_l, dist_l, sim_l, score_threshold=score_threshold
                )

            if plan.execution_path == "quantized_binary" and self._binary_quantizer is not None and self._quantized_binary is not None:
                b_db = self._quantized_binary[candidate_indices] if candidate_indices is not None else self._quantized_binary
                b_dists, b_sims = self._binary_quantizer.compute_batch_distances(q_arr, b_db)
                n_cand = b_db.shape[0]
                k = min(n_results, n_cand)
                can_rescore = self.keep_raw_vectors and self._vbuf.size == total
                oversample_mult = int(self.index_params.get("rescore_multiplier", 25))
                oversample = min(n_cand, max(k * oversample_mult, 256)) if can_rescore else k

                idx_l, dist_l, sim_l = [], [], []
                for q_idx in range(n_queries):
                    loc = top_k_indices(b_dists[q_idx], oversample)
                    g_top = candidate_indices[loc] if candidate_indices is not None else loc
                    if can_rescore and len(g_top) > 0:
                        rs_vecs = self._vbuf.view[g_top]
                        rs_inv = self._vbuf.inv_norms[g_top] if self.metric == "cosine" else None
                        rs_sq = self._vbuf.sq_norms[g_top] if self.metric == "euclidean" else None
                        rs_d, rs_s = compute_distances(
                            q_arr[q_idx : q_idx + 1],
                            rs_vecs,
                            metric=self.metric,
                            db_inv_norms=rs_inv,
                            db_sq_norms=rs_sq,
                        )
                        rs_ord = top_k_indices(rs_d[0], min(k, len(g_top)))
                        idx_l.append(g_top[rs_ord])
                        dist_l.append(rs_d[0, rs_ord])
                        sim_l.append(rs_s[0, rs_ord])
                    else:
                        idx_l.append(g_top[:k])
                        dist_l.append(b_dists[q_idx, loc[:k]])
                        sim_l.append(b_sims[q_idx, loc[:k]])
                return self._build_query_result(
                    n_queries, idx_l, dist_l, sim_l, score_threshold=score_threshold
                )

            # --- YOL 6: Sabit RAM Bütçeli Disk Akış Taraması (`disk_stream`) ---
            if plan.execution_path == "disk_stream":
                idx_l, dist_l, sim_l = self._disk_scanner.search(
                    q_arr,
                    self._vbuf.view,
                    metric=self.metric,
                    k=min(n_results, total),
                    inv_norms=self._vbuf.inv_norms if self.metric == "cosine" else None,
                    sq_norms=self._vbuf.sq_norms if self.metric == "euclidean" else None,
                    candidate_indices=candidate_indices,
                )
                return self._build_query_result(
                    n_queries, idx_l, dist_l, sim_l, score_threshold=score_threshold
                )

            # --- YOL 7: Optimize Exact Flat BLAS (inv_norms / sq_norms) ---
            if candidate_indices is not None:
                sub_vectors = self._vbuf.view[candidate_indices]
                sub_inv = self._vbuf.inv_norms[candidate_indices] if self.metric == "cosine" else None
                sub_sq = self._vbuf.sq_norms[candidate_indices] if self.metric == "euclidean" else None
            else:
                sub_vectors = self._vbuf.view
                sub_inv = self._vbuf.inv_norms if self.metric == "cosine" else None
                sub_sq = self._vbuf.sq_norms if self.metric == "euclidean" else None

            dists, sims = compute_distances(
                q_arr,
                sub_vectors,
                metric=self.metric,
                db_inv_norms=sub_inv,
                db_sq_norms=sub_sq,
            )

            k = min(n_results, sub_vectors.shape[0])
            idx_l, dist_l, sim_l = [], [], []
            for q_idx in range(n_queries):
                loc = top_k_indices(dists[q_idx], k)
                g_top = candidate_indices[loc] if candidate_indices is not None else loc
                idx_l.append(g_top)
                dist_l.append(dists[q_idx, loc])
                sim_l.append(sims[q_idx, loc])

            return self._build_query_result(
                n_queries, idx_l, dist_l, sim_l, score_threshold=score_threshold
            )

    def search(
        self,
        query_vector: VectorLike,
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
        ef_search: Optional[int] = None,
        nprobe: Optional[int] = None,
        strategy: str = "auto",
        filter_strategy: str = "auto",
    ) -> List[SearchResultItem]:
        """Tek sorgu için ergonomik arama arayüzü: List[SearchResultItem] döndürür."""
        res = self.query(
            query_embeddings=query_vector,
            n_results=limit,
            where=filter,
            score_threshold=score_threshold,
            ef_search=ef_search,
            nprobe=nprobe,
            strategy=strategy,
            filter_strategy=filter_strategy,
        )
        return [
            SearchResultItem(
                id=res.ids[0][i],
                score=res.similarities[0][i],
                distance=res.distances[0][i],
                document=res.documents[0][i],
                metadata=res.metadatas[0][i],
            )
            for i in range(len(res.ids[0]))
        ]

    # ------------------------------------------------------------------ #
    # Diğer CRUD Metotları (get, delete, count, peek, checkpoint, stats)
    # ------------------------------------------------------------------ #
    def get(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        with self._lock.read_lock():
            matched_indices: List[int] = []

            if ids is not None:
                seen = set()
                for s_id in ids:
                    if s_id in seen:
                        continue
                    idx = self._id_to_idx.get(s_id)
                    if idx is None:
                        continue
                    if where is not None and not matches_filter(self._metadatas[idx], where):
                        continue
                    seen.add(s_id)
                    matched_indices.append(idx)
                    if limit is not None and len(matched_indices) >= limit:
                        break
            elif where is not None:
                cand = self._meta_index.query_indices(where)
                if limit is not None:
                    cand = cand[:limit]
                matched_indices = cand.tolist()
            else:
                n_all = len(self._ids)
                end = n_all if limit is None else min(n_all, max(0, limit))
                matched_indices = list(range(end))

            return {
                "ids": [self._ids[i] for i in matched_indices],
                "documents": [self._documents[i] for i in matched_indices],
                "metadatas": [self._metadatas[i] for i in matched_indices],
            }

    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Kayıtları siler, tamponu sıkıştırır ve atomik olarak diske yazar."""
        if ids is None and where is None:
            raise ValueError("Silme için ya 'ids' ya da 'where' filtresi belirtilmelidir.")

        with self._lock.write_lock():
            to_delete_indices = set()
            if ids is not None:
                for s_id in ids:
                    idx = self._id_to_idx.get(s_id)
                    if idx is not None:
                        to_delete_indices.add(idx)

            if where is not None:
                cand = self._meta_index.query_indices(where)
                to_delete_indices.update(cand.tolist())

            if not to_delete_indices:
                return 0

            total_before = len(self._ids)
            deleted_count = len(to_delete_indices)
            keep_indices = [i for i in range(total_before) if i not in to_delete_indices]
            self._data_epoch += 1

            has_raw = self._vbuf.size == total_before
            if has_raw:
                self._vbuf.keep_indices(keep_indices)
            else:
                keep_arr = np.asarray(keep_indices, dtype=np.int64)
                if self._quantized_int8 is not None:
                    self._quantized_int8 = np.ascontiguousarray(self._quantized_int8[keep_arr])
                    if self._sq_scaled_sq is not None:
                        self._sq_scaled_sq = np.ascontiguousarray(self._sq_scaled_sq[keep_arr])
                    if self._sq_recon_inv is not None:
                        self._sq_recon_inv = np.ascontiguousarray(self._sq_recon_inv[keep_arr])
                if self._quantized_binary is not None:
                    self._quantized_binary = np.ascontiguousarray(self._quantized_binary[keep_arr])
                if self._quantized_pq_t is not None:
                    self._quantized_pq_t = np.ascontiguousarray(self._quantized_pq_t[:, keep_arr])

            if keep_indices:
                self._ids = [self._ids[i] for i in keep_indices]
                self._documents = [self._documents[i] for i in keep_indices]
                self._metadatas = [self._metadatas[i] for i in keep_indices]
                self._id_to_idx = {s_id: new_i for new_i, s_id in enumerate(self._ids)}
            else:
                self._ids.clear()
                self._documents.clear()
                self._metadatas.clear()
                self._id_to_idx.clear()

            if self._storage.is_persistent:
                self._save_meta_locked()
                if has_raw:
                    self._storage.save_vectors(self.name, self._vbuf.view, compute_crc=True)
                else:
                    disk_vecs = self._storage.load_vectors(
                        self.name, self.dimension, use_mmap=False, verify_crc=False
                    )
                    if disk_vecs.shape[0] == total_before:
                        kept_disk = (
                            np.ascontiguousarray(disk_vecs[keep_indices])
                            if keep_indices
                            else np.empty((0, self.dimension or 0), dtype=disk_vecs.dtype)
                        )
                        self._storage.save_vectors(self.name, kept_disk, compute_crc=True)
                compacted_items = [
                    (self._ids[i], i, self._documents[i], self._metadatas[i])
                    for i in range(len(self._ids))
                ]
                self._storage.replace_all_items_after_compaction(self.name, compacted_items)

            self._meta_index.rebuild(self._metadatas)
            if has_raw:
                self._rebuild_quantization_locked()
            self._rebuild_ann_index_locked()
            self._invalidate_norm_cache()
            return deleted_count

    def checkpoint(self) -> None:
        """
        Tam CRC32 sağlama toplamı ile vektörleri ve aktif ANN/Disk indeksini atomik kaydeder.
        """
        with self._lock.write_lock():
            if not self._storage.is_persistent:
                return
            if self._vbuf.size == len(self._ids):
                if self._vbuf.is_mmap:
                    arr_copy = np.ascontiguousarray(self._vbuf.view)
                    self._vbuf.load_array(arr_copy, is_mmap=False)
                self._storage.save_vectors(self.name, self._vbuf.view, compute_crc=True)
            self._save_meta_locked()
            self._save_ann_index_locked()

    def close(self) -> None:
        """Değişmiş indeksleri diske kaydedip açık tüm memmap tutamaçlarını kapatır."""
        with self._lock.write_lock():
            if self._storage.is_persistent and self._index_dirty and len(self._ids) > 0:
                try:
                    self._save_meta_locked()
                    self._save_ann_index_locked()
                except Exception:
                    pass
            if self._disk_index is not None:
                self._disk_index.close()
            self._vbuf.release_mmap()

    def get_all_ids(self) -> List[str]:
        with self._lock.read_lock():
            return list(self._ids)

    def count(self) -> int:
        with self._lock.read_lock():
            return len(self._ids)

    def peek(self, limit: int = 5) -> Dict[str, Any]:
        return self.get(limit=limit)

    def stats(self) -> Dict[str, Any]:
        """Koleksiyonun gerçek RAM, disk ve indeks istatistiklerini döndürür."""
        with self._lock.read_lock():
            q_bytes = 0
            if self._quantized_int8 is not None:
                q_bytes += int(self._quantized_int8.nbytes)
                if self._sq_scaled_sq is not None:
                    q_bytes += int(self._sq_scaled_sq.nbytes)
                if self._sq_recon_inv is not None:
                    q_bytes += int(self._sq_recon_inv.nbytes)
            if self._quantized_binary is not None:
                q_bytes += int(self._quantized_binary.nbytes)
            if self._quantized_pq_t is not None:
                q_bytes += int(self._quantized_pq_t.nbytes)
                if self._pq_quantizer is not None and self._pq_quantizer.codebooks is not None:
                    q_bytes += int(self._pq_quantizer.codebooks.nbytes)

            idx_bytes = 0
            if self._ivf_pq_index is not None:
                idx_bytes += self._ivf_pq_index.memory_bytes()
            if self._disk_index is not None:
                idx_bytes += self._disk_index.ram_bytes()

            return {
                "name": self.name,
                "count": len(self._ids),
                "dimension": self.dimension,
                "metric": self.metric,
                "index_type": self.index_type,
                "effective_index": self._effective_index_type(),
                "quantization": self.quantization,
                "storage_dtype": self.storage_dtype,
                "is_mmap": self._vbuf.is_mmap or (self._disk_index is not None),
                "data_epoch": self._data_epoch,
                "index_loaded_from_disk": self._index_loaded_from_disk,
                "vector_ram_mb": self._vbuf.memory_bytes() / (1024 * 1024),
                "quantized_ram_mb": q_bytes / (1024 * 1024),
                "index_ram_mb": idx_bytes / (1024 * 1024),
                "vector_disk_mb": self._storage.get_vector_file_size(self.name) / (1024 * 1024),
                "metadata_index": self._meta_index.stats(),
            }

    def __repr__(self) -> str:
        return (
            f"<Collection(name='{self.name}', count={self.count()}, "
            f"dimension={self.dimension}, metric='{self.metric}', "
            f"index='{self.index_type}', quant='{self.quantization}')>"
        )
