"""
hisamdb.collection
------------------------
Koleksiyon sınıfı:
- Vektör depolama (`VectorBuffer`, `float32`/`float16`, `mmap`)
- Ters metadata indeksi (`MetadataIndex` — tembel yüklemeli, hızlı açılış)
- İndeks motorları (`Flat`, `IVF`, `IVF-PQ`, `HNSW`, `DiskIndex`, `DiskStreamingScanner`)
- Kuantizasyon (`float16`, `int8`, `binary`, `pq`)
- Maliyet tabanlı `QueryOptimizer` (`explain()`)
- Sparse / BM25 + Hybrid Search (`RRF`, `min_max`, `z_score`) + RAG ilkelleri
- İkili WAL (`WriteAheadLog`) + Tombstone (<1 ms silme) + Otomatik/Manuel `compact()`
- RWLock eşzamanlılık güvenliği

v0.2.0, v0.3.0 ve v0.4.0 API'leri ile %100 geriye dönük uyumludur.
"""

import os
import threading
import time
import uuid
from concurrent.futures import Future
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np

from hisamdb.concurrency import (
    BackgroundTaskManager,
    BatchQueryExecutor,
    RWLock,
)
from hisamdb.filtering import MetadataIndex, matches_filter
from hisamdb.indexes import (
    DiskIndex,
    DiskStreamingScanner,
    HNSWIndex,
    IVFIndex,
    IVFPQIndex,
    QueryOptimizer,
    is_hnsw_available,
)
from hisamdb.metrics import (
    compute_distances,
    top_k_indices,
    validate_metric,
    validate_vectors,
)
from hisamdb.quantization import (
    BinaryQuantizer,
    ProductQuantizer,
    ScalarQuantizer,
)
from hisamdb.search import (
    BM25Index,
    SparseVector,
    SparseVectorIndex,
    apply_reranker,
    deduplicate_results,
    fuse_rrf,
    fuse_weighted,
    group_results_by_field,
)
from hisamdb.storage import (
    OP_CHECKPOINT,
    OP_COMPACT,
    OP_DELETE,
    StorageEngine,
    VectorBuffer,
    WriteAheadLog,
)
from hisamdb.types import (
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
    Yüksek performanslı, düşük bellek tüketimli, hibrit arama ve RAG destekli,
    thread-safe vektör koleksiyonu.
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
        query_executor: Optional[BatchQueryExecutor] = None,
        bg_tasks: Optional[BackgroundTaskManager] = None,
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
        self.index_params: Dict[str, Any] = dict(index_params or {})
        if use_mmap:
            self.index_params["_use_mmap"] = True
        self.use_mmap: bool = bool(self.index_params.get("_use_mmap", use_mmap))
        if not keep_raw_vectors:
            self.index_params["_keep_raw_vectors"] = False
        self.keep_raw_vectors: bool = bool(
            self.index_params.get("_keep_raw_vectors", keep_raw_vectors)
        )
        self._data_epoch: int = int(data_epoch)
        self._compaction_epoch: int = 0
        self._index_dirty: bool = False
        self._col_cache_dirty: bool = False
        self._index_loaded_from_disk: bool = False

        # Tombstone sıkıştırma politikası
        self._compaction_threshold: float = float(self.index_params.get("compaction_threshold", 0.25))
        self._auto_compact_min_deleted: int = int(self.index_params.get("auto_compact_min_deleted", 64))
        self._auto_compact_async: bool = bool(self.index_params.get("auto_compact_async", False))

        self._lock = RWLock()
        self._lazy_init_lock = threading.Lock()
        self._owns_executor = query_executor is None
        self._query_executor = query_executor or BatchQueryExecutor()
        self._owns_bg = bg_tasks is None
        self._bg_tasks = bg_tasks or BackgroundTaskManager(max_workers=2)
        self._storage.ensure_collection_storage(self.name)

        # İkili WAL (kalıcı modda etkin; varsayılan "normal" = OS flush, "full" = her yazımda fsync)
        wal_path = self._storage.get_index_file_path(self.name, ".wal")
        wal_sync_opt = self.index_params.get("wal_sync", "normal")
        wal_sync_full = (
            wal_sync_opt
            if isinstance(wal_sync_opt, bool)
            else str(wal_sync_opt).lower() == "full"
        )
        self._wal: Optional[WriteAheadLog] = (
            WriteAheadLog(wal_path, sync_on_write=wal_sync_full)
            if wal_path is not None
            else None
        )

        # Dinamik bitişik vektör tamponu
        buf_dtype = np.float16 if self.storage_dtype == "float16" else np.float32
        self._vbuf = VectorBuffer(dimension=self.dimension, dtype=buf_dtype)

        # Slot tabanlı kayıt yapıları + tombstone bitmask
        self._ids: List[str] = []
        self._id_to_idx: Dict[str, int] = {}
        self._documents: List[Optional[str]] = []
        self._metadatas: List[Optional[Metadata]] = []
        self._sparse_vectors: List[Optional[SparseVector]] = []
        self._deleted_mask: np.ndarray = np.zeros(0, dtype=bool)
        self._deleted_count: int = 0

        # Tembel (lazy) yardımcı indeksler — açılışta sıfır gereksiz CPU yükü (Sorun #1 çözümü)
        self._meta_index = MetadataIndex()
        self._meta_index_dirty: bool = False

        self._bm25_index = BM25Index(
            k1=float(self.index_params.get("bm25_k1", 1.5)),
            b=float(self.index_params.get("bm25_b", 0.75)),
        )
        self._bm25_dirty: bool = False

        self._sparse_index = SparseVectorIndex()
        self._sparse_dirty: bool = False

        # Geriye dönük uyumluluk bayrağı
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
        self._quantized_pq_t: Optional[np.ndarray] = None

        # Diskten mevcut verileri ve (varsa) güncel indeksleri yükle
        self._load_from_storage()

    # ------------------------------------------------------------------ #
    # Tembel İndeks Başlatıcıları (Double-Checked Thread-Safe!)
    # ------------------------------------------------------------------ #
    def _ensure_meta_index_locked(self) -> None:
        if not self._meta_index_dirty:
            return
        with self._lazy_init_lock:
            if not self._meta_index_dirty:
                return
            deleted_set = (
                set(np.flatnonzero(self._deleted_mask).tolist())
                if self._deleted_count > 0
                else None
            )
            self._meta_index.rebuild(self._metadatas, deleted_slots=deleted_set)
            self._meta_index_dirty = False

    def _ensure_bm25_index_locked(self) -> None:
        if not self._bm25_dirty:
            return
        with self._lazy_init_lock:
            if not self._bm25_dirty:
                return
            self._bm25_index.rebuild(
                self._documents,
                deleted_mask=self._deleted_mask if self._deleted_count > 0 else None,
            )
            self._bm25_dirty = False

    def _ensure_sparse_index_locked(self) -> None:
        if not self._sparse_dirty:
            return
        with self._lazy_init_lock:
            if not self._sparse_dirty:
                return
            self._sparse_index.rebuild(
                self._sparse_vectors,
                deleted_mask=self._deleted_mask if self._deleted_count > 0 else None,
            )
            self._sparse_dirty = False

    def _alive_count_locked(self) -> int:
        return len(self._ids) - self._deleted_count

    # ------------------------------------------------------------------ #
    # Geriye Dönük Uyumluluk Özellikleri (_vectors, _vectors_normalized)
    # ------------------------------------------------------------------ #
    @property
    def _vectors(self) -> np.ndarray:
        if self._deleted_count == 0:
            return self._vbuf.view
        return self._vbuf.view[~self._deleted_mask]

    @_vectors.setter
    def _vectors(self, value: np.ndarray) -> None:
        self._vbuf.load_array(np.asarray(value, dtype=self._vbuf.dtype), is_mmap=False)

    @property
    def _vectors_normalized(self) -> Optional[np.ndarray]:
        if not self._norm_cache_valid or self._vbuf.size == 0 or self._alive_count_locked() == 0:
            return None
        v32 = self._vbuf.view.astype(np.float32, copy=False)
        normed = (v32 * self._vbuf.inv_norms[:, None]).astype(np.float32, copy=False)
        return normed if self._deleted_count == 0 else normed[~self._deleted_mask]

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
    # Disk Yükleme, WAL Replay, Tombstone Kurtarma ve Anında İndeks Yükleme
    # ------------------------------------------------------------------ #
    def _replay_wal_on_load(self) -> bool:
        """WAL'daki yarım çerçeveleri budar ve SQLite'a yansımamış DELETE kayıtlarını oynatır."""
        if self._wal is None or not self._storage.is_persistent:
            return False
        records, _ = self._wal.replay(truncate_corrupt_tail=True)
        if not records:
            return False
        # Son checkpoint/compact sonrası silinmiş ancak çökme nedeniyle SQLite'ta kalmış ID'leri temizle
        pending_delete_ids: List[str] = []
        for rec in records:
            if rec.op in (OP_CHECKPOINT, OP_COMPACT):
                pending_delete_ids.clear()
            elif rec.op == OP_DELETE:
                ids_del = rec.payload.get("ids")
                if ids_del:
                    pending_delete_ids.extend(ids_del)
        if pending_delete_ids:
            self._storage.invalidate_items_columnar_cache(self.name)
            self._storage.delete_items_by_ids(self.name, pending_delete_ids)
            return True
        return False

    def _save_col_cache_locked(self) -> None:
        if not self._storage.is_persistent:
            return
        n_total = len(self._ids)
        alive = self._alive_count_locked()
        if alive == 0:
            self._storage.invalidate_items_columnar_cache(self.name)
            self._col_cache_dirty = False
            return
        if self._deleted_count == 0:
            self._storage.save_items_columnar_cache(
                self.name,
                self._data_epoch,
                self._ids,
                list(range(n_total)),
                self._documents,
                self._metadatas,
                self._sparse_vectors,
            )
        else:
            keep = np.flatnonzero(~self._deleted_mask).tolist()
            self._storage.save_items_columnar_cache(
                self.name,
                self._data_epoch,
                [self._ids[i] for i in keep],
                keep,
                [self._documents[i] for i in keep],
                [self._metadatas[i] for i in keep],
                [self._sparse_vectors[i] for i in keep],
            )
        self._col_cache_dirty = False

    def _load_from_storage(self) -> None:
        """Diskteki verileri sütunsal önbellekten (~10-40 ms) veya SQLite'tan açar; WAL ve indeksleri senkronize eder."""
        wal_replayed = self._replay_wal_on_load()
        cached = (
            None
            if wal_replayed
            else self._storage.load_items_columnar_cache(self.name, expected_epoch=self._data_epoch)
        )
        if cached is not None:
            ids, int_ids, docs, metas, sparses = cached
            self._col_cache_dirty = False
        else:
            ids, int_ids, docs, metas, sparses = self._storage.load_all_items_columnar(self.name)
            self._col_cache_dirty = len(ids) > 0
        n_alive = len(ids)

        if n_alive == 0:
            self._vbuf = VectorBuffer(
                dimension=self.dimension,
                dtype=np.float16 if self.storage_dtype == "float16" else np.float32,
            )
            self._deleted_mask = np.zeros(0, dtype=bool)
            self._deleted_count = 0
            return

        eff = self._effective_index_type()
        load_as_mmap = (
            self.use_mmap
            or eff == "disk"
            or (eff == "ivf_pq" and not self.keep_raw_vectors)
        )
        vecs = self._storage.load_vectors(
            self.name, self.dimension, use_mmap=load_as_mmap, verify_crc=not load_as_mmap
        )
        max_int_id = max(int_ids)
        needed_slots = max_int_id + 1
        n_vecs = vecs.shape[0]

        if n_vecs < needed_slots:
            raise DataCorruptionError(
                f"Koleksiyon '{self.name}' veri tutarsızlığı: "
                f"SQLite maksimum slot ({needed_slots}) > vektör sayısı ({n_vecs})."
            )
        if n_vecs > needed_slots:
            # Çökme kurtarması veya sondaki tombstone temizliği
            if isinstance(vecs, np.memmap):
                mm = getattr(vecs, "_mmap", None)
                if mm is not None:
                    mm.close()
                self._storage.truncate_vectors(self.name, needed_slots)
                vecs = self._storage.load_vectors(
                    self.name, self.dimension, use_mmap=load_as_mmap, verify_crc=False
                )
            else:
                self._storage.truncate_vectors(self.name, needed_slots)
                vecs = np.ascontiguousarray(vecs[:needed_slots])

        total_slots = needed_slots
        if self.dimension is None and vecs.ndim == 2 and vecs.shape[1] > 0:
            self.dimension = int(vecs.shape[1])

        self._vbuf.load_array(vecs, is_mmap=isinstance(vecs, np.memmap))

        if n_alive == total_slots and int_ids[-1] == total_slots - 1:
            # Hızlı yol (tombstone yok — sıfır kopya!)
            self._ids = ids
            self._documents = docs
            self._metadatas = metas
            self._sparse_vectors = sparses
            self._id_to_idx = {s_id: i for i, s_id in enumerate(ids)}
            self._deleted_mask = np.zeros(total_slots, dtype=bool)
            self._deleted_count = 0
        else:
            # Tombstone boşluklarını slot dizisine yerleştir
            self._ids = [""] * total_slots
            self._documents = [None] * total_slots
            self._metadatas = [None] * total_slots
            self._sparse_vectors = [None] * total_slots
            self._deleted_mask = np.ones(total_slots, dtype=bool)
            self._id_to_idx = {}
            for k in range(n_alive):
                slot = int_ids[k]
                self._ids[slot] = ids[k]
                self._documents[slot] = docs[k]
                self._metadatas[slot] = metas[k]
                self._sparse_vectors[slot] = sparses[k]
                self._deleted_mask[slot] = False
                self._id_to_idx[ids[k]] = slot
            self._deleted_count = total_slots - n_alive

        # Sorun #1 çözümü: MetadataIndex, BM25 ve Sparse indeksler tembel (lazy) kurulur!
        self._meta_index_dirty = True
        self._bm25_dirty = True
        self._sparse_dirty = True

        if n_alive > 0:
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
        total_slots = len(self._ids)
        n_alive = self._alive_count_locked()
        if eff == "flat" or n_alive == 0 or self.dimension is None:
            return False

        try:
            if eff == "hnsw" and is_hnsw_available():
                bin_p = self._storage.get_index_file_path(self.name, ".hnsw.bin")
                meta_p = self._storage.get_index_file_path(self.name, ".hnsw.meta.json")
                if bin_p and meta_p and os.path.exists(bin_p) and os.path.exists(meta_p):
                    idx, saved_epoch = HNSWIndex.load_with_meta(bin_p, meta_p, verify_crc=True)
                    if (
                        saved_epoch == self._data_epoch
                        and idx.count in (total_slots, n_alive)
                        and idx.dimension == self.dimension
                        and idx.metric == self.metric
                    ):
                        # Tombstone slotlar varsa HNSW'de de işaretle
                        if self._deleted_count > 0:
                            for dead_slot in np.flatnonzero(self._deleted_mask):
                                idx.mark_deleted(int(dead_slot))
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
                        and idx._total_items == total_slots
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
                        and idx.total_items == total_slots
                        and idx._dimension == self.dimension
                        and idx.metric == self.metric
                    ):
                        self._ivf_pq_index = idx
                        self._index_loaded_from_disk = True
                        self._index_dirty = False
                        if not self.keep_raw_vectors:
                            self._vbuf.release_mmap()
                            self._vbuf = VectorBuffer(dimension=self.dimension, dtype=self._vbuf.dtype)
                        return True
            elif eff == "disk":
                d_vec = self._storage.get_index_file_path(self.name, ".mvdbdisk")
                d_meta = self._storage.get_index_file_path(self.name, ".disk.meta")
                if d_vec and d_meta and os.path.exists(d_vec) and os.path.exists(d_meta):
                    idx, saved_epoch = DiskIndex.load(d_vec, d_meta)
                    if (
                        saved_epoch == self._data_epoch
                        and idx._total_items == total_slots
                        and idx._dimension == self.dimension
                        and idx.metric == self.metric
                    ):
                        self._disk_index = idx
                        self._index_loaded_from_disk = True
                        self._index_dirty = False
                        return True
        except Exception:
            return False
        return False

    def _save_ann_index_locked(self) -> None:
        if not self._storage.is_persistent or self._alive_count_locked() == 0:
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
    # Kuantizasyon & ANN İndeks Yönetimi (write_lock altında)
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
        if self._alive_count_locked() < _AUTO_ANN_THRESHOLD:
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
        total_slots = len(self._ids)
        n_alive = self._alive_count_locked()
        if n_alive == 0 or self.dimension is None or eff == "flat":
            return

        if source_vectors is not None:
            v_src = source_vectors
        elif self._vbuf.size == total_slots:
            v_src = self._vbuf.view
        elif self._storage.is_persistent:
            v_src = self._storage.load_vectors(self.name, self.dimension, use_mmap=False, verify_crc=False)
        else:
            return

        if eff == "ivf":
            v32 = v_src.astype(np.float32, copy=False)
            default_clusters = max(1, min(int(np.sqrt(total_slots)), 1024))
            n_clusters = int(self.index_params.get("n_clusters", default_clusters))
            nprobe = int(self.index_params.get("nprobe", max(1, min(n_clusters // 8, 16))))
            ivf = IVFIndex(n_clusters=min(n_clusters, total_slots), nprobe=nprobe, metric=self.metric)
            ivf.train(v32, n_iter=int(self.index_params.get("n_iter", 15)))
            self._ivf_index = ivf
            self._index_dirty = True
        elif eff == "ivf_pq":
            v32 = v_src.astype(np.float32, copy=False)
            default_clusters = max(1, min(int(np.sqrt(total_slots)), 512))
            n_clusters = int(self.index_params.get("n_clusters", default_clusters))
            nprobe = int(self.index_params.get("nprobe", max(1, min(n_clusters // 8, 16))))
            ivfpq = IVFPQIndex(
                n_clusters=min(n_clusters, total_slots),
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
                initial_capacity=max(1024, total_slots),
            )
            if self._deleted_count == 0:
                hnsw.add_items(v32, np.arange(total_slots, dtype=np.int64))
            else:
                alive_slots = np.flatnonzero(~self._deleted_mask).astype(np.int64)
                hnsw.add_items(v32[alive_slots], alive_slots)
            self._hnsw_index = hnsw
            self._index_dirty = True
        elif eff == "disk":
            default_clusters = max(1, min(int(np.sqrt(total_slots)), 512))
            n_clusters = int(self.index_params.get("n_clusters", default_clusters))
            nprobe = int(self.index_params.get("nprobe", max(1, min(n_clusters // 8, 16))))
            d_vec = self._storage.get_index_file_path(self.name, ".mvdbdisk")
            d_meta = self._storage.get_index_file_path(self.name, ".disk.meta")
            d_idx = DiskIndex(
                n_clusters=min(n_clusters, total_slots),
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
                if self._alive_count_locked() >= int(self.index_params.get("min_train_size", 64)):
                    self._rebuild_ann_index_locked()
            else:
                self._ivf_index.add_vectors(new_vecs_f32, start_idx=start_idx)
                self._index_dirty = True
        elif eff == "ivf_pq":
            if self._ivf_pq_index is None or not self._ivf_pq_index.is_trained:
                if self._alive_count_locked() >= int(self.index_params.get("min_train_size", 64)):
                    src = new_vecs_f32 if start_idx == 0 else None
                    self._rebuild_ann_index_locked(source_vectors=src)
            else:
                self._ivf_pq_index.add_vectors(new_vecs_f32, start_idx=start_idx)
                self._index_dirty = True
        elif eff == "disk":
            if self._alive_count_locked() >= int(self.index_params.get("min_train_size", 32)):
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
            if self._deleted_count > 0:
                self._compact_locked(rebuild_ann=False)
            self._rebuild_ann_index_locked()
            self._save_meta_locked()
            if save:
                self._save_ann_index_locked()

    # ------------------------------------------------------------------ #
    # CRUD: Ekleme, Güncelleme & RAG Doküman Yönetimi
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
        sparse_vectors: Optional[List[Optional[SparseVector]]] = None,
    ) -> None:
        """
        Koleksiyona yeni vektörler (+ opsiyonel doküman, metadata ve sparse vektör) ekler.
        O(M) artımlı bellek ve disk yazımı + WAL + hata anında bellek geri alması (rollback).
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
        if sparse_vectors is not None and len(sparse_vectors) != n_items:
            raise ValueError("Sparse vektör sayısı ID sayısıyla eşit olmalıdır.")

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
            new_sparses = [sparse_vectors[i] if sparse_vectors is not None else None for i in range(n_items)]

            for i, item_id in enumerate(ids):
                self._ids.append(item_id)
                self._id_to_idx[item_id] = start_idx + i
                self._documents.append(new_docs[i])
                self._metadatas.append(new_metas[i])
                self._sparse_vectors.append(new_sparses[i])

            prev_mask = self._deleted_mask
            self._deleted_mask = np.concatenate([prev_mask, np.zeros(n_items, dtype=bool)])

            # 2. Disk delta kalıcılığı + WAL (hata anında tam rollback!)
            try:
                if self._storage.is_persistent:
                    self._storage.invalidate_items_columnar_cache(self.name)
                    self._col_cache_dirty = True
                    if self._wal is not None:
                        self._wal.append_add(count=n_items, start_slot=start_idx, epoch=self._data_epoch)
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
                        (ids[i], start_idx + i, new_docs[i], new_metas[i], new_sparses[i])
                        for i in range(n_items)
                    ]
                    self._storage.save_items_batch(self.name, delta_items)
            except Exception:
                if keep_in_vbuf:
                    self._vbuf.truncate(prev_size)
                del self._ids[prev_size:]
                del self._documents[prev_size:]
                del self._metadatas[prev_size:]
                del self._sparse_vectors[prev_size:]
                self._deleted_mask = prev_mask
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

            # 3. Yardımcı indeksleri (tembel değilse artımlı) güncelle
            if not self._meta_index_dirty:
                self._meta_index.add_batch(new_metas)
            if not self._bm25_dirty and documents is not None:
                for i, doc in enumerate(new_docs):
                    if doc:
                        self._bm25_index.add_document(start_idx + i, doc)
            if not self._sparse_dirty and sparse_vectors is not None:
                for i, sv in enumerate(new_sparses):
                    if sv:
                        self._sparse_index.add(start_idx + i, sv)

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

    def add_documents(
        self,
        documents: List[str],
        embeddings: BatchVectors,
        metadatas: Optional[List[Metadata]] = None,
        ids: Optional[List[str]] = None,
        doc_ids: Optional[List[str]] = None,
        parent_ids: Optional[List[str]] = None,
        sparse_vectors: Optional[List[Optional[SparseVector]]] = None,
        tenant_id: Optional[str] = None,
    ) -> List[str]:
        """
        RAG iş akışları için doküman/chunk ekleme yardımcısı.
        Otomatik ID üretimi, `_doc_id`, `_parent_id`, `_chunk_index` ve `_tenant_id`
        etiketlemesi yapar; eklenen ID listesini döndürür.
        """
        n = len(documents)
        if ids is None:
            gen_ids = [f"doc_{uuid.uuid4().hex[:16]}" for _ in range(n)]
        else:
            if len(ids) != n:
                raise ValueError("ids sayısı documents sayısına eşit olmalıdır.")
            gen_ids = list(ids)

        if metadatas is not None and len(metadatas) != n:
            raise ValueError("metadatas sayısı documents sayısına eşit olmalıdır.")
        if doc_ids is not None and len(doc_ids) != n:
            raise ValueError("doc_ids sayısı documents sayısına eşit olmalıdır.")
        if parent_ids is not None and len(parent_ids) != n:
            raise ValueError("parent_ids sayısı documents sayısına eşit olmalıdır.")

        chunk_counters: Dict[str, int] = {}
        enriched_metas: List[Metadata] = []
        for i in range(n):
            meta: Metadata = dict(metadatas[i]) if (metadatas is not None and metadatas[i] is not None) else {}
            if doc_ids is not None:
                did = doc_ids[i]
                meta["_doc_id"] = did
                c_idx = chunk_counters.get(did, 0)
                meta.setdefault("_chunk_index", c_idx)
                chunk_counters[did] = c_idx + 1
            elif "_doc_id" not in meta:
                meta["_doc_id"] = gen_ids[i]
                meta.setdefault("_chunk_index", 0)
            if parent_ids is not None:
                meta["_parent_id"] = parent_ids[i]
            if tenant_id is not None:
                meta["_tenant_id"] = tenant_id
            enriched_metas.append(meta)

        self.add(
            ids=gen_ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=enriched_metas,
            sparse_vectors=sparse_vectors,
        )
        return gen_ids

    def upsert(
        self,
        ids: List[str],
        embeddings: Optional[BatchVectors] = None,
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Metadata]] = None,
        *,
        vectors: Optional[BatchVectors] = None,
        metadata: Optional[List[Metadata]] = None,
        sparse_vectors: Optional[List[Optional[SparseVector]]] = None,
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
        if sparse_vectors is not None and len(sparse_vectors) != n_items:
            raise ValueError("Sparse vektör sayısı ID sayısıyla eşit olmalıdır.")

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
            to_add_sparses: List[Optional[SparseVector]] = []
            updated_rows: List[Tuple[str, int, Optional[str], Optional[Metadata], Optional[SparseVector], np.ndarray]] = []

            for item_id, i in dedup_order.items():
                doc = documents[i] if documents is not None else None
                meta = raw_metas[i] if raw_metas is not None else None
                sp = sparse_vectors[i] if sparse_vectors is not None else None
                vec = arr[i]

                if item_id in self._id_to_idx:
                    idx = self._id_to_idx[item_id]
                    if self._vbuf.size == len(self._ids):
                        self._vbuf.update_row(idx, vec)
                    if documents is not None:
                        self._documents[idx] = doc
                        if not self._bm25_dirty:
                            self._bm25_index.update_document(idx, doc)
                    if raw_metas is not None:
                        self._metadatas[idx] = meta
                        if not self._meta_index_dirty:
                            self._meta_index.update(idx, meta)
                    if sparse_vectors is not None:
                        self._sparse_vectors[idx] = sp
                        if not self._sparse_dirty:
                            self._sparse_index.update(idx, sp)
                    updated_rows.append(
                        (item_id, idx, self._documents[idx], self._metadatas[idx], self._sparse_vectors[idx], vec)
                    )
                    if self._hnsw_index is not None:
                        self._hnsw_index.update_item(vec, idx)
                        self._index_dirty = True
                else:
                    to_add_ids.append(item_id)
                    to_add_vecs.append(vec)
                    to_add_docs.append(doc)
                    to_add_metas.append(meta)
                    to_add_sparses.append(sp)

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
                    self._sparse_vectors.append(to_add_sparses[i])
                    if not self._bm25_dirty and to_add_docs[i]:
                        self._bm25_index.add_document(start_idx + i, to_add_docs[i])
                    if not self._sparse_dirty and to_add_sparses[i]:
                        self._sparse_index.add(start_idx + i, to_add_sparses[i])
                self._deleted_mask = np.concatenate([self._deleted_mask, np.zeros(len(to_add_ids), dtype=bool)])
                if not self._meta_index_dirty:
                    self._meta_index.add_batch(to_add_metas)

            if self._storage.is_persistent:
                self._storage.invalidate_items_columnar_cache(self.name)
                self._col_cache_dirty = True
                self._save_meta_locked()
                for _, idx, _, _, _, vec in updated_rows:
                    v_disk = self._vbuf.view[idx] if self._vbuf.size == len(self._ids) else vec
                    self._storage.update_vector_row(self.name, idx, v_disk)
                if to_add_ids and new_block is not None:
                    v_append = self._vbuf.view[start_idx:end_idx] if self._vbuf.size == len(self._ids) else new_block
                    self._storage.append_vectors_delta(
                        self.name, v_append, expected_prev_count=prev_size
                    )
                delta_items = [
                    (s_id, idx, d, m, sp) for s_id, idx, d, m, sp, _ in updated_rows
                ] + [
                    (to_add_ids[i], start_idx + i, to_add_docs[i], to_add_metas[i], to_add_sparses[i])
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
        alive = self._alive_count_locked()
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
            total_vectors=alive,
            dimension=self.dimension or 1,
            k=k,
            effective_index=eff,
            quantization=self.quantization,
            is_mmap=self._vbuf.is_mmap,
            has_filter=bool(where),
            candidate_count=len(candidate_indices) if candidate_indices is not None else alive,
            has_raw_vectors=(self._vbuf.size == len(self._ids) and alive > 0),
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
        """Sorguyu çalıştırmadan Query Optimizer'ın seçeceği yürütme planını döndürür."""
        with self._lock.read_lock():
            cand = None
            if where:
                self._ensure_meta_index_locked()
                cand = self._meta_index.query_indices(where)
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
    # Dense Sorgu Motoru (query / search)
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

    def _make_allowed_mask_locked(self, candidate_indices: Optional[np.ndarray]) -> Optional[np.ndarray]:
        total_slots = len(self._ids)
        if candidate_indices is not None:
            mask = np.zeros(total_slots, dtype=bool)
            mask[candidate_indices] = True
            if self._deleted_count > 0:
                mask &= ~self._deleted_mask
            return mask
        if self._deleted_count > 0:
            return ~self._deleted_mask
        return None

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
        """Query Optimizer güdümlü yüksek hızlı benzerlik araması."""
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
            total_slots = len(self._ids)
            alive = self._alive_count_locked()
            if alive == 0:
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
                self._ensure_meta_index_locked()
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

            # --- YOL 1: HNSW ---
            if plan.execution_path == "hnsw" and self._hnsw_index is not None:
                labels, dists, sims = self._hnsw_index.search(
                    q_arr, k=min(n_results, alive), ef_search=plan.adaptive_ef_search
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

            # --- YOL 2: DiskIndex ---
            if plan.execution_path == "disk_partitioned" and self._disk_index is not None:
                allowed_mask = self._make_allowed_mask_locked(candidate_indices)
                idx_l, dist_l, sim_l = self._disk_index.search_batch(
                    q_arr,
                    k=min(n_results, alive),
                    nprobe=plan.adaptive_nprobe,
                    allowed_mask=allowed_mask,
                )
                return self._build_query_result(
                    n_queries, idx_l, dist_l, sim_l, score_threshold=score_threshold
                )

            # --- YOL 3: IVF-PQ ---
            if plan.execution_path == "ivf_pq" and self._ivf_pq_index is not None:
                allowed_mask = self._make_allowed_mask_locked(candidate_indices)
                idx_l, dist_l = self._ivf_pq_index.search_batch(
                    q_arr,
                    k=min(n_results, alive),
                    nprobe=plan.adaptive_nprobe,
                    allowed_mask=allowed_mask,
                )
                sim_l = [self._distances_to_similarities(d) for d in dist_l]
                return self._build_query_result(
                    n_queries, idx_l, dist_l, sim_l, score_threshold=score_threshold
                )

            # --- YOL 4: IVF ---
            if plan.execution_path == "ivf" and self._ivf_index is not None:
                allowed_mask = self._make_allowed_mask_locked(candidate_indices)
                idx_l, dist_l = self._ivf_index.search_batch(
                    q_arr,
                    k=min(n_results, alive),
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
                if candidate_indices is None and self._deleted_count > 0:
                    dists[:, self._deleted_mask] = np.float32(np.inf)
                k = min(n_results, alive if candidate_indices is None else len(candidate_indices))
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
                if candidate_indices is None and self._deleted_count > 0:
                    dists[:, self._deleted_mask] = np.float32(np.inf)
                k = min(n_results, alive if candidate_indices is None else len(candidate_indices))
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
                if candidate_indices is None and self._deleted_count > 0:
                    b_dists[:, self._deleted_mask] = np.float32(np.inf)
                n_cand = alive if candidate_indices is None else len(candidate_indices)
                k = min(n_results, n_cand)
                can_rescore = self.keep_raw_vectors and self._vbuf.size == total_slots
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

            # --- YOL 6: Sabit RAM Bütçeli Disk Akış Taraması ---
            if plan.execution_path == "disk_stream":
                stream_cands = candidate_indices
                if stream_cands is None and self._deleted_count > 0:
                    stream_cands = np.flatnonzero(~self._deleted_mask).astype(np.int64)
                idx_l, dist_l, sim_l = self._disk_scanner.search(
                    q_arr,
                    self._vbuf.view,
                    metric=self.metric,
                    k=min(n_results, alive),
                    inv_norms=self._vbuf.inv_norms if self.metric == "cosine" else None,
                    sq_norms=self._vbuf.sq_norms if self.metric == "euclidean" else None,
                    candidate_indices=stream_cands,
                )
                return self._build_query_result(
                    n_queries, idx_l, dist_l, sim_l, score_threshold=score_threshold
                )

            # --- YOL 7: Optimize Exact Flat BLAS (tombstone'da sıfır matris kopyası!) ---
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
            if candidate_indices is None and self._deleted_count > 0:
                dists[:, self._deleted_mask] = np.float32(np.inf)

            k = min(n_results, alive if candidate_indices is None else len(candidate_indices))
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

    def batch_search(
        self,
        query_vectors: BatchVectors,
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        filters: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        score_threshold: Optional[float] = None,
        ef_search: Optional[int] = None,
        nprobe: Optional[int] = None,
        strategy: str = "auto",
        filter_strategy: str = "auto",
        max_workers: Optional[int] = None,
    ) -> List[List[SearchResultItem]]:
        """
        Çok çekirdekli paralel toplu benzerlik araması (`BatchQueryExecutor`).
        Paylaşımlı `filter` veya sorgu başına farklı `filters` listesi destekler.
        Flat exact modunda (filtre paylaşımlıysa) doğrudan çok çekirdekli BLAS SGEMM kullanır.
        """
        q_arr = validate_vectors(query_vectors, dtype=np.float32)
        n_q = q_arr.shape[0]
        if n_q == 0:
            return []
        if filters is not None and len(filters) != n_q:
            raise ValueError(
                f"filters uzunluğu ({len(filters)}) sorgu vektör sayısına ({n_q}) eşit olmalıdır."
            )

        def _query_res_to_items(res: QueryResult) -> List[List[SearchResultItem]]:
            out: List[List[SearchResultItem]] = []
            for qi in range(len(res.ids)):
                out.append(
                    [
                        SearchResultItem(
                            id=res.ids[qi][j],
                            score=res.similarities[qi][j],
                            distance=res.distances[qi][j],
                            document=res.documents[qi][j],
                            metadata=res.metadatas[qi][j],
                        )
                        for j in range(len(res.ids[qi]))
                    ]
                )
            return out

        # Veri-odaklı otomatik paralellik yönlendirmesi (v0.6.0 GIL/BLAS analizi):
        # - HNSW: saf C++ GIL-serbest graf taraması -> çok çekirdekli havuzda 2.5x+ hızlanır (~20K QPS!)
        # - Flat BLAS / IVF / IVF-PQ / DiskIndex: dahili OpenBLAS veya kısa Python/NumPy çekirdekleri
        #   nedeniyle tek işçi (0 GIL çekişmesi) varsayılan olarak en yüksek QPS'i verir;
        #   kullanıcı `max_workers` açıkça verirse o sayıya uyulur.
        with self._lock.read_lock():
            eff = self._effective_index_type()
            use_single_thread_default = (eff != "hnsw")

        if filters is None and (
            n_q == 1
            or max_workers == 1
            or (max_workers is None and use_single_thread_default)
        ):
            res = self.query(
                query_embeddings=q_arr,
                n_results=limit,
                where=filter,
                score_threshold=score_threshold,
                ef_search=ef_search,
                nprobe=nprobe,
                strategy=strategy,
                filter_strategy=filter_strategy,
            )
            return _query_res_to_items(res)

        if filters is None:
            idx_list = list(range(n_q))

            def _run_shared_chunk(idxs: Sequence[int]) -> List[List[SearchResultItem]]:
                sub_q = q_arr[idxs[0] : idxs[-1] + 1]
                res = self.query(
                    query_embeddings=sub_q,
                    n_results=limit,
                    where=filter,
                    score_threshold=score_threshold,
                    ef_search=ef_search,
                    nprobe=nprobe,
                    strategy=strategy,
                    filter_strategy=filter_strategy,
                )
                return _query_res_to_items(res)

            return self._query_executor.map_chunks(_run_shared_chunk, idx_list, max_workers=max_workers)

        # Sorgu başına farklı filtreler
        eff_workers = (
            max_workers
            if max_workers is not None
            else (1 if use_single_thread_default else None)
        )
        pairs = [(i, filters[i]) for i in range(n_q)]

        def _run_per_filter_chunk(chunk: Sequence[Tuple[int, Optional[Dict[str, Any]]]]) -> List[List[SearchResultItem]]:
            out_chunk: List[List[SearchResultItem]] = []
            for qi, flt in chunk:
                out_chunk.append(
                    self.search(
                        query_vector=q_arr[qi],
                        limit=limit,
                        filter=flt,
                        score_threshold=score_threshold,
                        ef_search=ef_search,
                        nprobe=nprobe,
                        strategy=strategy,
                        filter_strategy=filter_strategy,
                    )
                )
            return out_chunk

        return self._query_executor.map_chunks(_run_per_filter_chunk, pairs, max_workers=eff_workers)

    def batch_hybrid_search(
        self,
        queries: Sequence[Dict[str, Any]],
        limit: int = 10,
        max_workers: Optional[int] = None,
        **default_kwargs: Any,
    ) -> List[List[SearchResultItem]]:
        """
        Çoklu heterojen hibrit sorguları (Dense + BM25 + Sparse) yürütür.
        Harici I/O tabanlı `reranker` kullanılıyorsa veya `max_workers` açıkça verilmişse
        çoklu iş parçacığıyla paralel çalışır; saf bellek-içi CPU sorgularında GIL çekişmesini
        önlemek için varsayılan olarak en hızlı tek-işçi (2,200+ QPS) yolunu seçer.
        """
        if not queries:
            return []

        has_reranker = default_kwargs.get("reranker") is not None or any(
            q.get("reranker") is not None for q in queries
        )
        eff_workers = max_workers if max_workers is not None else (None if has_reranker else 1)

        def _run_hybrid_chunk(chunk: Sequence[Dict[str, Any]]) -> List[List[SearchResultItem]]:
            out_chunk: List[List[SearchResultItem]] = []
            for q_spec in chunk:
                merged_kw = dict(default_kwargs)
                merged_kw.setdefault("limit", limit)
                merged_kw.update(q_spec)
                out_chunk.append(self.hybrid_search(**merged_kw))
            return out_chunk

        return self._query_executor.map_chunks(_run_hybrid_chunk, queries, max_workers=eff_workers)

    # ------------------------------------------------------------------ #
    # Faz 3: Sparse / BM25, Hybrid Search (RRF/Weighted), Multi-Vector & RAG
    # ------------------------------------------------------------------ #
    def _items_from_slots_locked(self, slot_scores: List[Tuple[int, float]]) -> List[SearchResultItem]:
        return [
            SearchResultItem(
                id=self._ids[slot],
                score=float(sc),
                distance=float(-sc),
                document=self._documents[slot],
                metadata=self._metadatas[slot],
            )
            for slot, sc in slot_scores
        ]

    def sparse_search(
        self,
        query_text: Optional[str] = None,
        query_sparse: Optional[SparseVector] = None,
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResultItem]:
        """
        Okapi BM25 (query_text) veya seyrek embedding (query_sparse) araması yapar.
        İkisi birlikte verilirse RRF ile birleştirir.
        """
        if query_text is None and query_sparse is None:
            raise ValueError("sparse_search için 'query_text' veya 'query_sparse' gereklidir.")
        if limit <= 0:
            return []

        with self._lock.read_lock():
            if self._alive_count_locked() == 0:
                return []

            allowed_set: Optional[Set[int]] = None
            if filter:
                self._ensure_meta_index_locked()
                cand = self._meta_index.query_indices(filter)
                if len(cand) == 0:
                    return []
                allowed_set = set(cand.tolist())

            del_mask = self._deleted_mask if self._deleted_count > 0 else None
            results_lists: List[List[Tuple[int, float]]] = []

            if query_text is not None:
                self._ensure_bm25_index_locked()
                idxs, scores = self._bm25_index.search(
                    query_text, k=limit * 2 if query_sparse is not None else limit,
                    allowed_set=allowed_set, deleted_mask=del_mask,
                )
                results_lists.append([(int(i), float(s)) for i, s in zip(idxs, scores)])

            if query_sparse is not None:
                self._ensure_sparse_index_locked()
                idxs, scores = self._sparse_index.search(
                    query_sparse, k=limit * 2 if query_text is not None else limit,
                    allowed_set=allowed_set, deleted_mask=del_mask,
                )
                results_lists.append([(int(i), float(s)) for i, s in zip(idxs, scores)])

            if len(results_lists) == 1:
                merged = results_lists[0][:limit]
            else:
                merged = fuse_rrf([[s for s, _ in rl] for rl in results_lists], limit=limit)

            if score_threshold is not None:
                merged = [(slot, sc) for slot, sc in merged if sc >= score_threshold]

            return self._items_from_slots_locked(merged)

    def hybrid_search(
        self,
        query_vector: Optional[VectorLike] = None,
        query_text: Optional[str] = None,
        query_sparse: Optional[SparseVector] = None,
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        alpha: float = 0.5,
        beta: float = 0.5,
        sparse_weight: Optional[float] = None,
        fusion: str = "rrf",
        rrf_k: int = 60,
        score_threshold: Optional[float] = None,
        reranker: Any = None,
        deduplicate: bool = False,
        dedup_threshold: float = 0.95,
        group_by: Optional[str] = None,
        max_per_group: int = 1,
        candidate_multiplier: int = 4,
    ) -> List[SearchResultItem]:
        """
        Dense + BM25 + Sparse Hibrit Arama motoru.

        Args:
            query_vector: Dense sorgu vektörü (opsiyonel).
            query_text: BM25 anahtar kelime sorgusu (opsiyonel).
            query_sparse: Seyrek embedding sorgusu `{token: weight}` (opsiyonel).
            alpha: Dense ağırlığı (varsayılan: 0.5).
            beta: BM25 / Sparse ağırlığı (varsayılan: 0.5).
            fusion: 'rrf' (varsayılan, ölçekten bağımsız), 'weighted' / 'min_max', veya 'z_score'.
            reranker: Harici reranker fonksiyonu veya `.rerank(query, docs)` nesnesi.
            deduplicate: True ise metin ve kosinüs benzerliğine göre tekilleştirir.
            group_by: Sonuçları metadata alanına (ör. '_doc_id') göre gruplar.
        """
        if query_vector is None and query_text is None and query_sparse is None:
            raise ValueError("hybrid_search için en az biri gereklidir: query_vector, query_text veya query_sparse.")
        if limit <= 0:
            return []

        cand_k = max(limit * max(2, int(candidate_multiplier)), 20)
        scored_lists: List[List[Tuple[int, float]]] = []
        weights: List[float] = []

        # 1. Dense adayları
        if query_vector is not None:
            d_res = self.query(
                query_embeddings=query_vector,
                n_results=cand_k,
                where=filter,
            )
            with self._lock.read_lock():
                d_pairs = [
                    (self._id_to_idx[item_id], float(sim))
                    for item_id, sim in zip(d_res.ids[0], d_res.similarities[0])
                    if item_id in self._id_to_idx
                ]
            scored_lists.append(d_pairs)
            weights.append(float(alpha))

        # 2. BM25 ve/veya Sparse adayları
        with self._lock.read_lock():
            if self._alive_count_locked() == 0:
                return []

            allowed_set: Optional[Set[int]] = None
            if filter:
                self._ensure_meta_index_locked()
                cand = self._meta_index.query_indices(filter)
                if len(cand) == 0:
                    return []
                allowed_set = set(cand.tolist())

            del_mask = self._deleted_mask if self._deleted_count > 0 else None

            if query_text is not None:
                self._ensure_bm25_index_locked()
                b_idx, b_sc = self._bm25_index.search(
                    query_text, k=cand_k, allowed_set=allowed_set, deleted_mask=del_mask
                )
                scored_lists.append([(int(i), float(s)) for i, s in zip(b_idx, b_sc)])
                weights.append(float(beta))

            if query_sparse is not None:
                self._ensure_sparse_index_locked()
                s_idx, s_sc = self._sparse_index.search(
                    query_sparse, k=cand_k, allowed_set=allowed_set, deleted_mask=del_mask
                )
                scored_lists.append([(int(i), float(s)) for i, s in zip(s_idx, s_sc)])
                weights.append(float(sparse_weight if sparse_weight is not None else beta))

            # 3. Füzyon
            pool_limit = cand_k if (reranker is not None or deduplicate or group_by is not None) else limit
            if fusion == "rrf":
                fused = fuse_rrf(
                    [[slot for slot, _ in sl] for sl in scored_lists],
                    weights=weights,
                    rrf_k=rrf_k,
                    limit=pool_limit,
                )
            elif fusion in ("weighted", "min_max"):
                fused = fuse_weighted(scored_lists, weights=weights, normalization="min_max", limit=pool_limit)
            elif fusion == "z_score":
                fused = fuse_weighted(scored_lists, weights=weights, normalization="z_score", limit=pool_limit)
            else:
                raise ValueError(f"Geçersiz fusion yöntemi: '{fusion}'. Desteklenenler: 'rrf', 'weighted', 'min_max', 'z_score'")

            if score_threshold is not None and reranker is None:
                fused = [(slot, sc) for slot, sc in fused if sc >= score_threshold]

            items = self._items_from_slots_locked(fused)

            vecs_map: Optional[Dict[str, np.ndarray]] = None
            if deduplicate and self._vbuf.size == len(self._ids):
                vecs_map = {
                    it.id: self._vbuf.view[self._id_to_idx[it.id]]
                    for it in items
                    if it.id in self._id_to_idx
                }

        # 4. Harici Reranker (kilidin dışında çalışır -> harici API/model bloklaması okuyucuları bekletmez!)
        if reranker is not None:
            items = apply_reranker(query_text or "", items, reranker, limit=None)
            if score_threshold is not None:
                items = [it for it in items if it.score >= score_threshold]

        # 5. Tekilleştirme ve Doküman Gruplama
        if deduplicate:
            items = deduplicate_results(items, vectors_by_id=vecs_map, similarity_threshold=dedup_threshold)
        if group_by is not None:
            items = group_results_by_field(items, group_by=group_by, max_per_group=max_per_group)

        return items[:limit]

    def multi_vector_search(
        self,
        query_vectors: BatchVectors,
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        aggregation: str = "rrf",
        weights: Optional[List[float]] = None,
        rrf_k: int = 60,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResultItem]:
        """
        Birden fazla sorgu vektörüyle (örn. HyDE, çoklu sorgu genişletme) arama yapar
        ve sonuçları 'rrf', 'max' veya 'mean' stratejisiyle birleştirir.
        """
        q_arr = validate_vectors(query_vectors, dtype=np.float32)
        n_q = q_arr.shape[0]
        if limit <= 0 or n_q == 0:
            return []

        cand_k = max(limit * 4, 20)
        res = self.query(query_embeddings=q_arr, n_results=cand_k, where=filter)

        with self._lock.read_lock():
            per_q_lists: List[List[Tuple[int, float]]] = []
            for qi in range(n_q):
                pairs = [
                    (self._id_to_idx[item_id], float(sim))
                    for item_id, sim in zip(res.ids[qi], res.similarities[qi])
                    if item_id in self._id_to_idx
                ]
                per_q_lists.append(pairs)

            if aggregation == "rrf":
                fused = fuse_rrf(
                    [[slot for slot, _ in pl] for pl in per_q_lists],
                    weights=weights,
                    rrf_k=rrf_k,
                    limit=limit,
                )
            elif aggregation in ("max", "mean"):
                w_list = [1.0] * n_q if weights is None else [float(w) for w in weights]
                agg_map: Dict[int, List[float]] = {}
                for pl, w in zip(per_q_lists, w_list):
                    for slot, sc in pl:
                        agg_map.setdefault(slot, []).append(sc * w)
                if aggregation == "max":
                    fused = sorted(
                        [(slot, max(vals)) for slot, vals in agg_map.items()],
                        key=lambda x: x[1],
                        reverse=True,
                    )[:limit]
                else:
                    w_tot = sum(w_list) or 1.0
                    fused = sorted(
                        [(slot, sum(vals) / w_tot) for slot, vals in agg_map.items()],
                        key=lambda x: x[1],
                        reverse=True,
                    )[:limit]
            else:
                raise ValueError(f"Geçersiz aggregation: '{aggregation}'. Desteklenenler: 'rrf', 'max', 'mean'")

            if score_threshold is not None:
                fused = [(slot, sc) for slot, sc in fused if sc >= score_threshold]
            return self._items_from_slots_locked(fused)

    def retrieve_with_context(
        self,
        query_vector: Optional[VectorLike] = None,
        query_text: Optional[str] = None,
        query_sparse: Optional[SparseVector] = None,
        limit: int = 5,
        window: int = 1,
        filter: Optional[Dict[str, Any]] = None,
        fusion: str = "rrf",
        alpha: float = 0.5,
        beta: float = 0.5,
        reranker: Any = None,
    ) -> List[Dict[str, Any]]:
        """
        Parent-Child ve komşu chunk bağlamı (context window) getirme.
        Eşleşen her chunk için aynı `_doc_id` (veya `_parent_id`) altındaki
        `[chunk_index - window, chunk_index + window]` komşularını birleştirir.
        """
        hits = self.hybrid_search(
            query_vector=query_vector,
            query_text=query_text,
            query_sparse=query_sparse,
            limit=limit,
            filter=filter,
            alpha=alpha,
            beta=beta,
            fusion=fusion,
            reranker=reranker,
        )
        if not hits:
            return []

        enriched: List[Dict[str, Any]] = []
        with self._lock.read_lock():
            self._ensure_meta_index_locked()
            for hit in hits:
                meta = hit.metadata or {}
                parent_id = meta.get("_parent_id")
                doc_id = meta.get("_doc_id")
                c_idx = meta.get("_chunk_index")

                group_key = "_parent_id" if parent_id is not None else ("_doc_id" if doc_id is not None else None)
                group_val = parent_id if parent_id is not None else doc_id

                neighbors: List[SearchResultItem] = [hit]
                if group_key is not None and group_val is not None:
                    sibling_slots = self._meta_index.query_indices({group_key: group_val})
                    sib_pairs: List[Tuple[int, SearchResultItem]] = []
                    for s_slot in sibling_slots:
                        si = int(s_slot)
                        s_meta = self._metadatas[si] or {}
                        s_cidx = s_meta.get("_chunk_index", 0)
                        if isinstance(c_idx, int) and isinstance(s_cidx, int):
                            if abs(s_cidx - c_idx) > window:
                                continue
                        sib_pairs.append(
                            (
                                int(s_cidx) if isinstance(s_cidx, int) else 0,
                                SearchResultItem(
                                    id=self._ids[si],
                                    score=hit.score if self._ids[si] == hit.id else 0.0,
                                    distance=hit.distance if self._ids[si] == hit.id else 0.0,
                                    document=self._documents[si],
                                    metadata=self._metadatas[si],
                                ),
                            )
                        )
                    if sib_pairs:
                        sib_pairs.sort(key=lambda x: x[0])
                        neighbors = [p[1] for p in sib_pairs]

                context_text = "\n".join(nb.document for nb in neighbors if nb.document)
                enriched.append(
                    {
                        "item": hit,
                        "doc_id": doc_id,
                        "parent_id": parent_id,
                        "neighbors": neighbors,
                        "context": context_text,
                    }
                )
        return enriched

    # ------------------------------------------------------------------ #
    # CRUD: get, Tombstone delete (<1 ms!), compact, checkpoint, stats
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
                self._ensure_meta_index_locked()
                cand = self._meta_index.query_indices(where)
                if limit is not None:
                    cand = cand[:limit]
                matched_indices = cand.tolist()
            else:
                if self._deleted_count == 0:
                    n_all = len(self._ids)
                    end = n_all if limit is None else min(n_all, max(0, limit))
                    matched_indices = list(range(end))
                else:
                    alive_arr = np.flatnonzero(~self._deleted_mask)
                    if limit is not None:
                        alive_arr = alive_arr[: max(0, limit)]
                    matched_indices = alive_arr.tolist()

            return {
                "ids": [self._ids[i] for i in matched_indices],
                "documents": [self._documents[i] for i in matched_indices],
                "metadatas": [self._metadatas[i] for i in matched_indices],
            }

    def get_vectors(self, ids: List[str]) -> Tuple[List[str], np.ndarray]:
        """
        Verilen ID listesindeki kayıtların (bulunan_idler, float32 vektör matrisi) çiftini
        tek bir okuma kilidi altında sıfır yeniden-gömme maliyetiyle döndürür.
        (LangChain / LlamaIndex vektörize MMR ve yeniden sıralama için optimize edilmiştir.)
        """
        with self._lock.read_lock():
            found_ids: List[str] = []
            slots: List[int] = []
            for s_id in ids:
                idx = self._id_to_idx.get(s_id)
                if idx is not None and not self._deleted_mask[idx]:
                    found_ids.append(s_id)
                    slots.append(idx)

            dim = self.dimension or 0
            if not slots:
                return found_ids, np.empty((0, dim), dtype=np.float32)

            slot_arr = np.asarray(slots, dtype=np.int64)
            total_slots = len(self._ids)
            if self._vbuf.size == total_slots:
                vecs = np.ascontiguousarray(self._vbuf.view[slot_arr], dtype=np.float32)
            elif self._quantized_int8 is not None and self._sq_quantizer is not None:
                vecs = np.ascontiguousarray(self._sq_quantizer.decode(self._quantized_int8[slot_arr]), dtype=np.float32)
            elif self._quantized_pq_t is not None and self._pq_quantizer is not None:
                vecs = np.ascontiguousarray(
                    self._pq_quantizer.decode(np.ascontiguousarray(self._quantized_pq_t[:, slot_arr].T)),
                    dtype=np.float32,
                )
            else:
                vecs = np.empty((0, dim), dtype=np.float32)
            return found_ids, vecs

    def _compact_locked(self, rebuild_ann: bool = True) -> None:
        """
        Tombstone (silinmiş) slotları fiziksel olarak temizler, `.mvdbvec` ve SQLite'ı
        atomik sıkıştırır ve WAL'ı sıfırlar.
        """
        total_slots = len(self._ids)
        if self._deleted_count == 0:
            return

        keep_indices = np.flatnonzero(~self._deleted_mask).tolist()
        has_raw = self._vbuf.size == total_slots

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
            self._sparse_vectors = [self._sparse_vectors[i] for i in keep_indices]
            self._id_to_idx = {s_id: new_i for new_i, s_id in enumerate(self._ids)}
        else:
            self._ids.clear()
            self._documents.clear()
            self._metadatas.clear()
            self._sparse_vectors.clear()
            self._id_to_idx.clear()

        self._deleted_mask = np.zeros(len(self._ids), dtype=bool)
        self._deleted_count = 0
        self._data_epoch += 1
        self._compaction_epoch += 1

        if self._storage.is_persistent:
            self._storage.invalidate_items_columnar_cache(self.name)
            self._col_cache_dirty = True
            self._save_meta_locked()
            if has_raw:
                self._storage.save_vectors(self.name, self._vbuf.view, compute_crc=True)
            else:
                disk_vecs = self._storage.load_vectors(
                    self.name, self.dimension, use_mmap=False, verify_crc=False
                )
                if disk_vecs.shape[0] == total_slots:
                    kept_disk = (
                        np.ascontiguousarray(disk_vecs[keep_indices])
                        if keep_indices
                        else np.empty((0, self.dimension or 0), dtype=disk_vecs.dtype)
                    )
                    self._storage.save_vectors(self.name, kept_disk, compute_crc=True)
            compacted_items = [
                (self._ids[i], i, self._documents[i], self._metadatas[i], self._sparse_vectors[i])
                for i in range(len(self._ids))
            ]
            self._storage.replace_all_items_after_compaction(self.name, compacted_items)
            if self._wal is not None:
                self._wal.truncate()

        self._meta_index_dirty = True
        self._bm25_dirty = True
        self._sparse_dirty = True
        if has_raw:
            self._rebuild_quantization_locked()
        if rebuild_ann:
            self._rebuild_ann_index_locked()
        self._invalidate_norm_cache()

    def compact(self) -> int:
        """Silinmiş (tombstone) alanları fiziksel olarak geri kazanır. Temizlenen slot sayısını döndürür."""
        with self._lock.write_lock():
            reclaimed = self._deleted_count
            if reclaimed > 0:
                self._compact_locked(rebuild_ann=True)
                self._save_ann_index_locked()
            return reclaimed

    def compact_async(self) -> Future:
        """
        Tombstone sıkıştırmasını arka plan iş parçacığında (coalesced) başlatır.
        Dönen `Future.result()` temizlenen slot sayısını verir.
        """
        return self._bg_tasks.submit_coalesced(f"compact:{self.name}", self.compact)

    def rebuild_index_async(self) -> Future:
        """
        Okuyucuları bloklamadan (non-blocking snapshot-and-swap) ANN/Disk indeksini
        arka planda yeniden inşa eder; inşa sırasında gelen yeni vektör deltalarını
        otomatik olarak yeni indekse ekler.
        Dönen `Future.result()` -> `True` (başarılı takas) veya `False` (eşzamanlı compaction).
        """
        return self._bg_tasks.submit_coalesced(
            f"rebuild:{self.name}", self._rebuild_index_nonblocking
        )

    def _rebuild_index_nonblocking(self) -> bool:
        # 1. Kısa read_lock: tutarlı vektör anlık görüntüsü al ve kilidi HEMEN bırak!
        with self._lock.read_lock():
            eff = self._effective_index_type()
            total_slots_snap = len(self._ids)
            n_alive = self._alive_count_locked()
            if n_alive == 0 or self.dimension is None or eff == "flat":
                return False
            comp_epoch_start = self._compaction_epoch
            deleted_mask_snap = self._deleted_mask.copy() if self._deleted_count > 0 else None
            has_raw = self._vbuf.size == total_slots_snap
            if has_raw:
                vecs_snap = np.ascontiguousarray(self._vbuf.view, dtype=np.float32)
            elif self._storage.is_persistent:
                vecs_snap = self._storage.load_vectors(
                    self.name, self.dimension, use_mmap=False, verify_crc=False
                ).astype(np.float32, copy=False)
            else:
                return False
            dim = self.dimension

        # 2. KİLİT DIŞINDA (sıfır okuyucu bloklaması!): ağır eğitimi yap
        new_ivf: Optional[IVFIndex] = None
        new_ivfpq: Optional[IVFPQIndex] = None
        new_hnsw: Optional[HNSWIndex] = None
        new_disk: Optional[DiskIndex] = None
        staged_disk_vec: Optional[str] = None
        staged_disk_meta: Optional[str] = None

        if eff == "ivf":
            default_clusters = max(1, min(int(np.sqrt(total_slots_snap)), 1024))
            n_clusters = int(self.index_params.get("n_clusters", default_clusters))
            nprobe = int(self.index_params.get("nprobe", max(1, min(n_clusters // 8, 16))))
            new_ivf = IVFIndex(n_clusters=min(n_clusters, total_slots_snap), nprobe=nprobe, metric=self.metric)
            new_ivf.train(vecs_snap, n_iter=int(self.index_params.get("n_iter", 15)))
        elif eff == "ivf_pq":
            default_clusters = max(1, min(int(np.sqrt(total_slots_snap)), 512))
            n_clusters = int(self.index_params.get("n_clusters", default_clusters))
            nprobe = int(self.index_params.get("nprobe", max(1, min(n_clusters // 8, 16))))
            new_ivfpq = IVFPQIndex(
                n_clusters=min(n_clusters, total_slots_snap),
                nprobe=nprobe,
                n_subvectors=self.index_params.get("pq_subvectors"),
                n_centroids=int(self.index_params.get("pq_centroids", 256)),
                metric=self.metric,
            )
            new_ivfpq.train(
                vecs_snap,
                n_iter=int(self.index_params.get("n_iter", 12)),
                pq_iter=int(self.index_params.get("pq_iter", 12)),
            )
        elif eff == "hnsw" and is_hnsw_available():
            new_hnsw = HNSWIndex(
                dimension=dim,
                metric=self.metric,
                M=int(self.index_params.get("M", 16)),
                ef_construction=int(self.index_params.get("ef_construction", 128)),
                ef_search=int(self.index_params.get("ef_search", 64)),
                initial_capacity=max(1024, total_slots_snap),
            )
            if deleted_mask_snap is None:
                new_hnsw.add_items(vecs_snap, np.arange(total_slots_snap, dtype=np.int64))
            else:
                alive_slots = np.flatnonzero(~deleted_mask_snap).astype(np.int64)
                new_hnsw.add_items(vecs_snap[alive_slots], alive_slots)
        elif eff == "disk":
            default_clusters = max(1, min(int(np.sqrt(total_slots_snap)), 512))
            n_clusters = int(self.index_params.get("n_clusters", default_clusters))
            nprobe = int(self.index_params.get("nprobe", max(1, min(n_clusters // 8, 16))))
            d_vec = self._storage.get_index_file_path(self.name, ".mvdbdisk")
            d_meta = self._storage.get_index_file_path(self.name, ".disk.meta")
            tag = f".bg.{os.getpid()}.{time.time_ns()}"
            staged_disk_vec = (d_vec + tag) if d_vec else None
            staged_disk_meta = (d_meta + tag) if d_meta else None
            new_disk = DiskIndex(
                n_clusters=min(n_clusters, total_slots_snap),
                nprobe=nprobe,
                metric=self.metric,
                storage_dtype=self.storage_dtype,
            )
            new_disk.build(
                vecs_snap,
                disk_vec_path=staged_disk_vec,
                meta_path=staged_disk_meta,
                epoch=self._data_epoch,
                n_iter=int(self.index_params.get("n_iter", 12)),
            )
        else:
            return False

        # 3. Mikrosaniyelik write_lock: yeni gelen deltaları ekle ve atomik takas et!
        with self._lock.write_lock():
            if self._compaction_epoch != comp_epoch_start:
                # İnşa sürerken compaction yapılmış -> bayat indeksi güvenle çöpe at
                if new_disk is not None:
                    new_disk.close()
                for p in (staged_disk_vec, staged_disk_meta):
                    if p and os.path.exists(p):
                        try:
                            os.remove(p)
                        except OSError:
                            pass
                return False

            cur_total = len(self._ids)
            delta_n = cur_total - total_slots_snap
            delta_vecs: Optional[np.ndarray] = None
            if delta_n > 0:
                if self._vbuf.size == cur_total:
                    delta_vecs = np.ascontiguousarray(self._vbuf.view[total_slots_snap:cur_total], dtype=np.float32)
                elif self._storage.is_persistent:
                    all_disk = self._storage.load_vectors(self.name, dim, use_mmap=False, verify_crc=False)
                    delta_vecs = np.ascontiguousarray(all_disk[total_slots_snap:cur_total], dtype=np.float32)

            if new_ivf is not None:
                if delta_vecs is not None and delta_n > 0:
                    new_ivf.add_vectors(delta_vecs, start_idx=total_slots_snap)
                self._ivf_index = new_ivf
            elif new_ivfpq is not None:
                if delta_vecs is not None and delta_n > 0:
                    new_ivfpq.add_vectors(delta_vecs, start_idx=total_slots_snap)
                self._ivf_pq_index = new_ivfpq
                if not self.keep_raw_vectors:
                    self._vbuf.release_mmap()
                    self._vbuf = VectorBuffer(dimension=dim, dtype=self._vbuf.dtype)
            elif new_hnsw is not None:
                if delta_vecs is not None and delta_n > 0:
                    new_hnsw.add_items(
                        delta_vecs, np.arange(total_slots_snap, cur_total, dtype=np.int64)
                    )
                if self._deleted_count > 0:
                    for dead_slot in np.flatnonzero(self._deleted_mask):
                        new_hnsw.mark_deleted(int(dead_slot))
                self._hnsw_index = new_hnsw
            elif new_disk is not None:
                # DiskIndex dosyalarını atomik yerine taşı
                final_vec = self._storage.get_index_file_path(self.name, ".mvdbdisk")
                final_meta = self._storage.get_index_file_path(self.name, ".disk.meta")
                if self._disk_index is not None:
                    self._disk_index.close()
                if staged_disk_vec and staged_disk_meta and final_vec and final_meta:
                    new_disk.close()
                    os.replace(staged_disk_vec, final_vec)
                    os.replace(staged_disk_meta, final_meta)
                    reloaded, _ = DiskIndex.load(final_vec, final_meta)
                    self._disk_index = reloaded
                else:
                    self._disk_index = new_disk
                if delta_vecs is not None and delta_n > 0:
                    self._rebuild_ann_index_locked()

            self._index_dirty = True
            return True

    def wait_for_background(self, timeout: Optional[float] = None) -> None:
        """Bekleyen tüm arka plan compaction / indeks inşa görevlerinin bitmesini bekler."""
        self._bg_tasks.wait_all(timeout=timeout)

    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Yüksek hızlı Tombstone (mantıksal) silme (<1 ms!) + WAL günlüğü.
        Tombstone oranı eşiği aştığında otomatik sıkıştırma (compaction) çalışır.
        """
        if ids is None and where is None:
            raise ValueError("Silme için ya 'ids' ya da 'where' filtresi belirtilmelidir.")

        with self._lock.write_lock():
            to_delete_slots: Set[int] = set()
            if ids is not None:
                for s_id in ids:
                    idx = self._id_to_idx.get(s_id)
                    if idx is not None:
                        to_delete_slots.add(idx)

            if where is not None:
                self._ensure_meta_index_locked()
                cand = self._meta_index.query_indices(where)
                to_delete_slots.update(cand.tolist())

            if not to_delete_slots:
                return 0

            deleted_count = len(to_delete_slots)
            deleted_ids = [self._ids[s] for s in to_delete_slots]
            slots_list = sorted(to_delete_slots)

            # 1. Önce WAL'a yaz (crash-safe! Kalıcı dosya tutamacı ile <0.1 ms)
            if self._wal is not None:
                self._wal.append_delete(deleted_ids, slots_list, epoch=self._data_epoch)

            # 2. O(silinen_sayısı) tombstone işaretleme (tam dosya yeniden yazımı YOK!)
            for slot in slots_list:
                s_id = self._ids[slot]
                self._id_to_idx.pop(s_id, None)
                self._ids[slot] = ""
                self._documents[slot] = None
                self._metadatas[slot] = None
                self._sparse_vectors[slot] = None
                self._deleted_mask[slot] = True
                if not self._meta_index_dirty:
                    self._meta_index.remove(slot)
                if not self._bm25_dirty:
                    self._bm25_index.remove_document(slot)
                if not self._sparse_dirty:
                    self._sparse_index.remove(slot)
                if self._hnsw_index is not None:
                    self._hnsw_index.mark_deleted(slot)

            self._deleted_count += deleted_count
            self._invalidate_norm_cache()

            # 3. Tek atomik SQLite silme işlemi (O(silinen_sayısı)! İndeks epoch'u korunur)
            if self._storage.is_persistent:
                if not self._col_cache_dirty:
                    self._storage.invalidate_items_columnar_cache(self.name)
                    self._col_cache_dirty = True
                self._storage.delete_items_by_ids(self.name, deleted_ids)

            # 4. Koleksiyon tamamen boşaldıysa veya tombstone oranı eşiği aştıysa sıkıştır
            total_slots = len(self._ids)
            alive_now = total_slots - self._deleted_count
            trigger_bg_compact = False
            if alive_now == 0:
                self._compact_locked(rebuild_ann=True)
            elif (
                self._deleted_count >= self._auto_compact_min_deleted
                and (self._deleted_count / max(1, total_slots)) >= self._compaction_threshold
            ):
                if self._auto_compact_async:
                    trigger_bg_compact = True
                else:
                    self._compact_locked(rebuild_ann=True)

        if trigger_bg_compact:
            self.compact_async()

        return deleted_count

    def checkpoint(self) -> None:
        """
        Varsa tombstone'ları sıkıştırır, tam CRC32 sağlama toplamı ile vektörleri ve
        aktif ANN/Disk indeksini atomik kaydeder ve WAL'ı temizler.
        """
        with self._lock.write_lock():
            if not self._storage.is_persistent:
                return
            if self._deleted_count > 0:
                self._compact_locked(rebuild_ann=True)
            elif self._vbuf.size == len(self._ids):
                if self._vbuf.is_mmap:
                    arr_copy = np.ascontiguousarray(self._vbuf.view)
                    self._vbuf.load_array(arr_copy, is_mmap=False)
                self._storage.save_vectors(self.name, self._vbuf.view, compute_crc=True)
            self._save_meta_locked()
            self._save_ann_index_locked()
            self._save_col_cache_locked()
            if self._wal is not None:
                self._wal.truncate()

    def close(self) -> None:
        """Arka plan görevlerini bekler, değişmiş indeksleri kaydedip tüm tutamaçları kapatır."""
        self.wait_for_background()
        with self._lock.write_lock():
            if self._storage.is_persistent and self._alive_count_locked() > 0:
                if self._index_dirty:
                    try:
                        self._save_meta_locked()
                        self._save_ann_index_locked()
                    except Exception:
                        pass
                if self._col_cache_dirty:
                    try:
                        self._save_col_cache_locked()
                    except Exception:
                        pass
            if self._disk_index is not None:
                self._disk_index.close()
            if self._wal is not None:
                self._wal.close()
            self._vbuf.release_mmap()
        if self._owns_bg:
            self._bg_tasks.shutdown(wait=True)
        if self._owns_executor:
            self._query_executor.shutdown(wait=True)

    def get_all_ids(self) -> List[str]:
        with self._lock.read_lock():
            if self._deleted_count == 0:
                return list(self._ids)
            return [self._ids[i] for i in range(len(self._ids)) if not self._deleted_mask[i]]

    def count(self) -> int:
        with self._lock.read_lock():
            return self._alive_count_locked()

    def peek(self, limit: int = 5) -> Dict[str, Any]:
        return self.get(limit=limit)

    def stats(self) -> Dict[str, Any]:
        """Koleksiyonun gerçek RAM, disk, tombstone, WAL ve indeks istatistiklerini döndürür."""
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
                "count": self._alive_count_locked(),
                "total_slots": len(self._ids),
                "tombstones": self._deleted_count,
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
                "wal_bytes": self._wal.size_bytes() if self._wal is not None else 0,
                "bm25_docs": self._bm25_index.num_docs if not self._bm25_dirty else None,
                "sparse_vectors": self._sparse_index.num_vectors if not self._sparse_dirty else None,
                "metadata_index": self._meta_index.stats() if not self._meta_index_dirty else {"lazy_pending": True},
            }

    def __repr__(self) -> str:
        return (
            f"<Collection(name='{self.name}', count={self.count()}, "
            f"dimension={self.dimension}, metric='{self.metric}', "
            f"index='{self.index_type}', quant='{self.quantization}')>"
        )
