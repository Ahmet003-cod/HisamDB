"""
microvectordb.collection
------------------------
Koleksiyon sınıfı: Vektör depolama (VectorBuffer), ters metadata indeksi
(MetadataIndex), indeks motorları (Flat / IVF / HNSW), kuantizasyon
(Float16 / INT8 / Binary), RWLock eşzamanlılık güvenliği ve delta kalıcılık
yönetiminin ana merkezi.

v0.2.0 ChromaDB/FAISS API'si ile %100 geriye dönük uyumludur.
"""

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from microvectordb.concurrency import RWLock
from microvectordb.filtering import MetadataIndex, matches_filter
from microvectordb.indexes import HNSWIndex, IVFIndex, is_hnsw_available
from microvectordb.metrics import (
    compute_distances,
    top_k_indices,
    validate_metric,
    validate_vectors,
)
from microvectordb.quantization import BinaryQuantizer, ScalarQuantizer
from microvectordb.storage import StorageEngine, VectorBuffer
from microvectordb.types import (
    BatchVectors,
    DataCorruptionError,
    DimensionMismatchError,
    IndexType,
    InvalidVectorError,
    Metadata,
    MetricType,
    QuantizationType,
    QueryResult,
    SearchResultItem,
    StorageDType,
    VectorDBError,
    VectorLike,
)

_VALID_INDEX_TYPES = ("flat", "ivf", "hnsw", "auto")
_VALID_QUANTIZATIONS = ("none", "float16", "int8", "binary")
_VALID_STORAGE_DTYPES = ("float32", "float16")
_EXACT_FILTER_THRESHOLD = 2048  # Bu sayının altındaki filtrelenmiş adaylarda exact BLAS daha hızlıdır
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

        # Geriye dönük uyumluluk bayrağı (v0.2.0 testleri _vectors_normalized durumunu denetler;
        # gerçek sorgular 2x RAM tüketmeden _vbuf.inv_norms kullanır!)
        self._norm_cache_valid: bool = False

        # Opsiyonel ANN indeksleri
        self._ivf_index: Optional[IVFIndex] = None
        self._hnsw_index: Optional[HNSWIndex] = None

        # Opsiyonel INT8 / Binary kuantizasyon durumları
        self._scalar_quantizer: Optional[ScalarQuantizer] = None
        self._quantized_int8: Optional[np.ndarray] = None
        self._sq_scaled_sq: Optional[np.ndarray] = None
        self._sq_recon_inv: Optional[np.ndarray] = None

        self._binary_quantizer: Optional[BinaryQuantizer] = None
        self._quantized_binary: Optional[np.ndarray] = None

        # Diskten mevcut verileri yükle
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
        """
        Geriye dönük uyumluluk özelliği:
        Normal sorgularda (N, D) ikinci kopya oluşturulmaz (`inv_norms` kullanılır).
        Yalnızca dışarıdan açıkça erişildiğinde ve kosinüs önbelleği geçerliyse üretilir.
        """
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
    # Disk Yükleme ve Çökme Kurtarma
    # ------------------------------------------------------------------ #
    def _load_from_storage(self) -> None:
        """Diskteki verileri yükler; yarım kalmış vektör eklemelerini güvenle kurtarır."""
        items = self._storage.load_all_items(self.name)
        if not items:
            self._vbuf = VectorBuffer(
                dimension=self.dimension,
                dtype=np.float16 if self.storage_dtype == "float16" else np.float32,
            )
            return

        vecs = self._storage.load_vectors(
            self.name, self.dimension, use_mmap=self.use_mmap, verify_crc=True
        )
        n_items = len(items)
        n_vecs = vecs.shape[0]

        if n_vecs > n_items:
            # Çökme kurtarması: Vektör dosyasına yazılıp SQLite'a commit edilmeyen satırları kırp
            if isinstance(vecs, np.memmap):
                mm = getattr(vecs, "_mmap", None)
                if mm is not None:
                    mm.close()
                self._storage.truncate_vectors(self.name, n_items)
                vecs = self._storage.load_vectors(
                    self.name, self.dimension, use_mmap=self.use_mmap, verify_crc=False
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

        # Kuantizasyon ve indeksleri yeniden inşa et
        if self._vbuf.size > 0:
            self._rebuild_quantization_locked()
            self._rebuild_ann_index_locked()

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

    def _effective_index_type(self) -> str:
        if self.index_type != "auto":
            return self.index_type
        if self._vbuf.size < _AUTO_ANN_THRESHOLD:
            return "flat"
        return "hnsw" if is_hnsw_available() else "ivf"

    def _rebuild_ann_index_locked(self) -> None:
        eff = self._effective_index_type()
        self._ivf_index = None
        self._hnsw_index = None

        n = self._vbuf.size
        if n == 0 or self.dimension is None or eff == "flat":
            return

        v32 = self._vbuf.view.astype(np.float32, copy=False)
        if eff == "ivf":
            default_clusters = max(1, min(int(np.sqrt(n)), 1024))
            n_clusters = int(self.index_params.get("n_clusters", default_clusters))
            nprobe = int(self.index_params.get("nprobe", max(1, min(n_clusters // 8, 16))))
            ivf = IVFIndex(n_clusters=min(n_clusters, n), nprobe=nprobe, metric=self.metric)
            ivf.train(v32, n_iter=int(self.index_params.get("n_iter", 15)))
            self._ivf_index = ivf
        elif eff == "hnsw":
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
        elif eff == "ivf":
            if self._ivf_index is None or not self._ivf_index.is_trained:
                if self._vbuf.size >= int(self.index_params.get("min_train_size", 64)):
                    self._rebuild_ann_index_locked()
            else:
                self._ivf_index.add_vectors(new_vecs_f32, start_idx=start_idx)

    def build_index(
        self,
        index_type: Optional[IndexType] = None,
        **params: Any,
    ) -> None:
        """Koleksiyon üzerinde indeksi (yeniden) kurar."""
        with self._lock.write_lock():
            if index_type is not None:
                if index_type not in _VALID_INDEX_TYPES:
                    raise ValueError(f"Geçersiz index_type: '{index_type}'")
                self.index_type = index_type
            if params:
                self.index_params.update(params)
            self._rebuild_ann_index_locked()
            self._storage.save_collection_meta(
                self.name,
                self.dimension,
                self.metric,
                self.created_at,
                index_type=self.index_type,
                quantization=self.quantization,
                storage_dtype=self.storage_dtype,
            )

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

        # Batch içi benzersizlik kontrolü (RAM'e dokunmadan önce!)
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
            self.dimension = new_dim
            keep_in_vbuf = self.keep_raw_vectors or self.quantization not in ("int8", "binary")

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
                        self._storage.save_collection_meta(
                            self.name,
                            self.dimension,
                            self.metric,
                            self.created_at,
                            index_type=self.index_type,
                            quantization=self.quantization,
                            storage_dtype=self.storage_dtype,
                        )
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
                # Atomik bellek geri alması (rollback)
                if keep_in_vbuf:
                    self._vbuf.truncate(prev_size)
                del self._ids[prev_size:]
                del self._documents[prev_size:]
                del self._metadatas[prev_size:]
                for item_id in ids:
                    self._id_to_idx.pop(item_id, None)
                self.dimension = prev_dim
                if self._storage.is_persistent:
                    try:
                        self._storage.truncate_vectors(self.name, prev_size)
                    except Exception:
                        pass
                raise

            # 3. İndeksleri ve önbellekleri güncelle
            self._meta_index.add_batch(new_metas)
            if not keep_in_vbuf and prev_size == 0:
                # İlk eklemede doğrudan arr üzerinden fit + encode (raw RAM tutmadan!)
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

        # Aynı upsert çağrısında aynı ID birden fazla kez geçerse sonuncusunu geçerli kıl
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
                else:
                    to_add_ids.append(item_id)
                    to_add_vecs.append(vec)
                    to_add_docs.append(doc)
                    to_add_metas.append(meta)

            prev_size = self._vbuf.size
            new_block: Optional[np.ndarray] = None
            start_idx = prev_size
            end_idx = prev_size

            if to_add_ids:
                new_block = np.ascontiguousarray(np.stack(to_add_vecs, axis=0), dtype=np.float32)
                start_idx, end_idx = self._vbuf.append(new_block)
                for i, item_id in enumerate(to_add_ids):
                    self._ids.append(item_id)
                    self._id_to_idx[item_id] = start_idx + i
                    self._documents.append(to_add_docs[i])
                    self._metadatas.append(to_add_metas[i])
                self._meta_index.add_batch(to_add_metas)

            # Delta kalıcılık
            if self._storage.is_persistent:
                self._storage.save_collection_meta(
                    self.name,
                    self.dimension,
                    self.metric,
                    self.created_at,
                    index_type=self.index_type,
                    quantization=self.quantization,
                    storage_dtype=self.storage_dtype,
                )
                for _, idx, _, _, _ in updated_rows:
                    self._storage.update_vector_row(self.name, idx, self._vbuf.view[idx])
                if to_add_ids:
                    self._storage.append_vectors_delta(
                        self.name,
                        self._vbuf.view[start_idx:end_idx],
                        expected_prev_count=prev_size,
                    )
                delta_items = [
                    (s_id, idx, d, m) for s_id, idx, d, m, _ in updated_rows
                ] + [
                    (to_add_ids[i], start_idx + i, to_add_docs[i], to_add_metas[i])
                    for i in range(len(to_add_ids))
                ]
                self._storage.save_items_batch(self.name, delta_items)

            # İndeks / kuantizasyon senkronizasyonu
            if updated_rows:
                if self.quantization in ("int8", "binary"):
                    self._rebuild_quantization_locked()
                if self._ivf_index is not None:
                    self._ivf_index.rebuild(self._vbuf.view.astype(np.float32, copy=False))
            elif new_block is not None:
                self._append_quantization_locked(new_block)
                self._append_ann_index_locked(new_block, start_idx=start_idx)

            self._invalidate_norm_cache()

    # ------------------------------------------------------------------ #
    # Metadata Filtreleme (Geriye Dönük Uyumlu Yardımcı)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _matches_filter(item_meta: Optional[Metadata], where: Optional[Dict[str, Any]]) -> bool:
        """Tek kayıt için metadata filtre denetimi ($eq..$nin, $and, $or, $not, nested)."""
        return matches_filter(item_meta, where)

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
    ) -> QueryResult:
        """
        Yüksek hızlı benzerlik araması (Nearest Neighbor Search).
        Ters metadata indeksi, `inv_norms`/`sq_norms` önbellekleri, kuantizasyon ve
        aktif ANN indeksinden (Flat / IVF / HNSW) otomatik yararlanır.
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

            # 1. Metadata filtresi varsa ters indeks ile adayları O(eşleşme) sürede bul
            candidate_indices: Optional[np.ndarray] = None
            if where:
                candidate_indices = self._meta_index.query_indices(where)
                if len(candidate_indices) == 0:
                    return QueryResult(
                        ids=[[] for _ in range(n_queries)],
                        distances=[[] for _ in range(n_queries)],
                        similarities=[[] for _ in range(n_queries)],
                        documents=[[] for _ in range(n_queries)],
                        metadatas=[[] for _ in range(n_queries)],
                    )

            # 2. Akıllı Sorgu Yönlendirmesi (HNSW / IVF / Quantized / Flat)
            eff_index = self._effective_index_type()
            use_exact_for_small_filter = (
                candidate_indices is not None
                and (len(candidate_indices) <= _EXACT_FILTER_THRESHOLD or len(candidate_indices) < total * 0.05)
            )

            # --- YOL A: HNSW ---
            if (
                eff_index == "hnsw"
                and self._hnsw_index is not None
                and not use_exact_for_small_filter
            ):
                filter_fn = None
                if candidate_indices is not None:
                    allowed_set = set(candidate_indices.tolist())
                    filter_fn = allowed_set.__contains__
                    k_req = min(n_results, len(allowed_set))
                else:
                    k_req = min(n_results, total)

                labels, dists, sims = self._hnsw_index.search(
                    q_arr, k=k_req, ef_search=ef_search, filter_fn=filter_fn
                )
                return self._build_query_result(
                    n_queries,
                    [labels[i] for i in range(n_queries)],
                    [dists[i] for i in range(n_queries)],
                    [sims[i] for i in range(n_queries)],
                    score_threshold=score_threshold,
                )

            # --- YOL B: IVF ---
            if (
                eff_index == "ivf"
                and self._ivf_index is not None
                and self._ivf_index.is_trained
                and not use_exact_for_small_filter
            ):
                allowed_mask = None
                if candidate_indices is not None:
                    allowed_mask = np.zeros(total, dtype=bool)
                    allowed_mask[candidate_indices] = True
                idx_lists, dist_lists = self._ivf_index.search_batch(
                    q_arr,
                    k=min(n_results, total),
                    nprobe=nprobe,
                    allowed_mask=allowed_mask,
                )
                sim_lists = [self._distances_to_similarities(d) for d in dist_lists]
                return self._build_query_result(
                    n_queries, idx_lists, dist_lists, sim_lists, score_threshold=score_threshold
                )

            # --- YOL C: Kuantize Flat Arama (INT8 / Binary) ---
            if self.quantization == "int8" and self._scalar_quantizer is not None and self._quantized_int8 is not None:
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
                idx_lists, dist_lists, sim_lists = [], [], []
                for q_idx in range(n_queries):
                    local_top = top_k_indices(dists[q_idx], k)
                    global_top = candidate_indices[local_top] if candidate_indices is not None else local_top
                    idx_lists.append(global_top)
                    dist_lists.append(dists[q_idx, local_top])
                    sim_lists.append(sims[q_idx, local_top])
                return self._build_query_result(
                    n_queries, idx_lists, dist_lists, sim_lists, score_threshold=score_threshold
                )

            if self.quantization == "binary" and self._binary_quantizer is not None and self._quantized_binary is not None:
                if candidate_indices is not None:
                    b_db = self._quantized_binary[candidate_indices]
                else:
                    b_db = self._quantized_binary

                b_dists, b_sims = self._binary_quantizer.compute_batch_distances(q_arr, b_db)
                n_cand = b_db.shape[0]
                k = min(n_results, n_cand)
                oversample_mult = int(self.index_params.get("rescore_multiplier", 25))
                oversample = min(n_cand, max(k * oversample_mult, 256)) if (self.keep_raw_vectors and self._vbuf.size == total) else k
                can_rescore = self.keep_raw_vectors and self._vbuf.size == total

                idx_lists, dist_lists, sim_lists = [], [], []
                for q_idx in range(n_queries):
                    local_top = top_k_indices(b_dists[q_idx], oversample)
                    global_top = candidate_indices[local_top] if candidate_indices is not None else local_top
                    if can_rescore and len(global_top) > 0:
                        # Ham float32/float16 vektörlerle kesin yeniden skorlama (rescore)
                        rs_vecs = self._vbuf.view[global_top]
                        rs_inv = self._vbuf.inv_norms[global_top] if self.metric == "cosine" else None
                        rs_sq = self._vbuf.sq_norms[global_top] if self.metric == "euclidean" else None
                        rs_d, rs_s = compute_distances(
                            q_arr[q_idx : q_idx + 1],
                            rs_vecs,
                            metric=self.metric,
                            db_inv_norms=rs_inv,
                            db_sq_norms=rs_sq,
                        )
                        rs_ord = top_k_indices(rs_d[0], min(k, len(global_top)))
                        idx_lists.append(global_top[rs_ord])
                        dist_lists.append(rs_d[0, rs_ord])
                        sim_lists.append(rs_s[0, rs_ord])
                    else:
                        idx_lists.append(global_top[:k])
                        dist_lists.append(b_dists[q_idx, local_top[:k]])
                        sim_lists.append(b_sims[q_idx, local_top[:k]])

                return self._build_query_result(
                    n_queries, idx_lists, dist_lists, sim_lists, score_threshold=score_threshold
                )

            # --- YOL D: Optimize Exact Flat BLAS (inv_norms / sq_norms) ---
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
            idx_lists, dist_lists, sim_lists = [], [], []
            for q_idx in range(n_queries):
                local_top = top_k_indices(dists[q_idx], k)
                global_top = candidate_indices[local_top] if candidate_indices is not None else local_top
                idx_lists.append(global_top)
                dist_lists.append(dists[q_idx, local_top])
                sim_lists.append(sims[q_idx, local_top])

            return self._build_query_result(
                n_queries, idx_lists, dist_lists, sim_lists, score_threshold=score_threshold
            )

    def search(
        self,
        query_vector: VectorLike,
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
        ef_search: Optional[int] = None,
        nprobe: Optional[int] = None,
    ) -> List[SearchResultItem]:
        """
        Tek sorgu için ergonomik arama arayüzü: List[SearchResultItem] döndürür.
        """
        res = self.query(
            query_embeddings=query_vector,
            n_results=limit,
            where=filter,
            score_threshold=score_threshold,
            ef_search=ef_search,
            nprobe=nprobe,
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
        """
        Kayıtları ID veya metadata filtresine göre O(|ids|) / O(eşleşme) sürede getirir.
        """
        with self._lock.read_lock():
            matched_indices: List[int] = []

            if ids is not None:
                # O(|ids|) doğrudan hash erişimi (v0.2.0'daki O(N * |ids|) yerine!)
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

            # Bellek yapılarını sıkıştır (mmap varsa önce serbest bırakılır -> Windows güvenli)
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

            # Diskte atomik güncelleme
            if self._storage.is_persistent:
                if has_raw:
                    self._storage.save_vectors(self.name, self._vbuf.view, compute_crc=True)
                else:
                    disk_vecs = self._storage.load_vectors(self.name, self.dimension, use_mmap=False, verify_crc=False)
                    if disk_vecs.shape[0] == total_before:
                        kept_disk = np.ascontiguousarray(disk_vecs[keep_indices]) if keep_indices else np.empty((0, self.dimension or 0), dtype=disk_vecs.dtype)
                        self._storage.save_vectors(self.name, kept_disk, compute_crc=True)
                compacted_items = [
                    (self._ids[i], i, self._documents[i], self._metadatas[i])
                    for i in range(len(self._ids))
                ]
                self._storage.replace_all_items_after_compaction(self.name, compacted_items)

            # Yardımcı indeksleri yeniden kur
            self._meta_index.rebuild(self._metadatas)
            if has_raw:
                self._rebuild_quantization_locked()
            self._rebuild_ann_index_locked()
            self._invalidate_norm_cache()
            return deleted_count

    def checkpoint(self) -> None:
        """
        Tam CRC32 sağlama toplamı ile atomik disk checkpoint'i oluşturur ve
        varsa parçalanmış yapıları sıkıştırır.
        """
        with self._lock.write_lock():
            if not self._storage.is_persistent:
                return
            # Eğer mmap bağlıysa atomik yazmadan önce belleğe al
            if self._vbuf.is_mmap:
                arr_copy = np.ascontiguousarray(self._vbuf.view)
                self._vbuf.load_array(arr_copy, is_mmap=False)
            self._storage.save_collection_meta(
                self.name,
                self.dimension,
                self.metric,
                self.created_at,
                index_type=self.index_type,
                quantization=self.quantization,
                storage_dtype=self.storage_dtype,
            )
            self._storage.save_vectors(self.name, self._vbuf.view, compute_crc=True)

    def close(self) -> None:
        """Açık memmap tutamaçlarını güvenle kapatır."""
        with self._lock.write_lock():
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
            return {
                "name": self.name,
                "count": len(self._ids),
                "dimension": self.dimension,
                "metric": self.metric,
                "index_type": self.index_type,
                "effective_index": self._effective_index_type(),
                "quantization": self.quantization,
                "storage_dtype": self.storage_dtype,
                "is_mmap": self._vbuf.is_mmap,
                "vector_ram_mb": self._vbuf.memory_bytes() / (1024 * 1024),
                "quantized_ram_mb": q_bytes / (1024 * 1024),
                "vector_disk_mb": self._storage.get_vector_file_size(self.name) / (1024 * 1024),
                "metadata_index": self._meta_index.stats(),
            }

    def __repr__(self) -> str:
        return (
            f"<Collection(name='{self.name}', count={self.count()}, "
            f"dimension={self.dimension}, metric='{self.metric}', "
            f"index='{self.index_type}', quant='{self.quantization}')>"
        )
