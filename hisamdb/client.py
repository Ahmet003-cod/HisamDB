"""
hisamdb.client
--------------------
Veritabanı istemcileri: `PersistentClient`, `EphemeralClient` ve birleşik `MicroVectorDB`.
ChromaDB uyumlu API + context manager (`with client:`) + thread-safe koleksiyon yönetimi.
"""

import threading
from typing import Any, Dict, List, Optional

from hisamdb.collection import Collection
from hisamdb.concurrency import BackgroundTaskManager, BatchQueryExecutor
from hisamdb.metrics import validate_metric
from hisamdb.search import TenantCollection
from hisamdb.storage import StorageEngine, validate_collection_name
from hisamdb.types import (
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    IndexType,
    MetricType,
    QuantizationType,
    StorageDType,
)


class BaseClient:
    """Temel vektör veritabanı istemcisi."""

    def __init__(
        self,
        storage: StorageEngine,
        use_mmap: bool = False,
        max_workers: Optional[int] = None,
    ):
        self._storage = storage
        self._default_use_mmap = bool(use_mmap)
        self._collections: Dict[str, Collection] = {}
        self._lock = threading.RLock()
        self._query_executor = BatchQueryExecutor(max_workers=max_workers)
        self._bg_tasks = BackgroundTaskManager(max_workers=2)
        self._load_existing_collections()

    def _load_existing_collections(self) -> None:
        """Diskte veya hafızada önceden var olan koleksiyonları başlatır."""
        metas = self._storage.load_collections_meta()
        for meta in metas:
            name = meta["name"]
            meta_params = meta.get("index_params") or {}
            coll = Collection(
                name=name,
                storage=self._storage,
                dimension=meta["dimension"],
                metric=meta["metric"],
                created_at=meta["created_at"],
                index_type=meta.get("index_type", "flat"),
                quantization=meta.get("quantization", "none"),
                storage_dtype=meta.get("storage_dtype", "float32"),
                use_mmap=bool(self._default_use_mmap or meta_params.get("_use_mmap", False)),
                keep_raw_vectors=bool(meta_params.get("_keep_raw_vectors", True)),
                index_params=meta_params,
                data_epoch=meta.get("data_epoch", 0),
                query_executor=self._query_executor,
                bg_tasks=self._bg_tasks,
            )
            self._collections[name] = coll

    def create_collection(
        self,
        name: str,
        dimension: Optional[int] = None,
        metric: MetricType = "cosine",
        index_type: IndexType = "flat",
        quantization: QuantizationType = "none",
        storage_dtype: StorageDType = "float32",
        use_mmap: Optional[bool] = None,
        keep_raw_vectors: bool = True,
        index_params: Optional[Dict[str, Any]] = None,
    ) -> Collection:
        """Yeni bir koleksiyon oluşturur. Varsa hata fırlatır."""
        validate_collection_name(name)
        validate_metric(metric)

        with self._lock:
            if name in self._collections:
                raise CollectionAlreadyExistsError(f"'{name}' adlı koleksiyon zaten mevcut.")

            coll = Collection(
                name=name,
                storage=self._storage,
                dimension=dimension,
                metric=metric,
                index_type=index_type,
                quantization=quantization,
                storage_dtype=storage_dtype,
                use_mmap=self._default_use_mmap if use_mmap is None else use_mmap,
                keep_raw_vectors=keep_raw_vectors,
                index_params=index_params,
                query_executor=self._query_executor,
                bg_tasks=self._bg_tasks,
            )
            self._collections[name] = coll
            self._storage.save_collection_meta(
                name,
                dimension,
                metric,
                coll.created_at,
                index_type=coll.index_type,
                quantization=coll.quantization,
                storage_dtype=coll.storage_dtype,
                data_epoch=coll._data_epoch,
                index_params=coll.index_params,
            )
            return coll

    def get_collection(self, name: str) -> Collection:
        """Mevcut bir koleksiyonu getirir. Yoksa hata fırlatır."""
        with self._lock:
            if name not in self._collections:
                raise CollectionNotFoundError(f"'{name}' adlı koleksiyon bulunamadı.")
            return self._collections[name]

    def get_or_create_collection(
        self,
        name: str,
        dimension: Optional[int] = None,
        metric: MetricType = "cosine",
        index_type: IndexType = "flat",
        quantization: QuantizationType = "none",
        storage_dtype: StorageDType = "float32",
        use_mmap: Optional[bool] = None,
        keep_raw_vectors: bool = True,
        index_params: Optional[Dict[str, Any]] = None,
    ) -> Collection:
        """Koleksiyon varsa getirir, yoksa yenisini oluşturur."""
        with self._lock:
            if name in self._collections:
                return self._collections[name]
            return self.create_collection(
                name=name,
                dimension=dimension,
                metric=metric,
                index_type=index_type,
                quantization=quantization,
                storage_dtype=storage_dtype,
                use_mmap=use_mmap,
                keep_raw_vectors=keep_raw_vectors,
                index_params=index_params,
            )

    def list_collections(self) -> List[str]:
        """Tüm koleksiyon isimlerinin listesini döndürür."""
        with self._lock:
            return list(self._collections.keys())

    def delete_collection(self, name: str) -> None:
        """Koleksiyonu ve ilişkili tüm verilerini diskten güvenle siler."""
        with self._lock:
            if name not in self._collections:
                raise CollectionNotFoundError(f"'{name}' adlı koleksiyon bulunamadı.")
            coll = self._collections[name]
            coll.close()
            self._storage.delete_collection_storage(name)
            del self._collections[name]

    def reset(self) -> None:
        """Tüm koleksiyonları ve veritabanını sıfırlar."""
        with self._lock:
            for name in list(self._collections.keys()):
                self.delete_collection(name)

    def get_tenant_collection(self, name: str, tenant_id: str) -> TenantCollection:
        """Çok kiracılı (Multi-Tenant) izole koleksiyon görünümü döndürür."""
        return TenantCollection(self.get_collection(name), tenant_id=tenant_id)

    def checkpoint(self) -> None:
        """Tüm koleksiyonlar için tam CRC32 atomik checkpoint yazar."""
        with self._lock:
            for coll in self._collections.values():
                coll.checkpoint()

    def create_snapshot(self, snapshot_dir: str) -> Dict[str, Any]:
        """
        Tüm koleksiyonları checkpoint edip SHA-256 doğrulamalı tutarlı anlık görüntü alır.
        """
        with self._lock:
            self.checkpoint()
            return self._storage.create_snapshot(snapshot_dir)

    @classmethod
    def restore_snapshot(
        cls,
        snapshot_dir: str,
        target_dir: str,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """SHA-256 manifestini doğrulayarak snapshot'ı `target_dir` konumuna geri yükler."""
        return StorageEngine.restore_snapshot(snapshot_dir, target_dir, overwrite=overwrite)

    def wait_for_background(self, timeout: Optional[float] = None) -> None:
        """Tüm koleksiyonlardaki bekleyen arka plan görevlerinin tamamlanmasını bekler."""
        self._bg_tasks.wait_all(timeout=timeout)

    def close(self) -> None:
        """Arka plan görevlerini tamamlar, tüm açık mmap/SQLite/havuz kaynaklarını kapatır."""
        self._bg_tasks.shutdown(wait=True)
        with self._lock:
            for coll in self._collections.values():
                coll.close()
            self._storage.close()
        self._query_executor.shutdown(wait=True)

    def __enter__(self) -> "BaseClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class PersistentClient(BaseClient):
    """
    Kalıcı (disk tabanlı) Vektör Veritabanı İstemcisi.

    Kullanım:
        with hisamdb.PersistentClient(path="./my_db") as client:
            collection = client.get_or_create_collection("docs")
    """

    def __init__(
        self,
        path: str = "./hisamdb_data",
        use_mmap: bool = False,
        max_workers: Optional[int] = None,
    ):
        storage = StorageEngine(persist_dir=path)
        super().__init__(storage=storage, use_mmap=use_mmap, max_workers=max_workers)
        self.path = path

    def __repr__(self) -> str:
        return f"<PersistentClient(path='{self.path}', collections={len(self._collections)})>"


class EphemeralClient(BaseClient):
    """
    Bellek içi (In-Memory) Vektör Veritabanı İstemcisi.
    Sıfır disk I/O ve sıfır SQLite serileştirme maliyeti ile maksimum hız sunar.
    """

    def __init__(self, max_workers: Optional[int] = None) -> None:
        storage = StorageEngine(persist_dir=None)
        super().__init__(storage=storage, use_mmap=False, max_workers=max_workers)

    def __repr__(self) -> str:
        return f"<EphemeralClient(collections={len(self._collections)})>"


class HisamDB(BaseClient):
    """
    Birleşik HisamDB istemcisi.
    `path` verilirse kalıcı (disk tabanlı), `None` ise bellek içi çalışır.

    Örnek:
        db = HisamDB("./database")
        coll = db.create_collection("knowledge_base", dimension=768, metric="cosine", index_type="hnsw")
    """

    def __init__(
        self,
        path: Optional[str] = None,
        use_mmap: bool = False,
        max_workers: Optional[int] = None,
    ):
        storage = StorageEngine(persist_dir=path)
        super().__init__(storage=storage, use_mmap=use_mmap, max_workers=max_workers)
        self.path = path

    def __repr__(self) -> str:
        mode = f"path='{self.path}'" if self.path else "in-memory"
        return f"<HisamDB({mode}, collections={len(self._collections)})>"


# Geriye dönük uyumluluk takma adı
MicroVectorDB = HisamDB
