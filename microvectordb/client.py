"""
microvectordb.client
--------------------
Veritabanı istemcisi (Client).
ChromaDB'nin Client API'si ile tam uyumlu 'PersistentClient' ve 'EphemeralClient' sınıflarını sunar.
"""

from typing import Dict, List, Optional
from microvectordb.collection import Collection
from microvectordb.storage import StorageEngine
from microvectordb.types import (
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    MetricType,
)


class BaseClient:
    """Temel vektör veritabanı istemcisi."""

    def __init__(self, storage: StorageEngine):
        self._storage = storage
        self._collections: Dict[str, Collection] = {}
        self._load_existing_collections()

    def _load_existing_collections(self) -> None:
        """Diskte veya hafızada önceden var olan koleksiyonları başlatır."""
        metas = self._storage.load_collections_meta()
        for meta in metas:
            name = meta["name"]
            coll = Collection(
                name=name,
                storage=self._storage,
                dimension=meta["dimension"],
                metric=meta["metric"],
                created_at=meta["created_at"],
            )
            self._collections[name] = coll

    def create_collection(
        self,
        name: str,
        dimension: Optional[int] = None,
        metric: MetricType = "cosine",
    ) -> Collection:
        """
        Yeni bir koleksiyon oluşturur. Varsa hata fırlatır.
        """
        if name in self._collections:
            raise CollectionAlreadyExistsError(f"'{name}' adlı koleksiyon zaten mevcut.")

        coll = Collection(
            name=name,
            storage=self._storage,
            dimension=dimension,
            metric=metric,
        )
        self._collections[name] = coll
        self._storage.save_collection_meta(name, dimension, metric, coll.created_at)
        return coll

    def get_collection(self, name: str) -> Collection:
        """
        Mevcut bir koleksiyonu getirir. Yoksa hata fırlatır.
        """
        if name not in self._collections:
            raise CollectionNotFoundError(f"'{name}' adlı koleksiyon bulunamadı.")
        return self._collections[name]

    def get_or_create_collection(
        self,
        name: str,
        dimension: Optional[int] = None,
        metric: MetricType = "cosine",
    ) -> Collection:
        """
        Koleksiyon varsa getirir, yoksa yenisini oluşturur.
        (ChromaDB'deki get_or_create_collection ile aynı davranış).
        """
        if name in self._collections:
            return self._collections[name]
        return self.create_collection(name=name, dimension=dimension, metric=metric)

    def list_collections(self) -> List[str]:
        """Tüm koleksiyon isimlerinin listesini döndürür."""
        return list(self._collections.keys())

    def delete_collection(self, name: str) -> None:
        """
        Koleksiyonu ve ilişkili tüm verilerini diskten siler.
        """
        if name not in self._collections:
            raise CollectionNotFoundError(f"'{name}' adlı koleksiyon bulunamadı.")

        self._storage.delete_collection_storage(name)
        del self._collections[name]

    def reset(self) -> None:
        """Tüm koleksiyonları ve veritabanını sıfırlar."""
        for name in list(self._collections.keys()):
            self.delete_collection(name)


class PersistentClient(BaseClient):
    """
    Kalıcı (disk tabanlı) Vektör Veritabanı İstemcisi.
    Veriler belirtilen klasörde güvenle saklanır ve program yeniden başlatıldığında aynen korunur.

    Kullanım:
        client = microvectordb.PersistentClient(path="./my_db")
        collection = client.get_or_create_collection("docs")
    """

    def __init__(self, path: str = "./microvectordb_data"):
        storage = StorageEngine(persist_dir=path)
        super().__init__(storage=storage)
        self.path = path

    def __repr__(self) -> str:
        return f"<PersistentClient(path='{self.path}', collections={len(self._collections)})>"


class EphemeralClient(BaseClient):
    """
    Bellek içi (In-Memory) Vektör Veritabanı İstemcisi.
    Diske hiçbir şey yazmaz, test ve geçici işlemler için son derece hızlıdır.

    Kullanım:
        client = microvectordb.EphemeralClient()
    """

    def __init__(self):
        storage = StorageEngine(persist_dir=None)
        super().__init__(storage=storage)

    def __repr__(self) -> str:
        return f"<EphemeralClient(collections={len(self._collections)})>"
