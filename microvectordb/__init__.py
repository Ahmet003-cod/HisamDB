"""
microvectordb
=============
Yüksek performanslı, düşük bellek tüketimli, güvenilir ve geliştirici dostu
Python Vektör Veritabanı.

Hızlı Başlangıç:
----------------
>>> from microvectordb import MicroVectorDB
>>> db = MicroVectorDB("./my_db")
>>> collection = db.get_or_create_collection(name="articles", metric="cosine")
>>> collection.add(
...     ids=["id1", "id2"],
...     embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
...     documents=["Yapay zeka makalesi", "Uzay bilimi"],
...     metadatas=[{"kategori": "ai"}, {"kategori": "uzay"}]
... )
>>> results = collection.search(query_vector=[0.1, 0.2, 0.3], limit=1, filter={"kategori": "ai"})
"""

from microvectordb.client import (
    BaseClient,
    EphemeralClient,
    MicroVectorDB,
    PersistentClient,
)
from microvectordb.collection import Collection
from microvectordb.concurrency import RWLock
from microvectordb.filtering import MetadataIndex
from microvectordb.indexes import HNSWIndex, IVFIndex, is_hnsw_available
from microvectordb.quantization import BinaryQuantizer, Float16Quantizer, ScalarQuantizer
from microvectordb.storage import StorageEngine, VectorBuffer
from microvectordb.types import (
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    DataCorruptionError,
    DimensionMismatchError,
    IndexType,
    InvalidCollectionNameError,
    InvalidMetricError,
    InvalidVectorError,
    MetricType,
    QuantizationType,
    QueryResult,
    SearchResultItem,
    StorageDType,
    VectorDBError,
)

__version__ = "0.3.0"
__all__ = [
    "MicroVectorDB",
    "PersistentClient",
    "EphemeralClient",
    "BaseClient",
    "Collection",
    "IVFIndex",
    "HNSWIndex",
    "is_hnsw_available",
    "ScalarQuantizer",
    "Float16Quantizer",
    "BinaryQuantizer",
    "MetadataIndex",
    "StorageEngine",
    "VectorBuffer",
    "RWLock",
    "QueryResult",
    "SearchResultItem",
    "VectorDBError",
    "CollectionNotFoundError",
    "CollectionAlreadyExistsError",
    "DimensionMismatchError",
    "InvalidMetricError",
    "InvalidVectorError",
    "InvalidCollectionNameError",
    "DataCorruptionError",
    "MetricType",
    "IndexType",
    "QuantizationType",
    "StorageDType",
]
