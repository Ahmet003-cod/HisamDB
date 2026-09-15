"""
microvectordb
=============
Hafif, yüksek performanslı, kalıcı ve ChromaDB/FAISS benzeri bağımsız Python Vektör Veritabanı.

Hızlı Başlangıç:
----------------
>>> import microvectordb
>>> client = microvectordb.PersistentClient(path="./my_db")
>>> collection = client.get_or_create_collection(name="articles", metric="cosine")
>>> collection.add(
...     ids=["id1", "id2"],
...     embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
...     documents=["Yapay zeka makalesi", "Uzay bilimi"],
...     metadatas=[{"kategori": "ai"}, {"kategori": "uzay"}]
... )
>>> results = collection.query(
...     query_embeddings=[[0.1, 0.2, 0.3]],
...     n_results=1,
...     where={"kategori": "ai"}
... )
"""

from microvectordb.client import BaseClient, EphemeralClient, PersistentClient
from microvectordb.collection import Collection
from microvectordb.ivf_index import IVFIndex
from microvectordb.quantization import ScalarQuantizer
from microvectordb.types import (
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    DimensionMismatchError,
    InvalidMetricError,
    MetricType,
    QueryResult,
    VectorDBError,
)

__version__ = "0.2.0"
__all__ = [
    "PersistentClient",
    "EphemeralClient",
    "BaseClient",
    "Collection",
    "IVFIndex",
    "ScalarQuantizer",
    "QueryResult",
    "VectorDBError",
    "CollectionNotFoundError",
    "CollectionAlreadyExistsError",
    "DimensionMismatchError",
    "InvalidMetricError",
    "MetricType",
]
