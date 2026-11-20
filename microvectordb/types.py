"""
microvectordb.types
-------------------
Veri yapıları, tip ipuçları ve özel hata sınıfları.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Union
import numpy as np

# Desteklenen metrikler
MetricType = Literal["cosine", "dot_product", "euclidean"]

# Desteklenen indeks, kuantizasyon ve depolama tipleri
IndexType = Literal["flat", "ivf", "hnsw", "auto"]
QuantizationType = Literal["none", "float16", "int8", "binary"]
StorageDType = Literal["float32", "float16"]

# Vektör tipleri (list of floats veya numpy array)
VectorLike = Union[List[float], np.ndarray]
BatchVectors = Union[List[List[float]], List[np.ndarray], np.ndarray]

# Metadata tipi
Metadata = Dict[str, Any]


@dataclass
class QueryResult:
    """Arama sonuçları veri yapısı (ChromaDB stili)."""
    ids: List[List[str]]
    distances: List[List[float]]
    similarities: List[List[float]]
    documents: List[List[Optional[str]]]
    metadatas: List[List[Optional[Metadata]]]

    def to_dict(self) -> Dict[str, Any]:
        """Sözlük formatına dönüştürür."""
        return {
            "ids": self.ids,
            "distances": self.distances,
            "similarities": self.similarities,
            "documents": self.documents,
            "metadatas": self.metadatas,
        }

    def __getitem__(self, item: str) -> Any:
        """Sözlük gibi erişim sağlar: results['ids']."""
        return getattr(self, item)


@dataclass
class SearchResultItem:
    """Tekil arama sonucu öğesi (ergonomik collection.search() API'si için)."""
    id: str
    score: float
    distance: float
    document: Optional[str] = None
    metadata: Optional[Metadata] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "score": self.score,
            "distance": self.distance,
            "document": self.document,
            "metadata": self.metadata,
        }


# Özel Hata Sınıfları
class VectorDBError(Exception):
    """microvectordb temel hata sınıfı."""
    pass


class CollectionNotFoundError(VectorDBError):
    """Koleksiyon bulunamadığında fırlatılır."""
    pass


class CollectionAlreadyExistsError(VectorDBError):
    """Aynı isimde koleksiyon zaten varken fırlatılır."""
    pass


class DimensionMismatchError(VectorDBError):
    """Vektör boyutu koleksiyon boyutuyla uyuşmadığında fırlatılır."""
    pass


class InvalidMetricError(VectorDBError):
    """Geçersiz benzerlik metriği seçildiğinde fırlatılır."""
    pass


class InvalidVectorError(VectorDBError):
    """NaN, Inf veya geçersiz şekle sahip vektör verildiğinde fırlatılır."""
    pass


class InvalidCollectionNameError(VectorDBError):
    """Geçersiz veya güvensiz koleksiyon adı verildiğinde fırlatılır."""
    pass


class DataCorruptionError(VectorDBError):
    """Diskteki veri dosyalarında bütünlük veya sağlama toplamı hatası tespit edildiğinde fırlatılır."""
    pass
