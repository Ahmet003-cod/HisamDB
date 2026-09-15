"""
microvectordb.types
-------------------
Veri yapıları, tip ipuçları ve özel hata sınıfları.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union
import numpy as np

# Desteklenen metrikler
MetricType = Literal["cosine", "dot_product", "euclidean"]

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
