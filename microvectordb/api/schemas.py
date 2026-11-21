"""
microvectordb.api.schemas
-------------------------
FastAPI üretim sunucusu için katı Pydantic v2 istek/yanıt şemaları.
- Sonlu float doğrulaması (NaN / Inf HTTP 422 ile reddedilir)
- Sınırlı `limit`, `nprobe`, `ef_search` ve batch boyut koruması
"""

import math
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


def _check_finite_vector(vec: List[float]) -> List[float]:
    if not vec:
        raise ValueError("Vektör boş olamaz.")
    for v in vec:
        if not math.isfinite(v):
            raise ValueError("Vektörler NaN veya Inf içeremez.")
    return vec


class CreateCollectionRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    dimension: Optional[int] = Field(default=None, ge=1, le=65536)
    metric: Literal["cosine", "euclidean", "dot_product"] = "cosine"
    index_type: Literal["flat", "ivf", "ivf_pq", "hnsw", "disk", "auto"] = "flat"
    quantization: Literal["none", "float16", "int8", "binary", "pq"] = "none"
    storage_dtype: Literal["float32", "float16"] = "float32"
    use_mmap: Optional[bool] = None
    keep_raw_vectors: bool = True
    index_params: Optional[Dict[str, Any]] = None
    get_or_create: bool = False


class AddItemsRequest(BaseModel):
    ids: List[str] = Field(..., min_length=1)
    vectors: List[List[float]] = Field(..., min_length=1)
    documents: Optional[List[Optional[str]]] = None
    metadatas: Optional[List[Optional[Dict[str, Any]]]] = None
    sparse_vectors: Optional[List[Optional[Dict[Union[str, int], float]]]] = None

    @field_validator("vectors")
    @classmethod
    def _validate_vectors(cls, vecs: List[List[float]]) -> List[List[float]]:
        for v in vecs:
            _check_finite_vector(v)
        return vecs


class AddDocumentsRequest(BaseModel):
    documents: List[str] = Field(..., min_length=1)
    embeddings: List[List[float]] = Field(..., min_length=1)
    metadatas: Optional[List[Optional[Dict[str, Any]]]] = None
    ids: Optional[List[str]] = None
    doc_ids: Optional[List[str]] = None
    parent_ids: Optional[List[str]] = None
    sparse_vectors: Optional[List[Optional[Dict[Union[str, int], float]]]] = None

    @field_validator("embeddings")
    @classmethod
    def _validate_embeddings(cls, vecs: List[List[float]]) -> List[List[float]]:
        for v in vecs:
            _check_finite_vector(v)
        return vecs


class SearchRequest(BaseModel):
    query_vector: List[float] = Field(..., min_length=1)
    limit: int = Field(default=10, ge=1, le=10_000)
    filter: Optional[Dict[str, Any]] = None
    score_threshold: Optional[float] = None
    ef_search: Optional[int] = Field(default=None, ge=1, le=10_000)
    nprobe: Optional[int] = Field(default=None, ge=1, le=10_000)
    strategy: str = "auto"
    filter_strategy: str = "auto"

    @field_validator("query_vector")
    @classmethod
    def _validate_q(cls, v: List[float]) -> List[float]:
        return _check_finite_vector(v)


class BatchSearchRequest(BaseModel):
    query_vectors: List[List[float]] = Field(..., min_length=1)
    limit: int = Field(default=10, ge=1, le=10_000)
    filter: Optional[Dict[str, Any]] = None
    filters: Optional[List[Optional[Dict[str, Any]]]] = None
    score_threshold: Optional[float] = None
    ef_search: Optional[int] = Field(default=None, ge=1, le=10_000)
    nprobe: Optional[int] = Field(default=None, ge=1, le=10_000)
    strategy: str = "auto"
    filter_strategy: str = "auto"
    max_workers: Optional[int] = Field(default=None, ge=1, le=64)

    @field_validator("query_vectors")
    @classmethod
    def _validate_qs(cls, vecs: List[List[float]]) -> List[List[float]]:
        for v in vecs:
            _check_finite_vector(v)
        return vecs


class SparseSearchRequest(BaseModel):
    query_text: Optional[str] = None
    query_sparse: Optional[Dict[Union[str, int], float]] = None
    limit: int = Field(default=10, ge=1, le=10_000)
    filter: Optional[Dict[str, Any]] = None
    score_threshold: Optional[float] = None


class HybridSearchRequest(BaseModel):
    query_vector: Optional[List[float]] = None
    query_text: Optional[str] = None
    query_sparse: Optional[Dict[Union[str, int], float]] = None
    limit: int = Field(default=10, ge=1, le=10_000)
    filter: Optional[Dict[str, Any]] = None
    alpha: float = Field(default=0.5, ge=0.0, le=100.0)
    beta: float = Field(default=0.5, ge=0.0, le=100.0)
    sparse_weight: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    fusion: Literal["rrf", "weighted", "z_score"] = "rrf"
    rrf_k: int = Field(default=60, ge=1, le=10_000)
    score_threshold: Optional[float] = None
    deduplicate: bool = False
    dedup_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    group_by: Optional[str] = None
    max_per_group: int = Field(default=1, ge=1, le=1000)

    @field_validator("query_vector")
    @classmethod
    def _validate_qv(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is not None:
            return _check_finite_vector(v)
        return v


class BatchHybridSearchRequest(BaseModel):
    queries: List[HybridSearchRequest] = Field(..., min_length=1)
    limit: int = Field(default=10, ge=1, le=10_000)
    max_workers: Optional[int] = Field(default=None, ge=1, le=64)


class RetrieveContextRequest(BaseModel):
    query_vector: Optional[List[float]] = None
    query_text: Optional[str] = None
    query_sparse: Optional[Dict[Union[str, int], float]] = None
    limit: int = Field(default=5, ge=1, le=1000)
    window: int = Field(default=1, ge=0, le=20)
    filter: Optional[Dict[str, Any]] = None
    fusion: Literal["rrf", "weighted", "z_score"] = "rrf"
    alpha: float = Field(default=0.5, ge=0.0, le=100.0)
    beta: float = Field(default=0.5, ge=0.0, le=100.0)

    @field_validator("query_vector")
    @classmethod
    def _validate_qv(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is not None:
            return _check_finite_vector(v)
        return v


class ExplainRequest(BaseModel):
    n_results: int = Field(default=10, ge=1, le=10_000)
    where: Optional[Dict[str, Any]] = None
    ef_search: Optional[int] = Field(default=None, ge=1, le=10_000)
    nprobe: Optional[int] = Field(default=None, ge=1, le=10_000)
    strategy: str = "auto"
    filter_strategy: str = "auto"


class GetItemsRequest(BaseModel):
    ids: Optional[List[str]] = None
    where: Optional[Dict[str, Any]] = None
    limit: Optional[int] = Field(default=None, ge=1, le=100_000)


class DeleteItemsRequest(BaseModel):
    ids: Optional[List[str]] = None
    where: Optional[Dict[str, Any]] = None


class SnapshotRequest(BaseModel):
    snapshot_name: str = Field(..., min_length=1, max_length=128)


class SearchResultItemResponse(BaseModel):
    id: str
    score: float
    distance: float
    document: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
