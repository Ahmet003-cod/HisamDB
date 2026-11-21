"""
microvectordb
=============
Yüksek performanslı, düşük bellek tüketimli, hibrit arama ve RAG destekli,
WAL korumalı ve geliştirici dostu Python Vektör Veritabanı.
"""

from microvectordb.client import (
    BaseClient,
    EphemeralClient,
    MicroVectorDB,
    PersistentClient,
)
from microvectordb.collection import Collection
from microvectordb.concurrency import (
    BackgroundTaskManager,
    BatchQueryExecutor,
    RWLock,
)
from microvectordb.concurrency.async_client import (
    AsyncCollection,
    AsyncMicroVectorDB,
    AsyncTenantCollection,
)
from microvectordb.filtering import MetadataIndex
from microvectordb.indexes import (
    DiskIndex,
    DiskStreamingScanner,
    HNSWIndex,
    IVFIndex,
    IVFPQIndex,
    QueryOptimizer,
    is_hnsw_available,
)
from microvectordb.quantization import (
    BinaryQuantizer,
    Float16Quantizer,
    ProductQuantizer,
    ScalarQuantizer,
)
from microvectordb.search import (
    BM25Index,
    SparseVector,
    SparseVectorIndex,
    TenantCollection,
    apply_reranker,
    chunk_text,
    deduplicate_results,
    fuse_rrf,
    fuse_weighted,
    group_results_by_field,
    normalize_scores,
    tokenize_text,
)
from microvectordb.storage import StorageEngine, VectorBuffer, WriteAheadLog
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
    QueryPlan,
    QueryResult,
    SearchResultItem,
    StorageDType,
    VectorDBError,
)

__version__ = "0.6.0"
__all__ = [
    "MicroVectorDB",
    "AsyncMicroVectorDB",
    "PersistentClient",
    "EphemeralClient",
    "BaseClient",
    "Collection",
    "AsyncCollection",
    "TenantCollection",
    "AsyncTenantCollection",
    "BatchQueryExecutor",
    "BackgroundTaskManager",
    "IVFIndex",
    "IVFPQIndex",
    "HNSWIndex",
    "DiskIndex",
    "DiskStreamingScanner",
    "QueryOptimizer",
    "is_hnsw_available",
    "ScalarQuantizer",
    "Float16Quantizer",
    "BinaryQuantizer",
    "ProductQuantizer",
    "BM25Index",
    "SparseVectorIndex",
    "SparseVector",
    "tokenize_text",
    "normalize_scores",
    "fuse_rrf",
    "fuse_weighted",
    "chunk_text",
    "apply_reranker",
    "deduplicate_results",
    "group_results_by_field",
    "MetadataIndex",
    "StorageEngine",
    "VectorBuffer",
    "WriteAheadLog",
    "RWLock",
    "QueryPlan",
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
