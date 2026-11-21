"""
HisamDB (hisamdb)
=================
Cutting-Edge, yüksek performanslı, ultra düşük bellek tüketimli, hibrit arama ve RAG destekli,
WAL korumalı ve geliştirici dostu Python Vektör Veritabanı.
"""

from hisamdb.api.client import (
    RemoteAPIError,
    RemoteCollection,
    RemoteHisamDB,
    RemoteMicroVectorDB,
)
from hisamdb.client import (
    BaseClient,
    EphemeralClient,
    HisamDB,
    MicroVectorDB,
    PersistentClient,
)
from hisamdb.collection import Collection
from hisamdb.concurrency import (
    BackgroundTaskManager,
    BatchQueryExecutor,
    RWLock,
)
from hisamdb.concurrency.async_client import (
    AsyncCollection,
    AsyncHisamDB,
    AsyncMicroVectorDB,
    AsyncTenantCollection,
)
from hisamdb.filtering import MetadataIndex
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
    is_native_simd_available,
    native_simd_path,
    set_native_simd_enabled,
)
from hisamdb.quantization import (
    BinaryQuantizer,
    Float16Quantizer,
    ProductQuantizer,
    ScalarQuantizer,
)
from hisamdb.search import (
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
from hisamdb.storage import StorageEngine, VectorBuffer, WriteAheadLog
from hisamdb.types import (
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

__version__ = "1.0.0"
__all__ = [
    "HisamDB",
    "AsyncHisamDB",
    "RemoteHisamDB",
    "MicroVectorDB",
    "AsyncMicroVectorDB",
    "RemoteMicroVectorDB",
    "RemoteCollection",
    "RemoteAPIError",
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
    "is_native_simd_available",
    "native_simd_path",
    "set_native_simd_enabled",
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
