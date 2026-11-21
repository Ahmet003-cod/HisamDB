"""
hisamdb.concurrency.async_client
--------------------------------------
Standart kütüphane `asyncio` tabanlı bloklamayan (non-blocking) asenkron istemci:
- `AsyncHisamDB`
- `AsyncCollection`
- `AsyncTenantCollection`

FastAPI, LangChain, AI Agent ve yüksek eşzamanlılıklı `asyncio.gather()` iş akışları için
sıfır ek bağımlılıkla (yalnızca stdlib asyncio + ThreadPoolExecutor) tam async erişim sunar.
"""

import asyncio
import functools
from typing import Any, Dict, List, Optional, Sequence, Union

from hisamdb.client import HisamDB
from hisamdb.collection import Collection
from hisamdb.concurrency.executor import BatchQueryExecutor
from hisamdb.search.rag import TenantCollection
from hisamdb.search.sparse import SparseVector
from hisamdb.types import (
    BatchVectors,
    IndexType,
    Metadata,
    MetricType,
    QuantizationType,
    QueryPlan,
    QueryResult,
    SearchResultItem,
    StorageDType,
    VectorLike,
)


class AsyncCollection:
    """`Collection` için bloklamayan `asyncio` sarmalayıcısı."""

    def __init__(self, collection: Collection, executor: BatchQueryExecutor) -> None:
        self._collection = collection
        self._executor = executor

    @property
    def name(self) -> str:
        return self._collection.name

    @property
    def dimension(self) -> Optional[int]:
        return self._collection.dimension

    @property
    def metric(self) -> MetricType:
        return self._collection.metric

    @property
    def sync_collection(self) -> Collection:
        return self._collection

    async def _run(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        pool = self._executor._get_pool()
        call = functools.partial(fn, *args, **kwargs) if kwargs else (
            functools.partial(fn, *args) if args else fn
        )
        return await loop.run_in_executor(pool, call)

    async def add(
        self,
        ids: List[str],
        embeddings: Optional[BatchVectors] = None,
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Metadata]] = None,
        *,
        vectors: Optional[BatchVectors] = None,
        metadata: Optional[List[Metadata]] = None,
        sparse_vectors: Optional[List[Optional[SparseVector]]] = None,
    ) -> None:
        await self._run(
            self._collection.add,
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            vectors=vectors,
            metadata=metadata,
            sparse_vectors=sparse_vectors,
        )

    async def add_documents(
        self,
        documents: List[str],
        embeddings: BatchVectors,
        metadatas: Optional[List[Metadata]] = None,
        ids: Optional[List[str]] = None,
        doc_ids: Optional[List[str]] = None,
        parent_ids: Optional[List[str]] = None,
        sparse_vectors: Optional[List[Optional[SparseVector]]] = None,
        tenant_id: Optional[str] = None,
    ) -> List[str]:
        return await self._run(
            self._collection.add_documents,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
            doc_ids=doc_ids,
            parent_ids=parent_ids,
            sparse_vectors=sparse_vectors,
            tenant_id=tenant_id,
        )

    async def upsert(
        self,
        ids: List[str],
        embeddings: Optional[BatchVectors] = None,
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Metadata]] = None,
        *,
        vectors: Optional[BatchVectors] = None,
        metadata: Optional[List[Metadata]] = None,
        sparse_vectors: Optional[List[Optional[SparseVector]]] = None,
    ) -> None:
        await self._run(
            self._collection.upsert,
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            vectors=vectors,
            metadata=metadata,
            sparse_vectors=sparse_vectors,
        )

    async def query(self, *args: Any, **kwargs: Any) -> QueryResult:
        return await self._run(self._collection.query, *args, **kwargs)

    async def search(
        self,
        query_vector: VectorLike,
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
        ef_search: Optional[int] = None,
        nprobe: Optional[int] = None,
        strategy: str = "auto",
        filter_strategy: str = "auto",
    ) -> List[SearchResultItem]:
        return await self._run(
            self._collection.search,
            query_vector=query_vector,
            limit=limit,
            filter=filter,
            score_threshold=score_threshold,
            ef_search=ef_search,
            nprobe=nprobe,
            strategy=strategy,
            filter_strategy=filter_strategy,
        )

    async def batch_search(
        self,
        query_vectors: BatchVectors,
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        filters: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        score_threshold: Optional[float] = None,
        ef_search: Optional[int] = None,
        nprobe: Optional[int] = None,
        strategy: str = "auto",
        filter_strategy: str = "auto",
        max_workers: Optional[int] = None,
    ) -> List[List[SearchResultItem]]:
        return await self._run(
            self._collection.batch_search,
            query_vectors=query_vectors,
            limit=limit,
            filter=filter,
            filters=filters,
            score_threshold=score_threshold,
            ef_search=ef_search,
            nprobe=nprobe,
            strategy=strategy,
            filter_strategy=filter_strategy,
            max_workers=max_workers,
        )

    async def sparse_search(self, *args: Any, **kwargs: Any) -> List[SearchResultItem]:
        return await self._run(self._collection.sparse_search, *args, **kwargs)

    async def hybrid_search(self, *args: Any, **kwargs: Any) -> List[SearchResultItem]:
        return await self._run(self._collection.hybrid_search, *args, **kwargs)

    async def batch_hybrid_search(
        self,
        queries: Sequence[Dict[str, Any]],
        limit: int = 10,
        max_workers: Optional[int] = None,
        **default_kwargs: Any,
    ) -> List[List[SearchResultItem]]:
        return await self._run(
            self._collection.batch_hybrid_search,
            queries=queries,
            limit=limit,
            max_workers=max_workers,
            **default_kwargs,
        )

    async def multi_vector_search(self, *args: Any, **kwargs: Any) -> List[SearchResultItem]:
        return await self._run(self._collection.multi_vector_search, *args, **kwargs)

    async def retrieve_with_context(self, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        return await self._run(self._collection.retrieve_with_context, *args, **kwargs)

    async def get(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        return await self._run(self._collection.get, ids=ids, where=where, limit=limit)

    async def get_all_ids(self) -> List[str]:
        return await self._run(self._collection.get_all_ids)

    async def peek(self, limit: int = 5) -> Dict[str, Any]:
        return await self._run(self._collection.peek, limit=limit)

    async def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> int:
        return await self._run(self._collection.delete, ids=ids, where=where)

    async def compact(self) -> int:
        return await self._run(self._collection.compact)

    async def compact_async(self) -> int:
        fut = self._collection.compact_async()
        return await asyncio.wrap_future(fut)

    async def rebuild_index_async(self) -> bool:
        fut = self._collection.rebuild_index_async()
        return await asyncio.wrap_future(fut)

    async def checkpoint(self) -> None:
        await self._run(self._collection.checkpoint)

    async def explain(self, *args: Any, **kwargs: Any) -> QueryPlan:
        return await self._run(self._collection.explain, *args, **kwargs)

    async def count(self) -> int:
        return await self._run(self._collection.count)

    async def stats(self) -> Dict[str, Any]:
        return await self._run(self._collection.stats)

    def __repr__(self) -> str:
        return f"<AsyncCollection(name='{self.name}', metric='{self.metric}')>"


class AsyncTenantCollection:
    """`TenantCollection` için bloklamayan `asyncio` sarmalayıcısı."""

    def __init__(self, tenant_coll: TenantCollection, executor: BatchQueryExecutor) -> None:
        self._tenant = tenant_coll
        self._executor = executor

    @property
    def tenant_id(self) -> str:
        return self._tenant.tenant_id

    async def _run(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        pool = self._executor._get_pool()
        call = functools.partial(fn, *args, **kwargs) if kwargs else (
            functools.partial(fn, *args) if args else fn
        )
        return await loop.run_in_executor(pool, call)

    async def add(self, *args: Any, **kwargs: Any) -> None:
        await self._run(self._tenant.add, *args, **kwargs)

    async def add_documents(self, *args: Any, **kwargs: Any) -> List[str]:
        return await self._run(self._tenant.add_documents, *args, **kwargs)

    async def query(self, *args: Any, **kwargs: Any) -> QueryResult:
        return await self._run(self._tenant.query, *args, **kwargs)

    async def search(self, *args: Any, **kwargs: Any) -> List[SearchResultItem]:
        return await self._run(self._tenant.search, *args, **kwargs)

    async def hybrid_search(self, *args: Any, **kwargs: Any) -> List[SearchResultItem]:
        return await self._run(self._tenant.hybrid_search, *args, **kwargs)

    async def get(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return await self._run(self._tenant.get, *args, **kwargs)

    async def delete(self, *args: Any, **kwargs: Any) -> int:
        return await self._run(self._tenant.delete, *args, **kwargs)

    async def count(self) -> int:
        return await self._run(self._tenant.count)

    def __repr__(self) -> str:
        return f"<AsyncTenantCollection(tenant='{self.tenant_id}')>"


class AsyncHisamDB:
    """
    Bloklamayan `asyncio` HisamDB istemcisi.

    Kullanım:
        async with AsyncHisamDB("./database") as db:
            coll = await db.get_or_create_collection("docs", dimension=128, index_type="hnsw")
            await coll.add(ids=..., vectors=...)
            results = await asyncio.gather(*(coll.search(q) for q in queries))
    """

    def __init__(
        self,
        path: Optional[str] = None,
        use_mmap: bool = False,
        max_workers: Optional[int] = None,
    ) -> None:
        self._db = HisamDB(path=path, use_mmap=use_mmap, max_workers=max_workers)
        self._executor = self._db._query_executor
        self.path = path

    @property
    def sync_client(self) -> HisamDB:
        return self._db

    async def _run(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        pool = self._executor._get_pool()
        call = functools.partial(fn, *args, **kwargs) if kwargs else (
            functools.partial(fn, *args) if args else fn
        )
        return await loop.run_in_executor(pool, call)

    async def create_collection(
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
    ) -> AsyncCollection:
        coll = await self._run(
            self._db.create_collection,
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
        return AsyncCollection(coll, self._executor)

    async def get_collection(self, name: str) -> AsyncCollection:
        coll = await self._run(self._db.get_collection, name=name)
        return AsyncCollection(coll, self._executor)

    async def get_or_create_collection(
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
    ) -> AsyncCollection:
        coll = await self._run(
            self._db.get_or_create_collection,
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
        return AsyncCollection(coll, self._executor)

    async def get_tenant_collection(self, name: str, tenant_id: str) -> AsyncTenantCollection:
        tc = await self._run(self._db.get_tenant_collection, name=name, tenant_id=tenant_id)
        return AsyncTenantCollection(tc, self._executor)

    async def list_collections(self) -> List[str]:
        return await self._run(self._db.list_collections)

    async def delete_collection(self, name: str) -> None:
        await self._run(self._db.delete_collection, name=name)

    async def reset(self) -> None:
        await self._run(self._db.reset)

    async def checkpoint(self) -> None:
        await self._run(self._db.checkpoint)

    async def create_snapshot(self, snapshot_dir: str) -> Dict[str, Any]:
        return await self._run(self._db.create_snapshot, snapshot_dir=snapshot_dir)

    @classmethod
    async def restore_snapshot(
        cls,
        snapshot_dir: str,
        target_dir: str,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            functools.partial(
                HisamDB.restore_snapshot,
                snapshot_dir=snapshot_dir,
                target_dir=target_dir,
                overwrite=overwrite,
            ),
        )

    async def close(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._db.close)

    async def __aenter__(self) -> "AsyncHisamDB":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    def __repr__(self) -> str:
        mode = f"path='{self.path}'" if self.path else "in-memory"
        return f"<AsyncHisamDB({mode})>"


# Geriye dönük uyumluluk takma adı
AsyncMicroVectorDB = AsyncHisamDB
