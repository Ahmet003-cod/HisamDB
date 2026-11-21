"""
microvectordb.api.server
------------------------
Üretim sınıfı FastAPI REST sunucusu:
- Lifespan yönetimli `AsyncMicroVectorDB` çekirdeği (temiz açılış ve kapatma)
- Hash'lenmiş API anahtarları (`Authorization: Bearer` / `X-API-Key`), kapsam (`read`/`write`/`admin`)
  ve sunucu tarafı katı çok-kiracılı (`tenant_id`) yalıtım
- Kimlik/IP bazlı Token Bucket hız sınırlama (`429 Too Many Requests` + `Retry-After`)
- `/health`, `/ready` ve Prometheus `/metrics` (text exposition v0.0.4)
- Hassas verileri maskeleyen loglama ve güvenli hata eşleme (stack trace sızdırmaz)

Başlatma:
    python -m microvectordb.api.server --host 0.0.0.0 --port 8000 --data-dir ./mvdb_data
"""

import argparse
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, PlainTextResponse
except ImportError as exc:
    raise ImportError(
        "MicroVectorDB REST sunucusu için opsiyonel 'server' eklentisi gereklidir: "
        'pip install "microvectordb[server]" (fastapi + uvicorn + pydantic)'
    ) from exc

import microvectordb
from microvectordb.api.metrics import MetricsCollector
from microvectordb.api.schemas import (
    AddDocumentsRequest,
    AddItemsRequest,
    BatchHybridSearchRequest,
    BatchSearchRequest,
    CreateCollectionRequest,
    DeleteItemsRequest,
    ExplainRequest,
    GetItemsRequest,
    HybridSearchRequest,
    RetrieveContextRequest,
    SearchRequest,
    SnapshotRequest,
    SparseSearchRequest,
)
from microvectordb.api.security import (
    APIKeyRegistry,
    AuthPrincipal,
    SanitizingLogFilter,
    TokenBucketRateLimiter,
)
from microvectordb.concurrency.async_client import AsyncCollection, AsyncMicroVectorDB
from microvectordb.storage.engine import validate_collection_name
from microvectordb.types import (
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    DataCorruptionError,
    DimensionMismatchError,
    InvalidCollectionNameError,
    InvalidMetricError,
    InvalidVectorError,
    SearchResultItem,
    VectorDBError,
)

logger = logging.getLogger("microvectordb.api.server")
logger.addFilter(SanitizingLogFilter())


@dataclass
class ServerConfig:
    """MicroVectorDB REST sunucusu yapılandırması."""

    data_dir: Optional[str] = "./microvectordb_server_data"
    snapshot_dir: Optional[str] = None
    use_mmap: bool = False
    max_workers: Optional[int] = None
    require_auth: bool = False
    api_key_registry: Optional[APIKeyRegistry] = None
    rate_limit_rps: float = 0.0
    rate_limit_burst: int = 0
    max_batch_size: int = 10_000
    cors_origins: List[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "ServerConfig":
        data_dir_raw = os.environ.get("MVDB_DATA_DIR", "./microvectordb_server_data")
        data_dir = None if data_dir_raw.lower() in (":memory:", "none", "") else data_dir_raw
        snap_dir = os.environ.get("MVDB_SNAPSHOT_DIR") or (
            os.path.join(data_dir, "_snapshots") if data_dir else None
        )
        require_auth = os.environ.get("MVDB_REQUIRE_AUTH", "false").lower() in ("1", "true", "yes")
        registry = APIKeyRegistry.from_env_spec(
            os.environ.get("MVDB_API_KEYS"), require_auth=require_auth
        )
        cors_raw = os.environ.get("MVDB_CORS_ORIGINS", "")
        cors_origins = [o.strip() for o in cors_raw.split(",") if o.strip()]
        mw_raw = os.environ.get("MVDB_MAX_WORKERS")
        return cls(
            data_dir=data_dir,
            snapshot_dir=snap_dir,
            use_mmap=os.environ.get("MVDB_USE_MMAP", "false").lower() in ("1", "true", "yes"),
            max_workers=int(mw_raw) if mw_raw else None,
            require_auth=require_auth,
            api_key_registry=registry,
            rate_limit_rps=float(os.environ.get("MVDB_RATE_LIMIT_RPS", "0")),
            rate_limit_burst=int(os.environ.get("MVDB_RATE_LIMIT_BURST", "0")),
            max_batch_size=int(os.environ.get("MVDB_MAX_BATCH_SIZE", "10000")),
            cors_origins=cors_origins,
        )


def _scope_filter(flt: Optional[Dict[str, Any]], tenant_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not tenant_id:
        return flt
    cond = {"_tenant_id": tenant_id}
    if not flt:
        return cond
    return {"$and": [cond, flt]}


def _serialize_item(it: SearchResultItem) -> Dict[str, Any]:
    return {
        "id": it.id,
        "score": it.score,
        "distance": it.distance,
        "document": it.document,
        "metadata": it.metadata,
    }


def _extract_api_key(
    authorization: Optional[str], x_api_key: Optional[str]
) -> Optional[str]:
    if x_api_key:
        return x_api_key.strip()
    if authorization:
        parts = authorization.strip().split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip()
    return None


def create_app(config: Optional[ServerConfig] = None) -> FastAPI:
    """Üretim sınıfı MicroVectorDB FastAPI uygulamasını oluşturur."""
    cfg = config or ServerConfig.from_env()
    registry = cfg.api_key_registry or APIKeyRegistry(require_auth=cfg.require_auth)
    limiter = TokenBucketRateLimiter(rate_rps=cfg.rate_limit_rps, burst=cfg.rate_limit_burst)
    metrics = MetricsCollector()

    state: Dict[str, Any] = {"db": None, "ready": False}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        db = AsyncMicroVectorDB(
            path=cfg.data_dir,
            use_mmap=cfg.use_mmap,
            max_workers=cfg.max_workers,
        )
        state["db"] = db
        state["ready"] = True
        if not registry.enabled:
            logger.warning(
                "MicroVectorDB REST sunucusu kimlik doğrulama OLMADAN yerel geliştirme modunda başlatıldı. "
                "Üretim için MVDB_API_KEYS veya MVDB_REQUIRE_AUTH=true ayarlayın."
            )
        try:
            yield
        finally:
            state["ready"] = False
            await db.close()
            state["db"] = None

    app = FastAPI(
        title="MicroVectorDB REST API",
        version=microvectordb.__version__,
        description="Yüksek performanslı, düşük bellekli, hibrit arama ve RAG destekli vektör veritabanı sunucusu.",
        lifespan=lifespan,
    )

    if cfg.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cfg.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # ------------------------------------------------------------------ #
    # Hata Eşleme (Stack Trace Sızdırmaz)
    # ------------------------------------------------------------------ #
    @app.exception_handler(CollectionNotFoundError)
    async def _handle_not_found(_req: Request, exc: CollectionNotFoundError):
        return JSONResponse(status_code=404, content={"detail": str(exc), "error": "collection_not_found"})

    @app.exception_handler(CollectionAlreadyExistsError)
    async def _handle_conflict(_req: Request, exc: CollectionAlreadyExistsError):
        return JSONResponse(status_code=409, content={"detail": str(exc), "error": "collection_already_exists"})

    @app.exception_handler(DimensionMismatchError)
    @app.exception_handler(InvalidVectorError)
    @app.exception_handler(InvalidMetricError)
    @app.exception_handler(InvalidCollectionNameError)
    @app.exception_handler(ValueError)
    async def _handle_bad_request(_req: Request, exc: Exception):
        return JSONResponse(status_code=400, content={"detail": str(exc), "error": "invalid_request"})

    @app.exception_handler(DataCorruptionError)
    async def _handle_corruption(_req: Request, exc: DataCorruptionError):
        logger.error("Veri bütünlüğü hatası: %s", str(exc))
        return JSONResponse(status_code=500, content={"detail": "Veri bütünlüğü hatası tespit edildi.", "error": "data_corruption"})

    @app.exception_handler(VectorDBError)
    async def _handle_vdb_error(_req: Request, exc: VectorDBError):
        return JSONResponse(status_code=400, content={"detail": str(exc), "error": "vectordb_error"})

    # ------------------------------------------------------------------ #
    # Gözlemlenebilirlik + Rate Limiting Ara Katmanı
    # ------------------------------------------------------------------ #
    @app.middleware("http")
    async def _metrics_and_ratelimit_middleware(request: Request, call_next):
        t0 = time.perf_counter()
        path = request.url.path

        # Sağlık/metrik uç noktaları hız sınırlamasından muaftır
        rl_headers: Dict[str, str] = {}
        if limiter.enabled and path.startswith("/v1/"):
            api_key = _extract_api_key(
                request.headers.get("authorization"), request.headers.get("x-api-key")
            )
            client_host = request.client.host if request.client else "unknown"
            bucket_id = f"key:{hash(api_key)}" if api_key else f"ip:{client_host}"
            allowed, remaining, retry_after = limiter.allow(bucket_id)
            if not allowed:
                metrics.record_rate_limited()
                dur = time.perf_counter() - t0
                metrics.record_request(request.method, path, 429, dur)
                return JSONResponse(
                    status_code=429,
                    content={"detail": "İstek hız sınırı aşıldı.", "error": "rate_limit_exceeded"},
                    headers={
                        "Retry-After": f"{max(0.05, retry_after):.2f}",
                        "X-RateLimit-Remaining": "0",
                    },
                )
            rl_headers["X-RateLimit-Remaining"] = f"{int(remaining)}"

        response = await call_next(request)
        dur = time.perf_counter() - t0
        route_obj = request.scope.get("route")
        route_tmpl = getattr(route_obj, "path", path)
        metrics.record_request(request.method, route_tmpl, response.status_code, dur)
        for hk, hv in rl_headers.items():
            response.headers[hk] = hv
        return response

    # ------------------------------------------------------------------ #
    # Bağımlılıklar: DB Erişimi & Kapsam Bazlı Kimlik Doğrulama
    # ------------------------------------------------------------------ #
    def _get_db() -> AsyncMicroVectorDB:
        db = state["db"]
        if db is None or not state["ready"]:
            raise HTTPException(status_code=503, detail="Veritabanı hazır değil.")
        return db

    def _require_scope(scope: str, disallow_tenant_key: bool = False):
        async def _dep(
            authorization: Optional[str] = Header(default=None),
            x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
        ) -> AuthPrincipal:
            presented = _extract_api_key(authorization, x_api_key)
            principal = registry.verify(presented)
            if principal is None:
                metrics.record_auth_failure()
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Geçersiz veya eksik API anahtarı.",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            if not principal.has_scope(scope):
                metrics.record_auth_failure()
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Bu işlem için '{scope}' yetkisi gereklidir.",
                )
            if disallow_tenant_key and principal.tenant_id is not None:
                metrics.record_auth_failure()
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Kiracıya özel API anahtarı sunucu/koleksiyon yönetim işlemi yapamaz.",
                )
            return principal

        return _dep

    def _check_batch_limit(n: int) -> None:
        if n > cfg.max_batch_size:
            raise HTTPException(
                status_code=413,
                detail=f"Batch boyutu ({n}) izin verilen üst sınırı ({cfg.max_batch_size}) aşıyor.",
            )

    # ------------------------------------------------------------------ #
    # Sağlık & Gözlemlenebilirlik Uç Noktaları
    # ------------------------------------------------------------------ #
    @app.get("/health", tags=["ops"])
    async def health():
        return {
            "status": "ok",
            "version": microvectordb.__version__,
            "uptime_s": round(metrics.uptime_seconds, 3),
        }

    @app.get("/ready", tags=["ops"])
    async def ready():
        db = state["db"]
        if db is None or not state["ready"]:
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        colls = await db.list_collections()
        return {"status": "ready", "collections": len(colls)}

    @app.get("/metrics", tags=["ops"], response_class=PlainTextResponse)
    async def prometheus_metrics(_p: AuthPrincipal = Depends(_require_scope("read"))):
        db = _get_db()
        names = await db.list_collections()
        stats_list: List[Dict[str, Any]] = []
        for nm in names:
            c = await db.get_collection(nm)
            stats_list.append(await c.stats())
        return PlainTextResponse(
            content=metrics.render_prometheus(stats_list),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    # ------------------------------------------------------------------ #
    # Koleksiyon Yönetimi (/v1/collections)
    # ------------------------------------------------------------------ #
    @app.get("/v1/collections", tags=["collections"])
    async def list_collections(_p: AuthPrincipal = Depends(_require_scope("read"))):
        db = _get_db()
        return {"collections": await db.list_collections()}

    @app.post("/v1/collections", status_code=201, tags=["collections"])
    async def create_collection(
        req: CreateCollectionRequest,
        _p: AuthPrincipal = Depends(_require_scope("write", disallow_tenant_key=True)),
    ):
        db = _get_db()
        fn = db.get_or_create_collection if req.get_or_create else db.create_collection
        coll = await fn(
            name=req.name,
            dimension=req.dimension,
            metric=req.metric,
            index_type=req.index_type,
            quantization=req.quantization,
            storage_dtype=req.storage_dtype,
            use_mmap=req.use_mmap,
            keep_raw_vectors=req.keep_raw_vectors,
            index_params=req.index_params,
        )
        return await coll.stats()

    @app.get("/v1/collections/{name}", tags=["collections"])
    async def get_collection_stats(
        name: str,
        principal: AuthPrincipal = Depends(_require_scope("read")),
    ):
        db = _get_db()
        if principal.tenant_id:
            tc = await db.get_tenant_collection(name, principal.tenant_id)
            return {
                "name": name,
                "tenant_id": principal.tenant_id,
                "count": await tc.count(),
            }
        coll = await db.get_collection(name)
        return await coll.stats()

    @app.delete("/v1/collections/{name}", tags=["collections"])
    async def delete_collection(
        name: str,
        _p: AuthPrincipal = Depends(_require_scope("admin", disallow_tenant_key=True)),
    ):
        db = _get_db()
        await db.delete_collection(name)
        return {"deleted": name}

    @app.post("/v1/collections/{name}/compact", tags=["collections"])
    async def compact_collection(
        name: str,
        background: bool = False,
        _p: AuthPrincipal = Depends(_require_scope("admin", disallow_tenant_key=True)),
    ):
        db = _get_db()
        coll = await db.get_collection(name)
        if background:
            coll.sync_collection.compact_async()
            return {"status": "scheduled", "collection": name}
        reclaimed = await coll.compact()
        return {"status": "completed", "collection": name, "reclaimed_slots": reclaimed}

    @app.post("/v1/collections/{name}/rebuild", tags=["collections"])
    async def rebuild_collection_index(
        name: str,
        wait: bool = True,
        _p: AuthPrincipal = Depends(_require_scope("admin", disallow_tenant_key=True)),
    ):
        db = _get_db()
        coll = await db.get_collection(name)
        if not wait:
            coll.sync_collection.rebuild_index_async()
            return {"status": "scheduled", "collection": name}
        swapped = await coll.rebuild_index_async()
        return {"status": "completed", "collection": name, "swapped": swapped}

    @app.post("/v1/collections/{name}/checkpoint", tags=["collections"])
    async def checkpoint_collection(
        name: str,
        _p: AuthPrincipal = Depends(_require_scope("admin", disallow_tenant_key=True)),
    ):
        db = _get_db()
        coll = await db.get_collection(name)
        await coll.checkpoint()
        return {"status": "ok", "collection": name}

    # ------------------------------------------------------------------ #
    # CRUD & Doküman Ekleme
    # ------------------------------------------------------------------ #
    @app.post("/v1/collections/{name}/add", tags=["vectors"])
    async def add_items(
        name: str,
        req: AddItemsRequest,
        principal: AuthPrincipal = Depends(_require_scope("write")),
    ):
        _check_batch_limit(len(req.ids))
        db = _get_db()
        if principal.tenant_id:
            tc = await db.get_tenant_collection(name, principal.tenant_id)
            await tc.add(
                ids=req.ids,
                vectors=req.vectors,
                documents=req.documents,  # type: ignore[arg-type]
                metadatas=req.metadatas,  # type: ignore[arg-type]
                sparse_vectors=req.sparse_vectors,
            )
            return {"added": len(req.ids), "tenant_id": principal.tenant_id}

        coll = await db.get_collection(name)
        await coll.add(
            ids=req.ids,
            vectors=req.vectors,
            documents=req.documents,  # type: ignore[arg-type]
            metadatas=req.metadatas,  # type: ignore[arg-type]
            sparse_vectors=req.sparse_vectors,
        )
        return {"added": len(req.ids)}

    @app.post("/v1/collections/{name}/documents", tags=["rag"])
    async def add_documents(
        name: str,
        req: AddDocumentsRequest,
        principal: AuthPrincipal = Depends(_require_scope("write")),
    ):
        _check_batch_limit(len(req.documents))
        db = _get_db()
        coll = await db.get_collection(name)
        gen_ids = await coll.add_documents(
            documents=req.documents,
            embeddings=req.embeddings,
            metadatas=req.metadatas,  # type: ignore[arg-type]
            ids=req.ids,
            doc_ids=req.doc_ids,
            parent_ids=req.parent_ids,
            sparse_vectors=req.sparse_vectors,
            tenant_id=principal.tenant_id,
        )
        return {"ids": gen_ids, "added": len(gen_ids)}

    @app.post("/v1/collections/{name}/upsert", tags=["vectors"])
    async def upsert_items(
        name: str,
        req: AddItemsRequest,
        principal: AuthPrincipal = Depends(_require_scope("write")),
    ):
        _check_batch_limit(len(req.ids))
        db = _get_db()
        coll = await db.get_collection(name)
        metas = req.metadatas
        if principal.tenant_id:
            # Güvenlik: Kiracı B, Kiracı A'ya ait mevcut bir ID'yi üzerine yazamamalı!
            existing = await coll.get(ids=req.ids)
            for eid, emeta in zip(existing["ids"], existing["metadatas"]):
                owner = emeta.get("_tenant_id") if isinstance(emeta, dict) else None
                if owner != principal.tenant_id:
                    raise HTTPException(
                        status_code=403,
                        detail=f"ID '{eid}' başka bir kiracıya ait; güncelleme engellendi.",
                    )
            tagged_metas = []
            for i in range(len(req.ids)):
                base = dict(metas[i]) if (metas is not None and metas[i] is not None) else {}
                base["_tenant_id"] = principal.tenant_id
                tagged_metas.append(base)
            metas = tagged_metas

        await coll.upsert(
            ids=req.ids,
            vectors=req.vectors,
            documents=req.documents,  # type: ignore[arg-type]
            metadatas=metas,  # type: ignore[arg-type]
            sparse_vectors=req.sparse_vectors,
        )
        return {"upserted": len(req.ids)}

    @app.post("/v1/collections/{name}/get", tags=["vectors"])
    async def get_items(
        name: str,
        req: GetItemsRequest,
        principal: AuthPrincipal = Depends(_require_scope("read")),
    ):
        db = _get_db()
        if principal.tenant_id:
            tc = await db.get_tenant_collection(name, principal.tenant_id)
            return await tc.get(ids=req.ids, where=req.where, limit=req.limit)
        coll = await db.get_collection(name)
        return await coll.get(ids=req.ids, where=req.where, limit=req.limit)

    @app.post("/v1/collections/{name}/delete", tags=["vectors"])
    async def delete_items(
        name: str,
        req: DeleteItemsRequest,
        principal: AuthPrincipal = Depends(_require_scope("write")),
    ):
        db = _get_db()
        if principal.tenant_id:
            tc = await db.get_tenant_collection(name, principal.tenant_id)
            deleted = await tc.delete(ids=req.ids, where=req.where)
            return {"deleted": deleted}
        coll = await db.get_collection(name)
        deleted = await coll.delete(ids=req.ids, where=req.where)
        return {"deleted": deleted}

    # ------------------------------------------------------------------ #
    # Arama & RAG Uç Noktaları
    # ------------------------------------------------------------------ #
    @app.post("/v1/collections/{name}/search", tags=["search"])
    async def search_vectors(
        name: str,
        req: SearchRequest,
        principal: AuthPrincipal = Depends(_require_scope("read")),
    ):
        db = _get_db()
        coll = await db.get_collection(name)
        results = await coll.search(
            query_vector=req.query_vector,
            limit=req.limit,
            filter=_scope_filter(req.filter, principal.tenant_id),
            score_threshold=req.score_threshold,
            ef_search=req.ef_search,
            nprobe=req.nprobe,
            strategy=req.strategy,
            filter_strategy=req.filter_strategy,
        )
        return {"results": [_serialize_item(r) for r in results]}

    @app.post("/v1/collections/{name}/batch_search", tags=["search"])
    async def batch_search_vectors(
        name: str,
        req: BatchSearchRequest,
        principal: AuthPrincipal = Depends(_require_scope("read")),
    ):
        _check_batch_limit(len(req.query_vectors))
        db = _get_db()
        coll = await db.get_collection(name)
        eff_filter = _scope_filter(req.filter, principal.tenant_id)
        eff_filters = (
            [_scope_filter(f, principal.tenant_id) for f in req.filters]
            if req.filters is not None
            else None
        )
        batches = await coll.batch_search(
            query_vectors=req.query_vectors,
            limit=req.limit,
            filter=eff_filter,
            filters=eff_filters,
            score_threshold=req.score_threshold,
            ef_search=req.ef_search,
            nprobe=req.nprobe,
            strategy=req.strategy,
            filter_strategy=req.filter_strategy,
            max_workers=req.max_workers,
        )
        return {"results": [[_serialize_item(r) for r in row] for row in batches]}

    @app.post("/v1/collections/{name}/sparse_search", tags=["search"])
    async def sparse_search(
        name: str,
        req: SparseSearchRequest,
        principal: AuthPrincipal = Depends(_require_scope("read")),
    ):
        db = _get_db()
        coll = await db.get_collection(name)
        results = await coll.sparse_search(
            query_text=req.query_text,
            query_sparse=req.query_sparse,
            limit=req.limit,
            filter=_scope_filter(req.filter, principal.tenant_id),
            score_threshold=req.score_threshold,
        )
        return {"results": [_serialize_item(r) for r in results]}

    @app.post("/v1/collections/{name}/hybrid_search", tags=["search"])
    async def hybrid_search(
        name: str,
        req: HybridSearchRequest,
        principal: AuthPrincipal = Depends(_require_scope("read")),
    ):
        db = _get_db()
        coll = await db.get_collection(name)
        results = await coll.hybrid_search(
            query_vector=req.query_vector,
            query_text=req.query_text,
            query_sparse=req.query_sparse,
            limit=req.limit,
            filter=_scope_filter(req.filter, principal.tenant_id),
            alpha=req.alpha,
            beta=req.beta,
            sparse_weight=req.sparse_weight,
            fusion=req.fusion,
            rrf_k=req.rrf_k,
            score_threshold=req.score_threshold,
            deduplicate=req.deduplicate,
            dedup_threshold=req.dedup_threshold,
            group_by=req.group_by,
            max_per_group=req.max_per_group,
        )
        return {"results": [_serialize_item(r) for r in results]}

    @app.post("/v1/collections/{name}/batch_hybrid_search", tags=["search"])
    async def batch_hybrid_search(
        name: str,
        req: BatchHybridSearchRequest,
        principal: AuthPrincipal = Depends(_require_scope("read")),
    ):
        _check_batch_limit(len(req.queries))
        db = _get_db()
        coll = await db.get_collection(name)
        specs = []
        for q in req.queries:
            d = q.model_dump()
            d["filter"] = _scope_filter(d.get("filter"), principal.tenant_id)
            specs.append(d)
        batches = await coll.batch_hybrid_search(
            queries=specs, limit=req.limit, max_workers=req.max_workers
        )
        return {"results": [[_serialize_item(r) for r in row] for row in batches]}

    @app.post("/v1/collections/{name}/retrieve_context", tags=["rag"])
    async def retrieve_context(
        name: str,
        req: RetrieveContextRequest,
        principal: AuthPrincipal = Depends(_require_scope("read")),
    ):
        db = _get_db()
        coll = await db.get_collection(name)
        hits = await coll.retrieve_with_context(
            query_vector=req.query_vector,
            query_text=req.query_text,
            query_sparse=req.query_sparse,
            limit=req.limit,
            window=req.window,
            filter=_scope_filter(req.filter, principal.tenant_id),
            fusion=req.fusion,
            alpha=req.alpha,
            beta=req.beta,
        )
        serialized = []
        for h in hits:
            serialized.append(
                {
                    "item": _serialize_item(h["item"]),
                    "doc_id": h.get("doc_id"),
                    "parent_id": h.get("parent_id"),
                    "context": h["context"],
                    "neighbors": [_serialize_item(nb) for nb in h["neighbors"]],
                }
            )
        return {"results": serialized}

    @app.post("/v1/collections/{name}/explain", tags=["search"])
    async def explain_query(
        name: str,
        req: ExplainRequest,
        principal: AuthPrincipal = Depends(_require_scope("read")),
    ):
        db = _get_db()
        coll = await db.get_collection(name)
        plan = await coll.explain(
            n_results=req.n_results,
            where=_scope_filter(req.where, principal.tenant_id),
            ef_search=req.ef_search,
            nprobe=req.nprobe,
            strategy=req.strategy,
            filter_strategy=req.filter_strategy,
        )
        return asdict(plan)

    # ------------------------------------------------------------------ #
    # Yönetim: Güvenli Snapshot (Path-Traversal Korumalı)
    # ------------------------------------------------------------------ #
    @app.post("/v1/admin/snapshot", tags=["admin"])
    async def create_snapshot(
        req: SnapshotRequest,
        _p: AuthPrincipal = Depends(_require_scope("admin", disallow_tenant_key=True)),
    ):
        db = _get_db()
        if not cfg.snapshot_dir:
            raise HTTPException(
                status_code=400,
                detail="Snapshot yalnızca kalıcı (data_dir / snapshot_dir tanımlı) sunucuda alınabilir.",
            )
        validate_collection_name(req.snapshot_name)
        base = os.path.abspath(cfg.snapshot_dir)
        os.makedirs(base, exist_ok=True)
        target = os.path.abspath(os.path.join(base, req.snapshot_name))
        if os.path.commonpath([base, target]) != base or target == base:
            raise HTTPException(status_code=400, detail="Geçersiz snapshot adı.")
        manifest = await db.create_snapshot(target)
        return {"snapshot_name": req.snapshot_name, "files": len(manifest["files"]), "manifest": manifest}

    return app


def main() -> None:
    """Komut satırından MicroVectorDB REST sunucusunu başlatır."""
    import uvicorn

    parser = argparse.ArgumentParser(description="MicroVectorDB Production REST Server")
    parser.add_argument("--host", default=os.environ.get("MVDB_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MVDB_PORT", "8000")))
    parser.add_argument("--data-dir", default=os.environ.get("MVDB_DATA_DIR", "./microvectordb_server_data"))
    parser.add_argument("--use-mmap", action="store_true")
    parser.add_argument("--max-workers", type=int, default=None)
    args = parser.parse_args()

    cfg = ServerConfig.from_env()
    cfg.data_dir = None if args.data_dir.lower() in (":memory:", "none", "") else args.data_dir
    if cfg.data_dir and not os.environ.get("MVDB_SNAPSHOT_DIR"):
        cfg.snapshot_dir = os.path.join(cfg.data_dir, "_snapshots")
    if args.use_mmap:
        cfg.use_mmap = True
    if args.max_workers is not None:
        cfg.max_workers = args.max_workers

    app = create_app(cfg)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
