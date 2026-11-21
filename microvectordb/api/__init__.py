"""
microvectordb.api
-----------------
REST Sunucusu (opsiyonel FastAPI + Uvicorn) ve Sıfır-Bağımlılıklı Python HTTP SDK paketi.
"""

from microvectordb.api.client import RemoteAPIError, RemoteCollection, RemoteMicroVectorDB
from microvectordb.api.metrics import MetricsCollector
from microvectordb.api.security import (
    APIKeyRegistry,
    AuthPrincipal,
    SanitizingLogFilter,
    TokenBucketRateLimiter,
    hash_api_key,
    mask_secret,
)

__all__ = [
    "RemoteMicroVectorDB",
    "RemoteCollection",
    "RemoteAPIError",
    "APIKeyRegistry",
    "AuthPrincipal",
    "TokenBucketRateLimiter",
    "SanitizingLogFilter",
    "MetricsCollector",
    "hash_api_key",
    "mask_secret",
]

try:
    from microvectordb.api.server import ServerConfig, create_app

    __all__ += ["ServerConfig", "create_app"]
except ImportError:
    pass
