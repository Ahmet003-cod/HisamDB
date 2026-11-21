"""
hisamdb.api
-----------------
REST Sunucusu (opsiyonel FastAPI + Uvicorn) ve Sıfır-Bağımlılıklı Python HTTP SDK paketi.
"""

from hisamdb.api.client import (
    RemoteAPIError,
    RemoteCollection,
    RemoteHisamDB,
    RemoteMicroVectorDB,
)
from hisamdb.api.metrics import MetricsCollector
from hisamdb.api.security import (
    APIKeyRegistry,
    AuthPrincipal,
    SanitizingLogFilter,
    TokenBucketRateLimiter,
    hash_api_key,
    mask_secret,
)

__all__ = [
    "RemoteHisamDB",
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
    from hisamdb.api.server import ServerConfig, create_app

    __all__ += ["ServerConfig", "create_app"]
except ImportError:
    pass
