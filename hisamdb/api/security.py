"""
hisamdb.api.security
--------------------------
Üretim sunucusu güvenlik katmanı:
- Hash'lenmiş API anahtarları (SHA-256 + sabit zamanlı `hmac.compare_digest`,
  düz metin anahtarlar bellekte tutulmaz veya loglanmaz)
- Kapsam (scope: read / write / admin) ve kiracı (tenant_id) bazlı yetkilendirme
- Thread-safe Token Bucket hız sınırlayıcı (Rate Limiter)
- Hassas veri/anahtar maskeleyen log filtresi (`mask_secret`, `SanitizingLogFilter`)
"""

import hashlib
import hmac
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

_VALID_SCOPES = frozenset({"read", "write", "admin"})
_SECRET_PATTERN = re.compile(
    r"((?:Bearer\s+|X-API-Key['\"]?\s*[:=]\s*['\"]?)[A-Za-z0-9_\-\.:]+|\b(?:hisam|mvdb)_[a-z0-9_\-]{4,})",
)


def hash_api_key(raw_key: str) -> str:
    """Ham API anahtarını SHA-256 özetine dönüştürür (`sha256:<hex>`)."""
    if raw_key.startswith("sha256:") and len(raw_key) == 71:
        return raw_key.lower()
    digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def mask_secret(value: str) -> str:
    """Anahtar/token dizelerini loglarda güvenle maskeler."""
    if not value:
        return ""
    if len(value) <= 8:
        return "****"
    return f"{value[:4]}****{value[-4:]}"


class SanitizingLogFilter(logging.Filter):
    """Log mesajlarındaki Bearer token ve API anahtarlarını otomatik maskeler."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = _SECRET_PATTERN.sub("[REDACTED]", record.msg)
        if record.args:
            if isinstance(record.args, tuple):
                record.args = tuple(
                    _SECRET_PATTERN.sub("[REDACTED]", a) if isinstance(a, str) else a
                    for a in record.args
                )
            elif isinstance(record.args, dict):
                record.args = {
                    k: (_SECRET_PATTERN.sub("[REDACTED]", v) if isinstance(v, str) else v)
                    for k, v in record.args.items()
                }
        return True


@dataclass(frozen=True)
class AuthPrincipal:
    """Kimliği doğrulanmış istemci öznesi (kapsam ve isteğe bağlı kiracı kilidiyle)."""

    key_id: str
    scopes: Set[str] = field(default_factory=lambda: {"read", "write", "admin"})
    tenant_id: Optional[str] = None

    def has_scope(self, required: str) -> bool:
        if "admin" in self.scopes:
            return True
        return required in self.scopes


class APIKeyRegistry:
    """
    Hash'lenmiş API anahtarı deposu ve sabit-zamanlı doğrulayıcı.
    """

    def __init__(self, require_auth: bool = False) -> None:
        self.require_auth = bool(require_auth)
        # key_hash ("sha256:...") -> AuthPrincipal
        self._keys: Dict[str, AuthPrincipal] = {}
        self._lock = threading.RLock()

    @property
    def enabled(self) -> bool:
        return self.require_auth or len(self._keys) > 0

    def add_key(
        self,
        raw_or_hashed_key: str,
        key_id: str = "default",
        scopes: Optional[Iterable[str]] = None,
        tenant_id: Optional[str] = None,
    ) -> str:
        """
        Ham veya `sha256:<hex>` anahtarı kaydeder. Düz metin anahtar saklanmaz;
        yalnızca SHA-256 özeti saklanır ve geri döndürülür.
        """
        if not raw_or_hashed_key or not raw_or_hashed_key.strip():
            raise ValueError("API anahtarı boş olamaz.")
        norm_scopes = set(scopes) if scopes is not None else {"read", "write", "admin"}
        invalid = norm_scopes - _VALID_SCOPES
        if invalid:
            raise ValueError(f"Geçersiz yetki kapsamları: {invalid}. Geçerliler: {sorted(_VALID_SCOPES)}")
        if tenant_id is not None and not tenant_id.strip():
            raise ValueError("tenant_id boş metin olamaz.")

        hashed = hash_api_key(raw_or_hashed_key.strip())
        principal = AuthPrincipal(
            key_id=key_id,
            scopes=norm_scopes,
            tenant_id=tenant_id.strip() if tenant_id else None,
        )
        with self._lock:
            self._keys[hashed] = principal
        return hashed

    def verify(self, presented_key: Optional[str]) -> Optional[AuthPrincipal]:
        """
        Gelen anahtarı sabit zamanlı (`hmac.compare_digest`) karşılaştırmayla doğrular.
        Kimlik doğrulama devre dışıysa varsayılan yerel yönetici öznesi döndürür.
        """
        if not self.enabled:
            return AuthPrincipal(key_id="anonymous-local", scopes={"read", "write", "admin"}, tenant_id=None)
        if not presented_key:
            return None
        candidate = hash_api_key(presented_key.strip())
        with self._lock:
            matched: Optional[AuthPrincipal] = None
            for stored_hash, principal in self._keys.items():
                if hmac.compare_digest(candidate, stored_hash):
                    matched = principal
            return matched

    @classmethod
    def from_env_spec(cls, spec: Optional[str], require_auth: bool = False) -> "APIKeyRegistry":
        """
        Ortam değişkeni biçiminden anahtar listesi ayrıştırır:
            MVDB_API_KEYS="key1:admin:read,write,admin;key2:acme:read,write:acme_corp"
        Biçim her kayıt için (`;` ile ayrılmış):
            <raw_or_sha256_key>[:<key_id>[:<scope1,scope2>[:<tenant_id>]]]
        """
        reg = cls(require_auth=require_auth)
        if not spec or not spec.strip():
            return reg
        for idx, entry in enumerate(spec.split(";")):
            entry = entry.strip()
            if not entry:
                continue
            # sha256:<hex> ön ekini koruyarak ayrıştır
            if entry.startswith("sha256:"):
                rest = entry[7:]
                parts = rest.split(":")
                key_val = "sha256:" + parts[0]
                tail = parts[1:]
            else:
                parts = entry.split(":")
                key_val = parts[0]
                tail = parts[1:]

            key_id = tail[0].strip() if len(tail) >= 1 and tail[0].strip() else f"key_{idx}"
            scopes = (
                [s.strip() for s in tail[1].split(",") if s.strip()]
                if len(tail) >= 2 and tail[1].strip()
                else ["read", "write", "admin"]
            )
            tenant_id = tail[2].strip() if len(tail) >= 3 and tail[2].strip() else None
            reg.add_key(key_val, key_id=key_id, scopes=scopes, tenant_id=tenant_id)
        return reg


class TokenBucketRateLimiter:
    """
    Thread-safe, kimlik/IP başına Token Bucket hız sınırlayıcı (Rate Limiter).
    `rate_rps <= 0` ise sınırlama devre dışıdır.
    """

    def __init__(self, rate_rps: float = 0.0, burst: int = 0) -> None:
        self.rate_rps = float(rate_rps)
        self.burst = int(burst) if burst > 0 else max(1, int(self.rate_rps * 2))
        self._buckets: Dict[str, Tuple[float, float]] = {}  # key -> (tokens, last_ts)
        self._lock = threading.Lock()
        self._last_gc = time.monotonic()

    @property
    def enabled(self) -> bool:
        return self.rate_rps > 0.0

    def allow(self, client_key: str) -> Tuple[bool, float, float]:
        """
        İsteğe izin verilip verilmediğini kontrol eder.
        Dönüş: (allowed, remaining_tokens, retry_after_seconds)
        """
        if not self.enabled:
            return True, float(self.burst), 0.0

        now = time.monotonic()
        with self._lock:
            # Bellek sızıntısını önlemek için periyodik boş kova temizliği
            if len(self._buckets) > 4096 and (now - self._last_gc) > 60.0:
                stale_cutoff = now - max(60.0, (self.burst / self.rate_rps) * 2.0)
                self._buckets = {k: v for k, v in self._buckets.items() if v[1] >= stale_cutoff}
                self._last_gc = now

            tokens, last_ts = self._buckets.get(client_key, (float(self.burst), now))
            elapsed = max(0.0, now - last_ts)
            tokens = min(float(self.burst), tokens + elapsed * self.rate_rps)

            if tokens >= 1.0:
                tokens -= 1.0
                self._buckets[client_key] = (tokens, now)
                return True, tokens, 0.0

            retry_after = (1.0 - tokens) / self.rate_rps
            self._buckets[client_key] = (tokens, now)
            return False, 0.0, retry_after
