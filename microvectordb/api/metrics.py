"""
microvectordb.api.metrics
-------------------------
Sıfır harici bağımlılıklı Prometheus metin dışa aktarım (text exposition v0.0.4)
toplayıcısı (`/metrics` uç noktası için).
"""

import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

_DEFAULT_BUCKETS: Tuple[float, ...] = (
    0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0
)


def _escape_label(val: str) -> str:
    return val.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


class MetricsCollector:
    """Thread-safe HTTP ve veritabanı Prometheus metrik toplayıcısı."""

    def __init__(self, buckets: Tuple[float, ...] = _DEFAULT_BUCKETS) -> None:
        self._buckets = buckets
        self._start_time = time.time()
        self._lock = threading.Lock()
        # (method, route, status) -> count
        self._req_counts: Dict[Tuple[str, str, int], int] = defaultdict(int)
        # (method, route) -> [bucket_counts..., sum, total_count]
        self._lat_hist: Dict[Tuple[str, str], List[float]] = {}
        self._rate_limited: int = 0
        self._auth_failures: int = 0

    @property
    def uptime_seconds(self) -> float:
        return max(0.0, time.time() - self._start_time)

    def record_request(self, method: str, route: str, status: int, duration_s: float) -> None:
        m = method.upper()
        with self._lock:
            self._req_counts[(m, route, int(status))] += 1
            key = (m, route)
            entry = self._lat_hist.get(key)
            if entry is None:
                entry = [0.0] * (len(self._buckets) + 2)
                self._lat_hist[key] = entry
            for i, b in enumerate(self._buckets):
                if duration_s <= b:
                    entry[i] += 1.0
            entry[-2] += float(duration_s)  # sum
            entry[-1] += 1.0  # count

    def record_rate_limited(self) -> None:
        with self._lock:
            self._rate_limited += 1

    def record_auth_failure(self) -> None:
        with self._lock:
            self._auth_failures += 1

    def render_prometheus(self, collection_stats: List[Dict[str, Any]]) -> str:
        """Prometheus exposition format (0.0.4) metin çıktısı üretir."""
        lines: List[str] = []

        lines.append("# HELP mvdb_uptime_seconds MicroVectorDB sunucu çalışma süresi (saniye).")
        lines.append("# TYPE mvdb_uptime_seconds gauge")
        lines.append(f"mvdb_uptime_seconds {self.uptime_seconds:.3f}")

        with self._lock:
            req_snap = dict(self._req_counts)
            lat_snap = {k: list(v) for k, v in self._lat_hist.items()}
            rl_cnt = self._rate_limited
            auth_cnt = self._auth_failures

        lines.append("# HELP mvdb_http_requests_total Toplam HTTP istek sayısı.")
        lines.append("# TYPE mvdb_http_requests_total counter")
        for (m, route, status), cnt in sorted(req_snap.items()):
            lines.append(
                f'mvdb_http_requests_total{{method="{_escape_label(m)}",'
                f'route="{_escape_label(route)}",status="{status}"}} {cnt}'
            )

        lines.append("# HELP mvdb_http_request_duration_seconds HTTP istek gecikme dağılımı (saniye).")
        lines.append("# TYPE mvdb_http_request_duration_seconds histogram")
        for (m, route), entry in sorted(lat_snap.items()):
            base_lbl = f'method="{_escape_label(m)}",route="{_escape_label(route)}"'
            for i, b in enumerate(self._buckets):
                lines.append(
                    f'mvdb_http_request_duration_seconds_bucket{{{base_lbl},le="{b}"}} {int(entry[i])}'
                )
            lines.append(
                f'mvdb_http_request_duration_seconds_bucket{{{base_lbl},le="+Inf"}} {int(entry[-1])}'
            )
            lines.append(
                f"mvdb_http_request_duration_seconds_sum{{{base_lbl}}} {entry[-2]:.6f}"
            )
            lines.append(
                f"mvdb_http_request_duration_seconds_count{{{base_lbl}}} {int(entry[-1])}"
            )

        lines.append("# HELP mvdb_rate_limited_total Hız sınırına takılan (HTTP 429) toplam istek sayısı.")
        lines.append("# TYPE mvdb_rate_limited_total counter")
        lines.append(f"mvdb_rate_limited_total {rl_cnt}")

        lines.append("# HELP mvdb_auth_failures_total Başarısız kimlik doğrulama/yetkilendirme sayısı.")
        lines.append("# TYPE mvdb_auth_failures_total counter")
        lines.append(f"mvdb_auth_failures_total {auth_cnt}")

        lines.append("# HELP mvdb_collections_total Aktif koleksiyon sayısı.")
        lines.append("# TYPE mvdb_collections_total gauge")
        lines.append(f"mvdb_collections_total {len(collection_stats)}")

        lines.append("# HELP mvdb_vectors_total Koleksiyon başına aktif vektör sayısı.")
        lines.append("# TYPE mvdb_vectors_total gauge")
        for st in collection_stats:
            cname = _escape_label(str(st.get("name", "")))
            lines.append(f'mvdb_vectors_total{{collection="{cname}"}} {int(st.get("count", 0))}')

        lines.append("# HELP mvdb_tombstones_total Koleksiyon başına bekleyen tombstone sayısı.")
        lines.append("# TYPE mvdb_tombstones_total gauge")
        for st in collection_stats:
            cname = _escape_label(str(st.get("name", "")))
            lines.append(f'mvdb_tombstones_total{{collection="{cname}"}} {int(st.get("tombstones", 0))}')

        lines.append("# HELP mvdb_vector_ram_bytes Koleksiyon başına bellek içi vektör RAM kullanımı (bayt).")
        lines.append("# TYPE mvdb_vector_ram_bytes gauge")
        for st in collection_stats:
            cname = _escape_label(str(st.get("name", "")))
            ram_b = int(round(float(st.get("vector_ram_mb", 0.0)) * 1024 * 1024))
            lines.append(f'mvdb_vector_ram_bytes{{collection="{cname}"}} {ram_b}')

        return "\n".join(lines) + "\n"
