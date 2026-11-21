# ---- Stage 1: Wheel / Dependency Builder ----
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml setup.py requirements.txt README.md LICENSE ./
COPY microvectordb ./microvectordb

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install ".[all]"

# ---- Stage 2: Minimal Non-Root Production Runtime ----
FROM python:3.11-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MVDB_HOST="0.0.0.0" \
    MVDB_PORT="8000" \
    MVDB_DATA_DIR="/data" \
    MVDB_SNAPSHOT_DIR="/data/_snapshots"

RUN groupadd --gid 10001 mvdb \
    && useradd --uid 10001 --gid 10001 --shell /usr/sbin/nologin --create-home mvdb \
    && mkdir -p /data /data/_snapshots \
    && chown -R mvdb:mvdb /data

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
USER mvdb

EXPOSE 8000
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["microvectordb-server", "--host", "0.0.0.0", "--port", "8000", "--data-dir", "/data"]
