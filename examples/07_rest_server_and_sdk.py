"""
07_rest_server_and_sdk.py
-------------------------
MicroVectorDB v0.7.0 (Faz 5) Üretim REST Sunucusu + Python HTTP SDK örneği:
1. Hash'lenmiş API anahtarı + Rate Limiter + Prometheus /metrics ile gömülü FastAPI sunucusu başlatır.
2. Sıfır-bağımlılıklı `RemoteMicroVectorDB` SDK'sı (HTTP/1.1 Keep-Alive) ile bağlanır.
3. Koleksiyon oluşturma, doküman ekleme, Dense/Hybrid/Batch arama, Tombstone silme,
   compaction, snapshot ve `/metrics` okuma işlemlerini uçtan uca gösterir.
"""

import os
import shutil
import socket
import tempfile
import threading
import time
import uvicorn

import microvectordb
from microvectordb import RemoteMicroVectorDB
from microvectordb.api import APIKeyRegistry, ServerConfig, create_app


def main() -> None:
    tmp_dir = tempfile.mkdtemp(prefix="mvdb_ex07_")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    reg = APIKeyRegistry(require_auth=True)
    reg.add_key("mvdb_demo_admin_key", key_id="admin", scopes={"read", "write", "admin"})
    cfg = ServerConfig(
        data_dir=os.path.join(tmp_dir, "data"),
        snapshot_dir=os.path.join(tmp_dir, "snapshots"),
        api_key_registry=reg,
        require_auth=True,
        rate_limit_rps=500.0,
        rate_limit_burst=500,
    )
    app = create_app(cfg)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.02)

    try:
        print(f"=== MicroVectorDB v{microvectordb.__version__} REST Sunucusu & Python SDK Demosu ===\n")
        url = f"http://127.0.0.1:{port}"
        with RemoteMicroVectorDB(url, api_key="mvdb_demo_admin_key") as db:
            for _ in range(50):
                try:
                    if db.health()["status"] == "ok":
                        break
                except Exception:
                    time.sleep(0.05)

            print("[1] Sağlık Kontrolü:", db.health(), "| Ready:", db.ready())
            coll = db.get_or_create_collection("articles", dimension=4, metric="cosine")
            coll.add_documents(
                ids=["a1", "a2", "a3"],
                documents=[
                    "İstanbul yapay zeka zirvesi ve vektör veritabanları",
                    "Üretim sunucusu hata kodu ERR-8080 kurtarma kılavuzu",
                    "Docker ve Prometheus gözlemlenebilirlik mimarisi",
                ],
                embeddings=[
                    [0.95, 0.10, 0.05, 0.0],
                    [0.30, 0.92, 0.05, 0.0],
                    [0.05, 0.10, 0.98, 0.0],
                ],
                doc_ids=["doc_1", "doc_2", "doc_3"],
            )
            hits = coll.hybrid_search(
                query_vector=[0.6, 0.8, 0.05, 0.0],
                query_text="ERR-8080 kurtarma",
                limit=2,
                fusion="rrf",
            )
            print("[2] REST Hibrit Arama İlk Sonuç:", hits[0].id, "| skor=", round(hits[0].score, 4))
            snap = db.create_snapshot("demo_snap")
            print("[3] SHA-256 Snapshot alındı:", snap["snapshot_name"], f"({snap['files']} dosya)")
            metric_lines = [ln for ln in db.metrics().splitlines() if ln.startswith("mvdb_vectors_total")]
            print("[4] Prometheus /metrics örneği:", metric_lines)
    finally:
        server.should_exit = True
        th.join(timeout=5)
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
