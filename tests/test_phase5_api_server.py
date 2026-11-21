"""
test_phase5_api_server.py
-------------------------
MicroVectorDB v0.7.0 (Faz 5 — FastAPI Üretim Sunucusu, Güvenlik, Rate Limiting,
Prometheus /metrics ve Sıfır-Bağımlılıklı RemoteMicroVectorDB Python SDK) test paketi:
1. /health, /ready ve Prometheus /metrics (exposition v0.0.4)
2. Tam REST CRUD, Dense/Batch/Sparse/Hybrid/RAG Context, Explain, Compact, Rebuild, Snapshot
3. Hash'lenmiş API anahtarları (sha256:), kapsam (read/write/admin) ve sunucu tarafı
   katı Multi-Tenant yalıtımı (çapraz kiracı okuma/silme/upsert saldırı testi!)
4. Token Bucket hız sınırlama (HTTP 429 + Retry-After), girdi doğrulama (NaN/413/404/409),
   snapshot path-traversal koruması ve log maskeleme
5. Canlı Uvicorn HTTP sunucusu + RemoteMicroVectorDB (Keep-Alive SDK) uçtan uca testi
"""

import logging
import os
import shutil
import socket
import tempfile
import threading
import time
import unittest

try:
    from fastapi.testclient import TestClient
    import uvicorn
    HAS_SERVER_DEPS = True
except ImportError:
    HAS_SERVER_DEPS = False

from microvectordb import (
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    RemoteAPIError,
    RemoteMicroVectorDB,
)
from microvectordb.api.security import (
    APIKeyRegistry,
    SanitizingLogFilter,
    hash_api_key,
    mask_secret,
)

if HAS_SERVER_DEPS:
    from microvectordb.api.server import ServerConfig, create_app


@unittest.skipUnless(HAS_SERVER_DEPS, "fastapi/uvicorn/httpx kurulu değil")
class TestPhase5APIServer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="mvdb_api_test_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_ops_health_ready_and_prometheus_metrics(self):
        cfg = ServerConfig(
            data_dir=os.path.join(self.temp_dir, "db_ops"),
            snapshot_dir=os.path.join(self.temp_dir, "snaps"),
        )
        app = create_app(cfg)
        with TestClient(app) as client:
            r_h = client.get("/health")
            self.assertEqual(r_h.status_code, 200)
            self.assertEqual(r_h.json()["status"], "ok")

            r_r = client.get("/ready")
            self.assertEqual(r_r.status_code, 200)
            self.assertEqual(r_r.json()["status"], "ready")

            client.post("/v1/collections", json={"name": "mtest", "dimension": 3})
            client.post("/v1/collections/mtest/add", json={"ids": ["1"], "vectors": [[1, 0, 0]]})

            r_m = client.get("/metrics")
            self.assertEqual(r_m.status_code, 200)
            body = r_m.text
            self.assertIn("mvdb_http_requests_total", body)
            self.assertIn("mvdb_http_request_duration_seconds_bucket", body)
            self.assertIn('mvdb_vectors_total{collection="mtest"} 1', body)
            self.assertIn("mvdb_collections_total 1", body)

    def test_full_rest_crud_search_hybrid_rag_and_maintenance(self):
        cfg = ServerConfig(
            data_dir=os.path.join(self.temp_dir, "db_crud"),
            snapshot_dir=os.path.join(self.temp_dir, "snaps_crud"),
        )
        app = create_app(cfg)
        with TestClient(app) as client:
            # Koleksiyon oluştur + listele + stats
            r = client.post(
                "/v1/collections",
                json={"name": "kb", "dimension": 4, "metric": "cosine", "index_type": "flat"},
            )
            self.assertEqual(r.status_code, 201)
            self.assertEqual(client.get("/v1/collections").json()["collections"], ["kb"])

            # Çakışma (409) & yok (404)
            self.assertEqual(client.post("/v1/collections", json={"name": "kb", "dimension": 4}).status_code, 409)
            self.assertEqual(client.get("/v1/collections/nonexistent").status_code, 404)

            # Doküman ekleme (Parent-Child RAG)
            r_doc = client.post(
                "/v1/collections/kb/documents",
                json={
                    "ids": ["d0", "d1", "d2"],
                    "documents": [
                        "Bölüm 0: İstanbul kurulum girişi",
                        "Bölüm 1: Sunucu hatası ERR-7711 çözüm adımları",
                        "Bölüm 2: Yedekleme doğrulama özeti",
                    ],
                    "embeddings": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.3, 0.95, 0.0, 0.0],
                        [0.0, 0.1, 0.99, 0.0],
                    ],
                    "doc_ids": ["manual_1", "manual_1", "manual_1"],
                    "metadatas": [{"tag": "ops", "yr": 2026}] * 3,
                    "sparse_vectors": [{"kurulum": 1.0}, {"err7711": 4.0}, {"yedek": 1.0}],
                },
            )
            self.assertEqual(r_doc.status_code, 200)
            self.assertEqual(r_doc.json()["added"], 3)

            # Dense, Batch, Sparse, Hybrid, Batch-Hybrid, Retrieve-Context, Explain
            r_s = client.post("/v1/collections/kb/search", json={"query_vector": [1, 0, 0, 0], "limit": 2})
            self.assertEqual(r_s.status_code, 200)
            self.assertEqual(r_s.json()["results"][0]["id"], "d0")

            r_bs = client.post(
                "/v1/collections/kb/batch_search",
                json={"query_vectors": [[1, 0, 0, 0], [0, 0, 1, 0]], "limit": 1},
            )
            self.assertEqual(r_bs.status_code, 200)
            self.assertEqual([row[0]["id"] for row in r_bs.json()["results"]], ["d0", "d2"])

            r_sp = client.post("/v1/collections/kb/sparse_search", json={"query_text": "ERR-7711", "limit": 1})
            self.assertEqual(r_sp.json()["results"][0]["id"], "d1")

            r_hyb = client.post(
                "/v1/collections/kb/hybrid_search",
                json={
                    "query_vector": [0.7, 0.7, 0, 0],
                    "query_text": "ERR-7711",
                    "query_sparse": {"err7711": 2.0},
                    "limit": 2,
                    "fusion": "rrf",
                },
            )
            self.assertEqual(r_hyb.status_code, 200)
            self.assertEqual(r_hyb.json()["results"][0]["id"], "d1")

            r_bhyb = client.post(
                "/v1/collections/kb/batch_hybrid_search",
                json={"queries": [{"query_text": "ERR-7711"}, {"query_text": "İstanbul"}], "limit": 1},
            )
            self.assertEqual(r_bhyb.status_code, 200)
            self.assertEqual(r_bhyb.json()["results"][0][0]["id"], "d1")

            r_ctx = client.post(
                "/v1/collections/kb/retrieve_context",
                json={"query_vector": [0.3, 0.95, 0, 0], "limit": 1, "window": 1},
            )
            self.assertEqual(r_ctx.status_code, 200)
            self.assertEqual(len(r_ctx.json()["results"][0]["neighbors"]), 3)

            r_exp = client.post("/v1/collections/kb/explain", json={"n_results": 2, "where": {"tag": "ops"}})
            self.assertEqual(r_exp.status_code, 200)
            self.assertIn("execution_path", r_exp.json())

            # Upsert + Get + Delete + Compact + Checkpoint + Snapshot
            self.assertEqual(
                client.post(
                    "/v1/collections/kb/upsert",
                    json={"ids": ["d2"], "vectors": [[0, 0, 0, 1]], "documents": ["Güncellenmiş bölüm 2"]},
                ).status_code,
                200,
            )
            self.assertEqual(
                client.post("/v1/collections/kb/get", json={"ids": ["d2"]}).json()["documents"][0],
                "Güncellenmiş bölüm 2",
            )
            self.assertEqual(client.post("/v1/collections/kb/delete", json={"ids": ["d0"]}).json()["deleted"], 1)
            self.assertEqual(client.post("/v1/collections/kb/compact").json()["reclaimed_slots"], 1)
            self.assertEqual(client.post("/v1/collections/kb/checkpoint").status_code, 200)

            r_sn = client.post("/v1/admin/snapshot", json={"snapshot_name": "snap_v1"})
            self.assertEqual(r_sn.status_code, 200)
            self.assertGreaterEqual(r_sn.json()["files"], 2)

            # Snapshot path-traversal engellenmeli!
            self.assertEqual(
                client.post("/v1/admin/snapshot", json={"snapshot_name": "../escape"}).status_code, 400
            )
            self.assertEqual(client.delete("/v1/collections/kb").status_code, 200)

    def test_security_hashed_api_keys_scopes_and_tenant_isolation(self):
        admin_key = "mvdb_admin_secret_999"
        admin_hashed = hash_api_key(admin_key)
        self.assertTrue(admin_hashed.startswith("sha256:"))

        spec = (
            f"{admin_hashed}:admin:read,write,admin;"
            "mvdb_ro_key_123:reader:read;"
            "mvdb_acme_key_1:acme:read,write:acme_corp;"
            "mvdb_beta_key_2:beta:read,write:beta_ltd"
        )
        reg = APIKeyRegistry.from_env_spec(spec, require_auth=True)
        cfg = ServerConfig(data_dir=None, api_key_registry=reg, require_auth=True)
        app = create_app(cfg)

        h_admin = {"Authorization": f"Bearer {admin_key}"}
        h_ro = {"X-API-Key": "mvdb_ro_key_123"}
        h_acme = {"Authorization": "Bearer mvdb_acme_key_1"}
        h_beta = {"Authorization": "Bearer mvdb_beta_key_2"}

        with TestClient(app) as client:
            # 1. Eksik veya geçersiz anahtar -> 401
            self.assertEqual(client.get("/v1/collections").status_code, 401)
            self.assertEqual(client.get("/v1/collections", headers={"X-API-Key": "wrong"}).status_code, 401)
            # /health her zaman 200
            self.assertEqual(client.get("/health").status_code, 200)

            # 2. Admin paylaşımlı koleksiyon oluşturur
            self.assertEqual(
                client.post("/v1/collections", json={"name": "saas", "dimension": 3}, headers=h_admin).status_code,
                201,
            )

            # 3. Read-only anahtar yazamaz -> 403
            self.assertEqual(
                client.post("/v1/collections/saas/add", json={"ids": ["x"], "vectors": [[1, 0, 0]]}, headers=h_ro).status_code,
                403,
            )
            self.assertEqual(client.get("/v1/collections", headers=h_ro).status_code, 200)

            # 4. ACME ve Beta kendi verilerini ekler
            self.assertEqual(
                client.post(
                    "/v1/collections/saas/add",
                    json={"ids": ["a1"], "vectors": [[1, 0, 0]], "documents": ["ACME gizli bilanço"]},
                    headers=h_acme,
                ).status_code,
                200,
            )
            self.assertEqual(
                client.post(
                    "/v1/collections/saas/add",
                    json={"ids": ["b1"], "vectors": [[1, 0, 0]], "documents": ["Beta gizli plan"]},
                    headers=h_beta,
                ).status_code,
                200,
            )

            # 5. Çapraz kiracı saldırısı: Beta, _tenant_id filtresiyle bile ACME'yi göremez, silemez, üzerine yazamaz!
            r_search_b = client.post(
                "/v1/collections/saas/search",
                json={"query_vector": [1, 0, 0], "limit": 10, "filter": {"_tenant_id": "acme_corp"}},
                headers=h_beta,
            )
            self.assertEqual(r_search_b.status_code, 200)
            self.assertEqual(r_search_b.json()["results"], [])

            r_hyb_b = client.post(
                "/v1/collections/saas/hybrid_search",
                json={"query_vector": [1, 0, 0], "query_text": "ACME bilanço", "limit": 10},
                headers=h_beta,
            )
            self.assertNotIn("a1", [x["id"] for x in r_hyb_b.json()["results"]])

            self.assertEqual(
                client.post("/v1/collections/saas/delete", json={"ids": ["a1"]}, headers=h_beta).json()["deleted"],
                0,
            )
            self.assertEqual(
                client.post(
                    "/v1/collections/saas/upsert",
                    json={"ids": ["a1"], "vectors": [[0, 1, 0]], "documents": ["HACKED"]},
                    headers=h_beta,
                ).status_code,
                403,
            )
            # Kiracı anahtarı koleksiyon silemez -> 403
            self.assertEqual(client.delete("/v1/collections/saas", headers=h_beta).status_code, 403)

    def test_rate_limiting_validation_batch_limit_and_log_masking(self):
        # Log maskeleme testi
        self.assertEqual(mask_secret("mvdb_secret_123456"), "mvdb****3456")
        flt = SanitizingLogFilter()
        rec = logging.LogRecord("test", logging.INFO, "", 0, "Auth failed for Bearer mvdb_secret_abc", (), None)
        flt.filter(rec)
        self.assertNotIn("mvdb_secret_abc", rec.msg)
        self.assertIn("[REDACTED]", rec.msg)

        cfg = ServerConfig(
            data_dir=None,
            rate_limit_rps=1.0,
            rate_limit_burst=3,
            max_batch_size=2,
        )
        app = create_app(cfg)
        with TestClient(app) as client:
            # 1. NaN vektör 400 veya 422 ile reddedilmeli
            client.post("/v1/collections", json={"name": "rl", "dimension": 2})
            r_nan = client.post(
                "/v1/collections/rl/add",
                content='{"ids":["1"],"vectors":[[1.0, NaN]]}',
                headers={"Content-Type": "application/json"},
            )
            self.assertIn(r_nan.status_code, (400, 422))

            # 2. Batch limit aşımı -> 413
            r_big = client.post(
                "/v1/collections/rl/add",
                json={"ids": ["1", "2", "3"], "vectors": [[1, 0], [0, 1], [1, 1]]},
            )
            self.assertEqual(r_big.status_code, 413)

            # 3. Rate limit (burst=3 tükendiğinde 429 + Retry-After!)
            statuses = [client.get("/v1/collections").status_code for _ in range(5)]
            self.assertIn(429, statuses)
            self.assertIn("mvdb_rate_limited_total", client.get("/metrics").text)

    def test_live_uvicorn_server_with_remote_sdk(self):
        # Boş localhost portu bul ve canlı uvicorn sunucusu başlat
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        reg = APIKeyRegistry(require_auth=True)
        reg.add_key("mvdb_live_test_key", key_id="sdk-admin", scopes={"read", "write", "admin"})
        cfg = ServerConfig(
            data_dir=os.path.join(self.temp_dir, "live_db"),
            snapshot_dir=os.path.join(self.temp_dir, "live_snaps"),
            api_key_registry=reg,
            require_auth=True,
        )
        app = create_app(cfg)
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        th = threading.Thread(target=server.run, daemon=True)
        th.start()
        for _ in range(200):
            if server.started:
                break
            time.sleep(0.02)

        # Sunucu hazır -> RemoteMicroVectorDB SDK ile test et
        url = f"http://127.0.0.1:{port}"
        with RemoteMicroVectorDB(url, api_key="mvdb_live_test_key") as sdk:
            for _ in range(50):
                try:
                    if sdk.health()["status"] == "ok":
                        break
                except Exception:
                    time.sleep(0.05)

            self.assertEqual(sdk.ready()["status"], "ready")
            coll = sdk.create_collection("docs", dimension=4, metric="cosine")
            with self.assertRaises(CollectionAlreadyExistsError):
                sdk.create_collection("docs", dimension=4)
            with self.assertRaises(CollectionNotFoundError):
                sdk.get_collection("does_not_exist")

            coll.add_documents(
                ids=["1", "2"],
                documents=["İstanbul veri merkezi", "Hata kodu ERR-5001 çözümü"],
                embeddings=[[1, 0, 0, 0], [0.2, 0.98, 0, 0]],
                doc_ids=["m1", "m1"],
            )
            self.assertEqual(coll.count(), 2)
            self.assertEqual(coll.search([1, 0, 0, 0], limit=1)[0].id, "1")
            self.assertEqual(
                coll.hybrid_search(query_vector=[0.6, 0.8, 0, 0], query_text="ERR-5001", limit=1)[0].id, "2"
            )
            self.assertEqual(len(coll.batch_search([[1, 0, 0, 0], [0, 1, 0, 0]], limit=1)), 2)
            self.assertEqual(len(coll.retrieve_with_context(query_vector=[1, 0, 0, 0], limit=1, window=1)[0]["neighbors"]), 2)
            self.assertIn("execution_path", coll.explain(n_results=2))
            self.assertEqual(coll.delete(ids=["1"]), 1)
            self.assertEqual(coll.compact()["reclaimed_slots"], 1)
            self.assertGreaterEqual(sdk.create_snapshot("sdk_snap")["files"], 2)
            self.assertIn("mvdb_vectors_total", sdk.metrics())
            sdk.delete_collection("docs")

        # Yanlış anahtarla RemoteAPIError(401) fırlatmalı
        with RemoteMicroVectorDB(url, api_key="bad_key") as bad_sdk:
            with self.assertRaises(RemoteAPIError) as ctx:
                bad_sdk.list_collections()
            self.assertEqual(ctx.exception.status_code, 401)

        server.should_exit = True
        th.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
