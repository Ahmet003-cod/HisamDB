"""
benchmarks/run_benchmarks_v07.py
--------------------------------
MicroVectorDB v0.7.0 (Faz 5 — FastAPI + Uvicorn Üretim REST Sunucusu & RemoteMicroVectorDB SDK)
gerçek uçtan uca HTTP loopback bilimsel benchmark paketi:
1. Tek istemci HTTP REST Dense search() (50K x 128D HNSW, Bearer Auth etkin)
2. Tek istemci HTTP REST batch_search(64) (50K x 128D HNSW -> toplam sorgu/sn)
3. Tek istemci HTTP REST hybrid_search() (10K doküman, Dense HNSW + BM25 + RRF)
4. Tek istemci HTTP REST delete() (10K kalıcı Tombstone + WAL)
5. Çoklu eşzamanlı HTTP istemcisi (4 ve 8 paralel RemoteMicroVectorDB istemci iş parçacığı)
"""

import csv
import json
import os
import platform
import shutil
import socket
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List
import numpy as np
import psutil
import uvicorn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hisamdb
from hisamdb import RemoteMicroVectorDB, is_hnsw_available
from hisamdb.api import APIKeyRegistry, ServerConfig, create_app


def make_clustered_data(n: int, d: int, n_topics: int = 64, n_queries: int = 256, seed: int = 42):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_topics, d)).astype(np.float32) * 2.5
    labels = rng.integers(0, n_topics, size=n)
    vecs = (centers[labels] + rng.standard_normal((n, d)).astype(np.float32) * 0.7).astype(np.float32)
    q_labels = rng.integers(0, n_topics, size=n_queries)
    queries = (centers[q_labels] + rng.standard_normal((n_queries, d)).astype(np.float32) * 0.7).astype(np.float32)
    return vecs, queries


def main() -> None:
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="mvdb_bench_v07_")

    meta = {
        "microvectordb_version": hisamdb.__version__,
        "python": platform.python_version(),
        "os": platform.platform(),
        "cpu_logical_cores": os.cpu_count(),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "numpy": np.__version__,
        "hnswlib": is_hnsw_available(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print("Benchmark v0.7.0 Metadata:", json.dumps(meta, indent=2))

    # Boş yerel port ayır
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    api_key = "mvdb_bench_secret_key_07"
    reg = APIKeyRegistry(require_auth=True)
    reg.add_key(api_key, key_id="bench-admin", scopes={"read", "write", "admin"})
    cfg = ServerConfig(
        data_dir=os.path.join(tmp_dir, "srv_data"),
        snapshot_dir=os.path.join(tmp_dir, "srv_snaps"),
        max_workers=8,
        api_key_registry=reg,
        require_auth=True,
        max_batch_size=50_000,
    )
    app = create_app(cfg)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.02)

    rows: List[Dict[str, Any]] = []
    base_url = f"http://127.0.0.1:{port}"

    try:
        with RemoteMicroVectorDB(base_url, api_key=api_key, timeout=60.0) as sdk:
            for _ in range(80):
                try:
                    if sdk.health()["status"] == "ok":
                        break
                except Exception:
                    time.sleep(0.05)

            n_vecs, dim = 50_000, 128
            vecs, queries = make_clustered_data(n_vecs, dim, n_topics=64, n_queries=256, seed=42)
            idx_type = "hnsw" if is_hnsw_available() else "ivf"

            # 1. 50K HNSW koleksiyonunu 5K bloklarla REST üzerinden yükle
            print(f"\n=== 1. REST SUNUCU VERİ YÜKLEME (50K x 128D {idx_type.upper()} + 10K Hibrit) ===")
            c_dense = sdk.create_collection(
                "dense_50k", dimension=dim, metric="cosine", index_type=idx_type,
                index_params={"M": 16, "ef_construction": 128, "ef_search": 64},
            )
            t_ing = time.perf_counter()
            for start in range(0, n_vecs, 5000):
                end = min(start + 5000, n_vecs)
                c_dense.add(ids=[f"v{i}" for i in range(start, end)], vectors=vecs[start:end])
            ing_s = time.perf_counter() - t_ing
            print(f"  50K x 128D HTTP REST yükleme: {ing_s:.2f}s ({n_vecs / ing_s:,.0f} vektör/sn)")

            # 10K Hibrit koleksiyonu
            n_docs = 10_000
            topics = [
                "veritabanı indeksleme hnsw performans",
                "kubernetes pod güvenlik sertifika",
                "yapay zeka rag hibrit arama",
                "dağıtık wal çökme kurtarma snapshot",
            ]
            c_hyb = sdk.create_collection(
                "hybrid_10k", dimension=dim, metric="cosine", index_type=idx_type,
                index_params={"M": 16, "ef_construction": 128, "ef_search": 64},
            )
            for start in range(0, n_docs, 2500):
                end = min(start + 2500, n_docs)
                c_hyb.add(
                    ids=[f"d{i}" for i in range(start, end)],
                    vectors=vecs[start:end],
                    documents=[
                        f"Belge {i}: {topics[i % len(topics)]} istanbul kod ERR-{1000 + i}"
                        for i in range(start, end)
                    ],
                )

            # Warmup
            c_dense.search(queries[0], limit=10)
            c_hyb.hybrid_search(query_vector=queries[0], query_text="istanbul ERR-1042", limit=10)

            print("\n=== 2. UÇTAN UCA HTTP REST PERFORMANSI (Bearer Auth + Keep-Alive SDK) ===")

            # 2A. Tek istemci HTTP Dense search()
            lats = []
            for i in range(200):
                ts = time.perf_counter()
                res = c_dense.search(queries[i], limit=10)
                lats.append((time.perf_counter() - ts) * 1000)
                assert len(res) == 10
            arr = np.array(lats)
            rows.append({
                "scenario": f"HTTP POST /search (50K {idx_type.upper()}, 1 client)",
                "clients": 1,
                "p50_ms": round(float(np.percentile(arr, 50)), 3),
                "p95_ms": round(float(np.percentile(arr, 95)), 3),
                "qps": round(1000.0 / float(arr.mean()), 1),
            })

            # 2B. Tek istemci HTTP batch_search(64)
            blats = []
            for b in range(10):
                sub = queries[(b * 16) % 192 : ((b * 16) % 192) + 64]
                ts = time.perf_counter()
                bres = c_dense.batch_search(sub, limit=10, max_workers=8)
                blats.append((time.perf_counter() - ts) * 1000)
                assert len(bres) == 64
            barr = np.array(blats)
            rows.append({
                "scenario": f"HTTP POST /batch_search(64) (50K {idx_type.upper()})",
                "clients": 1,
                "p50_ms": round(float(np.percentile(barr, 50)), 3),
                "p95_ms": round(float(np.percentile(barr, 95)), 3),
                "qps": round((64 * 1000.0) / float(barr.mean()), 1),
            })

            # 2C. Tek istemci HTTP hybrid_search() (10K Dense + BM25 + RRF)
            hlats = []
            for i in range(100):
                ts = time.perf_counter()
                hres = c_hyb.hybrid_search(
                    query_vector=queries[i],
                    query_text=f"{topics[i % len(topics)].split()[0]} ERR-{1000 + i * 13}",
                    limit=10,
                    fusion="rrf",
                )
                hlats.append((time.perf_counter() - ts) * 1000)
                assert len(hres) > 0
            harr = np.array(hlats)
            rows.append({
                "scenario": "HTTP POST /hybrid_search (10K HNSW+BM25+RRF, 1 client)",
                "clients": 1,
                "p50_ms": round(float(np.percentile(harr, 50)), 3),
                "p95_ms": round(float(np.percentile(harr, 95)), 3),
                "qps": round(1000.0 / float(harr.mean()), 1),
            })

            # 2D. Tek istemci HTTP delete() (Tombstone + WAL)
            dlats = []
            for i in range(25):
                ts = time.perf_counter()
                cnt = c_hyb.delete(ids=[f"d{i * 10}"])
                dlats.append((time.perf_counter() - ts) * 1000)
                assert cnt == 1
            darr = np.array(dlats)
            rows.append({
                "scenario": "HTTP POST /delete (10K Kalıcı Tombstone+WAL)",
                "clients": 1,
                "p50_ms": round(float(np.percentile(darr, 50)), 3),
                "p95_ms": round(float(np.percentile(darr, 95)), 3),
                "qps": round(1000.0 / float(darr.mean()), 1),
            })

            # 2E. Çoklu eşzamanlı HTTP istemcisi (4 ve 8 paralel Keep-Alive istemci)
            for n_clients in (4, 8):
                per_client_q = 60
                all_lats: List[float] = []
                lock = threading.Lock()

                def _client_worker(cid: int):
                    local_lats = []
                    with RemoteMicroVectorDB(base_url, api_key=api_key) as c_sdk:
                        col = c_sdk.get_collection("dense_50k")
                        for j in range(per_client_q):
                            ts = time.perf_counter()
                            col.search(queries[(cid * 31 + j) % 256], limit=10)
                            local_lats.append((time.perf_counter() - ts) * 1000)
                    with lock:
                        all_lats.extend(local_lats)

                t0 = time.perf_counter()
                with ThreadPoolExecutor(max_workers=n_clients) as ex:
                    futs = [ex.submit(_client_worker, c_id) for c_id in range(n_clients)]
                    for f in futs:
                        f.result()
                wall_s = time.perf_counter() - t0
                carr = np.array(all_lats)
                total_q = n_clients * per_client_q
                rows.append({
                    "scenario": f"HTTP POST /search ({n_clients} paralel istemci, 50K {idx_type.upper()})",
                    "clients": n_clients,
                    "p50_ms": round(float(np.percentile(carr, 50)), 3),
                    "p95_ms": round(float(np.percentile(carr, 95)), 3),
                    "qps": round(total_q / wall_s, 1),
                })

            for r in rows:
                print(
                    f"  {r['scenario']:<56} | P50={r['p50_ms']:>6.3f}ms P95={r['p95_ms']:>6.3f}ms | QPS={r['qps']:>8.1f}"
                )

            # Prometheus /metrics doğrulama örneği
            prom_txt = sdk.metrics()
            assert "mvdb_http_requests_total" in prom_txt

        json_path = os.path.join(out_dir, "benchmark_v0.7.0.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"metadata": meta, "http_benchmarks": rows}, f, indent=2, ensure_ascii=False)

        csv_path = os.path.join(out_dir, "benchmark_v0.7.0.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["scenario", "clients", "p50_ms", "p95_ms", "qps"])
            w.writeheader()
            for r in rows:
                w.writerow(r)

        print(f"\nSonuçlar kaydedildi:\n  - {json_path}\n  - {csv_path}")
    finally:
        server.should_exit = True
        th.join(timeout=5)
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
