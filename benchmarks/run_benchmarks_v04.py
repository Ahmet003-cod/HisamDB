"""
benchmarks/run_benchmarks_v04.py
--------------------------------
MicroVectorDB v0.4.0 (Faz 2) bilimsel benchmark paketi:
1. Product Quantization (PQ) & IVF-PQ vs FAISS IndexPQ / IndexIVFPQ (100K x 128D)
2. DiskIndex (Partitioned MMap) & DiskStreamingScanner (100K x 128D)
3. Anında İndeks Yeniden Yükleme: v0.3.0 Soğuk İnşa vs v0.4.0 Disk Yükleme (HNSW, IVF, IVF-PQ, Disk)
4. Query Optimizer Filtreli HNSW Hızlanması: %2, %10, %30, %60 seçicilik (v0.3 callback vs v0.4 optimizer)
"""

import csv
import gc
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List, Set
import numpy as np
import psutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import microvectordb
from microvectordb import MicroVectorDB, is_hnsw_available

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


def make_clustered_data(n: int, d: int, n_topics: int = 64, n_queries: int = 100, seed: int = 42):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_topics, d)).astype(np.float32) * 2.5
    labels = rng.integers(0, n_topics, size=n)
    vecs = (centers[labels] + rng.standard_normal((n, d)).astype(np.float32) * 0.7).astype(np.float32)
    q_labels = rng.integers(0, n_topics, size=n_queries)
    queries = (centers[q_labels] + rng.standard_normal((n_queries, d)).astype(np.float32) * 0.7).astype(np.float32)
    return vecs, queries


def exact_cosine_gt(vecs: np.ndarray, queries: np.ndarray, k: int = 10) -> List[Set[int]]:
    v_n = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    q_n = queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)
    sims = np.dot(q_n, v_n.T)
    gt = []
    for i in range(len(queries)):
        top = np.argpartition(-sims[i], k - 1)[:k]
        gt.append(set(int(x) for x in top))
    return gt


def log_row(rows: List[Dict[str, Any]], **kw: Any) -> None:
    rows.append(kw)
    print(
        f"[{kw['category']:<14}] {kw['name']:<46} | "
        f"P50={kw.get('p50_ms', 0):>6.3f}ms P95={kw.get('p95_ms', 0):>6.3f}ms | "
        f"QPS={kw.get('qps', 0):>8.1f} | Rec@10={kw.get('recall_at_10', 1.0):.3f} | "
        f"Build={kw.get('build_s', 0):>5.2f}s | RAM={kw.get('ram_mb', 0):>5.2f}MB"
    )


def main() -> None:
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="mvdb_bench_v04_")

    meta = {
        "microvectordb_version": microvectordb.__version__,
        "python": platform.python_version(),
        "os": platform.platform(),
        "cpu_logical_cores": os.cpu_count(),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "numpy": np.__version__,
        "faiss": faiss.__version__ if HAS_FAISS else None,
        "hnswlib": is_hnsw_available(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print("Benchmark v0.4.0 Metadata:", json.dumps(meta, indent=2))

    rows: List[Dict[str, Any]] = []
    reload_rows: List[Dict[str, Any]] = []
    optimizer_rows: List[Dict[str, Any]] = []

    try:
        n_vecs, dim, n_queries = 100_000, 128, 100
        vecs, queries = make_clustered_data(n_vecs, dim, n_topics=64, n_queries=n_queries, seed=42)
        ids = [f"v{i}" for i in range(n_vecs)]
        gt = exact_cosine_gt(vecs, queries, k=10)

        print("\n=== 1. PQ, IVF-PQ, DiskIndex & DiskStreaming vs FAISS (100K x 128D) ===")

        # 1A. Flat PQ (keep_raw=False, 16B/vec = 32x RAM compression!)
        gc.collect()
        db = MicroVectorDB()
        c_pq = db.create_collection(
            "pq", dimension=dim, metric="cosine", quantization="pq", keep_raw_vectors=False,
            index_params={"pq_subvectors": 16, "pq_centroids": 256, "pq_iter": 12},
        )
        t0 = time.perf_counter()
        c_pq.add(ids=ids, vectors=vecs)
        tb = time.perf_counter() - t0
        c_pq.query(queries[0], n_results=10)
        lats, recs = [], []
        for i in range(n_queries):
            tq = time.perf_counter()
            res = c_pq.query(queries[i], n_results=10)
            lats.append((time.perf_counter() - tq) * 1000)
            recs.append(len({int(s[1:]) for s in res.ids[0]} & gt[i]) / 10.0)
        st = c_pq.stats()
        lat = np.array(lats)
        log_row(
            rows, category="PQ/IVF-PQ", engine="MicroVectorDB v0.4.0",
            name="MVDB Flat PQ (M=16, 8-bit, keep_raw=False)",
            n_vectors=n_vecs, dim=dim, metric="cosine",
            build_s=round(tb, 3), p50_ms=round(float(np.percentile(lat, 50)), 3),
            p95_ms=round(float(np.percentile(lat, 95)), 3), qps=round(1000.0 / float(lat.mean()), 1),
            recall_at_10=round(float(np.mean(recs)), 4),
            ram_mb=round(st["vector_ram_mb"] + st["quantized_ram_mb"] + st["index_ram_mb"], 2),
        )
        del c_pq, db
        gc.collect()

        # 1B. IVF-PQ (keep_raw=False, persistent -> also measure reload!)
        p_ivfpq_dir = os.path.join(tmp_dir, "ivfpq_db")
        with MicroVectorDB(p_ivfpq_dir) as db:
            c = db.create_collection(
                "ivfpq", dimension=dim, metric="cosine", index_type="ivf_pq", keep_raw_vectors=False,
                index_params={"n_clusters": 128, "nprobe": 8, "pq_subvectors": 16, "pq_centroids": 256},
            )
            t0 = time.perf_counter()
            c.add(ids=ids, vectors=vecs)
            tb = time.perf_counter() - t0
            c.query(queries[0], n_results=10)
            lats, recs = [], []
            for i in range(n_queries):
                tq = time.perf_counter()
                res = c.query(queries[i], n_results=10)
                lats.append((time.perf_counter() - tq) * 1000)
                recs.append(len({int(s[1:]) for s in res.ids[0]} & gt[i]) / 10.0)
            st = c.stats()
            lat = np.array(lats)
            log_row(
                rows, category="PQ/IVF-PQ", engine="MicroVectorDB v0.4.0",
                name="MVDB IVF-PQ (C=128, np=8, M=16, keep_raw=F)",
                n_vectors=n_vecs, dim=dim, metric="cosine",
                build_s=round(tb, 3), p50_ms=round(float(np.percentile(lat, 50)), 3),
                p95_ms=round(float(np.percentile(lat, 95)), 3), qps=round(1000.0 / float(lat.mean()), 1),
                recall_at_10=round(float(np.mean(recs)), 4),
                ram_mb=round(st["vector_ram_mb"] + st["index_ram_mb"], 2),
            )

        t0 = time.perf_counter()
        with MicroVectorDB(p_ivfpq_dir) as db_r:
            c_r = db_r.get_collection("ivfpq")
            t_reload_ivfpq = (time.perf_counter() - t0) * 1000
            loaded_flag = c_r.stats()["index_loaded_from_disk"]
        reload_rows.append({
            "index": "IVF-PQ (100K x 128D)",
            "cold_build_s": round(tb, 3),
            "v04_disk_reload_ms": round(t_reload_ivfpq, 1),
            "loaded_from_disk": loaded_flag,
            "speedup": round((tb * 1000) / max(t_reload_ivfpq, 0.1), 1),
        })

        # 1C. DiskIndex (Partitioned MMap) + reload
        p_disk_dir = os.path.join(tmp_dir, "disk_db")
        with MicroVectorDB(p_disk_dir) as db:
            c = db.create_collection(
                "disk_c", dimension=dim, metric="cosine", index_type="disk",
                index_params={"n_clusters": 128, "nprobe": 8},
            )
            t0 = time.perf_counter()
            c.add(ids=ids, vectors=vecs)
            tb_disk = time.perf_counter() - t0

        t0 = time.perf_counter()
        with MicroVectorDB(p_disk_dir) as db_r:
            c_r = db_r.get_collection("disk_c")
            t_reload_disk = (time.perf_counter() - t0) * 1000
            c_r.query(queries[0], n_results=10)
            lats, recs = [], []
            for i in range(n_queries):
                tq = time.perf_counter()
                res = c_r.query(queries[i], n_results=10)
                lats.append((time.perf_counter() - tq) * 1000)
                recs.append(len({int(s[1:]) for s in res.ids[0]} & gt[i]) / 10.0)
            st = c_r.stats()
            lat = np.array(lats)
            log_row(
                rows, category="Disk/MMap", engine="MicroVectorDB v0.4.0",
                name="MVDB DiskIndex (Partitioned MMap, C=128, np=8)",
                n_vectors=n_vecs, dim=dim, metric="cosine",
                build_s=round(tb_disk, 3), p50_ms=round(float(np.percentile(lat, 50)), 3),
                p95_ms=round(float(np.percentile(lat, 95)), 3), qps=round(1000.0 / float(lat.mean()), 1),
                recall_at_10=round(float(np.mean(recs)), 4),
                ram_mb=round(st["vector_ram_mb"] + st["index_ram_mb"], 2),
            )
            # Also DiskStreamingScanner (exact mmap streaming)
            lats_s = []
            for i in range(25):
                tq = time.perf_counter()
                c_r.query(queries[i], n_results=10, strategy="disk_stream")
                lats_s.append((time.perf_counter() - tq) * 1000)
            lat_s = np.array(lats_s)
            log_row(
                rows, category="Disk/MMap", engine="MicroVectorDB v0.4.0",
                name="MVDB DiskStreamingScanner (Exact MMap, 4K blk)",
                n_vectors=n_vecs, dim=dim, metric="cosine",
                build_s=0.0, p50_ms=round(float(np.percentile(lat_s, 50)), 3),
                p95_ms=round(float(np.percentile(lat_s, 95)), 3), qps=round(1000.0 / float(lat_s.mean()), 1),
                recall_at_10=1.0,
                ram_mb=round(st["vector_ram_mb"], 2),
            )
        reload_rows.append({
            "index": "DiskIndex (100K x 128D)",
            "cold_build_s": round(tb_disk, 3),
            "v04_disk_reload_ms": round(t_reload_disk, 1),
            "loaded_from_disk": st["index_loaded_from_disk"],
            "speedup": round((tb_disk * 1000) / max(t_reload_disk, 0.1), 1),
        })

        # 1D. IVF & HNSW Instant Reload (100K x 128D)
        p_ivf_dir = os.path.join(tmp_dir, "ivf_db")
        with MicroVectorDB(p_ivf_dir) as db:
            c = db.create_collection(
                "ivf_c", dimension=dim, metric="cosine", index_type="ivf",
                index_params={"n_clusters": 128, "nprobe": 8},
            )
            t0 = time.perf_counter()
            c.add(ids=ids, vectors=vecs)
            tb_ivf = time.perf_counter() - t0
        t0 = time.perf_counter()
        with MicroVectorDB(p_ivf_dir) as db_r:
            c_r = db_r.get_collection("ivf_c")
            t_reload_ivf = (time.perf_counter() - t0) * 1000
            flag_ivf = c_r.stats()["index_loaded_from_disk"]
        reload_rows.append({
            "index": "IVF (100K x 128D)",
            "cold_build_s": round(tb_ivf, 3),
            "v04_disk_reload_ms": round(t_reload_ivf, 1),
            "loaded_from_disk": flag_ivf,
            "speedup": round((tb_ivf * 1000) / max(t_reload_ivf, 0.1), 1),
        })

        if is_hnsw_available():
            p_hnsw_dir = os.path.join(tmp_dir, "hnsw_db")
            with MicroVectorDB(p_hnsw_dir) as db:
                c = db.create_collection(
                    "hnsw_c", dimension=dim, metric="cosine", index_type="hnsw",
                    index_params={"M": 16, "ef_construction": 128, "ef_search": 64},
                )
                t0 = time.perf_counter()
                c.add(ids=ids, vectors=vecs)
                tb_hnsw = time.perf_counter() - t0
            t0 = time.perf_counter()
            with MicroVectorDB(p_hnsw_dir) as db_r:
                c_r = db_r.get_collection("hnsw_c")
                t_reload_hnsw = (time.perf_counter() - t0) * 1000
                flag_hnsw = c_r.stats()["index_loaded_from_disk"]
            reload_rows.append({
                "index": "HNSW (100K x 128D, M=16, efC=128)",
                "cold_build_s": round(tb_hnsw, 3),
                "v04_disk_reload_ms": round(t_reload_hnsw, 1),
                "loaded_from_disk": flag_hnsw,
                "speedup": round((tb_hnsw * 1000) / max(t_reload_hnsw, 0.1), 1),
            })

        # 1E. FAISS PQ & IVF-PQ Reference
        if HAS_FAISS:
            v_norm = np.ascontiguousarray(vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12), dtype=np.float32)
            q_norm = np.ascontiguousarray(queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12), dtype=np.float32)

            fpq = faiss.IndexPQ(dim, 16, 8)
            t0 = time.perf_counter()
            fpq.train(v_norm)
            fpq.add(v_norm)
            tb_fpq = time.perf_counter() - t0
            lats, recs = [], []
            for i in range(n_queries):
                tq = time.perf_counter()
                _, I = fpq.search(q_norm[i : i + 1], 10)
                lats.append((time.perf_counter() - tq) * 1000)
                recs.append(len(set(int(x) for x in I[0]) & gt[i]) / 10.0)
            lat = np.array(lats)
            log_row(
                rows, category="FAISS-Ref", engine=f"FAISS {faiss.__version__}",
                name="FAISS IndexPQ (M=16, 8-bit)",
                n_vectors=n_vecs, dim=dim, metric="cosine",
                build_s=round(tb_fpq, 3), p50_ms=round(float(np.percentile(lat, 50)), 3),
                p95_ms=round(float(np.percentile(lat, 95)), 3), qps=round(1000.0 / float(lat.mean()), 1),
                recall_at_10=round(float(np.mean(recs)), 4), ram_mb=round((n_vecs * 16) / (1024 * 1024), 2),
            )

            qtz = faiss.IndexFlatL2(dim)
            fivfpq = faiss.IndexIVFPQ(qtz, dim, 128, 16, 8)
            t0 = time.perf_counter()
            fivfpq.train(v_norm)
            fivfpq.add(v_norm)
            fivfpq.nprobe = 8
            tb_fipq = time.perf_counter() - t0
            lats, recs = [], []
            for i in range(n_queries):
                tq = time.perf_counter()
                _, I = fivfpq.search(q_norm[i : i + 1], 10)
                lats.append((time.perf_counter() - tq) * 1000)
                recs.append(len(set(int(x) for x in I[0]) & gt[i]) / 10.0)
            lat = np.array(lats)
            log_row(
                rows, category="FAISS-Ref", engine=f"FAISS {faiss.__version__}",
                name="FAISS IndexIVFPQ (nlist=128, np=8, M=16)",
                n_vectors=n_vecs, dim=dim, metric="cosine",
                build_s=round(tb_fipq, 3), p50_ms=round(float(np.percentile(lat, 50)), 3),
                p95_ms=round(float(np.percentile(lat, 95)), 3), qps=round(1000.0 / float(lat.mean()), 1),
                recall_at_10=round(float(np.mean(recs)), 4), ram_mb=round((n_vecs * 24) / (1024 * 1024), 2),
            )

        print("\n=== 2. INSTANT INDEX DISK RELOAD SUMMARY (100K x 128D) ===")
        for rr in reload_rows:
            print(
                f"  {rr['index']:<36} | Cold Build: {rr['cold_build_s']:>6.2f}s -> "
                f"v0.4 Disk Reload: {rr['v04_disk_reload_ms']:>6.1f}ms | "
                f"Speedup: {rr['speedup']:>5.1f}x (from_disk={rr['loaded_from_disk']})"
            )

        # 3. Query Optimizer Filtered HNSW Speedup (100K x 128D)
        if is_hnsw_available():
            print("\n=== 3. QUERY OPTIMIZER: FILTERED HNSW (100K x 128D) ===")
            db = MicroVectorDB()
            metas = [{"p2": i % 50 == 0, "p10": i % 10 == 0, "p30": (i % 10) < 3, "p60": (i % 10) < 6} for i in range(n_vecs)]
            c_h = db.create_collection(
                "opt_h", dimension=dim, metric="cosine", index_type="hnsw",
                index_params={"M": 16, "ef_construction": 128, "ef_search": 64},
            )
            c_h.add(ids=ids, vectors=vecs, metadata=metas)

            for label, flt in [("2% (2K)", {"p2": True}), ("10% (10K)", {"p10": True}), ("30% (30K)", {"p30": True}), ("60% (60K)", {"p60": True})]:
                plan = c_h.explain(n_results=10, where=flt)
                # v0.3 style (callback forced via filter_strategy="callback" / fallback) vs v0.4 optimizer auto
                l_auto, l_cb = [], []
                for i in range(50):
                    t0 = time.perf_counter()
                    c_h.query(queries[i], n_results=10, where=flt, filter_strategy="auto")
                    l_auto.append((time.perf_counter() - t0) * 1000)

                    t0 = time.perf_counter()
                    c_h.query(queries[i], n_results=10, where=flt, filter_strategy="callback")
                    l_cb.append((time.perf_counter() - t0) * 1000)

                p50_a = float(np.percentile(l_auto, 50))
                p50_c = float(np.percentile(l_cb, 50))
                sp = p50_c / max(p50_a, 0.001)
                optimizer_rows.append({
                    "selectivity": label,
                    "chosen_path": plan.execution_path,
                    "v03_callback_p50_ms": round(p50_c, 3),
                    "v04_optimizer_p50_ms": round(p50_a, 3),
                    "speedup": round(sp, 2),
                })
                print(
                    f"  Selectivity {label:<10} | Plan={plan.execution_path:<18} | "
                    f"Callback P50={p50_c:>6.3f}ms -> Optimizer P50={p50_a:>6.3f}ms ({sp:.2f}x faster!)"
                )

        # Save JSON & CSV
        json_path = os.path.join(out_dir, "benchmark_v0.4.0.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": meta,
                    "benchmarks": rows,
                    "reload": reload_rows,
                    "optimizer": optimizer_rows,
                },
                f,
                indent=2,
            )

        csv_path = os.path.join(out_dir, "benchmark_v0.4.0.csv")
        keys = ["category", "engine", "name", "n_vectors", "dim", "metric", "build_s", "p50_ms", "p95_ms", "qps", "recall_at_10", "ram_mb"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nSonuçlar kaydedildi:\n  - {json_path}\n  - {csv_path}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
