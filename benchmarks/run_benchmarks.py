"""
benchmarks/run_benchmarks.py
----------------------------
MicroVectorDB v0.3.0 bilimsel ve tekrarlanabilir performans benchmark paketi.
- v0.2.0 denetim senaryolarıyla birebir aynı tohum/boyutlarda karşılaştırma
- Kümelemeli gerçekçi embedding dağılımında MicroVectorDB vs FAISS karşılaştırması
- Otomatik JSON ve CSV rapor çıktısı
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
from typing import Any, Dict, List, Optional, Set
import numpy as np
import psutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import microvectordb
from microvectordb import (
    EphemeralClient,
    IVFIndex,
    MicroVectorDB,
    PersistentClient,
    ScalarQuantizer,
    is_hnsw_available,
)

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

PROC = psutil.Process(os.getpid())


def rss_mb() -> float:
    return PROC.memory_info().rss / (1024 * 1024)


def make_clustered_data(n: int, d: int, n_topics: int = 64, n_queries: int = 100, seed: int = 42):
    """Gerçekçi semantik embedding dağılımı (n_topics kümeli + gürültü)."""
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


def record_row(results: List[Dict[str, Any]], **kwargs: Any) -> None:
    results.append(kwargs)
    print(
        f"[{kwargs['category']:<12}] {kwargs['name']:<42} | "
        f"N={kwargs['n_vectors']:>7,} D={kwargs['dim']:>4} | "
        f"P50={kwargs.get('p50_ms', 0):>7.3f}ms P95={kwargs.get('p95_ms', 0):>7.3f}ms | "
        f"QPS={kwargs.get('qps', 0):>8.1f} | "
        f"Rec@10={kwargs.get('recall_at_10', 1.0):.3f} | "
        f"Build/Add={kwargs.get('build_s', 0):.3f}s | "
        f"dRAM={kwargs.get('ram_mb', 0):+.1f}MB"
    )


def bench_flat_gaussian(results: List[Dict[str, Any]]) -> None:
    print("\n=== 1. FLAT SEARCH & FILTERING (Gaussian — v0.2.0 Baseline ile Birebir) ===")
    configs = [
        (10_000, 128, "cosine", None, 100, "Flat Cosine"),
        (50_000, 128, "cosine", None, 100, "Flat Cosine"),
        (100_000, 128, "cosine", None, 100, "Flat Cosine"),
        (100_000, 128, "cosine", {"cat": "c0"}, 100, "Flat Cosine + Filter (eq 10%)"),
        (100_000, 128, "cosine", {"val": {"$gte": 900}}, 100, "Flat Cosine + Filter (range 10%)"),
        (100_000, 128, "euclidean", None, 100, "Flat Euclidean (cached sq_norms)"),
        (100_000, 128, "dot_product", None, 100, "Flat Dot Product"),
        (50_000, 768, "cosine", None, 50, "Flat Cosine 768D"),
        (50_000, 768, "euclidean", None, 50, "Flat Euclidean 768D (cached sq_norms)"),
    ]

    for n_vecs, dim, metric, where, n_queries, label in configs:
        gc.collect()
        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((n_vecs, dim)).astype(np.float32)
        queries = rng.standard_normal((n_queries, dim)).astype(np.float32)
        ids = [f"v{i}" for i in range(n_vecs)]
        metas = [{"cat": f"c{i % 10}", "val": i % 1000} for i in range(n_vecs)]

        client = EphemeralClient()
        coll = client.create_collection("bench", dimension=dim, metric=metric)

        rss_before = rss_mb()
        t0 = time.perf_counter()
        coll.add(ids=ids, embeddings=vecs, metadatas=metas)
        t_add = time.perf_counter() - t0

        # Warmup
        t0 = time.perf_counter()
        coll.query(query_embeddings=queries[0], n_results=10, where=where)
        first_ms = (time.perf_counter() - t0) * 1000
        rss_after = rss_mb()

        lats = []
        for i in range(n_queries):
            tq = time.perf_counter()
            coll.query(query_embeddings=queries[i], n_results=10, where=where)
            lats.append((time.perf_counter() - tq) * 1000)

        lat = np.array(lats)
        record_row(
            results,
            category="Flat-Gauss",
            engine="MicroVectorDB v0.3.0",
            name=label,
            n_vectors=n_vecs,
            dim=dim,
            metric=metric,
            filter=str(where) if where else "none",
            build_s=round(t_add, 4),
            insert_vps=round(n_vecs / max(t_add, 1e-6), 1),
            first_query_ms=round(first_ms, 3),
            p50_ms=round(float(np.percentile(lat, 50)), 3),
            p95_ms=round(float(np.percentile(lat, 95)), 3),
            p99_ms=round(float(np.percentile(lat, 99)), 3),
            qps=round(1000.0 / float(lat.mean()), 1),
            recall_at_10=1.0,
            ram_mb=round(coll.stats()["vector_ram_mb"], 2),
            rss_delta_mb=round(rss_after - rss_before, 2),
        )
        del coll, client, vecs, queries, ids, metas
        gc.collect()


def bench_quantization_and_ann_vs_faiss(results: List[Dict[str, Any]]) -> None:
    print("\n=== 2. ANN & QUANTIZATION vs FAISS (100K x 128D Gerçekçi Kümelemeli Veri) ===")
    n_vecs, dim, n_queries = 100_000, 128, 100
    vecs, queries = make_clustered_data(n_vecs, dim, n_topics=64, n_queries=n_queries, seed=42)
    ids = [f"v{i}" for i in range(n_vecs)]
    gt = exact_cosine_gt(vecs, queries, k=10)

    def eval_mvdb(
        label: str,
        index_type: str = "flat",
        quantization: str = "none",
        keep_raw_vectors: bool = True,
        index_params: Optional[dict] = None,
    ):
        gc.collect()
        db = MicroVectorDB()
        coll = db.create_collection(
            "c",
            dimension=dim,
            metric="cosine",
            index_type=index_type,  # type: ignore[arg-type]
            quantization=quantization,  # type: ignore[arg-type]
            keep_raw_vectors=keep_raw_vectors,
            index_params=index_params,
        )
        t0 = time.perf_counter()
        coll.add(ids=ids, vectors=vecs)
        t_build = time.perf_counter() - t0
        # Warmup
        coll.query(query_embeddings=queries[0], n_results=10)

        lats = []
        recalls = []
        for i in range(n_queries):
            tq = time.perf_counter()
            res = coll.query(query_embeddings=queries[i], n_results=10)
            lats.append((time.perf_counter() - tq) * 1000)
            got = {int(s[1:]) for s in res.ids[0]}
            recalls.append(len(got & gt[i]) / 10.0)

        lat = np.array(lats)
        st = coll.stats()
        ram_used = st["vector_ram_mb"] + st["quantized_ram_mb"]
        record_row(
            results,
            category="ANN/Quant",
            engine="MicroVectorDB v0.3.0",
            name=label,
            n_vectors=n_vecs,
            dim=dim,
            metric="cosine",
            filter="none",
            build_s=round(t_build, 4),
            insert_vps=round(n_vecs / max(t_build, 1e-6), 1),
            p50_ms=round(float(np.percentile(lat, 50)), 3),
            p95_ms=round(float(np.percentile(lat, 95)), 3),
            p99_ms=round(float(np.percentile(lat, 99)), 3),
            qps=round(1000.0 / float(lat.mean()), 1),
            recall_at_10=round(float(np.mean(recalls)), 4),
            ram_mb=round(ram_used, 2),
        )
        del coll, db
        gc.collect()

    eval_mvdb("MVDB Flat Float32", index_type="flat", quantization="none")
    eval_mvdb("MVDB Flat Float16 (2x less RAM)", index_type="flat", quantization="float16")
    eval_mvdb("MVDB Flat INT8 (keep_raw=False, 4x RAM)", index_type="flat", quantization="int8", keep_raw_vectors=False)
    eval_mvdb("MVDB Flat Binary (keep_raw=False, 32x RAM)", index_type="flat", quantization="binary", keep_raw_vectors=False)
    eval_mvdb("MVDB Flat Binary+Rescore (mult=50)", index_type="flat", quantization="binary", keep_raw_vectors=True, index_params={"rescore_multiplier": 50})
    eval_mvdb("MVDB IVF (C=128, nprobe=8)", index_type="ivf", index_params={"n_clusters": 128, "nprobe": 8})
    eval_mvdb("MVDB IVF (C=128, nprobe=16)", index_type="ivf", index_params={"n_clusters": 128, "nprobe": 16})
    if is_hnsw_available():
        eval_mvdb("MVDB HNSW (M=16, efC=128, efS=64)", index_type="hnsw", index_params={"M": 16, "ef_construction": 128, "ef_search": 64})
        eval_mvdb("MVDB HNSW (M=16, efC=128, efS=128)", index_type="hnsw", index_params={"M": 16, "ef_construction": 128, "ef_search": 128})

    if HAS_FAISS:
        v_norm = np.ascontiguousarray(vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12), dtype=np.float32)
        q_norm = np.ascontiguousarray(queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12), dtype=np.float32)

        def eval_faiss(label: str, idx: Any, train_first: bool = False):
            t0 = time.perf_counter()
            if train_first:
                idx.train(v_norm)
            idx.add(v_norm)
            t_build = time.perf_counter() - t0
            idx.search(q_norm[0:1], 10)
            lats = []
            recalls = []
            for i in range(n_queries):
                tq = time.perf_counter()
                _, I = idx.search(q_norm[i : i + 1], 10)
                lats.append((time.perf_counter() - tq) * 1000)
                recalls.append(len(set(int(x) for x in I[0]) & gt[i]) / 10.0)
            lat = np.array(lats)
            record_row(
                results,
                category="FAISS-Ref",
                engine=f"FAISS {faiss.__version__}",
                name=label,
                n_vectors=n_vecs,
                dim=dim,
                metric="cosine",
                filter="none",
                build_s=round(t_build, 4),
                insert_vps=round(n_vecs / max(t_build, 1e-6), 1),
                p50_ms=round(float(np.percentile(lat, 50)), 3),
                p95_ms=round(float(np.percentile(lat, 95)), 3),
                p99_ms=round(float(np.percentile(lat, 99)), 3),
                qps=round(1000.0 / float(lat.mean()), 1),
                recall_at_10=round(float(np.mean(recalls)), 4),
                ram_mb=round(v_norm.nbytes / (1024 * 1024), 2),
            )

        eval_faiss("FAISS IndexFlatIP (cosine)", faiss.IndexFlatIP(dim))
        ivf_q = faiss.IndexFlatIP(dim)
        f_ivf = faiss.IndexIVFFlat(ivf_q, dim, 128, faiss.METRIC_INNER_PRODUCT)
        f_ivf.nprobe = 8
        eval_faiss("FAISS IndexIVFFlat (nlist=128, np=8)", f_ivf, train_first=True)
        f_hnsw = faiss.IndexHNSWFlat(dim, 16, faiss.METRIC_INNER_PRODUCT)
        f_hnsw.hnsw.efConstruction = 128
        f_hnsw.hnsw.efSearch = 64
        eval_faiss("FAISS IndexHNSWFlat (M=16, efS=64)", f_hnsw)


def bench_standalone_sq_and_ivf_gaussian(results: List[Dict[str, Any]]) -> None:
    print("\n=== 3. STANDALONE ScalarQuantizer & IVFIndex (100K x 128D Gaussian — v0.2.0 Regresyon Kıyas) ===")
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100_000, 128)).astype(np.float32)
    Q = rng.standard_normal((50, 128)).astype(np.float32)

    gt = []
    for q in Q:
        d = np.einsum("ij,ij->i", X - q, X - q)
        gt.append(set(int(x) for x in np.argpartition(d, 9)[:10]))

    sq = ScalarQuantizer().fit(X)
    enc = sq.encode(X)
    scaled_sq, _ = sq.precompute_db_norms(enc)
    lats, recs = [], []
    for i in range(50):
        tq = time.perf_counter()
        qd = sq.compute_distances_quantized(Q[i], enc, metric="euclidean", precomputed_scaled_sq=scaled_sq)
        top = np.argpartition(qd, 9)[:10]
        lats.append((time.perf_counter() - tq) * 1000)
        recs.append(len(set(int(x) for x in top) & gt[i]) / 10.0)
    lat = np.array(lats)
    record_row(
        results,
        category="Standalone",
        engine="MicroVectorDB v0.3.0",
        name="ScalarQuantizer (overflow-free, tiled)",
        n_vectors=100_000,
        dim=128,
        metric="euclidean",
        filter="none",
        build_s=0.0,
        p50_ms=round(float(np.percentile(lat, 50)), 3),
        p95_ms=round(float(np.percentile(lat, 95)), 3),
        p99_ms=round(float(np.percentile(lat, 99)), 3),
        qps=round(1000.0 / float(lat.mean()), 1),
        recall_at_10=round(float(np.mean(recs)), 4),
        ram_mb=round(enc.nbytes / (1024 * 1024), 2),
    )

    ivf = IVFIndex(n_clusters=256, nprobe=8, metric="euclidean")
    t0 = time.perf_counter()
    ivf.train(X, n_iter=15)
    t_tr = time.perf_counter() - t0
    lats, recs = [], []
    for i in range(50):
        tq = time.perf_counter()
        idxs, _ = ivf.search(Q[i], X, k=10)
        lats.append((time.perf_counter() - tq) * 1000)
        recs.append(len(set(int(x) for x in idxs) & gt[i]) / 10.0)
    lat = np.array(lats)
    record_row(
        results,
        category="Standalone",
        engine="MicroVectorDB v0.3.0",
        name="IVFIndex (K-Means++, packed BLAS, Gauss)",
        n_vectors=100_000,
        dim=128,
        metric="euclidean",
        filter="none",
        build_s=round(t_tr, 3),
        p50_ms=round(float(np.percentile(lat, 50)), 3),
        p95_ms=round(float(np.percentile(lat, 95)), 3),
        p99_ms=round(float(np.percentile(lat, 99)), 3),
        qps=round(1000.0 / float(lat.mean()), 1),
        recall_at_10=round(float(np.mean(recs)), 4),
        ram_mb=round(X.nbytes / (1024 * 1024), 2),
    )


def bench_persistence_incremental(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    print("\n=== 4. INCREMENTAL PERSISTENCE, RELOAD & MMAP (20 x 500 = 10K x 128D) ===")
    tmp_dir = tempfile.mkdtemp(prefix="mvdb_bench_p_")
    try:
        rng = np.random.default_rng(42)
        batch_times = []
        with PersistentClient(path=tmp_dir) as client:
            coll = client.create_collection("inc_test", dimension=128, metric="cosine")
            t0 = time.perf_counter()
            for b in range(20):
                tb = time.perf_counter()
                vecs = rng.standard_normal((500, 128)).astype(np.float32)
                ids = [f"id_{b}_{i}" for i in range(500)]
                metas = [{"b": b, "i": i} for i in range(500)]
                coll.add(ids=ids, embeddings=vecs, metadatas=metas)
                batch_times.append((time.perf_counter() - tb) * 1000)
            total_s = time.perf_counter() - t0
            disk_vec_mb = coll.stats()["vector_disk_mb"]
            disk_db_mb = client._storage.get_db_file_size() / (1024 * 1024)

        # Reload (RAM mode)
        t0 = time.perf_counter()
        with PersistentClient(path=tmp_dir, use_mmap=False) as c_ram:
            reload_ram_ms = (time.perf_counter() - t0) * 1000
            t_del0 = time.perf_counter()
            c_ram.get_collection("inc_test").delete(ids=["id_0_0"])
            single_del_ms = (time.perf_counter() - t_del0) * 1000

        # Reload (mmap mode)
        t0 = time.perf_counter()
        with PersistentClient(path=tmp_dir, use_mmap=True) as c_mm:
            reload_mmap_ms = (time.perf_counter() - t0) * 1000
            mmap_ram_mb = c_mm.get_collection("inc_test").stats()["vector_ram_mb"]

        summary = {
            "total_insert_s": round(total_s, 4),
            "insert_vps": round(10_000 / total_s, 1),
            "batch_1_ms": round(batch_times[0], 2),
            "batch_10_ms": round(batch_times[9], 2),
            "batch_20_ms": round(batch_times[-1], 2),
            "batch_20_to_1_ratio": round(batch_times[-1] / max(batch_times[0], 0.01), 2),
            "disk_vec_mb": round(disk_vec_mb, 2),
            "disk_db_mb": round(disk_db_mb, 2),
            "reload_ram_ms": round(reload_ram_ms, 2),
            "reload_mmap_ms": round(reload_mmap_ms, 2),
            "mmap_vector_ram_mb": round(mmap_ram_mb, 3),
            "single_delete_ms": round(single_del_ms, 2),
        }
        print(
            f"20x500 Persistent Add: total={summary['total_insert_s']}s ({summary['insert_vps']:,.0f} v/s) | "
            f"B1={summary['batch_1_ms']}ms B10={summary['batch_10_ms']}ms B20={summary['batch_20_ms']}ms "
            f"(ratio={summary['batch_20_to_1_ratio']}x)"
        )
        print(
            f"Disk: .mvdbvec={summary['disk_vec_mb']}MB, sqlite={summary['disk_db_mb']}MB | "
            f"Reload RAM={summary['reload_ram_ms']}ms, mmap={summary['reload_mmap_ms']}ms "
            f"(mmap vec RAM={summary['mmap_vector_ram_mb']}MB) | Single Delete={summary['single_delete_ms']}ms"
        )
        return summary
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> None:
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)

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
    print("Benchmark Metadata:", json.dumps(meta, indent=2))

    results: List[Dict[str, Any]] = []
    bench_flat_gaussian(results)
    bench_quantization_and_ann_vs_faiss(results)
    bench_standalone_sq_and_ivf_gaussian(results)
    persist_summary = bench_persistence_incremental(results)

    # JSON kaydet
    json_path = os.path.join(out_dir, "benchmark_v0.3.0.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": meta, "benchmarks": results, "persistence": persist_summary}, f, indent=2)

    # CSV kaydet
    csv_path = os.path.join(out_dir, "benchmark_v0.3.0.csv")
    all_keys = [
        "category", "engine", "name", "n_vectors", "dim", "metric", "filter",
        "build_s", "insert_vps", "first_query_ms", "p50_ms", "p95_ms", "p99_ms",
        "qps", "recall_at_10", "ram_mb", "rss_delta_mb",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\nSonuçlar kaydedildi:\n  - {json_path}\n  - {csv_path}")


if __name__ == "__main__":
    main()
