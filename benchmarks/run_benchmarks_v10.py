"""
MicroVectorDB v1.0.0 — Faz 6 Bilimsel Ölçek, Yüksek Boyut ve Native SIMD Benchmark Paketi
=========================================================================================

Üç kapsamlı, yeniden üretilebilir benchmark:
  1. Native Rust SIMD (AVX2 + FMA3 + POPCNT) vs Saf NumPy vs C++ FAISS 1.13.2 (100K x 128D & 768D)
     - IVF-PQ (C=128, np=8, M=16), Flat PQ (M=16), ScalarQuantizer INT8, BinaryQuantizer 64-bit POPCNT
  2. Büyük Ölçek Eğrisi: 10K -> 100K -> 500K -> 1,000,000 vektör (128D, Cosine)
     - MVDB DiskIndex (MMap), MVDB IVF-PQ (Native SIMD), MVDB IVF, MVDB HNSW vs FAISS (IVFFlat, IVFPQ, HNSW)
  3. Yüksek Boyutlu LLM Embedding'leri: 384D (MiniLM), 768D (BGE/E5), 1536D (OpenAI)
     - Flat Float32, Float16, INT8 (Native SIMD vs NumPy), Binary, Binary+Rescore, DiskIndex, HNSW vs FAISS

Çıktılar:
  - benchmarks/results/benchmark_v1.0.0.json
  - benchmarks/results/benchmark_v1.0.0.csv
"""

from __future__ import annotations

import csv
import gc
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import psutil

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import faiss  # noqa: E402
import hisamdb  # noqa: E402
from hisamdb import (  # noqa: E402
    BinaryQuantizer,
    DiskIndex,
    Float16Quantizer,
    HNSWIndex,
    IVFIndex,
    IVFPQIndex,
    ProductQuantizer,
    ScalarQuantizer,
    is_native_simd_available,
    native_simd_path,
    set_native_simd_enabled,
)
from hisamdb.metrics import (  # noqa: E402
    compute_distances,
    compute_inv_norms,
    normalize_vectors,
    top_k_indices,
)


def make_clustered_vectors(
    n: int, dim: int, n_topics: int = 64, noise_norm: float = 0.65, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gerçek embedding uzaylarını (OpenAI/BGE/MiniLM) yansıtan boyut-bağımsız çok-kümeli manifold
    dağılımı üretir. Gürültü standart sapması `noise_norm / sqrt(dim)` olarak ölçeklenir; böylece
    128D, 384D, 768D ve 1536D boyutlarının tümünde küme içi kosinüs benzerliği ~0.70 civarında kalır.
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 1.0, size=(n_topics, dim)).astype(np.float32)
    centers = normalize_vectors(centers)
    elem_std = float(noise_norm / np.sqrt(dim))

    # DB vektörlerini bellek dostu bloklarla üret (1M x 128D için hızlı)
    vecs = np.empty((n, dim), dtype=np.float32)
    chunk = 100_000
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        sz = e - s
        c_idx = rng.integers(0, n_topics, size=sz)
        block = centers[c_idx] + rng.normal(0.0, elem_std, size=(sz, dim)).astype(np.float32)
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        np.divide(block, np.maximum(norms, np.float32(1e-8)), out=block)
        vecs[s:e] = block

    q_cnt = 100
    qc_idx = rng.integers(0, n_topics, size=q_cnt)
    queries = centers[qc_idx] + rng.normal(0.0, elem_std * 1.15, size=(q_cnt, dim)).astype(np.float32)
    queries = normalize_vectors(queries)
    return vecs, queries


def exact_ground_truth(vecs: np.ndarray, queries: np.ndarray, k: int = 10) -> np.ndarray:
    """FAISS IndexFlatIP ile kesin Cosine Top-K referans komşularını hesaplar."""
    dim = vecs.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(vecs)
    _, gt = idx.search(queries, k)
    del idx
    gc.collect()
    return gt


def compute_recall_at_k(found_list: List[np.ndarray], gt: np.ndarray, k: int = 10) -> float:
    hits = 0
    total = len(found_list) * k
    for i, found in enumerate(found_list):
        gt_set = set(int(x) for x in gt[i, :k])
        for x in found[:k]:
            if int(x) in gt_set:
                hits += 1
    return round(hits / max(1, total), 4)


def bench_queries(fn, queries: np.ndarray, warmup: int = 5) -> Tuple[float, float, float, List[np.ndarray]]:
    """Tek-sorgu P50, P95, QPS ve bulunan komşuları ölçer."""
    for i in range(min(warmup, len(queries))):
        fn(queries[i])
    lat_ms: List[float] = []
    results: List[np.ndarray] = []
    t_all0 = time.perf_counter()
    for i in range(len(queries)):
        t0 = time.perf_counter()
        idx = fn(queries[i])
        lat_ms.append((time.perf_counter() - t0) * 1000.0)
        results.append(np.asarray(idx, dtype=np.int64).ravel())
    elapsed = time.perf_counter() - t_all0
    arr = np.asarray(lat_ms, dtype=np.float64)
    p50 = round(float(np.percentile(arr, 50)), 4)
    p95 = round(float(np.percentile(arr, 95)), 4)
    qps = round(len(queries) / max(elapsed, 1e-9), 1)
    return p50, p95, qps, results


# ==============================================================================
# SUITE 1: Pure NumPy vs Compiled Rust Native SIMD vs C++ FAISS (100K)
# ==============================================================================
def run_suite1_simd(k: int = 10) -> List[Dict[str, Any]]:
    print("\n" + "=" * 78)
    print("SUITE 1: Pure NumPy vs Rust Native SIMD (AVX2+FMA+POPCNT) vs C++ FAISS 1.13.2")
    print("=" * 78)
    rows: List[Dict[str, Any]] = []

    n_vec, dim = 100_000, 128
    vecs, queries = make_clustered_vectors(n_vec, dim, n_topics=64, noise_norm=0.65, seed=101)
    gt = exact_ground_truth(vecs, queries, k=k)

    # --- 1A. IVF-PQ (100K x 128D, C=128, np=8, M=16) ---
    set_native_simd_enabled(True)
    t0 = time.perf_counter()
    ivfpq = IVFPQIndex(n_clusters=128, nprobe=8, n_subvectors=16, n_centroids=256, metric="cosine", seed=42)
    ivfpq.train(vecs, n_iter=12)
    ivfpq_build = round(time.perf_counter() - t0, 3)
    ivfpq_ram = round(ivfpq.memory_bytes() / (1024 * 1024), 2)

    for label, enabled in [
        ("MVDB IVF-PQ (Pure NumPy)", False),
        ("MVDB IVF-PQ (Rust Native SIMD)", True),
    ]:
        set_native_simd_enabled(enabled)
        p50, p95, qps, res = bench_queries(lambda q: ivfpq.search(q, k=k)[0], queries)
        rec = compute_recall_at_k(res, gt, k=k)
        rows.append({
            "group": "IVF-PQ (100K x 128D)",
            "engine": label,
            "n_vectors": n_vec,
            "dim": dim,
            "build_s": ivfpq_build,
            "p50_ms": p50,
            "p95_ms": p95,
            "qps": qps,
            "recall_at_10": rec,
            "ram_mb": ivfpq_ram,
        })
        print(f"  {label:<36} | P50={p50:6.3f} ms | P95={p95:6.3f} ms | QPS={qps:8.1f} | R@10={rec:.3f} | RAM={ivfpq_ram:.2f} MB")

    # FAISS IndexIVFPQ
    t0 = time.perf_counter()
    quant = faiss.IndexFlatIP(dim)
    f_ivfpq = faiss.IndexIVFPQ(quant, dim, 128, 16, 8, faiss.METRIC_INNER_PRODUCT)
    f_ivfpq.train(vecs)
    f_ivfpq.add(vecs)
    f_ivfpq.nprobe = 8
    f_build = round(time.perf_counter() - t0, 3)
    f_ram = round((n_vec * (16 + 8)) / (1024 * 1024), 2)
    p50, p95, qps, res = bench_queries(lambda q: f_ivfpq.search(q.reshape(1, -1), k)[1][0], queries)
    rec = compute_recall_at_k(res, gt, k=k)
    rows.append({
        "group": "IVF-PQ (100K x 128D)",
        "engine": "FAISS IndexIVFPQ (C++)",
        "n_vectors": n_vec,
        "dim": dim,
        "build_s": f_build,
        "p50_ms": p50,
        "p95_ms": p95,
        "qps": qps,
        "recall_at_10": rec,
        "ram_mb": f_ram,
    })
    print(f"  {'FAISS IndexIVFPQ (C++)':<36} | P50={p50:6.3f} ms | P95={p95:6.3f} ms | QPS={qps:8.1f} | R@10={rec:.3f} | RAM={f_ram:.2f} MB")
    del ivfpq, f_ivfpq, quant
    gc.collect()

    # --- 1B. Flat PQ (100K x 128D, M=16) ---
    t0 = time.perf_counter()
    pq = ProductQuantizer(n_subvectors=16, n_centroids=256, metric="cosine", n_iter=12, seed=42)
    pq.fit(vecs)
    pq_codes_t = np.ascontiguousarray(pq.encode(vecs).T)
    pq_build = round(time.perf_counter() - t0, 3)
    pq_ram = round((pq_codes_t.nbytes + pq.codebooks.nbytes) / (1024 * 1024), 2)

    for label, enabled in [
        ("MVDB Flat PQ (Pure NumPy)", False),
        ("MVDB Flat PQ (Rust Native SIMD)", True),
    ]:
        set_native_simd_enabled(enabled)
        p50, p95, qps, res = bench_queries(
            lambda q: top_k_indices(pq.compute_distances_adc(q, pq_codes_t, metric="cosine")[0], k),
            queries,
        )
        rec = compute_recall_at_k(res, gt, k=k)
        rows.append({
            "group": "Flat PQ (100K x 128D)",
            "engine": label,
            "n_vectors": n_vec,
            "dim": dim,
            "build_s": pq_build,
            "p50_ms": p50,
            "p95_ms": p95,
            "qps": qps,
            "recall_at_10": rec,
            "ram_mb": pq_ram,
        })
        print(f"  {label:<36} | P50={p50:6.3f} ms | P95={p95:6.3f} ms | QPS={qps:8.1f} | R@10={rec:.3f} | RAM={pq_ram:.2f} MB")

    t0 = time.perf_counter()
    f_pq = faiss.IndexPQ(dim, 16, 8, faiss.METRIC_INNER_PRODUCT)
    f_pq.train(vecs)
    f_pq.add(vecs)
    fpq_build = round(time.perf_counter() - t0, 3)
    p50, p95, qps, res = bench_queries(lambda q: f_pq.search(q.reshape(1, -1), k)[1][0], queries)
    rec = compute_recall_at_k(res, gt, k=k)
    rows.append({
        "group": "Flat PQ (100K x 128D)",
        "engine": "FAISS IndexPQ (C++)",
        "n_vectors": n_vec,
        "dim": dim,
        "build_s": fpq_build,
        "p50_ms": p50,
        "p95_ms": p95,
        "qps": qps,
        "recall_at_10": rec,
        "ram_mb": round((n_vec * 16) / (1024 * 1024), 2),
    })
    print(f"  {'FAISS IndexPQ (C++)':<36} | P50={p50:6.3f} ms | P95={p95:6.3f} ms | QPS={qps:8.1f} | R@10={rec:.3f}")
    del pq, pq_codes_t, f_pq
    gc.collect()

    # --- 1C. ScalarQuantizer INT8 (100K x 128D, Cosine) ---
    t0 = time.perf_counter()
    sq = ScalarQuantizer().fit(vecs)
    sq_codes = sq.encode(vecs)
    sq_scaled, sq_inv = sq.precompute_db_norms(sq_codes)
    sq_build = round(time.perf_counter() - t0, 3)
    sq_ram = round((sq_codes.nbytes + sq_scaled.nbytes + sq_inv.nbytes) / (1024 * 1024), 2)

    for label, enabled in [
        ("MVDB INT8 (Pure NumPy)", False),
        ("MVDB INT8 (Rust Native SIMD)", True),
    ]:
        set_native_simd_enabled(enabled)
        p50, p95, qps, res = bench_queries(
            lambda q: top_k_indices(
                sq.compute_distances_quantized(
                    q, sq_codes, metric="cosine",
                    precomputed_scaled_sq=sq_scaled,
                    precomputed_recon_inv=sq_inv,
                ),
                k,
            ),
            queries,
        )
        rec = compute_recall_at_k(res, gt, k=k)
        rows.append({
            "group": "INT8 Scalar (100K x 128D)",
            "engine": label,
            "n_vectors": n_vec,
            "dim": dim,
            "build_s": sq_build,
            "p50_ms": p50,
            "p95_ms": p95,
            "qps": qps,
            "recall_at_10": rec,
            "ram_mb": sq_ram,
        })
        print(f"  {label:<36} | P50={p50:6.3f} ms | P95={p95:6.3f} ms | QPS={qps:8.1f} | R@10={rec:.3f} | RAM={sq_ram:.2f} MB")

    # FAISS IndexScalarQuantizer (QT_8bit)
    t0 = time.perf_counter()
    f_sq = faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)
    f_sq.train(vecs)
    f_sq.add(vecs)
    fsq_build = round(time.perf_counter() - t0, 3)
    p50, p95, qps, res = bench_queries(lambda q: f_sq.search(q.reshape(1, -1), k)[1][0], queries)
    rec = compute_recall_at_k(res, gt, k=k)
    rows.append({
        "group": "INT8 Scalar (100K x 128D)",
        "engine": "FAISS IndexSQ8 (C++)",
        "n_vectors": n_vec,
        "dim": dim,
        "build_s": fsq_build,
        "p50_ms": p50,
        "p95_ms": p95,
        "qps": qps,
        "recall_at_10": rec,
        "ram_mb": round((n_vec * dim) / (1024 * 1024), 2),
    })
    print(f"  {'FAISS IndexSQ8 (C++)':<36} | P50={p50:6.3f} ms | P95={p95:6.3f} ms | QPS={qps:8.1f} | R@10={rec:.3f}")
    del sq, sq_codes, sq_scaled, sq_inv, f_sq, vecs, queries, gt
    gc.collect()

    # --- 1D. BinaryQuantizer 64-bit POPCNT (100K x 768D LLM dim) ---
    vecs768, queries768 = make_clustered_vectors(100_000, 768, n_topics=64, noise_norm=0.65, seed=202)
    gt768 = exact_ground_truth(vecs768, queries768, k=k)
    t0 = time.perf_counter()
    bq = BinaryQuantizer().fit(vecs768)
    b_codes = bq.encode(vecs768)
    bq_build = round(time.perf_counter() - t0, 3)
    bq_ram = round(b_codes.nbytes / (1024 * 1024), 2)
    q_bits_list = [bq.encode(queries768[i]) for i in range(len(queries768))]

    for label, enabled in [
        ("MVDB Binary (Pure NumPy)", False),
        ("MVDB Binary (Rust POPCNT SIMD)", True),
    ]:
        set_native_simd_enabled(enabled)
        # bench_queries wrapper over index
        q_idx_arr = np.arange(len(queries768))
        p50, p95, qps, res = bench_queries(
            lambda qi: top_k_indices(bq.hamming_distances(q_bits_list[int(qi)], b_codes), k),
            q_idx_arr,
        )
        rec = compute_recall_at_k(res, gt768, k=k)
        rows.append({
            "group": "Binary Hamming (100K x 768D)",
            "engine": label,
            "n_vectors": 100_000,
            "dim": 768,
            "build_s": bq_build,
            "p50_ms": p50,
            "p95_ms": p95,
            "qps": qps,
            "recall_at_10": rec,
            "ram_mb": bq_ram,
        })
        print(f"  {label:<36} | P50={p50:6.3f} ms | P95={p95:6.3f} ms | QPS={qps:8.1f} | R@10={rec:.3f} | RAM={bq_ram:.2f} MB")

    set_native_simd_enabled(True)
    del vecs768, queries768, gt768, bq, b_codes
    gc.collect()
    return rows


# ==============================================================================
# SUITE 2: Scale Curve (10K -> 100K -> 500K -> 1,000,000 x 128D)
# ==============================================================================
def run_suite2_scale(k: int = 10) -> List[Dict[str, Any]]:
    print("\n" + "=" * 78)
    print("SUITE 2: Büyük Ölçek Eğrisi — 10K -> 100K -> 500K -> 1,000,000 Vektör (128D)")
    print("=" * 78)
    set_native_simd_enabled(True)
    rows: List[Dict[str, Any]] = []
    dim = 128

    scale_configs = [
        (10_000,   64,  8),
        (100_000, 128, 12),
        (500_000, 256, 16),
        (1_000_000, 512, 24),
    ]

    tmpdir = tempfile.mkdtemp(prefix="mvdb_scale_1m_")
    try:
        for n_vec, n_clusters, nprobe in scale_configs:
            print(f"\n--- Ölçek: N = {n_vec:,} x {dim}D (C={n_clusters}, nprobe={nprobe}) ---")
            vecs, queries = make_clustered_vectors(n_vec, dim, n_topics=128, noise_norm=0.65, seed=77)
            queries = queries[:50]  # 50 sorgu (yeterli istatistiksel doğruluk + hızlı)
            gt = exact_ground_truth(vecs, queries, k=k)
            raw_mb = round(vecs.nbytes / (1024 * 1024), 2)

            # 1. MVDB DiskIndex (MMap Float32)
            d_vec_path = os.path.join(tmpdir, f"disk_{n_vec}.mvdbdisk")
            d_meta_path = os.path.join(tmpdir, f"disk_{n_vec}.meta")
            t0 = time.perf_counter()
            d_idx = DiskIndex(n_clusters=n_clusters, nprobe=nprobe, metric="cosine", storage_dtype="float32", seed=42)
            d_idx.build(vecs, disk_vec_path=d_vec_path, meta_path=d_meta_path, n_iter=10)
            b_s = round(time.perf_counter() - t0, 3)
            ram_mb = round(d_idx.ram_bytes() / (1024 * 1024), 2)
            p50, p95, qps, res = bench_queries(lambda q: d_idx.search_batch(q, k=k)[0][0], queries)
            rec = compute_recall_at_k(res, gt, k=k)
            rows.append({
                "n_vectors": n_vec, "dim": dim, "raw_mb": raw_mb,
                "engine": "MVDB DiskIndex (MMap)",
                "build_s": b_s, "p50_ms": p50, "p95_ms": p95, "qps": qps,
                "recall_at_10": rec, "ram_mb": ram_mb,
            })
            print(f"  {'MVDB DiskIndex (MMap)':<28} | build={b_s:6.2f}s | P50={p50:6.3f} ms | QPS={qps:8.1f} | R@10={rec:.3f} | RAM={ram_mb:6.2f} MB")
            d_idx.close()
            del d_idx
            for fp in (d_vec_path, d_meta_path):
                if os.path.exists(fp):
                    try:
                        os.remove(fp)
                    except OSError:
                        pass
            gc.collect()

            # 2. MVDB IVF-PQ (Rust Native SIMD, keep_raw=False)
            t0 = time.perf_counter()
            ipq = IVFPQIndex(n_clusters=n_clusters, nprobe=nprobe, n_subvectors=16, n_centroids=256, metric="cosine", seed=42)
            ipq.train(vecs, n_iter=10)
            b_s = round(time.perf_counter() - t0, 3)
            ram_mb = round(ipq.memory_bytes() / (1024 * 1024), 2)
            p50, p95, qps, res = bench_queries(lambda q: ipq.search(q, k=k)[0], queries)
            rec = compute_recall_at_k(res, gt, k=k)
            rows.append({
                "n_vectors": n_vec, "dim": dim, "raw_mb": raw_mb,
                "engine": "MVDB IVF-PQ (Native SIMD)",
                "build_s": b_s, "p50_ms": p50, "p95_ms": p95, "qps": qps,
                "recall_at_10": rec, "ram_mb": ram_mb,
            })
            print(f"  {'MVDB IVF-PQ (Native SIMD)':<28} | build={b_s:6.2f}s | P50={p50:6.3f} ms | QPS={qps:8.1f} | R@10={rec:.3f} | RAM={ram_mb:6.2f} MB")
            del ipq
            gc.collect()

            # 3. MVDB IVF (RAM Float32)
            t0 = time.perf_counter()
            ivf = IVFIndex(n_clusters=n_clusters, nprobe=nprobe, metric="cosine")
            ivf.train(vecs, n_iter=10)
            b_s = round(time.perf_counter() - t0, 3)
            ram_mb = round((vecs.nbytes + n_vec * 12) / (1024 * 1024), 2)
            p50, p95, qps, res = bench_queries(lambda q: ivf.search(q, k=k)[0], queries)
            rec = compute_recall_at_k(res, gt, k=k)
            rows.append({
                "n_vectors": n_vec, "dim": dim, "raw_mb": raw_mb,
                "engine": "MVDB IVF (Float32)",
                "build_s": b_s, "p50_ms": p50, "p95_ms": p95, "qps": qps,
                "recall_at_10": rec, "ram_mb": ram_mb,
            })
            print(f"  {'MVDB IVF (Float32)':<28} | build={b_s:6.2f}s | P50={p50:6.3f} ms | QPS={qps:8.1f} | R@10={rec:.3f} | RAM={ram_mb:6.2f} MB")
            del ivf
            gc.collect()

            # 4. MVDB HNSW (M=16, efC=100, efS=64)
            t0 = time.perf_counter()
            hnsw = HNSWIndex(dimension=dim, metric="cosine", M=16, ef_construction=100, ef_search=64, initial_capacity=n_vec)
            hnsw.add_items(vecs, np.arange(n_vec, dtype=np.int64))
            b_s = round(time.perf_counter() - t0, 3)
            # hnswlib bellek tahmini: float32 vektörler + M*2*4B grafik bağlantıları
            ram_mb = round((n_vec * (dim * 4 + 16 * 8 + 16)) / (1024 * 1024), 2)
            p50, p95, qps, res = bench_queries(lambda q: hnsw.search(q, k=k)[0][0], queries)
            rec = compute_recall_at_k(res, gt, k=k)
            rows.append({
                "n_vectors": n_vec, "dim": dim, "raw_mb": raw_mb,
                "engine": "MVDB HNSW (M=16, efS=64)",
                "build_s": b_s, "p50_ms": p50, "p95_ms": p95, "qps": qps,
                "recall_at_10": rec, "ram_mb": ram_mb,
            })
            print(f"  {'MVDB HNSW (M=16, efS=64)':<28} | build={b_s:6.2f}s | P50={p50:6.3f} ms | QPS={qps:8.1f} | R@10={rec:.3f} | RAM={ram_mb:6.2f} MB")
            del hnsw
            gc.collect()

            # 5. FAISS IndexIVFFlat
            t0 = time.perf_counter()
            q_ivf = faiss.IndexFlatIP(dim)
            f_ivf = faiss.IndexIVFFlat(q_ivf, dim, n_clusters, faiss.METRIC_INNER_PRODUCT)
            f_ivf.train(vecs)
            f_ivf.add(vecs)
            f_ivf.nprobe = nprobe
            b_s = round(time.perf_counter() - t0, 3)
            ram_mb = round((vecs.nbytes + n_vec * 8) / (1024 * 1024), 2)
            p50, p95, qps, res = bench_queries(lambda q: f_ivf.search(q.reshape(1, -1), k)[1][0], queries)
            rec = compute_recall_at_k(res, gt, k=k)
            rows.append({
                "n_vectors": n_vec, "dim": dim, "raw_mb": raw_mb,
                "engine": "FAISS IndexIVFFlat",
                "build_s": b_s, "p50_ms": p50, "p95_ms": p95, "qps": qps,
                "recall_at_10": rec, "ram_mb": ram_mb,
            })
            print(f"  {'FAISS IndexIVFFlat':<28} | build={b_s:6.2f}s | P50={p50:6.3f} ms | QPS={qps:8.1f} | R@10={rec:.3f} | RAM={ram_mb:6.2f} MB")
            del f_ivf, q_ivf
            gc.collect()

            # 6. FAISS IndexIVFPQ
            t0 = time.perf_counter()
            q_pq = faiss.IndexFlatIP(dim)
            f_pq = faiss.IndexIVFPQ(q_pq, dim, n_clusters, 16, 8, faiss.METRIC_INNER_PRODUCT)
            f_pq.train(vecs)
            f_pq.add(vecs)
            f_pq.nprobe = nprobe
            b_s = round(time.perf_counter() - t0, 3)
            ram_mb = round((n_vec * (16 + 8)) / (1024 * 1024), 2)
            p50, p95, qps, res = bench_queries(lambda q: f_pq.search(q.reshape(1, -1), k)[1][0], queries)
            rec = compute_recall_at_k(res, gt, k=k)
            rows.append({
                "n_vectors": n_vec, "dim": dim, "raw_mb": raw_mb,
                "engine": "FAISS IndexIVFPQ",
                "build_s": b_s, "p50_ms": p50, "p95_ms": p95, "qps": qps,
                "recall_at_10": rec, "ram_mb": ram_mb,
            })
            print(f"  {'FAISS IndexIVFPQ':<28} | build={b_s:6.2f}s | P50={p50:6.3f} ms | QPS={qps:8.1f} | R@10={rec:.3f} | RAM={ram_mb:6.2f} MB")
            del f_pq, q_pq
            gc.collect()

            # 7. FAISS IndexHNSWFlat (M=16, efC=100, efS=64)
            t0 = time.perf_counter()
            f_hnsw = faiss.IndexHNSWFlat(dim, 16, faiss.METRIC_INNER_PRODUCT)
            f_hnsw.hnsw.efConstruction = 100
            f_hnsw.hnsw.efSearch = 64
            f_hnsw.add(vecs)
            b_s = round(time.perf_counter() - t0, 3)
            ram_mb = round((n_vec * (dim * 4 + 16 * 8 + 16)) / (1024 * 1024), 2)
            p50, p95, qps, res = bench_queries(lambda q: f_hnsw.search(q.reshape(1, -1), k)[1][0], queries)
            rec = compute_recall_at_k(res, gt, k=k)
            rows.append({
                "n_vectors": n_vec, "dim": dim, "raw_mb": raw_mb,
                "engine": "FAISS IndexHNSWFlat",
                "build_s": b_s, "p50_ms": p50, "p95_ms": p95, "qps": qps,
                "recall_at_10": rec, "ram_mb": ram_mb,
            })
            print(f"  {'FAISS IndexHNSWFlat':<28} | build={b_s:6.2f}s | P50={p50:6.3f} ms | QPS={qps:8.1f} | R@10={rec:.3f} | RAM={ram_mb:6.2f} MB")
            del f_hnsw, vecs, queries, gt
            gc.collect()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return rows


# ==============================================================================
# SUITE 3: High-Dimensional LLM Embeddings (384D, 768D, 1536D)
# ==============================================================================
def run_suite3_highdim(k: int = 10) -> List[Dict[str, Any]]:
    print("\n" + "=" * 78)
    print("SUITE 3: Yüksek Boyutlu LLM Embedding'leri — 384D (MiniLM), 768D (BGE), 1536D (OpenAI)")
    print("=" * 78)
    set_native_simd_enabled(True)
    rows: List[Dict[str, Any]] = []

    configs = [
        ("MiniLM-384D", 50_000, 384, 128, 12),
        ("BGE/E5-768D", 50_000, 768, 128, 12),
        ("OpenAI-1536D", 25_000, 1536, 96, 10),
    ]

    tmpdir = tempfile.mkdtemp(prefix="mvdb_highdim_")
    try:
        for profile, n_vec, dim, n_clusters, nprobe in configs:
            print(f"\n--- Profil: {profile} ({n_vec:,} x {dim}D) ---")
            vecs, queries = make_clustered_vectors(n_vec, dim, n_topics=64, noise_norm=0.65, seed=303)
            queries = queries[:50]
            gt = exact_ground_truth(vecs, queries, k=k)
            raw_mb = round(vecs.nbytes / (1024 * 1024), 2)
            _, inv_norms = compute_inv_norms(vecs)

            # 1. MVDB Flat Float32
            p50, p95, qps, res = bench_queries(
                lambda q: top_k_indices(compute_distances(q.reshape(1, -1), vecs, metric="cosine", db_inv_norms=inv_norms)[0][0], k),
                queries,
            )
            rec = compute_recall_at_k(res, gt, k=k)
            rows.append({
                "profile": profile, "n_vectors": n_vec, "dim": dim, "raw_mb": raw_mb,
                "engine": "MVDB Flat (Float32)",
                "p50_ms": p50, "p95_ms": p95, "qps": qps, "recall_at_10": rec, "ram_mb": raw_mb,
            })
            print(f"  {'MVDB Flat (Float32)':<30} | P50={p50:6.3f} ms | QPS={qps:8.1f} | R@10={rec:.3f} | RAM={raw_mb:6.2f} MB")

            # 2. MVDB Float16
            f16 = Float16Quantizer()
            v16 = f16.encode(vecs)
            f16_ram = round(v16.nbytes / (1024 * 1024), 2)
            p50, p95, qps, res = bench_queries(
                lambda q: top_k_indices(f16.compute_batch_distances(q.reshape(1, -1), v16, metric="cosine", db_inv_norms=inv_norms)[0][0], k),
                queries,
            )
            rec = compute_recall_at_k(res, gt, k=k)
            rows.append({
                "profile": profile, "n_vectors": n_vec, "dim": dim, "raw_mb": raw_mb,
                "engine": "MVDB Float16 (2x RAM↓)",
                "p50_ms": p50, "p95_ms": p95, "qps": qps, "recall_at_10": rec, "ram_mb": f16_ram,
            })
            print(f"  {'MVDB Float16 (2x RAM↓)':<30} | P50={p50:6.3f} ms | QPS={qps:8.1f} | R@10={rec:.3f} | RAM={f16_ram:6.2f} MB")
            del v16, f16
            gc.collect()

            # 3. MVDB INT8 (Pure NumPy vs Rust Native SIMD)
            sq = ScalarQuantizer().fit(vecs)
            sq_codes = sq.encode(vecs)
            sq_sc, sq_inv = sq.precompute_db_norms(sq_codes)
            sq_ram = round((sq_codes.nbytes + sq_sc.nbytes + sq_inv.nbytes) / (1024 * 1024), 2)
            for label, enabled in [
                ("MVDB INT8 (Pure NumPy)", False),
                ("MVDB INT8 (Rust Native SIMD)", True),
            ]:
                set_native_simd_enabled(enabled)
                p50, p95, qps, res = bench_queries(
                    lambda q: top_k_indices(
                        sq.compute_distances_quantized(
                            q, sq_codes, metric="cosine",
                            precomputed_scaled_sq=sq_sc, precomputed_recon_inv=sq_inv,
                        ),
                        k,
                    ),
                    queries,
                )
                rec = compute_recall_at_k(res, gt, k=k)
                rows.append({
                    "profile": profile, "n_vectors": n_vec, "dim": dim, "raw_mb": raw_mb,
                    "engine": label,
                    "p50_ms": p50, "p95_ms": p95, "qps": qps, "recall_at_10": rec, "ram_mb": sq_ram,
                })
                print(f"  {label:<30} | P50={p50:6.3f} ms | QPS={qps:8.1f} | R@10={rec:.3f} | RAM={sq_ram:6.2f} MB")
            set_native_simd_enabled(True)
            del sq, sq_codes, sq_sc, sq_inv
            gc.collect()

            # 4. MVDB Binary (32x RAM↓) + Binary + Float32 Rescore (top 20x)
            bq = BinaryQuantizer().fit(vecs)
            b_codes = bq.encode(vecs)
            b_ram = round(b_codes.nbytes / (1024 * 1024), 2)
            q_bits = [bq.encode(queries[i]) for i in range(len(queries))]
            q_idx_arr = np.arange(len(queries))

            p50, p95, qps, res = bench_queries(
                lambda qi: top_k_indices(bq.hamming_distances(q_bits[int(qi)], b_codes), k),
                q_idx_arr,
            )
            rec = compute_recall_at_k(res, gt, k=k)
            rows.append({
                "profile": profile, "n_vectors": n_vec, "dim": dim, "raw_mb": raw_mb,
                "engine": "MVDB Binary (32x RAM↓)",
                "p50_ms": p50, "p95_ms": p95, "qps": qps, "recall_at_10": rec, "ram_mb": b_ram,
            })
            print(f"  {'MVDB Binary (32x RAM↓)':<30} | P50={p50:6.3f} ms | QPS={qps:8.1f} | R@10={rec:.3f} | RAM={b_ram:6.2f} MB")

            # Binary + Float32 Rescore (top 200 aday)
            oversample = 200
            def _bin_rescore(qi: int) -> np.ndarray:
                cand = top_k_indices(bq.hamming_distances(q_bits[int(qi)], b_codes), oversample)
                dists, _ = compute_distances(queries[int(qi)].reshape(1, -1), vecs[cand], metric="cosine", db_inv_norms=inv_norms[cand])
                loc = top_k_indices(dists[0], k)
                return cand[loc]

            p50, p95, qps, res = bench_queries(_bin_rescore, q_idx_arr)
            rec = compute_recall_at_k(res, gt, k=k)
            rows.append({
                "profile": profile, "n_vectors": n_vec, "dim": dim, "raw_mb": raw_mb,
                "engine": "MVDB Binary + Rescore (20x)",
                "p50_ms": p50, "p95_ms": p95, "qps": qps, "recall_at_10": rec, "ram_mb": b_ram,
            })
            print(f"  {'MVDB Binary + Rescore (20x)':<30} | P50={p50:6.3f} ms | QPS={qps:8.1f} | R@10={rec:.3f} | RAM={b_ram:6.2f} MB (+disk/raw)")
            del bq, b_codes
            gc.collect()

            # 5. MVDB DiskIndex (MMap Float32)
            dvp = os.path.join(tmpdir, f"hd_{dim}.mvdbdisk")
            dmp = os.path.join(tmpdir, f"hd_{dim}.meta")
            d_idx = DiskIndex(n_clusters=n_clusters, nprobe=nprobe, metric="cosine", storage_dtype="float32", seed=42)
            d_idx.build(vecs, disk_vec_path=dvp, meta_path=dmp, n_iter=10)
            d_ram = round(d_idx.ram_bytes() / (1024 * 1024), 2)
            p50, p95, qps, res = bench_queries(lambda q: d_idx.search_batch(q, k=k)[0][0], queries)
            rec = compute_recall_at_k(res, gt, k=k)
            rows.append({
                "profile": profile, "n_vectors": n_vec, "dim": dim, "raw_mb": raw_mb,
                "engine": "MVDB DiskIndex (MMap)",
                "p50_ms": p50, "p95_ms": p95, "qps": qps, "recall_at_10": rec, "ram_mb": d_ram,
            })
            print(f"  {'MVDB DiskIndex (MMap)':<30} | P50={p50:6.3f} ms | QPS={qps:8.1f} | R@10={rec:.3f} | RAM={d_ram:6.2f} MB")
            d_idx.close()
            del d_idx
            gc.collect()

            # 6. MVDB HNSW (M=16, efC=100, efS=64)
            hnsw = HNSWIndex(dimension=dim, metric="cosine", M=16, ef_construction=100, ef_search=64, initial_capacity=n_vec)
            hnsw.add_items(vecs, np.arange(n_vec, dtype=np.int64))
            h_ram = round((n_vec * (dim * 4 + 16 * 8 + 16)) / (1024 * 1024), 2)
            p50, p95, qps, res = bench_queries(lambda q: hnsw.search(q, k=k)[0][0], queries)
            rec = compute_recall_at_k(res, gt, k=k)
            rows.append({
                "profile": profile, "n_vectors": n_vec, "dim": dim, "raw_mb": raw_mb,
                "engine": "MVDB HNSW (M=16, efS=64)",
                "p50_ms": p50, "p95_ms": p95, "qps": qps, "recall_at_10": rec, "ram_mb": h_ram,
            })
            print(f"  {'MVDB HNSW (M=16, efS=64)':<30} | P50={p50:6.3f} ms | QPS={qps:8.1f} | R@10={rec:.3f} | RAM={h_ram:6.2f} MB")
            del hnsw
            gc.collect()

            # 7. FAISS IndexHNSWFlat
            f_hnsw = faiss.IndexHNSWFlat(dim, 16, faiss.METRIC_INNER_PRODUCT)
            f_hnsw.hnsw.efConstruction = 100
            f_hnsw.hnsw.efSearch = 64
            f_hnsw.add(vecs)
            p50, p95, qps, res = bench_queries(lambda q: f_hnsw.search(q.reshape(1, -1), k)[1][0], queries)
            rec = compute_recall_at_k(res, gt, k=k)
            rows.append({
                "profile": profile, "n_vectors": n_vec, "dim": dim, "raw_mb": raw_mb,
                "engine": "FAISS IndexHNSWFlat",
                "p50_ms": p50, "p95_ms": p95, "qps": qps, "recall_at_10": rec, "ram_mb": h_ram,
            })
            print(f"  {'FAISS IndexHNSWFlat':<30} | P50={p50:6.3f} ms | QPS={qps:8.1f} | R@10={rec:.3f} | RAM={h_ram:6.2f} MB")
            del f_hnsw, vecs, queries, gt, inv_norms
            gc.collect()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return rows


def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else "all"
    print(f"MicroVectorDB v1.0.0 — Faz 6 Benchmark Başlıyor (mod={only})...")
    print(f"Native Rust SIMD aktif: {is_native_simd_available()} ({native_simd_path()})")

    out_dir = os.path.join(REPO_ROOT, "benchmarks", "results")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "benchmark_v1.0.0.json")
    existing: Dict[str, Any] = {}
    if only != "all" and os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            existing = json.load(f)

    t_start = time.perf_counter()
    suite1 = run_suite1_simd() if only in ("all", "suite1") else existing.get("suite1_simd_vs_numpy_vs_faiss", [])
    suite2 = run_suite2_scale() if only in ("all", "suite2") else existing.get("suite2_scale_10k_to_1m", [])
    suite3 = run_suite3_highdim() if only in ("all", "suite3") else existing.get("suite3_high_dim_llm", [])
    total_s = round(time.perf_counter() - t_start, 2)

    payload = {
        "metadata": {
            "microvectordb_version": hisamdb.__version__,
            "python": platform.python_version(),
            "os": platform.platform(),
            "cpu_logical_cores": os.cpu_count(),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
            "numpy": np.__version__,
            "faiss": faiss.__version__,
            "native_simd_available": is_native_simd_available(),
            "dataset_distribution": "Dimension-scaled Clustered Gaussian Manifold (L2-normalized float32, noise_norm=0.65 -> intra-cluster cos ~0.70)",
            "total_benchmark_s": total_s,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "suite1_simd_vs_numpy_vs_faiss": suite1,
        "suite2_scale_10k_to_1m": suite2,
        "suite3_high_dim_llm": suite3,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(out_dir, "benchmark_v1.0.0.csv")
    csv_rows: List[Dict[str, Any]] = []
    for r in suite1:
        csv_rows.append({"suite": "suite1_simd", "label": r["group"], **r})
    for r in suite2:
        csv_rows.append({"suite": "suite2_scale", "label": f"{r['n_vectors']}x{r['dim']}D", **r})
    for r in suite3:
        csv_rows.append({"suite": "suite3_highdim", "label": r["profile"], **r})

    fieldnames = [
        "suite", "label", "engine", "n_vectors", "dim", "build_s",
        "p50_ms", "p95_ms", "qps", "recall_at_10", "ram_mb",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nTamamlandı ({total_s}s)!")
    print(f"  JSON: {json_path}")
    print(f"  CSV : {csv_path}")


if __name__ == "__main__":
    main()
