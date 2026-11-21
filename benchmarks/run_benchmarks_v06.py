"""
benchmarks/run_benchmarks_v06.py
--------------------------------
MicroVectorDB v0.6.0 (Faz 4 — Eşzamanlılık, Paralel Batch, Async ve Arka Plan Bakım)
bilimsel benchmark paketi:
1. Çok çekirdekli paralel batch_search() ölçeklenmesi (1 / 2 / 4 / 8 işçi, 100K x 128D: HNSW, IVF, IVF-PQ, DiskIndex)
2. Paralel batch_hybrid_search() (Dense HNSW + BM25 + RRF, 10K doküman, 256 sorgu)
3. AsyncMicroVectorDB + asyncio.gather() (256 eşzamanlı async sorgu, 100K x 128D HNSW)
4. Bloklamayan rebuild_index_async() sırasında okuyucu gecikme yalıtımı (50K x 128D IVF)
5. auto_compact_async=True vs False: yoğun silmede P99 çağrı gecikmesi (10K x 128D)
"""

import asyncio
import csv
import gc
import json
import os
import platform
import shutil
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List
import numpy as np
import psutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import microvectordb
from microvectordb import AsyncMicroVectorDB, MicroVectorDB, is_hnsw_available


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
    tmp_dir = tempfile.mkdtemp(prefix="mvdb_bench_v06_")

    meta = {
        "microvectordb_version": microvectordb.__version__,
        "python": platform.python_version(),
        "os": platform.platform(),
        "cpu_logical_cores": os.cpu_count(),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "numpy": np.__version__,
        "hnswlib": is_hnsw_available(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print("Benchmark v0.6.0 Metadata:", json.dumps(meta, indent=2))

    scaling_rows: List[Dict[str, Any]] = []
    async_rows: List[Dict[str, Any]] = []
    bg_rows: List[Dict[str, Any]] = []

    try:
        n_vecs, dim, n_queries = 100_000, 128, 256
        vecs, queries = make_clustered_data(n_vecs, dim, n_topics=64, n_queries=n_queries, seed=42)
        ids = [f"v{i}" for i in range(n_vecs)]
        worker_levels = [1, 2, 4, 8]

        # ============================================================== #
        # 1. ÇOK ÇEKİRDEKLİ PARALEL BATCH_SEARCH ÖLÇEKLENMESİ (100K)     #
        # ============================================================== #
        print("\n=== 1. ÇOK ÇEKİRDEKLİ PARALEL batch_search() ÖLÇEKLENMESİ (100K x 128D, 256 sorgu) ===")

        configs = []
        if is_hnsw_available():
            configs.append(("HNSW (M=16, efS=64)", "hnsw", {"M": 16, "ef_construction": 128, "ef_search": 64}, None, True))
        configs.extend([
            ("IVF (C=128, np=8, saf NumPy)", "ivf", {"n_clusters": 128, "nprobe": 8}, None, True),
            ("IVF-PQ (C=128, np=8, M=16)", "ivf_pq", {"n_clusters": 128, "nprobe": 8, "pq_subvectors": 16}, None, False),
            ("DiskIndex (MMap, C=128, np=8)", "disk", {"n_clusters": 128, "nprobe": 8}, os.path.join(tmp_dir, "disk_p4"), True),
        ])

        for label, idx_t, params, db_path, keep_raw in configs:
            gc.collect()
            with MicroVectorDB(db_path, max_workers=8) as db:
                coll = db.create_collection(
                    "c", dimension=dim, metric="cosine", index_type=idx_t,
                    keep_raw_vectors=keep_raw, index_params=params,
                )
                coll.add(ids=ids, vectors=vecs)
                coll.batch_search(queries[:8], limit=10, max_workers=1)  # warmup

                base_qps = None
                for w in worker_levels:
                    # 3 tekrarın medyanı (tutarlı ölçüm)
                    times_ms = []
                    for _ in range(3):
                        t0 = time.perf_counter()
                        res = coll.batch_search(queries, limit=10, max_workers=w)
                        times_ms.append((time.perf_counter() - t0) * 1000)
                        assert len(res) == n_queries
                    med_ms = float(np.median(times_ms))
                    qps = (n_queries * 1000.0) / med_ms
                    if base_qps is None:
                        base_qps = qps
                    sp = qps / base_qps
                    row = {
                        "workload": label,
                        "n_vectors": n_vecs,
                        "n_queries": n_queries,
                        "workers": w,
                        "batch_ms": round(med_ms, 2),
                        "qps": round(qps, 1),
                        "speedup": round(sp, 2),
                    }
                    scaling_rows.append(row)
                    print(
                        f"  {label:<32} | workers={w} | Batch({n_queries})={med_ms:>7.2f}ms | "
                        f"QPS={qps:>9.1f} | Hızlanma={sp:>4.2f}x"
                    )

        # ============================================================== #
        # 2. PARALEL BATCH_HYBRID_SEARCH (10K DOKÜMAN, 256 SORGU)        #
        # ============================================================== #
        print("\n=== 2. PARALEL batch_hybrid_search() ÖLÇEKLENMESİ (10K Doküman, 256 sorgu) ===")
        rng = np.random.default_rng(123)
        n_docs = 10_000
        topic_names = [
            "veritabanı indeksleme hnsw performans",
            "kubernetes pod ağ güvenlik sertifika",
            "yapay zeka büyük dil modeli rag",
            "dağıtık sistem wal çökme kurtarma",
            "finans fatura vergi mutabakat rapor",
            "lojistik depo sevkiyat istanbul ankara",
            "siber güvenlik yetkilendirme şifreleme kvkk",
            "bulut depolama yedekleme snapshot geri yükleme",
        ]
        doc_vecs = vecs[:n_docs]
        doc_ids = [f"doc_{i}" for i in range(n_docs)]
        docs_text = [
            f"Belge {i}: {topic_names[i % len(topic_names)]} sistem kodu ERR-{1000 + i} modül M{i % 200}"
            for i in range(n_docs)
        ]

        with MicroVectorDB(max_workers=8) as db_h:
            idx_h = "hnsw" if is_hnsw_available() else "flat"
            c_h = db_h.create_collection("hyb", dimension=dim, metric="cosine", index_type=idx_h)
            c_h.add(ids=doc_ids, vectors=doc_vecs, documents=docs_text)
            q_specs = [
                {
                    "query_vector": queries[i],
                    "query_text": f"{topic_names[i % len(topic_names)].split()[0]} ERR-{1000 + (i * 17) % n_docs}",
                    "fusion": "rrf",
                }
                for i in range(n_queries)
            ]
            c_h.batch_hybrid_search(q_specs[:8], limit=10, max_workers=1)

            base_qps = None
            for w in worker_levels:
                times_ms = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    res = c_h.batch_hybrid_search(q_specs, limit=10, max_workers=w)
                    times_ms.append((time.perf_counter() - t0) * 1000)
                    assert len(res) == n_queries
                med_ms = float(np.median(times_ms))
                qps = (n_queries * 1000.0) / med_ms
                if base_qps is None:
                    base_qps = qps
                sp = qps / base_qps
                row = {
                    "workload": f"Hybrid ({idx_h.upper()}+BM25+RRF, 10K)",
                    "n_vectors": n_docs,
                    "n_queries": n_queries,
                    "workers": w,
                    "batch_ms": round(med_ms, 2),
                    "qps": round(qps, 1),
                    "speedup": round(sp, 2),
                }
                scaling_rows.append(row)
                print(
                    f"  {row['workload']:<32} | workers={w} | Batch({n_queries})={med_ms:>7.2f}ms | "
                    f"QPS={qps:>9.1f} | Hızlanma={sp:>4.2f}x"
                )

        # ============================================================== #
        # 3. ASYNC MICROVECTORDB + asyncio.gather() FAN-OUT (100K HNSW)  #
        # ============================================================== #
        if is_hnsw_available():
            print("\n=== 3. AsyncMicroVectorDB + asyncio.gather() (100K x 128D HNSW) ===")

            async def _bench_async():
                async with AsyncMicroVectorDB(max_workers=8) as adb:
                    ac = await adb.create_collection(
                        "ah", dimension=dim, metric="cosine", index_type="hnsw",
                        index_params={"M": 16, "ef_construction": 128, "ef_search": 64},
                    )
                    await ac.add(ids=ids, vectors=vecs)
                    await ac.search(queries[0], limit=10)

                    for conc in (64, 256):
                        per_req_ms: List[float] = []

                        async def _one(qi: int):
                            ts = time.perf_counter()
                            r = await ac.search(queries[qi], limit=10)
                            per_req_ms.append((time.perf_counter() - ts) * 1000)
                            return r

                        t0 = time.perf_counter()
                        await asyncio.gather(*(_one(i) for i in range(conc)))
                        wall_ms = (time.perf_counter() - t0) * 1000
                        qps = (conc * 1000.0) / wall_ms
                        arr = np.array(per_req_ms)
                        arow = {
                            "scenario": f"AsyncMicroVectorDB gather({conc}) on 100K HNSW",
                            "concurrency": conc,
                            "wall_ms": round(wall_ms, 2),
                            "aggregate_qps": round(qps, 1),
                            "req_p50_ms": round(float(np.percentile(arr, 50)), 3),
                            "req_p95_ms": round(float(np.percentile(arr, 95)), 3),
                        }
                        async_rows.append(arow)
                        print(
                            f"  gather({conc:<3}) | Toplam={wall_ms:>6.2f}ms | "
                            f"Toplam QPS={qps:>8.1f} | İstek P50={arow['req_p50_ms']:.3f}ms P95={arow['req_p95_ms']:.3f}ms"
                        )

            asyncio.run(_bench_async())

        # ============================================================== #
        # 4. BLOKLAMAYAN ARKA PLAN rebuild_index_async() YALITIMI (50K)  #
        # ============================================================== #
        print("\n=== 4. BLOKLAMAYAN rebuild_index_async() OKUYUCU GECİKME YALITIMI (50K x 128D IVF) ===")
        with MicroVectorDB(max_workers=4) as db_bg:
            c_bg = db_bg.create_collection(
                "ivf_bg", dimension=dim, metric="cosine", index_type="ivf",
                index_params={"n_clusters": 128, "nprobe": 8},
            )
            c_bg.add(ids=ids[:50_000], vectors=vecs[:50_000])
            c_bg.search(queries[0], limit=10)

            # A. Bloklamayan rebuild_index_async() çalışırken sürekli okuyucu gecikmelerini ölç!
            read_lats: List[float] = []
            stop_flag = threading.Event()

            def _reader_loop():
                qi = 0
                while not stop_flag.is_set():
                    ts = time.perf_counter()
                    c_bg.search(queries[qi % n_queries], limit=10)
                    read_lats.append((time.perf_counter() - ts) * 1000)
                    qi += 1

            rt = threading.Thread(target=_reader_loop)
            rt.start()
            time.sleep(0.02)
            tb0 = time.perf_counter()
            fut = c_bg.rebuild_index_async()
            swapped = fut.result(timeout=60)
            build_ms = (time.perf_counter() - tb0) * 1000
            stop_flag.set()
            rt.join()

            r_arr = np.array(read_lats)
            bg_row = {
                "scenario": "50K IVF rebuild_index_async() sırasında eşzamanlı okuyucular",
                "swapped": bool(swapped),
                "rebuild_ms": round(build_ms, 1),
                "reads_completed_during_rebuild": int(len(r_arr)),
                "reader_p50_ms": round(float(np.percentile(r_arr, 50)), 3),
                "reader_p95_ms": round(float(np.percentile(r_arr, 95)), 3),
                "reader_max_ms": round(float(r_arr.max()), 3),
            }
            bg_rows.append(bg_row)
            print(
                f"  Arka Plan İnşa={build_ms:.1f}ms | İnşa sırasında tamamlanan okuma={len(r_arr)} | "
                f"Okuyucu P50={bg_row['reader_p50_ms']:.3f}ms P95={bg_row['reader_p95_ms']:.3f}ms Max={bg_row['reader_max_ms']:.2f}ms"
            )

        # ============================================================== #
        # 5. auto_compact_async=True vs False SİLME P99 GECİKMESİ (10K)  #
        # ============================================================== #
        print("\n=== 5. auto_compact_async=True vs False: YOĞUN SİLME P99 GECİKMESİ (10K x 128D) ===")
        for async_cmp in (False, True):
            p_cmp = os.path.join(tmp_dir, f"cmp_async_{async_cmp}")
            with MicroVectorDB(p_cmp) as db_c:
                c_c = db_c.create_collection(
                    "dc", dimension=dim, metric="cosine",
                    index_params={
                        "auto_compact_async": async_cmp,
                        "compaction_threshold": 0.25,
                        "auto_compact_min_deleted": 64,
                    },
                )
                c_c.add(ids=ids[:10_000], vectors=vecs[:10_000])
                del_lats = []
                # 25 batch x 100 kayıt = 2.500 silme (25. çağrı tam %25 eşiğini tetikler!)
                for b in range(25):
                    batch_ids = ids[b * 100 : (b + 1) * 100]
                    ts = time.perf_counter()
                    c_c.delete(ids=batch_ids)
                    del_lats.append((time.perf_counter() - ts) * 1000)
                trigger_ms = del_lats[-1]
                c_c.wait_for_background(timeout=30)
                arr = np.array(del_lats)
                row_c = {
                    "scenario": f"10K Yoğun Silme (25x100) auto_compact_async={async_cmp}",
                    "p50_ms": round(float(np.percentile(arr, 50)), 3),
                    "p95_ms": round(float(np.percentile(arr, 95)), 3),
                    "trigger_batch_ms": round(float(trigger_ms), 3),
                    "max_ms": round(float(arr.max()), 3),
                    "final_tombstones": c_c.stats()["tombstones"],
                }
                bg_rows.append(row_c)
                print(
                    f"  auto_compact_async={str(async_cmp):<5} | P50={row_c['p50_ms']:>6.3f}ms | "
                    f"Tetikleyen Silme={row_c['trigger_batch_ms']:>7.3f}ms | Max={row_c['max_ms']:>7.3f}ms | "
                    f"Son Tombstone={row_c['final_tombstones']}"
                )

        # Kaydet: JSON & CSV
        json_path = os.path.join(out_dir, "benchmark_v0.6.0.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": meta,
                    "parallel_batch_scaling": scaling_rows,
                    "async_gather_fanout": async_rows,
                    "background_maintenance": bg_rows,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        csv_path = os.path.join(out_dir, "benchmark_v0.6.0.csv")
        keys = ["workload", "n_vectors", "n_queries", "workers", "batch_ms", "qps", "speedup"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in scaling_rows:
                w.writerow(r)

        print(f"\nSonuçlar kaydedildi:\n  - {json_path}\n  - {csv_path}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
