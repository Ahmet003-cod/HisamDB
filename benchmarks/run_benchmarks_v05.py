"""
benchmarks/run_benchmarks_v05.py
--------------------------------
MicroVectorDB v0.5.0 (Faz 3 + Sorun #1/#2/#3 Çözümleri) bilimsel benchmark paketi:
1. Sorun #2: Önceden Hesaplanmış (Precomputed) Küme-Merkezli IVF-PQ & Batched matmul PQ vs FAISS (100K x 128D)
2. Sorun #1: 100K x 128D Sütunsal + orjson + Tembel Metadata/Norm Açılış Hızı (v0.4 -> v0.5)
3. Sorun #3: Tombstone + İkili WAL Tekil Kalıcı Silme (10K & 100K) + Compaction + SHA-256 Snapshot/Restore
4. Faz 3: 10.000 Dokümanlık Gerçekçi Kurumsal Külliyatta Dense vs BM25 vs Hibrit (RRF & Weighted) Hız ve Doğruluk
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

import hisamdb
from hisamdb import MicroVectorDB, is_hnsw_available

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    import orjson  # noqa: F401
    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False


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
        f"[{kw['category']:<13}] {kw['name']:<48} | "
        f"P50={kw.get('p50_ms', 0):>6.3f}ms P95={kw.get('p95_ms', 0):>6.3f}ms | "
        f"QPS={kw.get('qps', 0):>8.1f} | Rec@10={kw.get('recall_at_10', 1.0):.3f} | "
        f"RAM={kw.get('ram_mb', 0):>5.2f}MB"
    )


def main() -> None:
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="mvdb_bench_v05_")

    meta = {
        "microvectordb_version": hisamdb.__version__,
        "python": platform.python_version(),
        "os": platform.platform(),
        "cpu_logical_cores": os.cpu_count(),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "numpy": np.__version__,
        "faiss": faiss.__version__ if HAS_FAISS else None,
        "hnswlib": is_hnsw_available(),
        "orjson": HAS_ORJSON,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print("Benchmark v0.5.0 Metadata:", json.dumps(meta, indent=2))

    rows: List[Dict[str, Any]] = []
    reload_rows: List[Dict[str, Any]] = []
    delete_rows: List[Dict[str, Any]] = []
    hybrid_rows: List[Dict[str, Any]] = []

    try:
        n_vecs, dim, n_queries = 100_000, 128, 100
        vecs, queries = make_clustered_data(n_vecs, dim, n_topics=64, n_queries=n_queries, seed=42)
        ids = [f"v{i}" for i in range(n_vecs)]
        gt = exact_cosine_gt(vecs, queries, k=10)

        # ============================================================== #
        # 1. SORUN #2: PRECOMPUTED IVF-PQ & BATCHED MATMUL PQ vs FAISS   #
        # ============================================================== #
        print("\n=== 1. SORUN #2: PRECOMPUTED IVF-PQ & PQ vs FAISS (100K x 128D) ===")

        # 1A. IVF-PQ (keep_raw=False, persistent -> aynı zamanda Sorun #1 reload ölçümü!)
        p_ivfpq_dir = os.path.join(tmp_dir, "ivfpq_db")
        with MicroVectorDB(p_ivfpq_dir) as db:
            c = db.create_collection(
                "ivfpq", dimension=dim, metric="cosine", index_type="ivf_pq", keep_raw_vectors=False,
                index_params={"n_clusters": 128, "nprobe": 8, "pq_subvectors": 16, "pq_centroids": 256},
            )
            t0 = time.perf_counter()
            c.add(ids=ids, vectors=vecs)
            tb_ivfpq = time.perf_counter() - t0
            c.query(queries[0], n_results=10)  # warmup
            lats, recs = [], []
            for i in range(n_queries):
                tq = time.perf_counter()
                res = c.query(queries[i], n_results=10)
                lats.append((time.perf_counter() - tq) * 1000)
                recs.append(len({int(s[1:]) for s in res.ids[0]} & gt[i]) / 10.0)
            st = c.stats()
            lat = np.array(lats)
            log_row(
                rows, category="Issue#2-PQ", engine="MicroVectorDB v0.5.0",
                name="MVDB v0.5 IVF-PQ (Precomputed LUT, C=128, np=8)",
                n_vectors=n_vecs, dim=dim, metric="cosine",
                build_s=round(tb_ivfpq, 3), p50_ms=round(float(np.percentile(lat, 50)), 3),
                p95_ms=round(float(np.percentile(lat, 95)), 3), qps=round(1000.0 / float(lat.mean()), 1),
                recall_at_10=round(float(np.mean(recs)), 4),
                ram_mb=round(st["vector_ram_mb"] + st["index_ram_mb"], 2),
            )

        # Reload IVF-PQ (Sorun #1)
        t0 = time.perf_counter()
        with MicroVectorDB(p_ivfpq_dir) as db_r:
            c_r = db_r.get_collection("ivfpq")
            t_reload_ivfpq = (time.perf_counter() - t0) * 1000
            st_r = c_r.stats()
        reload_rows.append({
            "index": "IVF-PQ (100K x 128D, keep_raw=F)",
            "v04_reload_ms": 542.0,
            "v05_reload_ms": round(t_reload_ivfpq, 1),
            "speedup_vs_v04": round(542.0 / max(t_reload_ivfpq, 0.1), 2),
            "loaded_from_disk": st_r["index_loaded_from_disk"],
            "vec_ram_mb": round(st_r["vector_ram_mb"], 2),
        })

        # 1B. Flat PQ (batched np.matmul)
        gc.collect()
        db_pq = MicroVectorDB()
        c_pq = db_pq.create_collection(
            "pq", dimension=dim, metric="cosine", quantization="pq", keep_raw_vectors=False,
            index_params={"pq_subvectors": 16, "pq_centroids": 256, "pq_iter": 12},
        )
        t0 = time.perf_counter()
        c_pq.add(ids=ids, vectors=vecs)
        tb_pq = time.perf_counter() - t0
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
            rows, category="Issue#2-PQ", engine="MicroVectorDB v0.5.0",
            name="MVDB v0.5 Flat PQ (Batched matmul, M=16, 8-bit)",
            n_vectors=n_vecs, dim=dim, metric="cosine",
            build_s=round(tb_pq, 3), p50_ms=round(float(np.percentile(lat, 50)), 3),
            p95_ms=round(float(np.percentile(lat, 95)), 3), qps=round(1000.0 / float(lat.mean()), 1),
            recall_at_10=round(float(np.mean(recs)), 4),
            ram_mb=round(st["vector_ram_mb"] + st["quantized_ram_mb"] + st["index_ram_mb"], 2),
        )
        del c_pq, db_pq
        gc.collect()

        # 1C. FAISS Reference (same hardware & session)
        if HAS_FAISS:
            v_norm = np.ascontiguousarray(vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12), dtype=np.float32)
            q_norm = np.ascontiguousarray(queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12), dtype=np.float32)

            qtz = faiss.IndexFlatL2(dim)
            fivfpq = faiss.IndexIVFPQ(qtz, dim, 128, 16, 8)
            t0 = time.perf_counter()
            fivfpq.train(v_norm)
            fivfpq.add(v_norm)
            fivfpq.nprobe = 8
            tb_fipq = time.perf_counter() - t0
            fivfpq.search(q_norm[:1], 10)
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

        # ============================================================== #
        # 2. SORUN #1: 100K x 128D HIZLI AÇILIŞ (COLUMNAR + ORJSON + LAZY) #
        # ============================================================== #
        print("\n=== 2. SORUN #1: 100K x 128D HIZLI YENİDEN YÜKLEME (v0.4 -> v0.5) ===")

        # 2A. DiskIndex (no meta)
        p_disk_dir = os.path.join(tmp_dir, "disk_db")
        with MicroVectorDB(p_disk_dir) as db:
            c = db.create_collection("disk_c", dimension=dim, metric="cosine", index_type="disk", index_params={"n_clusters": 128, "nprobe": 8})
            c.add(ids=ids, vectors=vecs)
        t0 = time.perf_counter()
        with MicroVectorDB(p_disk_dir) as db_r:
            c_r = db_r.get_collection("disk_c")
            t_reload_disk = (time.perf_counter() - t0) * 1000
            st_d = c_r.stats()
        reload_rows.append({
            "index": "DiskIndex (100K x 128D, MMap)",
            "v04_reload_ms": 532.8,
            "v05_reload_ms": round(t_reload_disk, 1),
            "speedup_vs_v04": round(532.8 / max(t_reload_disk, 0.1), 2),
            "loaded_from_disk": st_d["index_loaded_from_disk"],
            "vec_ram_mb": round(st_d["vector_ram_mb"], 2),
        })

        # 2B. IVF (no meta)
        p_ivf_dir = os.path.join(tmp_dir, "ivf_db")
        with MicroVectorDB(p_ivf_dir) as db:
            c = db.create_collection("ivf_c", dimension=dim, metric="cosine", index_type="ivf", index_params={"n_clusters": 128, "nprobe": 8})
            c.add(ids=ids, vectors=vecs)
        t0 = time.perf_counter()
        with MicroVectorDB(p_ivf_dir) as db_r:
            c_r = db_r.get_collection("ivf_c")
            t_reload_ivf = (time.perf_counter() - t0) * 1000
            st_i = c_r.stats()
        reload_rows.append({
            "index": "IVF (100K x 128D)",
            "v04_reload_ms": 781.3,
            "v05_reload_ms": round(t_reload_ivf, 1),
            "speedup_vs_v04": round(781.3 / max(t_reload_ivf, 0.1), 2),
            "loaded_from_disk": st_i["index_loaded_from_disk"],
            "vec_ram_mb": round(st_i["vector_ram_mb"], 2),
        })

        # 2C. HNSW (no meta) + Tombstone sonrası reload doğrulaması!
        if is_hnsw_available():
            p_hnsw_dir = os.path.join(tmp_dir, "hnsw_db")
            with MicroVectorDB(p_hnsw_dir) as db:
                c = db.create_collection("hnsw_c", dimension=dim, metric="cosine", index_type="hnsw", index_params={"M": 16, "ef_construction": 128, "ef_search": 64})
                c.add(ids=ids, vectors=vecs)
            t0 = time.perf_counter()
            with MicroVectorDB(p_hnsw_dir) as db_r:
                c_r = db_r.get_collection("hnsw_c")
                t_reload_hnsw = (time.perf_counter() - t0) * 1000
                st_h = c_r.stats()
                # Tombstone silme testi de burada yapalım:
                c_r.delete(ids=["v42"])
            # Tombstone sonrası anında diskten yükleme doğrulaması:
            t0 = time.perf_counter()
            with MicroVectorDB(p_hnsw_dir) as db_r2:
                c_r2 = db_r2.get_collection("hnsw_c")
                t_reload_hnsw_ts = (time.perf_counter() - t0) * 1000
                assert c_r2.count() == n_vecs - 1
                assert c_r2.stats()["index_loaded_from_disk"] is True
            reload_rows.append({
                "index": "HNSW (100K x 128D, M=16, efC=128)",
                "v04_reload_ms": 940.6,
                "v05_reload_ms": round(t_reload_hnsw, 1),
                "speedup_vs_v04": round(940.6 / max(t_reload_hnsw, 0.1), 2),
                "loaded_from_disk": st_h["index_loaded_from_disk"],
                "vec_ram_mb": round(st_h["vector_ram_mb"], 2),
                "post_tombstone_reload_ms": round(t_reload_hnsw_ts, 1),
            })

        # 2D. 100K MMap + 100K zengin Metadata (cat, year, score)
        p_meta_dir = os.path.join(tmp_dir, "meta_db")
        metas_100k = [{"category": f"cat_{i % 20}", "year": 2020 + (i % 6), "score": i % 100} for i in range(n_vecs)]
        with MicroVectorDB(p_meta_dir) as db:
            c = db.create_collection("meta_c", dimension=dim, metric="cosine", use_mmap=True)
            c.add(ids=ids, vectors=vecs, metadata=metas_100k)
        t0 = time.perf_counter()
        with MicroVectorDB(p_meta_dir) as db_r:
            c_r = db_r.get_collection("meta_c")
            t_reload_meta = (time.perf_counter() - t0) * 1000
            st_m = c_r.stats()
        reload_rows.append({
            "index": "MMap + 100K Zengin Metadata (lazy idx)",
            "v04_reload_ms": 1150.0,  # referans tahmini değil, sadece açıklama için None bırakıyoruz
            "v05_reload_ms": round(t_reload_meta, 1),
            "speedup_vs_v04": None,
            "loaded_from_disk": True,
            "vec_ram_mb": round(st_m["vector_ram_mb"], 2),
            "lazy_meta_pending": st_m["metadata_index"].get("lazy_pending", False),
        })
        reload_rows[-1]["v04_reload_ms"] = None

        for rr in reload_rows:
            v04_s = f"{rr['v04_reload_ms']:>6.1f}ms" if rr["v04_reload_ms"] else "   N/A  "
            sp_s = f"{rr['speedup_vs_v04']:>5.2f}x" if rr["speedup_vs_v04"] else "  N/A "
            print(
                f"  {rr['index']:<40} | v0.4: {v04_s} -> v0.5: {rr['v05_reload_ms']:>6.1f}ms "
                f"| Hızlanma: {sp_s} | from_disk={rr['loaded_from_disk']} | vec_ram={rr['vec_ram_mb']:.2f}MB"
            )

        # ============================================================== #
        # 3. SORUN #3: TOMBSTONE SİLME (<1 ms), COMPACT & SNAPSHOT       #
        # ============================================================== #
        print("\n=== 3. SORUN #3: TOMBSTONE SİLME, COMPACTION & SHA-256 SNAPSHOT ===")

        # 3A. 10K x 128D (v0.3.0'da 181.9 ms idi!) — WAL normal & WAL full
        for mode_label, wal_mode in [("10K x 128D Tekil Silme (WAL normal)", "normal"), ("10K x 128D Tekil Silme (WAL full fsync)", "full")]:
            p_del = os.path.join(tmp_dir, f"del10k_{wal_mode}")
            with MicroVectorDB(p_del) as db:
                c = db.create_collection("del_c", dimension=dim, metric="cosine", index_params={"wal_sync": wal_mode})
                c.add(ids=ids[:10_000], vectors=vecs[:10_000])
                lats = []
                for i in range(25):
                    ts = time.perf_counter()
                    c.delete(ids=[f"v{i * 10}"])
                    lats.append((time.perf_counter() - ts) * 1000)
                lat = np.array(lats)
                tc = time.perf_counter()
                reclaimed = c.compact()
                t_compact = (time.perf_counter() - tc) * 1000
                delete_rows.append({
                    "scenario": mode_label,
                    "v03_p50_ms": 181.9,
                    "v05_p50_ms": round(float(np.percentile(lat, 50)), 3),
                    "v05_p95_ms": round(float(np.percentile(lat, 95)), 3),
                    "v05_min_ms": round(float(lat.min()), 3),
                    "speedup_vs_v03": round(181.9 / max(float(np.percentile(lat, 50)), 0.001), 1),
                    "compact_ms": round(t_compact, 2),
                    "reclaimed_slots": reclaimed,
                })

        # 3B. 100K x 128D tekil kalıcı silme (WAL normal)
        with MicroVectorDB(p_meta_dir) as db:
            c = db.get_collection("meta_c")
            lats = []
            for i in range(25):
                ts = time.perf_counter()
                c.delete(ids=[f"v{i * 10}"])
                lats.append((time.perf_counter() - ts) * 1000)
            lat = np.array(lats)
            delete_rows.append({
                "scenario": "100K x 128D Tekil Silme (WAL normal)",
                "v03_p50_ms": None,
                "v05_p50_ms": round(float(np.percentile(lat, 50)), 3),
                "v05_p95_ms": round(float(np.percentile(lat, 95)), 3),
                "v05_min_ms": round(float(lat.min()), 3),
                "speedup_vs_v03": None,
                "compact_ms": None,
                "reclaimed_slots": None,
            })

        # 3C. Snapshot Create & Restore (10K x 128D, SHA-256)
        p_snap_src = os.path.join(tmp_dir, "snap_src")
        p_snap_dst = os.path.join(tmp_dir, "snap_backup")
        p_snap_res = os.path.join(tmp_dir, "snap_restored")
        with MicroVectorDB(p_snap_src) as db:
            c = db.create_collection("sn", dimension=dim, metric="cosine")
            c.add(ids=ids[:10_000], vectors=vecs[:10_000])
            t0 = time.perf_counter()
            manifest = db.create_snapshot(p_snap_dst)
            t_snap_create = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        MicroVectorDB.restore_snapshot(p_snap_dst, p_snap_res)
        t_snap_restore = (time.perf_counter() - t0) * 1000
        delete_rows.append({
            "scenario": "10K x 128D SHA-256 Snapshot Create / Restore",
            "v03_p50_ms": None,
            "v05_p50_ms": round(t_snap_create, 2),
            "v05_p95_ms": round(t_snap_restore, 2),
            "v05_min_ms": len(manifest["files"]),
            "speedup_vs_v03": None,
            "compact_ms": None,
            "reclaimed_slots": None,
        })

        for dr in delete_rows:
            if "Snapshot" in dr["scenario"]:
                print(
                    f"  {dr['scenario']:<46} | Create={dr['v05_p50_ms']:>6.2f}ms | "
                    f"Restore(SHA-256)={dr['v05_p95_ms']:>6.2f}ms ({dr['v05_min_ms']} dosya)"
                )
            else:
                sp = f"{dr['speedup_vs_v03']:>6.1f}x" if dr["speedup_vs_v03"] else "   N/A"
                cmp_s = f" | Compact={dr['compact_ms']:.1f}ms" if dr["compact_ms"] else ""
                print(
                    f"  {dr['scenario']:<46} | P50={dr['v05_p50_ms']:>6.3f}ms P95={dr['v05_p95_ms']:>6.3f}ms "
                    f"| vs v0.3: {sp}{cmp_s}"
                )

        # ============================================================== #
        # 4. FAZ 3: 10K DOKÜMAN DENSE vs BM25 vs HİBRİT (RRF & WEIGHTED) #
        # ============================================================== #
        print("\n=== 4. FAZ 3: 10.000 DOKÜMAN DENSE vs BM25 vs HİBRİT ARAMA (128D) ===")
        rng = np.random.default_rng(123)
        n_docs = 10_000
        n_topics = 40
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
        centers = rng.standard_normal((n_topics, dim)).astype(np.float32) * 2.5
        doc_topics = rng.integers(0, n_topics, size=n_docs)
        # Gerçekçi RAG: aynı konudaki dokümanlar birbirine yakın embedding'lere sahip (sigma=0.20),
        # yalnızca benzersiz hata kodu / belge kimliği metin içinde ayrışır!
        doc_vecs = (centers[doc_topics] + rng.standard_normal((n_docs, dim)).astype(np.float32) * 0.20).astype(np.float32)
        doc_ids = [f"doc_{i}" for i in range(n_docs)]
        docs_text = [
            f"Belge {i}: {topic_names[doc_topics[i] % len(topic_names)]} sistem kodu ERR-{1000 + i} modül Mod-{i % 250} sürüm v{1 + (i % 5)}"
            for i in range(n_docs)
        ]
        sparse_vecs = [
            {f"err{1000 + i}": 3.5, f"mod{i % 250}": 1.5, f"topic{doc_topics[i]}": 1.0}
            for i in range(n_docs)
        ]

        db_h = MicroVectorDB()
        idx_t = "hnsw" if is_hnsw_available() else "flat"
        c_h = db_h.create_collection(
            "hybrid_kb", dimension=dim, metric="cosine", index_type=idx_t,
            index_params={"M": 16, "ef_construction": 128, "ef_search": 64},
        )
        t_add0 = time.perf_counter()
        c_h.add(ids=doc_ids, vectors=doc_vecs, documents=docs_text, sparse_vectors=sparse_vecs)
        t_add_h = time.perf_counter() - t_add0

        # 100 gerçekçi hibrit sorgu:
        # 50 sorgu = konu bazlı semantik sorgu + kesin hata kodu (ERR-xxxx) -> Dense konuyu bulur ama 250 dokümandan 1'ini ayıramaz!
        # 50 sorgu = dokümana çok yakın semantik parafraz sorgu (tam kod yok) -> BM25 zayıf, Dense çok güçlü!
        n_hq = 100
        target_indices = rng.choice(n_docs, size=n_hq, replace=False)
        q_vecs_h = []
        q_texts_h = []
        q_sparses_h = []
        for qi, tidx in enumerate(target_indices):
            if qi < 50:
                qv = (centers[doc_topics[tidx]] + rng.standard_normal(dim).astype(np.float32) * 0.25).astype(np.float32)
                qt = f"{topic_names[doc_topics[tidx] % len(topic_names)].split()[0]} ERR-{1000 + tidx}"
                qs = {f"err{1000 + tidx}": 2.0}
            else:
                qv = (doc_vecs[tidx] + rng.standard_normal(dim).astype(np.float32) * 0.04).astype(np.float32)
                qt = topic_names[doc_topics[tidx] % len(topic_names)]
                qs = {f"topic{doc_topics[tidx]}": 1.0}
            q_vecs_h.append(qv)
            q_texts_h.append(qt)
            q_sparses_h.append(qs)

        # Warmup
        c_h.search(q_vecs_h[0], limit=10)
        c_h.sparse_search(query_text=q_texts_h[0], limit=10)
        c_h.sparse_search(query_sparse=q_sparses_h[0], limit=10)
        c_h.hybrid_search(query_vector=q_vecs_h[0], query_text=q_texts_h[0], limit=10, fusion="rrf")

        def bench_hybrid_mode(mode_name: str, fn):
            lats, hits1, hits10 = [], [], []
            for qi in range(n_hq):
                target_id = doc_ids[int(target_indices[qi])]
                ts = time.perf_counter()
                res = fn(qi)
                lats.append((time.perf_counter() - ts) * 1000)
                res_ids = [r.id for r in res]
                hits1.append(1.0 if (res_ids and res_ids[0] == target_id) else 0.0)
                hits10.append(1.0 if target_id in res_ids else 0.0)
            lat = np.array(lats)
            row = {
                "mode": mode_name,
                "p50_ms": round(float(np.percentile(lat, 50)), 3),
                "p95_ms": round(float(np.percentile(lat, 95)), 3),
                "qps": round(1000.0 / float(lat.mean()), 1),
                "hit_at_1": round(float(np.mean(hits1)), 3),
                "hit_at_10": round(float(np.mean(hits10)), 3),
            }
            hybrid_rows.append(row)
            print(
                f"  {mode_name:<42} | P50={row['p50_ms']:>6.3f}ms P95={row['p95_ms']:>6.3f}ms | "
                f"QPS={row['qps']:>7.1f} | Hit@1={row['hit_at_1']:.3f} | Hit@10={row['hit_at_10']:.3f}"
            )

        bench_hybrid_mode(
            f"Dense-Only ({idx_t.upper()})",
            lambda qi: c_h.search(q_vecs_h[qi], limit=10),
        )
        bench_hybrid_mode(
            "BM25-Only (Okapi BM25)",
            lambda qi: c_h.sparse_search(query_text=q_texts_h[qi], limit=10),
        )
        bench_hybrid_mode(
            "Sparse-Vector-Only (Inverted Index)",
            lambda qi: c_h.sparse_search(query_sparse=q_sparses_h[qi], limit=10),
        )
        bench_hybrid_mode(
            "Hybrid (Dense + BM25, RRF)",
            lambda qi: c_h.hybrid_search(query_vector=q_vecs_h[qi], query_text=q_texts_h[qi], limit=10, fusion="rrf"),
        )
        bench_hybrid_mode(
            "Hybrid (Dense + BM25, Weighted Min-Max)",
            lambda qi: c_h.hybrid_search(query_vector=q_vecs_h[qi], query_text=q_texts_h[qi], limit=10, fusion="weighted", alpha=0.5, beta=0.5),
        )
        bench_hybrid_mode(
            "3-Way Hybrid (Dense + BM25 + Sparse, RRF)",
            lambda qi: c_h.hybrid_search(query_vector=q_vecs_h[qi], query_text=q_texts_h[qi], query_sparse=q_sparses_h[qi], limit=10, fusion="rrf"),
        )
        print(f"  (10K doküman + dense + BM25 + sparse toplam ekleme süresi: {t_add_h:.3f}s)")

        # Save JSON & CSV
        json_path = os.path.join(out_dir, "benchmark_v0.5.0.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": meta,
                    "issue2_pq_benchmarks": rows,
                    "issue1_reload_benchmarks": reload_rows,
                    "issue3_delete_wal_snapshot": delete_rows,
                    "phase3_hybrid_10k": {
                        "n_docs": n_docs,
                        "dim": dim,
                        "index_type": idx_t,
                        "ingest_s": round(t_add_h, 3),
                        "modes": hybrid_rows,
                    },
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        csv_path = os.path.join(out_dir, "benchmark_v0.5.0.csv")
        csv_rows = []
        for r in rows:
            csv_rows.append(r)
        for hr in hybrid_rows:
            csv_rows.append({
                "category": "Phase3-Hybrid",
                "engine": "MicroVectorDB v0.5.0",
                "name": hr["mode"],
                "n_vectors": n_docs,
                "dim": dim,
                "metric": "hybrid",
                "build_s": round(t_add_h, 3),
                "p50_ms": hr["p50_ms"],
                "p95_ms": hr["p95_ms"],
                "qps": hr["qps"],
                "recall_at_10": hr["hit_at_10"],
                "ram_mb": "",
            })
        keys = ["category", "engine", "name", "n_vectors", "dim", "metric", "build_s", "p50_ms", "p95_ms", "qps", "recall_at_10", "ram_mb"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in csv_rows:
                w.writerow(r)

        print(f"\nSonuçlar kaydedildi:\n  - {json_path}\n  - {csv_path}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
