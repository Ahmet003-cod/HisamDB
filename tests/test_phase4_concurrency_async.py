"""
test_phase4_concurrency_async.py
--------------------------------
MicroVectorDB v0.6.0 (Faz 4 — Eşzamanlılık, Paralel Batch, Arka Plan İşçileri ve Async İstemci)
kapsamlı birim, eşzamanlılık ve kaos/stres test paketi:
1. batch_search() paylaşımlı ve sorgu-başına filtrelerle sıralı search() ile %100 eşitlik
2. batch_hybrid_search() paralel yürütme doğruluğu
3. Bloklamayan rebuild_index_async(): inşa sürerken eşzamanlı okuma + yeni ekleme deltası + tombstone koruması
4. compact_async() görev birleştirme (coalescing) ve auto_compact_async=True
5. AsyncMicroVectorDB / AsyncCollection / AsyncTenantCollection + asyncio.gather()
6. 12 iş parçacıklı karma stres testi + soğuk açılışta tembel indeks yarış (lazy-init race) testi
"""

import asyncio
import os
import shutil
import tempfile
import threading
import time
import unittest
import numpy as np

from hisamdb import (
    AsyncMicroVectorDB,
    MicroVectorDB,
    is_hnsw_available,
)


class TestBatchParallelSearch(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.dim = 32
        self.n = 600
        self.vecs = rng.standard_normal((self.n, self.dim)).astype(np.float32)
        self.ids = [f"v{i}" for i in range(self.n)]
        self.docs = [
            f"Belge {i}: {'istanbul' if i % 3 == 0 else 'ankara'} sunucu kodu ERR-{100 + (i % 50)}"
            for i in range(self.n)
        ]
        self.metas = [{"grp": i % 5, "score": i % 100} for i in range(self.n)]
        self.queries = rng.standard_normal((24, self.dim)).astype(np.float32)

    def test_batch_search_exact_parity_shared_and_per_query_filters(self):
        index_types = ["flat", "ivf", "ivf_pq"]
        if is_hnsw_available():
            index_types.append("hnsw")

        for idx_t in index_types:
            with MicroVectorDB(max_workers=4) as db:
                params = {"n_clusters": 16, "nprobe": 4} if idx_t in ("ivf", "ivf_pq") else {}
                if idx_t == "ivf_pq":
                    params.update({"pq_subvectors": 8, "pq_centroids": 32})
                coll = db.create_collection(
                    f"c_{idx_t}", dimension=self.dim, metric="cosine", index_type=idx_t, index_params=params
                )
                coll.add(ids=self.ids, vectors=self.vecs, documents=self.docs, metadata=self.metas)

                # 1. Paylaşımlı filtre ile paralel batch_search vs sıralı search
                shared_flt = {"grp": 2}
                seq = [coll.search(self.queries[i], limit=5, filter=shared_flt) for i in range(len(self.queries))]
                par = coll.batch_search(self.queries, limit=5, filter=shared_flt, max_workers=4)
                self.assertEqual(len(par), len(seq))
                for i in range(len(seq)):
                    self.assertEqual([r.id for r in par[i]], [r.id for r in seq[i]], f"Mismatch in {idx_t} q={i}")

                # 2. Sorgu başına farklı filtrelerle paralel batch_search
                per_filters = [{"grp": i % 5} for i in range(len(self.queries))]
                seq_pf = [coll.search(self.queries[i], limit=5, filter=per_filters[i]) for i in range(len(self.queries))]
                par_pf = coll.batch_search(self.queries, limit=5, filters=per_filters, max_workers=4)
                for i in range(len(seq_pf)):
                    self.assertEqual([r.id for r in par_pf[i]], [r.id for r in seq_pf[i]])

    def test_batch_hybrid_search_parallel_parity(self):
        with MicroVectorDB(max_workers=4) as db:
            idx_t = "hnsw" if is_hnsw_available() else "flat"
            coll = db.create_collection("hyb_par", dimension=self.dim, metric="cosine", index_type=idx_t)
            coll.add(ids=self.ids, vectors=self.vecs, documents=self.docs, metadata=self.metas)

            q_specs = [
                {
                    "query_vector": self.queries[i],
                    "query_text": f"istanbul ERR-{100 + (i % 50)}",
                    "filter": {"grp": i % 5},
                    "fusion": "rrf" if i % 2 == 0 else "weighted",
                }
                for i in range(len(self.queries))
            ]
            seq = [coll.hybrid_search(limit=5, **spec) for spec in q_specs]
            par = coll.batch_hybrid_search(q_specs, limit=5, max_workers=4)
            self.assertEqual(len(par), len(seq))
            for i in range(len(seq)):
                self.assertEqual([r.id for r in par[i]], [r.id for r in seq[i]])


class TestBackgroundMaintenanceAndStress(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="mvdb_p4_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_nonblocking_rebuild_index_async_preserves_concurrent_adds_and_deletes(self):
        rng = np.random.default_rng(7)
        dim = 16
        vecs = rng.standard_normal((1500, dim)).astype(np.float32)
        ids = [f"v{i}" for i in range(1500)]

        for idx_t in (["ivf", "hnsw"] if is_hnsw_available() else ["ivf"]):
            db_path = os.path.join(self.temp_dir, f"bg_{idx_t}")
            with MicroVectorDB(db_path) as db:
                params = {"n_clusters": 16, "nprobe": 4} if idx_t == "ivf" else {"M": 16, "ef_construction": 64}
                coll = db.create_collection("bg_c", dimension=dim, metric="cosine", index_type=idx_t, index_params=params)
                coll.add(ids=ids, vectors=vecs)

                # Arka plan yeniden inşasını başlat ve inşa sürerken eşzamanlı okuma + yeni ekleme + silme yap!
                fut = coll.rebuild_index_async()
                read_hits = []
                for qi in range(10):
                    read_hits.append(coll.search(vecs[qi], limit=3))

                # İnşa sırasında gelen yeni vektör deltası + silme
                extra_vec = rng.standard_normal((1, dim)).astype(np.float32)
                coll.add(ids=["new_live"], vectors=extra_vec)
                coll.delete(ids=["v0"])

                swapped = fut.result(timeout=30)
                self.assertTrue(swapped)
                self.assertEqual(coll.count(), 1500)  # 1500 + 1 - 1

                # Yeni eklenen vektör bulunabilmeli, silinen v0 asla dönmemeli!
                res_new = coll.search(extra_vec[0], limit=5)
                self.assertIn("new_live", [r.id for r in res_new])
                res_del = coll.search(vecs[0], limit=10)
                self.assertNotIn("v0", [r.id for r in res_del])
                self.assertEqual(len(read_hits), 10)

    def test_compact_async_coalescing_and_auto_compact_async(self):
        rng = np.random.default_rng(19)
        dim = 8
        vecs = rng.standard_normal((200, dim)).astype(np.float32)
        ids = [f"k{i}" for i in range(200)]

        db_path = os.path.join(self.temp_dir, "auto_bg_cmp")
        with MicroVectorDB(db_path) as db:
            coll = db.create_collection(
                "ac", dimension=dim, metric="cosine",
                index_params={"auto_compact_async": True, "compaction_threshold": 0.25, "auto_compact_min_deleted": 32},
            )
            coll.add(ids=ids, vectors=vecs)

            # 80 kayıt sil -> auto_compact_async arka planda tetiklenmeli!
            deleted = coll.delete(ids=ids[:80])
            self.assertEqual(deleted, 80)
            self.assertEqual(coll.count(), 120)

            # Eşzamanlı ikinci compact_async() çağrısı güvenli şekilde çalışmalı
            f1 = coll.compact_async()
            f2 = coll.compact_async()
            f1.result(timeout=15)
            f2.result(timeout=15)
            coll.wait_for_background(timeout=15)

            st = coll.stats()
            self.assertEqual(st["count"], 120)
            self.assertEqual(st["tombstones"], 0)
            self.assertEqual(st["total_slots"], 120)

        # Yeniden açılışta tam bütünlük doğrulanmalı
        with MicroVectorDB(db_path) as db2:
            c2 = db2.get_collection("ac")
            self.assertEqual(c2.count(), 120)
            self.assertNotIn("k0", c2.get_all_ids())

    def test_12_thread_mixed_stress_and_lazy_init_race(self):
        """
        Soğuk açılış sonrası (tüm yardımcı indeksler ve mmap normları tembel beklerken)
        12 iş parçacığının aynı anda okuma, filtreleme, BM25, hibrit arama, yazma ve
        arka plan sıkıştırma yapmasını hatasız geçmelidir.
        """
        rng = np.random.default_rng(101)
        dim = 16
        vecs = rng.standard_normal((400, dim)).astype(np.float32)
        ids = [f"s{i}" for i in range(400)]
        docs = [f"Stres testi belgesi {i} istanbul modül M{i % 20}" for i in range(400)]
        metas = [{"grp": i % 4, "val": i} for i in range(400)]

        db_path = os.path.join(self.temp_dir, "stress_db")
        with MicroVectorDB(db_path) as db:
            coll = db.create_collection("st", dimension=dim, metric="cosine", use_mmap=True)
            coll.add(ids=ids, vectors=vecs, documents=docs, metadata=metas)

        # Soğuk yeniden aç: MetadataIndex, BM25Index, mmap normları henüz kurulmadı!
        with MicroVectorDB(db_path, max_workers=8) as db:
            coll = db.get_collection("st")
            self.assertEqual(coll.stats()["vector_ram_mb"], 0.0)
            self.assertEqual(coll.stats()["metadata_index"], {"lazy_pending": True})

            barrier = threading.Barrier(12)
            errors = []

            def reader_worker(tid: int):
                try:
                    local_rng = np.random.default_rng(500 + tid)
                    barrier.wait()
                    for step in range(25):
                        q = local_rng.standard_normal(dim).astype(np.float32)
                        if step % 3 == 0:
                            coll.search(q, limit=5, filter={"grp": step % 4})
                        elif step % 3 == 1:
                            coll.sparse_search(query_text=f"istanbul M{step % 20}", limit=5)
                        else:
                            coll.hybrid_search(query_vector=q, query_text="istanbul stres", limit=5, filter={"val": {"$gte": 10}})
                except Exception as exc:
                    errors.append((f"reader-{tid}", exc))

            def writer_worker(tid: int):
                try:
                    local_rng = np.random.default_rng(900 + tid)
                    barrier.wait()
                    for step in range(10):
                        nid = f"w{tid}_{step}"
                        nv = local_rng.standard_normal((1, dim)).astype(np.float32)
                        coll.add(
                            ids=[nid], vectors=nv,
                            documents=[f"Yeni yazıcı {tid} adım {step} istanbul"],
                            metadata=[{"grp": step % 4, "val": 1000 + step}],
                        )
                except Exception as exc:
                    errors.append((f"writer-{tid}", exc))

            def deleter_worker():
                try:
                    barrier.wait()
                    for step in range(15):
                        coll.delete(ids=[f"s{step}"])
                        if step == 10:
                            coll.compact_async()
                        time.sleep(0.002)
                except Exception as exc:
                    errors.append(("deleter", exc))

            def compactor_worker():
                try:
                    barrier.wait()
                    time.sleep(0.01)
                    coll.compact_async().result(timeout=15)
                except Exception as exc:
                    errors.append(("compactor", exc))

            threads = [threading.Thread(target=reader_worker, args=(i,)) for i in range(8)]
            threads += [threading.Thread(target=writer_worker, args=(i,)) for i in range(2)]
            threads.append(threading.Thread(target=deleter_worker))
            threads.append(threading.Thread(target=compactor_worker))

            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

            self.assertEqual(errors, [], f"Eşzamanlı stres hataları: {errors}")
            coll.wait_for_background(timeout=15)
            # 400 başlangıç + 20 yeni yazılan - 15 silinen = 405 kayıt!
            self.assertEqual(coll.count(), 405)


class TestAsyncClient(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="mvdb_async_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_async_client_full_lifecycle_and_gather_fanout(self):
        async def _scenario():
            db_path = os.path.join(self.temp_dir, "async_db")
            snap_path = os.path.join(self.temp_dir, "async_snap")
            restored_path = os.path.join(self.temp_dir, "async_restored")

            rng = np.random.default_rng(77)
            dim = 16
            vecs = rng.standard_normal((120, dim)).astype(np.float32)
            docs = [f"Async doküman {i} istanbul kod A-{i}" for i in range(120)]

            async with AsyncMicroVectorDB(db_path, max_workers=4) as db:
                coll = await db.create_collection("docs", dimension=dim, metric="cosine", index_type="ivf", index_params={"n_clusters": 8, "nprobe": 4})
                ids = await coll.add_documents(
                    documents=docs,
                    embeddings=vecs,
                    doc_ids=[f"d{i // 4}" for i in range(120)],
                )
                self.assertEqual(len(ids), 120)
                self.assertEqual(await coll.count(), 120)

                # 32 eşzamanlı asyncio.gather() sorgusu
                q_vecs = rng.standard_normal((16, dim)).astype(np.float32)
                dense_tasks = [coll.search(q_vecs[i], limit=5) for i in range(16)]
                hybrid_tasks = [
                    coll.hybrid_search(query_vector=q_vecs[i], query_text=f"istanbul A-{i}", limit=5)
                    for i in range(16)
                ]
                all_res = await asyncio.gather(*dense_tasks, *hybrid_tasks)
                self.assertEqual(len(all_res), 32)
                self.assertTrue(all(len(r) == 5 for r in all_res))

                # Async batch_search & batch_hybrid_search
                b_res = await coll.batch_search(q_vecs, limit=3)
                self.assertEqual(len(b_res), 16)
                bh_res = await coll.batch_hybrid_search(
                    [{"query_vector": q_vecs[i], "query_text": "istanbul"} for i in range(8)],
                    limit=3,
                )
                self.assertEqual(len(bh_res), 8)

                # Async arka plan rebuild & delete + compact_async
                rebuilt = await coll.rebuild_index_async()
                self.assertTrue(rebuilt)
                deleted = await coll.delete(ids=ids[:5])
                self.assertEqual(deleted, 5)
                reclaimed = await coll.compact_async()
                self.assertEqual(reclaimed, 5)
                self.assertEqual(await coll.count(), 115)

                # Async Multi-Tenant izolasyonu
                await db.create_collection("saas", dimension=4, metric="cosine")
                ta = await db.get_tenant_collection("saas", "alpha")
                tb = await db.get_tenant_collection("saas", "beta")
                await ta.add(ids=["a1"], vectors=[[1, 0, 0, 0]], documents=["Alpha gizli"])
                await tb.add(ids=["b1"], vectors=[[1, 0, 0, 0]], documents=["Beta gizli"])
                self.assertEqual(await ta.count(), 1)
                self.assertEqual(await tb.delete(ids=["a1"]), 0)

                # Async SHA-256 Snapshot & Restore
                manifest = await db.create_snapshot(snap_path)
                self.assertIn("vectordb.sqlite3", manifest["files"])

            await AsyncMicroVectorDB.restore_snapshot(snap_path, restored_path)
            async with AsyncMicroVectorDB(restored_path) as db_r:
                c_r = await db_r.get_collection("docs")
                self.assertEqual(await c_r.count(), 115)

        asyncio.run(_scenario())


if __name__ == "__main__":
    unittest.main()
