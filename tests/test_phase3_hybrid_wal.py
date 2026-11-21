"""
test_phase3_hybrid_wal.py
-------------------------
MicroVectorDB v0.5.0 (Faz 3 + Sorun #1/#2/#3 Çözümleri) kapsamlı test paketi:
- Tombstone (<1 ms) mantıksal silme, kalıcı reload, otomatik/manuel compaction
- İkili WAL (CRC32, yarım yazma budaması, crash replay) ve SHA-256 Snapshot/Restore
- Türkçe/Unicode uyumlu Okapi BM25 ve SparseVectorIndex kalıcılığı
- Hibrit Arama (RRF, Min-Max, Z-Score, 3-yollu füzyon)
- RAG ilkelleri (chunk_text, add_documents, retrieve_with_context, harici reranker,
  deduplicate, group_by, multi_vector_search, multi-tenant TenantCollection izolasyonu)
- Hızlı sütunsal yükleme & tembel MetadataIndex doğrulaması
"""

import os
import shutil
import tempfile
import unittest
import numpy as np

from microvectordb import (
    DataCorruptionError,
    MicroVectorDB,
    WriteAheadLog,
    chunk_text,
    is_hnsw_available,
    tokenize_text,
)
from microvectordb.storage.wal import OP_DELETE


class TestTombstoneWALAndSnapshots(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="mvdb_wal_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_tombstone_soft_delete_reload_and_compact(self):
        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((200, 16)).astype(np.float32)
        ids = [f"v{i}" for i in range(200)]
        docs = [f"Belge metni numara {i}" for i in range(200)]
        metas = [{"grp": i % 4} for i in range(200)]

        with MicroVectorDB(self.temp_dir) as db:
            coll = db.create_collection("ts_coll", dimension=16, metric="cosine")
            coll.add(ids=ids, vectors=vecs, documents=docs, metadata=metas)
            # 5 kayıt sil (< %25 ve < 64) -> tombstone olmalı, tam dosya yazılmamalı!
            deleted = coll.delete(ids=["v0", "v10", "v50", "v100", "v199"])
            self.assertEqual(deleted, 5)
            self.assertEqual(coll.count(), 195)
            st = coll.stats()
            self.assertEqual(st["tombstones"], 5)
            self.assertEqual(st["total_slots"], 200)
            self.assertGreater(st["wal_bytes"], 0)

            # Silinen ID'ler hiçbir sorguda görünmemeli
            self.assertEqual(coll.get(ids=["v0", "v10", "v5"])["ids"], ["v5"])
            res = coll.search(vecs[0], limit=10)
            self.assertNotIn("v0", [r.id for r in res])
            bm_res = coll.sparse_search(query_text="numara 50", limit=10)
            self.assertNotIn("v50", [r.id for r in bm_res])

        # Yeniden aç: tembel metadata indeksi + tombstone koruması doğrulanmalı
        with MicroVectorDB(self.temp_dir) as db2:
            coll2 = db2.get_collection("ts_coll")
            self.assertEqual(coll2.count(), 195)
            self.assertEqual(coll2.stats()["metadata_index"], {"lazy_pending": True})
            self.assertNotIn("v0", coll2.get_all_ids())
            self.assertNotIn("v100", coll2.get(where={"grp": 0})["ids"])
            self.assertEqual(coll2.stats()["tombstones"], 4)  # sondaki v199 açılışta budandı

            # Manuel compact()
            reclaimed = coll2.compact()
            self.assertEqual(reclaimed, 4)
            st2 = coll2.stats()
            self.assertEqual(st2["tombstones"], 0)
            self.assertEqual(st2["total_slots"], 195)
            self.assertEqual(st2["wal_bytes"], 0)

        # Compact sonrası yeniden aç
        with MicroVectorDB(self.temp_dir) as db3:
            coll3 = db3.get_collection("ts_coll")
            self.assertEqual(coll3.count(), 195)
            self.assertEqual(coll3.stats()["tombstones"], 0)

    def test_auto_compaction_on_heavy_delete(self):
        rng = np.random.default_rng(7)
        vecs = rng.standard_normal((200, 8)).astype(np.float32)
        ids = [f"k{i}" for i in range(200)]

        with MicroVectorDB(self.temp_dir) as db:
            coll = db.create_collection("auto_cmp", dimension=8, metric="cosine")
            coll.add(ids=ids, vectors=vecs)
            # 80 kayıt sil (%40 >= %25 ve >= 64) -> otomatik compact tetiklenmeli!
            deleted = coll.delete(ids=ids[:80])
            self.assertEqual(deleted, 80)
            self.assertEqual(coll.count(), 120)
            self.assertEqual(coll.stats()["tombstones"], 0)
            self.assertEqual(coll.stats()["total_slots"], 120)

    def test_wal_torn_write_recovery_and_delete_replay(self):
        # 1. Doğrudan WAL yarım-yazma budama testi
        wal_file = os.path.join(self.temp_dir, "test.wal")
        wal = WriteAheadLog(wal_file)
        wal.append_add(count=5, start_slot=0, epoch=1)
        wal.append_delete(ids=["a"], slots=[0], epoch=2)
        valid_size = os.path.getsize(wal_file)

        # Kuyruğa yarım/bozuk çerçeve ekle (elektrik kesintisi simülasyonu)
        with open(wal_file, "ab") as f:
            f.write(b"MWAL\x03\x00\x00\x00CORRUPT_TAIL")

        recs, truncated = wal.replay(truncate_corrupt_tail=True)
        self.assertEqual(len(recs), 2)
        self.assertGreater(truncated, 0)
        self.assertEqual(os.path.getsize(wal_file), valid_size)

        # 2. Çökme simülasyonu: WAL'a DELETE yazıldı ama SQLite'tan silinmeden kapandı
        db_dir = os.path.join(self.temp_dir, "replay_db")
        with MicroVectorDB(db_dir) as db:
            coll = db.create_collection("rep", dimension=4)
            coll.add(ids=["x1", "x2", "x3"], vectors=np.eye(4, dtype=np.float32)[:3])
            assert coll._wal is not None
            coll._wal.append(OP_DELETE, {"ids": ["x2"], "slots": [1], "epoch": 99})

        with MicroVectorDB(db_dir) as db2:
            c2 = db2.get_collection("rep")
            self.assertEqual(c2.count(), 2)
            self.assertEqual(set(c2.get_all_ids()), {"x1", "x3"})

    def test_snapshot_backup_restore_and_tamper_detection(self):
        src_dir = os.path.join(self.temp_dir, "live_db")
        snap_dir = os.path.join(self.temp_dir, "snap_1")
        restored_dir = os.path.join(self.temp_dir, "restored_db")

        with MicroVectorDB(src_dir) as db:
            coll = db.create_collection("docs", dimension=4, metric="cosine")
            coll.add(
                ids=["d1", "d2"],
                vectors=[[1, 0, 0, 0], [0, 1, 0, 0]],
                documents=["Alfa", "Beta"],
                sparse_vectors=[{"alfa": 2.0}, {"beta": 3.0}],
            )
            manifest = db.create_snapshot(snap_dir)
            self.assertIn("vectordb.sqlite3", manifest["files"])

        # Sağlam geri yükleme
        MicroVectorDB.restore_snapshot(snap_dir, restored_dir)
        with MicroVectorDB(restored_dir) as db_r:
            c_r = db_r.get_collection("docs")
            self.assertEqual(c_r.count(), 2)
            self.assertEqual(c_r.sparse_search(query_sparse={"beta": 1.0}, limit=1)[0].id, "d2")

        # Snapshot dosyasını boz -> SHA-256 doğrulaması DataCorruptionError fırlatmalı
        vec_files = [f for f in os.listdir(snap_dir) if f.endswith(".mvdbvec")]
        vpath = os.path.join(snap_dir, vec_files[0])
        with open(vpath, "r+b") as f:
            f.seek(0)
            b = f.read(1)
            f.seek(0)
            f.write(bytes([b[0] ^ 0xFF]))

        with self.assertRaises(DataCorruptionError):
            MicroVectorDB.restore_snapshot(snap_dir, os.path.join(self.temp_dir, "bad_restore"))

    def test_columnar_cache_corruption_falls_back_to_sqlite_and_preserves_mmap(self):
        db_dir = os.path.join(self.temp_dir, "col_cache_db")
        rng = np.random.default_rng(99)
        vecs = rng.standard_normal((50, 8)).astype(np.float32)
        ids = [f"c{i}" for i in range(50)]
        metas = [{"tag": f"t{i % 5}", "n": i} for i in range(50)]

        with MicroVectorDB(db_dir) as db:
            coll = db.create_collection("cc", dimension=8, metric="cosine", use_mmap=True)
            coll.add(ids=ids, vectors=vecs, metadata=metas)

        # Temiz açılış: .items.mvdbcol oluşturulmuş ve use_mmap korunmuş olmalı
        col_files = [f for f in os.listdir(db_dir) if f.endswith(".items.mvdbcol")]
        self.assertEqual(len(col_files), 1)
        with MicroVectorDB(db_dir) as db2:
            c2 = db2.get_collection("cc")
            self.assertEqual(c2.count(), 50)
            self.assertTrue(c2.stats()["is_mmap"])
            self.assertEqual(c2.stats()["vector_ram_mb"], 0.0)

        # .items.mvdbcol dosyasını boz -> CRC32 uyumsuzluğunda SQLite'tan sessizce ve eksiksiz yüklenmeli!
        cpath = os.path.join(db_dir, col_files[0])
        with open(cpath, "wb") as f:
            f.write(b"MCOL_CORRUPTED_GARBAGE_DATA")

        with MicroVectorDB(db_dir) as db3:
            c3 = db3.get_collection("cc")
            self.assertEqual(c3.count(), 50)
            self.assertEqual(c3.get(ids=["c12"])["metadatas"][0]["n"], 12)


class TestSparseHybridAndRAG(unittest.TestCase):
    def setUp(self):
        self.db = MicroVectorDB()
        self.coll = self.db.create_collection("kb", dimension=4, metric="cosine")
        self.coll.add(
            ids=["d1", "d2", "d3", "d4"],
            vectors=[
                [1.0, 0.0, 0.0, 0.0],  # Dense olarak sorguya en yakın
                [0.4, 0.9, 0.0, 0.0],  # Dense orta ama exact kod ERR-9042 içeriyor!
                [0.98, 0.05, 0.0, 0.0],  # d1 ile neredeyse özdeş vektör
                [0.0, 0.0, 1.0, 0.0],
            ],
            documents=[
                "Yapay zeka ve vektör veritabanı mimarisi İstanbul",
                "Sunucu bağlantı hatası ERR-9042 için yeniden başlatma prosedürü",
                "Yapay zeka ve vektör veritabanı mimarisi İstanbul",  # Birebir metin kopyası
                "Python ile Iğdır hava durumu ve güneş enerjisi raporu",
            ],
            metadata=[
                {"category": "ai", "_doc_id": "pdf_1", "_chunk_index": 0},
                {"category": "ops", "_doc_id": "pdf_2", "_chunk_index": 0},
                {"category": "ai", "_doc_id": "pdf_1", "_chunk_index": 1},
                {"category": "weather", "_doc_id": "pdf_3", "_chunk_index": 0},
            ],
            sparse_vectors=[
                {"ai": 1.5, "vector": 2.0},
                {"err9042": 5.0, "ops": 1.2},
                {"ai": 1.4, "vector": 1.9},
                {"weather": 3.0},
            ],
        )

    def test_turkish_tokenizer_and_bm25_search(self):
        toks = tokenize_text("İstanbul'da IĞDIR ve Şeker-Üzüm!")
        self.assertIn("istanbul", toks)
        self.assertIn("ığdır", toks)
        self.assertIn("şeker", toks)

        res = self.coll.sparse_search(query_text="istanbul mimarisi", limit=2)
        self.assertEqual(set(r.id for r in res), {"d1", "d3"})

        res_code = self.coll.sparse_search(query_text="ERR-9042", limit=1)
        self.assertEqual(res_code[0].id, "d2")

        res_sp = self.coll.sparse_search(query_sparse={"err9042": 1.0}, limit=1)
        self.assertEqual(res_sp[0].id, "d2")

    def test_hybrid_search_rrf_weighted_and_zscore(self):
        # Sorgu vektörü d1'e çok yakın ama metinde tam olarak "ERR-9042" arıyor
        q_vec = [0.7, 0.7, 0.0, 0.0]

        r_rrf = self.coll.hybrid_search(
            query_vector=q_vec,
            query_text="ERR-9042 bağlantı hatası",
            limit=3,
            alpha=0.4,
            beta=0.6,
            fusion="rrf",
        )
        self.assertEqual(r_rrf[0].id, "d2")

        r_weighted = self.coll.hybrid_search(
            query_vector=q_vec,
            query_text="ERR-9042",
            query_sparse={"err9042": 2.0},
            limit=3,
            alpha=0.2,
            beta=0.8,
            fusion="weighted",
        )
        self.assertEqual(r_weighted[0].id, "d2")

        r_z = self.coll.hybrid_search(
            query_vector=q_vec,
            query_text="ERR-9042",
            limit=3,
            fusion="z_score",
        )
        self.assertEqual(len(r_z), 3)

    def test_rag_deduplication_grouping_and_external_reranker(self):
        # Tekilleştirme: d1 ve d3 aynı metne sahip -> yalnızca biri kalmalı
        deduped = self.coll.hybrid_search(
            query_vector=[1.0, 0.0, 0.0, 0.0],
            query_text="yapay zeka",
            limit=4,
            deduplicate=True,
        )
        ids_d = [r.id for r in deduped]
        self.assertFalse("d1" in ids_d and "d3" in ids_d)

        # Doküman gruplama: pdf_1'den en fazla 1 parça gelmeli
        grouped = self.coll.hybrid_search(
            query_vector=[1.0, 0.0, 0.0, 0.0],
            limit=4,
            group_by="_doc_id",
            max_per_group=1,
        )
        pdf1_hits = [r for r in grouped if r.metadata and r.metadata.get("_doc_id") == "pdf_1"]
        self.assertEqual(len(pdf1_hits), 1)

        # Harici reranker nesnesi (.rerank)
        class DummyReranker:
            def rerank(self, query, documents):
                return [10.0 if "ERR-9042" in d else 0.1 for d in documents]

        reranked = self.coll.hybrid_search(
            query_vector=[1.0, 0.0, 0.0, 0.0],
            query_text="hata kodu",
            limit=2,
            reranker=DummyReranker(),
        )
        self.assertEqual(reranked[0].id, "d2")
        self.assertAlmostEqual(reranked[0].score, 10.0)

    def test_chunk_text_add_documents_and_parent_child_context(self):
        long_doc = "Bölüm 1 giriş metni. " * 10 + "Bölüm 2 kritik bilgi. " * 10 + "Bölüm 3 sonuç. " * 10
        chunks = chunk_text(long_doc, chunk_size=120, chunk_overlap=20)
        self.assertGreaterEqual(len(chunks), 3)

        db = MicroVectorDB()
        coll = db.create_collection("rag_ctx", dimension=3, metric="cosine")
        embs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ids = coll.add_documents(
            documents=["Parça 0: Başlangıç", "Parça 1: Çekirdek cevap", "Parça 2: Ek detay"],
            embeddings=embs,
            doc_ids=["manual_1", "manual_1", "manual_1"],
            parent_ids=["p_manual", "p_manual", "p_manual"],
        )
        self.assertEqual(len(ids), 3)

        # Ortadaki Parça 1'i bul, window=1 ile Parça 0 + 1 + 2 bağlamını otomatik birleştir!
        ctx_hits = coll.retrieve_with_context(query_vector=[0.0, 1.0, 0.0], limit=1, window=1)
        self.assertEqual(len(ctx_hits), 1)
        self.assertEqual(len(ctx_hits[0]["neighbors"]), 3)
        self.assertIn("Parça 0: Başlangıç", ctx_hits[0]["context"])
        self.assertIn("Parça 1: Çekirdek cevap", ctx_hits[0]["context"])
        self.assertIn("Parça 2: Ek detay", ctx_hits[0]["context"])

    def test_multi_vector_search(self):
        res_rrf = self.coll.multi_vector_search(
            query_vectors=[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            limit=3,
            aggregation="rrf",
        )
        self.assertTrue({"d1", "d4"}.issubset({r.id for r in res_rrf}))

        res_max = self.coll.multi_vector_search(
            query_vectors=[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            limit=2,
            aggregation="max",
        )
        self.assertEqual(len(res_max), 2)

    def test_multi_tenant_isolation(self):
        db = MicroVectorDB()
        db.create_collection("shared_saas", dimension=3, metric="cosine")
        t_a = db.get_tenant_collection("shared_saas", "tenant_a")
        t_b = db.get_tenant_collection("shared_saas", "tenant_b")

        t_a.add(ids=["a1", "a2"], vectors=[[1, 0, 0], [0.9, 0.1, 0]], documents=["A gizli veri", "A rapor"])
        t_b.add(ids=["b1"], vectors=[[1, 0, 0]], documents=["B gizli veri"])

        self.assertEqual(t_a.count(), 2)
        self.assertEqual(t_b.count(), 1)
        self.assertEqual([r.id for r in t_a.search([1, 0, 0], limit=5)], ["a1", "a2"])
        self.assertEqual([r.id for r in t_b.search([1, 0, 0], limit=5)], ["b1"])
        self.assertEqual([r.id for r in t_b.hybrid_search(query_vector=[1, 0, 0], query_text="gizli veri", limit=5)], ["b1"])

        # Tenant B, Tenant A'nın ID'sini silmeye çalışırsa 0 silinmeli!
        self.assertEqual(t_b.delete(ids=["a1"]), 0)
        self.assertEqual(t_a.count(), 2)
        self.assertEqual(t_a.delete(ids=["a1"]), 1)
        self.assertEqual(t_a.count(), 1)

    @unittest.skipUnless(is_hnsw_available(), "hnswlib kurulu değil")
    def test_hnsw_tombstone_delete(self):
        rng = np.random.default_rng(11)
        vecs = rng.standard_normal((100, 16)).astype(np.float32)
        ids = [f"h{i}" for i in range(100)]
        db = MicroVectorDB()
        coll = db.create_collection("h_ts", dimension=16, metric="cosine", index_type="hnsw")
        coll.add(ids=ids, vectors=vecs)
        coll.delete(ids=["h0"])
        self.assertEqual(coll.count(), 99)
        res = coll.search(vecs[0], limit=5)
        self.assertNotIn("h0", [r.id for r in res])


if __name__ == "__main__":
    unittest.main()
