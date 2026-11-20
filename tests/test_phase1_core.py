"""
test_phase1_core.py
-------------------
MicroVectorDB v0.3.0 (Faz 1) kapsamlı test paketi:
- Güvenlik (Path traversal, tablo adı çakışması)
- Veri bütünlüğü, CRC32 bozulma tespiti, çökme kurtarma, rollback, v0.2 .npy göçü
- Sınır değerler (boş koleksiyon, NaN/Inf, tekrarlanan ID, boyut uyuşmazlığı, 0/negatif limit)
- Ters metadata indeksi ve gelişmiş filtreler ($and, $or, $not, $exists, $contains, nested, karma tip)
- IVF (kth sınır düzeltmesi, tüm metrikler), HNSW, Float16, INT8 (taşmasız), Binary kuantizasyon
- Memory-mapped (use_mmap=True) okuma/yazma/kapatma
- Çok iş parçacıklı eşzamanlılık (RWLock)
- Yeni ergonomik API (MicroVectorDB, search, score_threshold, stats, context manager)
"""

import os
import shutil
import sqlite3
import tempfile
import threading
import unittest
import numpy as np

import microvectordb
from microvectordb import (
    BinaryQuantizer,
    DataCorruptionError,
    DimensionMismatchError,
    EphemeralClient,
    Float16Quantizer,
    HNSWIndex,
    IVFIndex,
    InvalidCollectionNameError,
    InvalidMetricError,
    InvalidVectorError,
    MicroVectorDB,
    PersistentClient,
    ScalarQuantizer,
    VectorDBError,
    is_hnsw_available,
)


class TestSecurityAndValidation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="mvdb_sec_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_table_name_collision_fixed(self):
        """'my-docs' ve 'my_docs' farklı tablolarda tamamen izole olmalıdır."""
        with PersistentClient(path=self.temp_dir) as client:
            c1 = client.create_collection("my-docs", dimension=3)
            c2 = client.create_collection("my_docs", dimension=3)

            c1.add(ids=["a"], embeddings=[[1.0, 0.0, 0.0]], documents=["from dash"])
            c2.add(ids=["b"], embeddings=[[0.0, 1.0, 0.0]], documents=["from underscore"])

            self.assertEqual(c1.count(), 1)
            self.assertEqual(c2.count(), 1)
            self.assertEqual(c1.get()["ids"], ["a"])
            self.assertEqual(c2.get()["ids"], ["b"])

            # Birini silmek diğerine dokunmamalı
            client.delete_collection("my-docs")
            self.assertEqual(c2.count(), 1)
            self.assertEqual(c2.get()["ids"], ["b"])

        # Yeniden açıldığında c2 sağlam olmalı
        with PersistentClient(path=self.temp_dir) as client2:
            self.assertEqual(client2.list_collections(), ["my_docs"])
            c2_reloaded = client2.get_collection("my_docs")
            self.assertEqual(c2_reloaded.count(), 1)
            self.assertEqual(c2_reloaded.get()["documents"], ["from underscore"])

    def test_path_traversal_safe_filename(self):
        """'../escape' gibi isimler persist_dir dışına dosya yazamamalıdır."""
        with PersistentClient(path=self.temp_dir) as client:
            coll = client.create_collection("../escape_attempt", dimension=2)
            coll.add(ids=["1"], embeddings=[[1.0, 0.0]])
            # Üst dizinde hiçbir dosya oluşmamış olmalı
            parent_dir = os.path.dirname(os.path.abspath(self.temp_dir))
            self.assertFalse(os.path.exists(os.path.join(parent_dir, "escape_attempt_vectors.npy")))
            # Tüm dosyalar self.temp_dir içinde olmalı
            files = os.listdir(self.temp_dir)
            self.assertTrue(any(f.endswith(".mvdbvec") for f in files))

    def test_invalid_collection_names_rejected(self):
        client = EphemeralClient()
        for bad in ["", "   ", "bad\x00name", "bad\nname", "x" * 200]:
            with self.assertRaises(InvalidCollectionNameError):
                client.create_collection(bad)

    def test_invalid_metric_rejected_at_creation(self):
        client = EphemeralClient()
        with self.assertRaises(InvalidMetricError):
            client.create_collection("bad_metric", dimension=3, metric="manhattan")  # type: ignore

    def test_nan_and_inf_vectors_rejected(self):
        client = EphemeralClient()
        coll = client.create_collection("finite_test", dimension=3)

        with self.assertRaises(InvalidVectorError):
            coll.add(ids=["a"], embeddings=[[1.0, float("nan"), 0.0]])

        with self.assertRaises(InvalidVectorError):
            coll.add(ids=["b"], embeddings=[[1.0, float("inf"), 0.0]])

        with self.assertRaises(InvalidVectorError):
            coll.add(ids=["c"], embeddings=[[1.0, float("-inf"), 0.0]])

        self.assertEqual(coll.count(), 0)

        coll.add(ids=["ok"], embeddings=[[1.0, 0.0, 0.0]])
        with self.assertRaises(InvalidVectorError):
            coll.query(query_embeddings=[[float("nan"), 0.0, 0.0]])

    def test_duplicate_ids_within_same_add_batch_rejected(self):
        client = EphemeralClient()
        coll = client.create_collection("dup_batch", dimension=2)
        with self.assertRaises(VectorDBError):
            coll.add(ids=["x", "x"], embeddings=[[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(coll.count(), 0)

    def test_edge_cases_empty_and_dimension_mismatch(self):
        client = EphemeralClient()
        coll = client.create_collection("edges", dimension=3)

        # Boş add no-op olmalı
        coll.add(ids=[], embeddings=[])
        self.assertEqual(coll.count(), 0)

        # Boş koleksiyonda sorgu
        res_empty = coll.query(query_embeddings=[[1.0, 0.0, 0.0]], n_results=5)
        self.assertEqual(res_empty.ids, [[]])

        # Tek vektör + sıfır vektör
        coll.add(ids=["v1", "zero"], embeddings=[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        res = coll.query(query_embeddings=[[1.0, 0.0, 0.0]], n_results=2)
        self.assertEqual(res.ids[0][0], "v1")
        self.assertAlmostEqual(res.similarities[0][1], 0.0, places=5)

        # n_results <= 0
        self.assertEqual(coll.query(query_embeddings=[[1.0, 0.0, 0.0]], n_results=0).ids, [[]])
        self.assertEqual(coll.query(query_embeddings=[[1.0, 0.0, 0.0]], n_results=-3).ids, [[]])

        # Yanlış sorgu boyutu
        with self.assertRaises(DimensionMismatchError):
            coll.query(query_embeddings=[[1.0, 0.0]], n_results=1)


class TestPersistenceIntegrityAndRecovery(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="mvdb_persist_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_incremental_append_upsert_delete_and_reload(self):
        rng = np.random.default_rng(7)
        with PersistentClient(path=self.temp_dir) as client:
            coll = client.create_collection("inc", dimension=16, metric="cosine")
            for b in range(5):
                vecs = rng.standard_normal((50, 16)).astype(np.float32)
                ids = [f"b{b}_{i}" for i in range(50)]
                metas = [{"batch": b, "idx": i} for i in range(50)]
                coll.add(ids=ids, embeddings=vecs, metadatas=metas)
            self.assertEqual(coll.count(), 250)

            # Upsert mevcut ve yeni
            new_v = np.ones(16, dtype=np.float32)
            coll.upsert(
                ids=["b0_0", "brand_new"],
                embeddings=[new_v, new_v * 2.0],
                documents=["updated", "fresh"],
                metadatas=[{"batch": 99}, {"batch": 100}],
            )
            self.assertEqual(coll.count(), 251)

            # Silme
            deleted = coll.delete(where={"batch": 1})
            self.assertEqual(deleted, 50)
            self.assertEqual(coll.count(), 201)
            coll.checkpoint()

        # Yeniden aç ve tüm veriyi doğrula
        with PersistentClient(path=self.temp_dir) as client2:
            coll2 = client2.get_collection("inc")
            self.assertEqual(coll2.count(), 201)
            got = coll2.get(ids=["b0_0", "brand_new"])
            self.assertEqual(got["documents"], ["updated", "fresh"])
            self.assertEqual(len(coll2.get(where={"batch": 1})["ids"]), 0)

    def test_crc32_corruption_detection(self):
        with PersistentClient(path=self.temp_dir) as client:
            coll = client.create_collection("crc_coll", dimension=4)
            coll.add(ids=["a", "b"], embeddings=[[1, 0, 0, 0], [0, 1, 0, 0]])
            coll.checkpoint()  # Data CRC32 yazılır

        # Diskteki .mvdbvec dosyasının veri baytını boz
        vec_files = [f for f in os.listdir(self.temp_dir) if f.endswith(".mvdbvec")]
        self.assertEqual(len(vec_files), 1)
        fpath = os.path.join(self.temp_dir, vec_files[0])
        with open(fpath, "r+b") as f:
            f.seek(64)  # İlk veri baytı
            b = f.read(1)
            f.seek(64)
            f.write(bytes([b[0] ^ 0xFF]))

        with self.assertRaises(DataCorruptionError):
            PersistentClient(path=self.temp_dir)

    def test_crash_recovery_truncates_uncommitted_vectors(self):
        """Vektör dosyasında SQLite'tan fazla satır varsa (çökme simülasyonu) güvenle kurtarılmalı."""
        with PersistentClient(path=self.temp_dir) as client:
            coll = client.create_collection("crash_coll", dimension=4)
            coll.add(ids=["a", "b"], embeddings=[[1, 0, 0, 0], [0, 1, 0, 0]])
            # Çökme simülasyonu: SQLite'a yazmadan vektör dosyasına fazladan 2 satır ekle
            extra = np.array([[0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
            client._storage.append_vectors_delta("crash_coll", extra, expected_prev_count=2)

        with PersistentClient(path=self.temp_dir) as client2:
            coll2 = client2.get_collection("crash_coll")
            self.assertEqual(coll2.count(), 2)
            self.assertEqual(coll2.get()["ids"], ["a", "b"])
            res = coll2.query(query_embeddings=[[1, 0, 0, 0]], n_results=1)
            self.assertEqual(res.ids[0][0], "a")

    def test_legacy_v02_npy_migration(self):
        """Eski v0.2.0 SQLite + .npy veritabanı sorunsuz açılıp göç ettirilmelidir."""
        db_path = os.path.join(self.temp_dir, "vectordb.sqlite3")
        conn = sqlite3.connect(db_path)
        with conn:
            conn.execute(
                "CREATE TABLE collections (name TEXT PRIMARY KEY, dimension INTEGER, metric TEXT NOT NULL, created_at REAL NOT NULL);"
            )
            conn.execute(
                "INSERT INTO collections VALUES ('legacy_docs', 3, 'cosine', 1700000000.0);"
            )
            conn.execute(
                "CREATE TABLE items_legacy_docs (id TEXT PRIMARY KEY, int_id INTEGER UNIQUE, document TEXT, metadata TEXT);"
            )
            conn.execute(
                """INSERT INTO items_legacy_docs VALUES ('old1', 0, 'Eski belge', '{"v": 2}');"""
            )
        conn.close()
        np.save(
            os.path.join(self.temp_dir, "legacy_docs_vectors.npy"),
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        )

        with PersistentClient(path=self.temp_dir) as client:
            coll = client.get_collection("legacy_docs")
            self.assertEqual(coll.count(), 1)
            res = coll.query(query_embeddings=[[1.0, 0.0, 0.0]], n_results=1)
            self.assertEqual(res.ids[0][0], "old1")
            self.assertEqual(res.metadatas[0][0]["v"], 2)

    def test_mmap_load_query_mutate_and_clean_close(self):
        with PersistentClient(path=self.temp_dir) as client:
            coll = client.create_collection("mmap_coll", dimension=8, metric="cosine")
            vecs = np.eye(8, dtype=np.float32)
            coll.add(ids=[f"id{i}" for i in range(8)], embeddings=vecs)

        with PersistentClient(path=self.temp_dir, use_mmap=True) as client2:
            coll2 = client2.get_collection("mmap_coll")
            self.assertTrue(coll2.stats()["is_mmap"])
            res = coll2.search(query_vector=np.eye(8, dtype=np.float32)[3], limit=1)
            self.assertEqual(res[0].id, "id3")
            # Mmap koleksiyonuna yazma -> sorunsuz writable tampona geçiş
            coll2.add(ids=["id8"], embeddings=[[1.0] * 8])
            self.assertEqual(coll2.count(), 9)
            self.assertFalse(coll2.stats()["is_mmap"])

    def test_large_chunked_delete(self):
        """1,200 ID silme (SQLite 500'lü chunking doğrulaması)."""
        with PersistentClient(path=self.temp_dir) as client:
            coll = client.create_collection("big_del", dimension=4)
            n = 1200
            coll.add(
                ids=[f"k{i}" for i in range(n)],
                embeddings=np.ones((n, 4), dtype=np.float32),
            )
            deleted = coll.delete(ids=[f"k{i}" for i in range(1100)])
            self.assertEqual(deleted, 1100)
            self.assertEqual(coll.count(), 100)

    def test_rollback_on_disk_failure(self):
        """Disk yazımı başarısız olduğunda bellek durumu tamamen geri alınmalıdır."""
        with PersistentClient(path=self.temp_dir) as client:
            coll = client.create_collection("rb_coll", dimension=3)
            coll.add(ids=["ok1"], embeddings=[[1.0, 0.0, 0.0]], documents=["first"])
            self.assertEqual(coll.count(), 1)

            orig_save = client._storage.save_items_batch

            def failing_save(*args, **kwargs):
                raise IOError("Simüle edilmiş disk dolu hatası")

            client._storage.save_items_batch = failing_save  # type: ignore[method-assign]
            try:
                with self.assertRaises(IOError):
                    coll.add(ids=["bad1", "bad2"], embeddings=[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            finally:
                client._storage.save_items_batch = orig_save  # type: ignore[method-assign]

            # RAM ve disk önceki tutarlı duruma dönmüş olmalı
            self.assertEqual(coll.count(), 1)
            self.assertEqual(coll.get()["ids"], ["ok1"])
            self.assertNotIn("bad1", coll._id_to_idx)

        with PersistentClient(path=self.temp_dir) as client2:
            c2 = client2.get_collection("rb_coll")
            self.assertEqual(c2.count(), 1)
            self.assertEqual(c2.get()["ids"], ["ok1"])


class TestAdvancedFiltering(unittest.TestCase):
    def setUp(self):
        self.client = EphemeralClient()
        self.coll = self.client.create_collection("filt", dimension=3, metric="cosine")
        self.coll.add(
            ids=["d1", "d2", "d3", "d4", "d5"],
            embeddings=[
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.8, 0.2, 0.0],
                [0.7, 0.3, 0.0],
                [0.6, 0.4, 0.0],
            ],
            metadatas=[
                {"cat": "ai", "year": 2024, "active": True, "score": 1, "tags": ["rag", "llm"], "author": {"country": "TR"}},
                {"cat": "ai", "year": 2022, "active": False, "score": 0, "tags": ["cv"], "author": {"country": "US"}},
                {"cat": "db", "year": 2025, "active": True, "score": "high", "tags": ["vector", "rag"], "author": {"country": "TR"}},
                {"cat": "web", "year": 2024, "active": True, "tags": ["react"]},
                None,
            ],
        )

    def test_logical_and_or_not(self):
        r_or = self.coll.query(
            query_embeddings=[[1.0, 0.0, 0.0]],
            n_results=10,
            where={"$or": [{"cat": "db"}, {"year": {"$lte": 2022}}]},
        )
        self.assertEqual(set(r_or.ids[0]), {"d2", "d3"})

        r_and = self.coll.query(
            query_embeddings=[[1.0, 0.0, 0.0]],
            n_results=10,
            where={"$and": [{"cat": "ai"}, {"year": {"$gte": 2024}}]},
        )
        self.assertEqual(r_and.ids[0], ["d1"])

        r_not = self.coll.query(
            query_embeddings=[[1.0, 0.0, 0.0]],
            n_results=10,
            where={"$not": {"cat": "ai"}},
        )
        self.assertEqual(set(r_not.ids[0]), {"d3", "d4"})

    def test_nested_dot_path_and_dict_filter(self):
        r_dot = self.coll.get(where={"author.country": "TR"})
        self.assertEqual(set(r_dot["ids"]), {"d1", "d3"})

        r_nested = self.coll.get(where={"author": {"country": "US"}})
        self.assertEqual(r_nested["ids"], ["d2"])

    def test_contains_exists_and_mixed_type_safety(self):
        # $contains liste
        r_tag = self.coll.get(where={"tags": {"$contains": "rag"}})
        self.assertEqual(set(r_tag["ids"]), {"d1", "d3"})

        # $exists
        r_ex = self.coll.get(where={"author.country": {"$exists": True}})
        self.assertEqual(set(r_ex["ids"]), {"d1", "d2", "d3"})
        r_no_ex = self.coll.get(where={"author.country": {"$exists": False}})
        self.assertEqual(set(r_no_ex["ids"]), {"d4", "d5"})

        # Karma tip güvenliği: d3'te score="high" (str) varken $gte: 1 çökmemeli!
        r_num = self.coll.get(where={"score": {"$gte": 1}})
        self.assertEqual(r_num["ids"], ["d1"])

        # bool vs int ayrımı: active=True, score=1 ile karışmamalı
        r_bool = self.coll.get(where={"score": True})
        self.assertEqual(r_bool["ids"], [])


class TestIndexesAndQuantization(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        # 16 kümeli gerçekçi embedding dağılımı
        centers = rng.standard_normal((16, 64)).astype(np.float32) * 3.0
        labels = rng.integers(0, 16, size=2000)
        self.vectors = (centers[labels] + rng.standard_normal((2000, 64)).astype(np.float32) * 0.5).astype(np.float32)
        self.queries = (centers[rng.integers(0, 16, size=20)] + rng.standard_normal((20, 64)).astype(np.float32) * 0.5).astype(np.float32)
        self.ids = [f"v{i}" for i in range(2000)]

    def test_ivf_nprobe_equals_n_clusters_does_not_crash(self):
        """v0.2.0'da ValueError fırlatan nprobe >= n_clusters regresyon testi."""
        ivf = IVFIndex(n_clusters=4, nprobe=4, metric="euclidean")
        ivf.train(self.vectors[:20])
        idxs, dists = ivf.search(self.vectors[0], self.vectors[:20], k=5)
        self.assertEqual(len(idxs), 5)
        self.assertEqual(idxs[0], 0)

    def test_ivf_collection_cosine_high_recall(self):
        db = MicroVectorDB()
        flat_c = db.create_collection("flat", dimension=64, metric="cosine", index_type="flat")
        ivf_c = db.create_collection(
            "ivf",
            dimension=64,
            metric="cosine",
            index_type="ivf",
            index_params={"n_clusters": 16, "nprobe": 4},
        )
        flat_c.add(ids=self.ids, vectors=self.vectors)
        ivf_c.add(ids=self.ids, vectors=self.vectors)

        recalls = []
        for q in self.queries:
            gt = {r.id for r in flat_c.search(q, limit=10)}
            got = {r.id for r in ivf_c.search(q, limit=10)}
            recalls.append(len(gt & got) / 10.0)
        self.assertGreaterEqual(float(np.mean(recalls)), 0.90)

    @unittest.skipUnless(is_hnsw_available(), "hnswlib kurulu değil")
    def test_hnsw_collection_all_metrics_and_filter(self):
        for metric in ("cosine", "euclidean", "dot_product"):
            db = MicroVectorDB()
            flat_c = db.create_collection("flat", dimension=64, metric=metric, index_type="flat")
            hnsw_c = db.create_collection(
                "hnsw",
                dimension=64,
                metric=metric,
                index_type="hnsw",
                index_params={"M": 16, "ef_construction": 100, "ef_search": 64},
            )
            metas = [{"grp": i % 4} for i in range(len(self.ids))]
            flat_c.add(ids=self.ids, vectors=self.vectors, metadata=metas)
            hnsw_c.add(ids=self.ids, vectors=self.vectors, metadata=metas)

            recalls = []
            for q in self.queries[:10]:
                gt = {r.id for r in flat_c.search(q, limit=10)}
                got = {r.id for r in hnsw_c.search(q, limit=10)}
                recalls.append(len(gt & got) / 10.0)
            self.assertGreaterEqual(float(np.mean(recalls)), 0.90, f"HNSW {metric} düşük recall")

            # Filtreli HNSW araması
            filt_res = hnsw_c.search(self.queries[0], limit=5, filter={"grp": 2})
            self.assertEqual(len(filt_res), 5)
            self.assertTrue(all(r.metadata["grp"] == 2 for r in filt_res))

    def test_scalar_quantizer_overflow_free_high_recall(self):
        """Taşma düzeltmesi sonrası ScalarQuantizer Recall@10 > 0.95 olmalıdır."""
        sq = ScalarQuantizer().fit(self.vectors)
        enc = sq.encode(self.vectors)
        scaled_sq, _ = sq.precompute_db_norms(enc)

        recalls = []
        for q in self.queries:
            real_d = np.sum((self.vectors - q) ** 2, axis=1)
            gt = set(np.argsort(real_d)[:10])
            qd = sq.compute_distances_quantized(q, enc, metric="euclidean", precomputed_scaled_sq=scaled_sq)
            self.assertTrue(np.all(qd >= 0.0))
            got = set(np.argsort(qd)[:10])
            recalls.append(len(gt & got) / 10.0)
        self.assertGreaterEqual(float(np.mean(recalls)), 0.95)

    def test_collection_quantizations_float16_int8_binary(self):
        db = MicroVectorDB()
        c_f32 = db.create_collection("f32", dimension=64, metric="cosine")
        c_f16 = db.create_collection("f16", dimension=64, metric="cosine", quantization="float16")
        c_i8 = db.create_collection("i8", dimension=64, metric="cosine", quantization="int8")
        c_bin = db.create_collection("bin", dimension=64, metric="cosine", quantization="binary", keep_raw_vectors=True)

        for c in (c_f32, c_f16, c_i8, c_bin):
            c.add(ids=self.ids, vectors=self.vectors)

        # Float16 RAM kullanımı Float32'nin yarısı olmalı
        self.assertLess(c_f16.stats()["vector_ram_mb"], c_f32.stats()["vector_ram_mb"] * 0.65)

        for name, coll, min_rec in [("float16", c_f16, 0.99), ("int8", c_i8, 0.95), ("binary+rescore", c_bin, 0.85)]:
            recalls = []
            for q in self.queries:
                gt = {r.id for r in c_f32.search(q, limit=10)}
                got = {r.id for r in coll.search(q, limit=10)}
                recalls.append(len(gt & got) / 10.0)
            self.assertGreaterEqual(float(np.mean(recalls)), min_rec, f"{name} düşük recall")

    def test_quantization_drop_raw_vectors_ram_savings(self):
        """keep_raw_vectors=False ile INT8 (~4x) ve Binary (~32x) gerçek RAM tasarrufu sağlamalıdır."""
        db = MicroVectorDB()
        c_f32 = db.create_collection("raw_f32", dimension=64, metric="cosine")
        c_i8 = db.create_collection("raw_i8", dimension=64, metric="cosine", quantization="int8", keep_raw_vectors=False)
        c_bin = db.create_collection("raw_bin", dimension=64, metric="cosine", quantization="binary", keep_raw_vectors=False)

        for c in (c_f32, c_i8, c_bin):
            c.add(ids=self.ids, vectors=self.vectors)

        f32_ram = c_f32.stats()["vector_ram_mb"]
        i8_st = c_i8.stats()
        bin_st = c_bin.stats()
        self.assertEqual(i8_st["vector_ram_mb"], 0.0)
        self.assertEqual(bin_st["vector_ram_mb"], 0.0)
        self.assertLess(i8_st["quantized_ram_mb"], f32_ram * 0.40)
        self.assertLess(bin_st["quantized_ram_mb"], f32_ram * 0.06)

        # Arama ve silme keep_raw_vectors=False modunda da çalışmalı
        res = c_i8.search(self.queries[0], limit=5)
        self.assertEqual(len(res), 5)
        deleted = c_i8.delete(ids=["v0", "v1"])
        self.assertEqual(deleted, 2)
        self.assertEqual(c_i8.count(), len(self.ids) - 2)

    def test_score_threshold_and_search_api(self):
        db = MicroVectorDB()
        coll = db.create_collection("rag", dimension=3, metric="cosine")
        coll.add(
            ids=["exact", "near", "far"],
            vectors=[[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.0, 0.0, 1.0]],
            documents=["d1", "d2", "d3"],
        )
        res = coll.search(query_vector=[1.0, 0.0, 0.0], limit=10, score_threshold=0.75)
        self.assertEqual([r.id for r in res], ["exact", "near"])


class TestConcurrencyRWLock(unittest.TestCase):
    def test_concurrent_readers_and_writers(self):
        """6 okuyucu + 2 yazıcı iş parçacığı çakışmadan ve hatasız çalışmalıdır."""
        rng = np.random.default_rng(123)
        db = MicroVectorDB()
        coll = db.create_collection("concurrent", dimension=16, metric="cosine")
        coll.add(
            ids=[f"init_{i}" for i in range(200)],
            vectors=rng.standard_normal((200, 16)).astype(np.float32),
            metadata=[{"t": i % 5} for i in range(200)],
        )

        errors = []
        stop = threading.Event()

        def writer(wid: int):
            try:
                local_rng = np.random.default_rng(1000 + wid)
                for step in range(25):
                    vecs = local_rng.standard_normal((10, 16)).astype(np.float32)
                    ids = [f"w{wid}_{step}_{j}" for j in range(10)]
                    metas = [{"t": (step + j) % 5} for j in range(10)]
                    coll.add(ids=ids, vectors=vecs, metadata=metas)
                    coll.upsert(ids=[ids[0]], vectors=vecs[:1], metadata=[{"t": 0}])
            except Exception as e:
                errors.append(e)

        def reader(rid: int):
            try:
                local_rng = np.random.default_rng(2000 + rid)
                while not stop.is_set():
                    q = local_rng.standard_normal(16).astype(np.float32)
                    res = coll.search(q, limit=5, filter={"t": rid % 5})
                    for item in res:
                        assert item.metadata is not None and item.metadata["t"] == (rid % 5)
            except Exception as e:
                errors.append(e)

        writers = [threading.Thread(target=writer, args=(w,)) for w in range(2)]
        readers = [threading.Thread(target=reader, args=(r,)) for r in range(6)]
        for t in readers + writers:
            t.start()
        for t in writers:
            t.join()
        stop.set()
        for t in readers:
            t.join()

        self.assertEqual(errors, [], f"Eşzamanlılık hatası: {errors}")
        self.assertEqual(coll.count(), 200 + 2 * 25 * 10)


if __name__ == "__main__":
    unittest.main()
