"""
test_phase2_indexing.py
-----------------------
MicroVectorDB v0.4.0 (Faz 2) kapsamlı birim ve entegrasyon testleri:
- Product Quantization (PQ) — tüm metrikler, bölünmeyen boyutlar, CRC32 serileştirme
- IVF-PQ (Residual Product Quantization) — yüksek recall, düşük RAM, disk kalıcılığı
- HNSW / IVF / IVF-PQ / DiskIndex anında diskten yükleme ve CRC32 bozulma geri dönüşü
- Out-of-Core DiskStreamingScanner ve bölümlenmiş MMap DiskIndex
- Maliyet tabanlı QueryOptimizer (`collection.explain()`) ve HNSW post-filter fallback
"""

import os
import shutil
import tempfile
import unittest
import numpy as np

from hisamdb import (
    DataCorruptionError,
    DiskIndex,
    DiskStreamingScanner,
    IVFIndex,
    IVFPQIndex,
    MicroVectorDB,
    ProductQuantizer,
    QueryOptimizer,
    is_hnsw_available,
)


def _make_clustered(n: int = 2000, d: int = 64, n_topics: int = 16, n_q: int = 20, seed: int = 42):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_topics, d)).astype(np.float32) * 3.0
    labels = rng.integers(0, n_topics, size=n)
    vecs = (centers[labels] + rng.standard_normal((n, d)).astype(np.float32) * 0.5).astype(np.float32)
    q_labels = rng.integers(0, n_topics, size=n_q)
    queries = (centers[q_labels] + rng.standard_normal((n_q, d)).astype(np.float32) * 0.5).astype(np.float32)
    return vecs, queries


class TestProductQuantizer(unittest.TestCase):
    def setUp(self):
        self.vecs, self.queries = _make_clustered(n=2000, d=64, n_topics=16, n_q=20)
        self.ids = [f"v{i}" for i in range(len(self.vecs))]

    def test_pq_all_metrics_and_non_divisible_dim(self):
        """PQ hem tam bölünen (64D) hem tam bölünmeyen (50D) boyutlarda yüksek recall vermelidir."""
        for metric in ("cosine", "euclidean", "dot_product"):
            pq = ProductQuantizer(n_subvectors=16, n_centroids=256, metric=metric, n_iter=15).fit(self.vecs)
            codes_t = pq.encode_transposed(self.vecs)
            self.assertEqual(codes_t.shape, (16, len(self.vecs)))
            self.assertEqual(codes_t.dtype, np.uint8)

            # Decode şekil kontrolü
            decoded = pq.decode(codes_t.T)
            self.assertEqual(decoded.shape, self.vecs.shape)

            # Exact referansla Recall@10 kontrolü (8-bit PQ M=16, 16x sıkıştırma)
            db = MicroVectorDB()
            flat = db.create_collection("flat", dimension=64, metric=metric)
            flat.add(ids=self.ids, vectors=self.vecs)
            recalls = []
            for q in self.queries[:10]:
                gt = {int(r.id[1:]) for r in flat.search(q, limit=10)}
                dists, _ = pq.compute_distances_adc(q, codes_t, metric=metric)
                got = set(np.argsort(dists)[:10].tolist())
                recalls.append(len(gt & got) / 10.0)
            self.assertGreaterEqual(float(np.mean(recalls)), 0.50, f"PQ {metric} düşük recall")

        # Tam bölünmeyen boyut (50D, M=8 -> padded_dim=56)
        v50, q50 = _make_clustered(n=500, d=50, n_topics=8, n_q=5)
        pq50 = ProductQuantizer(n_subvectors=8, n_centroids=32, metric="euclidean").fit(v50)
        self.assertEqual(pq50.padded_dim, 56)
        c50 = pq50.encode_transposed(v50)
        d50, _ = pq50.compute_distances_adc(q50[0], c50)
        self.assertEqual(d50.shape, (500,))
        self.assertEqual(pq50.decode(c50.T).shape, (500, 50))

    def test_pq_serialization_and_crc32_corruption(self):
        pq = ProductQuantizer(n_subvectors=8, n_centroids=32, metric="cosine").fit(self.vecs[:400])
        raw = pq.to_bytes()
        restored = ProductQuantizer.from_bytes(raw)
        np.testing.assert_allclose(pq.codebooks, restored.codebooks)

        # Son baytı boz -> DataCorruptionError
        corrupted = bytearray(raw)
        corrupted[-1] ^= 0xFF
        with self.assertRaises(DataCorruptionError):
            ProductQuantizer.from_bytes(bytes(corrupted))

    def test_collection_pq_drop_raw_ram_savings(self):
        db = MicroVectorDB()
        c_f32 = db.create_collection("f32", dimension=64, metric="cosine")
        c_pq = db.create_collection(
            "pq",
            dimension=64,
            metric="cosine",
            quantization="pq",
            keep_raw_vectors=False,
            index_params={"pq_subvectors": 16, "pq_centroids": 64},
        )
        c_f32.add(ids=self.ids, vectors=self.vecs)
        c_pq.add(ids=self.ids, vectors=self.vecs)

        st_pq = c_pq.stats()
        self.assertEqual(st_pq["vector_ram_mb"], 0.0)
        self.assertLess(st_pq["quantized_ram_mb"], c_f32.stats()["vector_ram_mb"] * 0.25)

        res = c_pq.search(self.queries[0], limit=5)
        self.assertEqual(len(res), 5)
        self.assertEqual(c_pq.delete(ids=["v0", "v1"]), 2)
        self.assertEqual(c_pq.count(), len(self.ids) - 2)


class TestIVFPQAndIndexPersistence(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="mvdb_p2_")
        self.vecs, self.queries = _make_clustered(n=2000, d=64, n_topics=16, n_q=15)
        self.ids = [f"v{i}" for i in range(len(self.vecs))]
        self.metas = [{"grp": i % 4, "val": i % 100} for i in range(len(self.vecs))]

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_ivfpq_all_metrics_and_incremental_add(self):
        for metric in ("cosine", "euclidean", "dot_product"):
            idx = IVFPQIndex(n_clusters=16, nprobe=4, n_subvectors=16, n_centroids=256, metric=metric)
            idx.train(self.vecs[:1500], n_iter=12, pq_iter=12)
            idx.add_vectors(self.vecs[1500:], start_idx=1500)
            self.assertEqual(idx.total_items, 2000)

            db = MicroVectorDB()
            flat = db.create_collection("flat", dimension=64, metric=metric)
            flat.add(ids=self.ids, vectors=self.vecs)
            recalls = []
            for q in self.queries[:10]:
                gt = {int(r.id[1:]) for r in flat.search(q, limit=10)}
                got_ids, _ = idx.search(q, k=10)
                recalls.append(len(gt & set(got_ids.tolist())) / 10.0)
            self.assertGreaterEqual(float(np.mean(recalls)), 0.65, f"IVF-PQ {metric} düşük recall")

    def test_ivf_instant_persistence_reload(self):
        with MicroVectorDB(self.temp_dir) as db:
            coll = db.create_collection(
                "ivf_p",
                dimension=64,
                metric="cosine",
                index_type="ivf",
                index_params={"n_clusters": 16, "nprobe": 4},
            )
            coll.add(ids=self.ids[:1500], vectors=self.vecs[:1500], metadata=self.metas[:1500])
            # Artımlı ekleme sonrası close() otomatik _compact_extras + save yapmalı
            coll.add(ids=self.ids[1500:], vectors=self.vecs[1500:], metadata=self.metas[1500:])
            expected = [r.id for r in coll.search(self.queries[0], limit=5)]

        with MicroVectorDB(self.temp_dir) as db2:
            coll2 = db2.get_collection("ivf_p")
            self.assertTrue(coll2.stats()["index_loaded_from_disk"])
            got = [r.id for r in coll2.search(self.queries[0], limit=5)]
            self.assertEqual(got, expected)

    def test_ivfpq_instant_persistence_reload(self):
        with MicroVectorDB(self.temp_dir) as db:
            coll = db.create_collection(
                "ivfpq_p",
                dimension=64,
                metric="cosine",
                index_type="ivf_pq",
                index_params={"n_clusters": 16, "nprobe": 4, "pq_subvectors": 16, "pq_centroids": 64},
            )
            coll.add(ids=self.ids, vectors=self.vecs, metadata=self.metas)
            expected = [r.id for r in coll.search(self.queries[0], limit=5)]

        with MicroVectorDB(self.temp_dir) as db2:
            coll2 = db2.get_collection("ivfpq_p")
            self.assertTrue(coll2.stats()["index_loaded_from_disk"])
            got = [r.id for r in coll2.search(self.queries[0], limit=5)]
            self.assertEqual(got, expected)

    @unittest.skipUnless(is_hnsw_available(), "hnswlib kurulu değil")
    def test_hnsw_instant_persistence_and_corruption_fallback(self):
        with MicroVectorDB(self.temp_dir) as db:
            coll = db.create_collection(
                "hnsw_p",
                dimension=64,
                metric="cosine",
                index_type="hnsw",
                index_params={"M": 16, "ef_construction": 100, "ef_search": 64},
            )
            coll.add(ids=self.ids, vectors=self.vecs, metadata=self.metas)
            expected = [r.id for r in coll.search(self.queries[0], limit=5)]

        # 1. Temiz yeniden yükleme -> diskten anında açılmalı
        with MicroVectorDB(self.temp_dir) as db2:
            coll2 = db2.get_collection("hnsw_p")
            self.assertTrue(coll2.stats()["index_loaded_from_disk"])
            got = [r.id for r in coll2.search(self.queries[0], limit=5)]
            self.assertEqual(got, expected)

        # 2. HNSW dosyasını boz -> CRC32 tespiti ile güvenle yeniden inşa etmeli (çökmeden!)
        hnsw_bins = [f for f in os.listdir(self.temp_dir) if f.endswith(".hnsw.bin")]
        self.assertEqual(len(hnsw_bins), 1)
        bpath = os.path.join(self.temp_dir, hnsw_bins[0])
        with open(bpath, "r+b") as f:
            b = f.read(1)
            f.seek(0)
            f.write(bytes([b[0] ^ 0xFF]))

        with MicroVectorDB(self.temp_dir) as db3:
            coll3 = db3.get_collection("hnsw_p")
            self.assertFalse(coll3.stats()["index_loaded_from_disk"])
            self.assertEqual(coll3.count(), len(self.ids))
            res3 = coll3.search(self.queries[0], limit=5)
            self.assertEqual(len(res3), 5)


class TestDiskIndexAndStreaming(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="mvdb_disk_")
        self.vecs, self.queries = _make_clustered(n=2000, d=64, n_topics=16, n_q=10)
        self.ids = [f"v{i}" for i in range(len(self.vecs))]
        self.metas = [{"grp": i % 5} for i in range(len(self.vecs))]

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_disk_streaming_scanner_matches_exact_flat(self):
        """Sabit-RAM DiskStreamingScanner tüm metriklerde Exact Flat ile %100 aynı sonucu vermelidir."""
        scanner = DiskStreamingScanner(block_rows=256)
        for metric in ("cosine", "euclidean", "dot_product"):
            db = MicroVectorDB()
            coll = db.create_collection("exact", dimension=64, metric=metric)
            coll.add(ids=self.ids, vectors=self.vecs, metadata=self.metas)

            # Filtresiz
            exact_res = coll.query(query_embeddings=self.queries[:3], n_results=10)
            stream_res = coll.query(query_embeddings=self.queries[:3], n_results=10, strategy="disk_stream")
            self.assertEqual(exact_res.ids, stream_res.ids, f"disk_stream {metric} exact ile uyuşmadı")

            # Filtreli
            ex_f = coll.query(query_embeddings=self.queries[:3], n_results=10, where={"grp": 2})
            st_f = coll.query(query_embeddings=self.queries[:3], n_results=10, where={"grp": 2}, strategy="disk_stream")
            self.assertEqual(ex_f.ids, st_f.ids)

            # Doğrudan scanner testi
            idxs, _, _ = scanner.search(self.queries[:2], self.vecs, metric=metric, k=5)
            self.assertEqual(len(idxs), 2)
            self.assertEqual(len(idxs[0]), 5)

    def test_disk_index_partitioned_mmap_persistence_and_low_ram(self):
        with MicroVectorDB(self.temp_dir) as db:
            flat = db.create_collection("flat", dimension=64, metric="cosine")
            disk_c = db.create_collection(
                "disk_c",
                dimension=64,
                metric="cosine",
                index_type="disk",
                index_params={"n_clusters": 16, "nprobe": 4},
            )
            flat.add(ids=self.ids, vectors=self.vecs, metadata=self.metas)
            disk_c.add(ids=self.ids, vectors=self.vecs, metadata=self.metas)

            recalls = []
            for q in self.queries:
                gt = {r.id for r in flat.search(q, limit=10)}
                got = {r.id for r in disk_c.search(q, limit=10)}
                recalls.append(len(gt & got) / 10.0)
            self.assertGreaterEqual(float(np.mean(recalls)), 0.90)

            # Filtreli DiskIndex araması
            f_res = disk_c.search(self.queries[0], limit=5, filter={"grp": 3})
            self.assertEqual(len(f_res), 5)
            self.assertTrue(all(r.metadata["grp"] == 3 for r in f_res))

        # Yeniden aç: mmap olarak diskten anında yüklenmeli ve çok düşük RAM kullanmalı
        with MicroVectorDB(self.temp_dir) as db2:
            d2 = db2.get_collection("disk_c")
            st = d2.stats()
            self.assertTrue(st["index_loaded_from_disk"])
            self.assertTrue(st["is_mmap"])
            # Ham float32 (2000*64*4 = 0.49 MB) RAM'de tutulmamalı
            self.assertLess(st["vector_ram_mb"], 0.05)
            res = d2.search(self.queries[0], limit=5)
            self.assertEqual(len(res), 5)


class TestQueryOptimizer(unittest.TestCase):
    def setUp(self):
        self.vecs, self.queries = _make_clustered(n=6000, d=32, n_topics=16, n_q=5)
        self.ids = [f"v{i}" for i in range(6000)]
        # rare: %1 (60 kayıt), mid: %25 (1500 kayıt), broad: %50 (3000 > 2048 kayıt)
        self.metas = [
            {"rare": (i < 60), "mid": (i % 4 == 0), "broad": (i % 2 == 0), "tail_only": (i >= 5400)}
            for i in range(6000)
        ]

    def test_explain_and_selectivity_routing(self):
        db = MicroVectorDB()
        coll = db.create_collection(
            "opt_ivf",
            dimension=32,
            metric="cosine",
            index_type="ivf",
            index_params={"n_clusters": 32, "nprobe": 4},
        )
        coll.add(ids=self.ids, vectors=self.vecs, metadata=self.metas)

        # 1. Boş eşleşme -> short-circuit empty
        p_empty = coll.explain(n_results=10, where={"rare": "never"})
        self.assertEqual(p_empty.execution_path, "empty")
        self.assertEqual(p_empty.estimated_scan_ops, 0)

        # 2. Düşük seçicilik (%1) -> pre_filter_exact
        p_rare = coll.explain(n_results=10, where={"rare": True})
        self.assertEqual(p_rare.execution_path, "flat_exact")
        self.assertEqual(p_rare.filter_strategy, "pre_filter_exact")

        # 3. Geniş filtre (%50) -> IVF + uyarlamalı nprobe
        p_broad = coll.explain(n_results=10, where={"broad": True})
        self.assertEqual(p_broad.execution_path, "ivf")
        self.assertGreaterEqual(p_broad.adaptive_nprobe, 4)

        # 4. Manuel strateji zorlama
        p_manual = coll.explain(n_results=10, strategy="exact")
        self.assertEqual(p_manual.execution_path, "flat_exact")

    @unittest.skipUnless(is_hnsw_available(), "hnswlib kurulu değil")
    def test_hnsw_post_filter_and_adversarial_fallback(self):
        db = MicroVectorDB()
        coll = db.create_collection(
            "opt_hnsw",
            dimension=32,
            metric="cosine",
            index_type="hnsw",
            index_params={"M": 16, "ef_construction": 100, "ef_search": 64},
        )
        coll.add(ids=self.ids, vectors=self.vecs, metadata=self.metas)

        # %1 seçicilik -> exact pre-filter
        p_rare = coll.explain(n_results=10, where={"rare": True})
        self.assertEqual(p_rare.execution_path, "flat_exact")

        # %50 seçicilik -> hnsw_post_filter (sıfır GIL callback!)
        p_broad = coll.explain(n_results=10, where={"broad": True})
        self.assertEqual(p_broad.execution_path, "hnsw_post_filter")
        res = coll.search(self.queries[0], limit=10, filter={"broad": True})
        self.assertEqual(len(res), 10)
        self.assertTrue(all(r.metadata["broad"] is True for r in res))

        # %10 seçicilikte korelasyonlu filtre -> post_filter veya fallback mutlaka tam k=10 dönmeli
        res_tail = coll.search(self.queries[0], limit=10, filter={"tail_only": True}, filter_strategy="post")
        self.assertEqual(len(res_tail), 10)
        self.assertTrue(all(r.metadata["tail_only"] is True for r in res_tail))


if __name__ == "__main__":
    unittest.main()
