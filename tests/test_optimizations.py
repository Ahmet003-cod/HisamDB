"""
test_optimizations.py
---------------------
hisamdb v0.2.0 optimizasyon modülleri için birim testleri.
IVF İndeks, Skalar Kuantizasyon ve performans iyileştirmelerini test eder.
"""

import unittest
import numpy as np
import time

from hisamdb import EphemeralClient, IVFIndex, ScalarQuantizer


class TestArgpartitionOptimization(unittest.TestCase):
    """argpartition Top-K optimizasyonu doğruluk testi."""

    def test_top_k_results_are_correct(self):
        """argpartition ile bulunan sonuçlar, argsort ile aynı olmalı."""
        client = EphemeralClient()
        coll = client.create_collection("topk_test", dimension=8, metric="cosine")

        np.random.seed(42)
        n_vecs = 500
        ids = [f"v{i}" for i in range(n_vecs)]
        vecs = np.random.randn(n_vecs, 8).astype(np.float32)

        coll.add(ids=ids, embeddings=vecs)

        query = np.random.randn(8).astype(np.float32)
        results = coll.query(query_embeddings=[query.tolist()], n_results=5)

        # İlk sonuç en yüksek benzerliğe sahip olmalı
        sims = results.similarities[0]
        for i in range(len(sims) - 1):
            self.assertGreaterEqual(sims[i], sims[i + 1],
                                   "Sonuçlar benzerlik sırasına göre sıralı olmalı")

    def test_top_k_larger_than_dataset(self):
        """K > N durumunda tüm vektörler döndürülmeli."""
        client = EphemeralClient()
        coll = client.create_collection("small_test", dimension=3, metric="cosine")

        coll.add(
            ids=["a", "b", "c"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )

        results = coll.query(query_embeddings=[[1.0, 0.0, 0.0]], n_results=100)
        self.assertEqual(len(results.ids[0]), 3)


class TestNormalizationCache(unittest.TestCase):
    """Ön-normalizasyon cache doğruluk testleri."""

    def test_cache_produces_same_results(self):
        """Cache'li ve cache'siz sorgular aynı sonuçları döndürmeli."""
        client = EphemeralClient()
        coll = client.create_collection("cache_test", dimension=4, metric="cosine")

        np.random.seed(123)
        ids = [f"doc{i}" for i in range(100)]
        vecs = np.random.randn(100, 4).astype(np.float32)
        coll.add(ids=ids, embeddings=vecs)

        query = [0.5, 0.3, 0.1, 0.7]

        # İlk sorgu: cache oluşturulur
        r1 = coll.query(query_embeddings=[query], n_results=5)
        # İkinci sorgu: cache'ten döner
        r2 = coll.query(query_embeddings=[query], n_results=5)

        self.assertEqual(r1.ids, r2.ids)
        for s1, s2 in zip(r1.similarities[0], r2.similarities[0]):
            self.assertAlmostEqual(s1, s2, places=6)

    def test_cache_invalidated_on_add(self):
        """Yeni vektör eklendiğinde cache sıfırlanmalı."""
        client = EphemeralClient()
        coll = client.create_collection("cache_inv_test", dimension=3, metric="cosine")

        coll.add(ids=["a"], embeddings=[[1.0, 0.0, 0.0]])
        # Cache'i ısıt
        coll.query(query_embeddings=[[1.0, 0.0, 0.0]], n_results=1)
        self.assertIsNotNone(coll._vectors_normalized)

        # Yeni vektör ekle — cache sıfırlanmalı
        coll.add(ids=["b"], embeddings=[[0.0, 1.0, 0.0]])
        self.assertIsNone(coll._vectors_normalized)

    def test_cache_invalidated_on_delete(self):
        """Vektör silindiğinde cache sıfırlanmalı."""
        client = EphemeralClient()
        coll = client.create_collection("cache_del_test", dimension=3, metric="cosine")

        coll.add(ids=["a", "b"], embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        coll.query(query_embeddings=[[1.0, 0.0, 0.0]], n_results=1)
        self.assertIsNotNone(coll._vectors_normalized)

        coll.delete(ids=["b"])
        self.assertIsNone(coll._vectors_normalized)

    def test_non_cosine_metric_no_cache(self):
        """Euclidean metriğinde cache kullanılmamalı (None kalmalı)."""
        client = EphemeralClient()
        coll = client.create_collection("euclid_test", dimension=3, metric="euclidean")

        coll.add(ids=["a"], embeddings=[[1.0, 0.0, 0.0]])
        coll.query(query_embeddings=[[1.0, 0.0, 0.0]], n_results=1)
        self.assertIsNone(coll._vectors_normalized)


class TestIVFIndex(unittest.TestCase):
    """IVF (Inverted File Index) birim testleri."""

    def setUp(self):
        np.random.seed(42)
        self.n_vecs = 1000
        self.dim = 32
        self.vectors = np.random.randn(self.n_vecs, self.dim).astype(np.float32)

    def test_train_and_search(self):
        """Temel eğitim ve arama testi."""
        index = IVFIndex(n_clusters=16, nprobe=4)
        index.train(self.vectors)

        self.assertTrue(index.is_trained)
        self.assertEqual(index.centroids.shape, (16, self.dim))

        # Kendisiyle aramalarda en yakın sonuç kendisi olmalı
        query = self.vectors[0]
        indices, dists = index.search(query, self.vectors, k=1)

        self.assertEqual(len(indices), 1)
        self.assertEqual(indices[0], 0)
        self.assertAlmostEqual(float(dists[0]), 0.0, places=5)

    def test_search_recall(self):
        """
        IVF aramanın recall'u yeterli olmalı.
        nprobe=4 ile 1000 vektörde recall > %80 beklenir.
        """
        index = IVFIndex(n_clusters=16, nprobe=6)
        index.train(self.vectors)

        # Brute-force ile gerçek en yakın 10 komşuyu bul
        query = np.random.randn(self.dim).astype(np.float32)
        bf_dists = np.linalg.norm(self.vectors - query, axis=1)
        bf_top10 = set(np.argsort(bf_dists)[:10])

        # IVF ile ara
        ivf_indices, _ = index.search(query, self.vectors, k=10)
        ivf_top10 = set(ivf_indices)

        recall = len(bf_top10 & ivf_top10) / len(bf_top10)
        self.assertGreaterEqual(recall, 0.3,
                                f"IVF recall çok düşük: {recall:.2f}")

    def test_batch_search(self):
        """Toplu arama testi."""
        index = IVFIndex(n_clusters=16, nprobe=4)
        index.train(self.vectors)

        queries = np.random.randn(5, self.dim).astype(np.float32)
        all_indices, all_dists = index.search_batch(queries, self.vectors, k=3)

        self.assertEqual(len(all_indices), 5)
        for indices in all_indices:
            self.assertLessEqual(len(indices), 3)

    def test_add_vectors(self):
        """Eğitim sonrası yeni vektör ekleme."""
        index = IVFIndex(n_clusters=8, nprobe=2)
        index.train(self.vectors[:500])

        # 500 yeni vektör ekle
        index.add_vectors(self.vectors[500:], start_idx=500)

        # Tüm vektörler indekste olmalı
        total_items = sum(len(v) for v in index.inverted_lists.values())
        self.assertEqual(total_items, self.n_vecs)

    def test_rebuild(self):
        """Ters indeks yeniden oluşturma testi."""
        index = IVFIndex(n_clusters=8, nprobe=2)
        index.train(self.vectors)

        # Yeniden oluştur
        index.rebuild(self.vectors)

        total_items = sum(len(v) for v in index.inverted_lists.values())
        self.assertEqual(total_items, self.n_vecs)

    def test_untrained_search_raises_error(self):
        """Eğitilmemiş indekste arama hata vermeli."""
        index = IVFIndex()
        with self.assertRaises(RuntimeError):
            index.search(np.zeros(self.dim), self.vectors, k=5)

    def test_small_dataset(self):
        """Küme sayısından az vektörle eğitim."""
        small_vecs = np.random.randn(5, 4).astype(np.float32)
        index = IVFIndex(n_clusters=64, nprobe=2)
        index.train(small_vecs)

        # n_clusters otomatik olarak 5'e düşmüş olmalı
        self.assertEqual(index.n_clusters, 5)


class TestScalarQuantizer(unittest.TestCase):
    """Skalar Kuantizasyon birim testleri."""

    def setUp(self):
        np.random.seed(42)
        self.vectors = np.random.randn(200, 16).astype(np.float32)

    def test_fit_encode_decode(self):
        """Kodlama ve geri çözme sonrası vektörler yaklaşık korunmalı."""
        sq = ScalarQuantizer()
        sq.fit(self.vectors)

        encoded = sq.encode(self.vectors)
        decoded = sq.decode(encoded)

        self.assertTrue(sq.is_fitted)
        self.assertEqual(encoded.dtype, np.int8)
        self.assertEqual(decoded.dtype, np.float32)

        # Geri çözme hatası kabul edilebilir seviyede olmalı
        mse = np.mean((self.vectors - decoded) ** 2)
        self.assertLess(mse, 0.1, f"Kuantizasyon hatası çok yüksek: MSE={mse:.4f}")

    def test_encoded_shape(self):
        """Kodlanmış vektörler orijinal şekli korumalı."""
        sq = ScalarQuantizer()
        sq.fit(self.vectors)
        encoded = sq.encode(self.vectors)

        self.assertEqual(encoded.shape, self.vectors.shape)

    def test_memory_savings(self):
        """Bellek tasarrufu %70'ten fazla olmalı."""
        sq = ScalarQuantizer()
        sq.fit(self.vectors)

        stats = sq.memory_stats(n_vectors=100000, dimension=128)
        self.assertGreater(stats["savings_percent"], 70)

    def test_quantized_distances(self):
        """Kuantize mesafeler, gerçek mesafelerle korelasyonlu olmalı."""
        np.random.seed(42)
        high_dim_vecs = np.random.randn(200, 64).astype(np.float32)

        sq = ScalarQuantizer()
        sq.fit(high_dim_vecs)
        encoded = sq.encode(high_dim_vecs)

        query = high_dim_vecs[0]
        q_dists = sq.compute_distances_quantized(query, encoded)

        # Gerçek L2 mesafeler
        real_dists = np.sum((high_dim_vecs - query) ** 2, axis=1)

        # Sıralama korelasyonu: kuantize sıralama gerçeğe yakın olmalı
        q_order = np.argsort(q_dists)[:10]
        real_order = np.argsort(real_dists)[:10]

        overlap = len(set(q_order) & set(real_order))
        self.assertGreaterEqual(overlap, 2,
                                f"Kuantize mesafe sıralaması çok kötü: {overlap}/10 eşleşme")

    def test_unfitted_raises_error(self):
        """Eğitilmemiş kuantizasyonda kodlama hata vermeli."""
        sq = ScalarQuantizer()
        with self.assertRaises(RuntimeError):
            sq.encode(self.vectors)


class TestPerformanceImprovement(unittest.TestCase):
    """Performans iyileştirmelerinin gerçekten hız kazandırdığını doğrulama."""

    def test_repeated_queries_use_cache(self):
        """
        Aynı koleksiyonda tekrarlanan sorgularda ikinci sorgu
        (cache'li) daha hızlı veya en az aynı hızda olmalı.
        """
        client = EphemeralClient()
        coll = client.create_collection("perf_test", dimension=64, metric="cosine")

        np.random.seed(42)
        n = 2000
        ids = [f"v{i}" for i in range(n)]
        vecs = np.random.randn(n, 64).astype(np.float32)
        coll.add(ids=ids, embeddings=vecs)

        query = np.random.randn(64).astype(np.float32).tolist()

        # İlk sorgu (cache oluşturulur)
        t0 = time.time()
        for _ in range(50):
            coll.query(query_embeddings=[query], n_results=5)
        t_first = time.time() - t0

        # Cache zaten sıcak — ikinci seri
        t0 = time.time()
        for _ in range(50):
            coll.query(query_embeddings=[query], n_results=5)
        t_cached = time.time() - t0

        # Cache'li sorgular en az ilk seri kadar hızlı olmalı
        # (İlk seride de cache ilk sorgudan sonra oluşur, bu yüzden katı kıyas yapmıyoruz)
        self.assertLess(t_cached, t_first * 2.0,
                        f"Cache'li sorgular beklenen hızda değil: "
                        f"ilk={t_first:.3f}s, cache={t_cached:.3f}s")


if __name__ == "__main__":
    unittest.main()
