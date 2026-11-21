"""
Faz 6 (v1.0.0) Birim & Parite Testleri:
  - Native Rust SIMD (AVX2/FMA/POPCNT) vs Saf NumPy bit/sayısal paritesi
    (ProductQuantizer, IVFPQIndex, ScalarQuantizer INT8, BinaryQuantizer)
  - Yüksek boyutlu LLM embedding (384D, 768D, 1536D) koleksiyon, sıkıştırma ve DiskIndex doğrulaması
"""

import shutil
import tempfile
import unittest

import numpy as np

import microvectordb
from microvectordb import (
    BinaryQuantizer,
    IVFPQIndex,
    MicroVectorDB,
    ProductQuantizer,
    ScalarQuantizer,
    is_native_simd_available,
    set_native_simd_enabled,
)
from microvectordb.metrics.native_simd import (
    binary_hamming_u64_native,
    int8_dot_f32_native,
    pq_adc_u16_native,
    pq_adc_u8_native,
)


class TestNativeSIMDParity(unittest.TestCase):
    """Native SIMD çekirdeği ile saf NumPy yolu arasındaki tam sayısal eşitliği doğrular."""

    def tearDown(self):
        set_native_simd_enabled(True)

    def test_low_level_kernels_exact_parity(self):
        """Doğrudan `pq_adc_u8`, `pq_adc_u16`, `int8_dot_f32`, `binary_hamming_u64` parite testi."""
        set_native_simd_enabled(True)
        if not is_native_simd_available():
            self.skipTest("mvdb_simd paylaşımlı kütüphanesi derlenmemiş; opsiyonel test atlandı.")

        rng = np.random.default_rng(42)

        # 1. pq_adc_u8
        m_cnt, k_cnt, n = 16, 256, 2000
        lut = rng.normal(size=(m_cnt, k_cnt)).astype(np.float32)
        codes_t = rng.integers(0, k_cnt, size=(m_cnt, n), dtype=np.uint8)
        bias = 1.25
        ref_u8 = np.full(n, bias, dtype=np.float32)
        for m in range(m_cnt):
            ref_u8 += lut[m][codes_t[m]]
        got_u8 = pq_adc_u8_native(lut, codes_t, bias=bias)
        self.assertIsNotNone(got_u8)
        np.testing.assert_allclose(got_u8, ref_u8, rtol=1e-5, atol=1e-4)

        # 2. pq_adc_u16 (single-shot flat LUT)
        flat_lut = lut.reshape(-1)
        offsets = (np.arange(m_cnt, dtype=np.uint16) * k_cnt)[:, None]
        codes_u16 = codes_t.astype(np.uint16) + offsets
        got_u16 = pq_adc_u16_native(flat_lut, codes_u16, bias=bias)
        self.assertIsNotNone(got_u16)
        np.testing.assert_allclose(got_u16, ref_u8, rtol=1e-5, atol=1e-4)

        # 3. int8_dot_f32 (hem 8'in katı hem tekil kalan boyutlar: 13, 63, 128, 384)
        for dim in (13, 63, 128, 384):
            q = rng.normal(size=(dim,)).astype(np.float32)
            db_i8 = rng.integers(-127, 128, size=(1024, dim), dtype=np.int8)
            ref_dot = db_i8.astype(np.float32) @ q
            got_dot = int8_dot_f32_native(q, db_i8)
            self.assertIsNotNone(got_dot)
            np.testing.assert_allclose(got_dot, ref_dot, rtol=1e-4, atol=1e-3)

        # 4. binary_hamming_u64 (tam bit eşitliği)
        for words in (1, 2, 6, 12, 24):
            q_w = rng.integers(0, np.iinfo(np.uint64).max, size=(words,), dtype=np.uint64)
            db_w = rng.integers(0, np.iinfo(np.uint64).max, size=(512, words), dtype=np.uint64)
            xor = np.bitwise_xor(db_w, q_w)
            ref_ham = np.zeros(512, dtype=np.uint32)
            for b in range(64):
                ref_ham += ((xor >> np.uint64(b)) & np.uint64(1)).sum(axis=1).astype(np.uint32)
            got_ham = binary_hamming_u64_native(q_w, db_w)
            self.assertIsNotNone(got_ham)
            np.testing.assert_array_equal(got_ham, ref_ham)

    def test_quantizers_and_ivfpq_native_vs_numpy_parity(self):
        """ScalarQuantizer, BinaryQuantizer, ProductQuantizer ve IVFPQIndex tüm metriklerde birebir parite."""
        rng = np.random.default_rng(100)
        vecs = rng.normal(size=(1500, 128)).astype(np.float32)
        queries = rng.normal(size=(10, 128)).astype(np.float32)

        # ScalarQuantizer (INT8)
        sq = ScalarQuantizer().fit(vecs)
        q_codes = sq.encode(vecs)
        scaled_sq, recon_inv = sq.precompute_db_norms(q_codes)
        for metric in ("cosine", "euclidean", "dot_product"):
            for q in queries[:3]:
                set_native_simd_enabled(False)
                d_np = sq.compute_distances_quantized(
                    q, q_codes, metric=metric,
                    precomputed_scaled_sq=scaled_sq,
                    precomputed_recon_inv=recon_inv,
                )
                set_native_simd_enabled(True)
                d_simd = sq.compute_distances_quantized(
                    q, q_codes, metric=metric,
                    precomputed_scaled_sq=scaled_sq,
                    precomputed_recon_inv=recon_inv,
                )
                np.testing.assert_allclose(d_simd, d_np, rtol=1e-4, atol=1e-3)

        # BinaryQuantizer (64-bit POPCNT vs uint8 LUT)
        for dim in (13, 128, 384, 768):
            v_d = rng.normal(size=(500, dim)).astype(np.float32)
            q_d = rng.normal(size=(dim,)).astype(np.float32)
            bq = BinaryQuantizer().fit(v_d)
            b_codes = bq.encode(v_d)
            q_bits = bq.encode(q_d)
            set_native_simd_enabled(False)
            h_np = bq.hamming_distances(q_bits, b_codes)
            set_native_simd_enabled(True)
            h_simd = bq.hamming_distances(q_bits, b_codes)
            np.testing.assert_array_equal(h_simd, h_np)

        # ProductQuantizer & IVFPQIndex
        for metric in ("cosine", "euclidean", "dot_product"):
            pq = ProductQuantizer(n_subvectors=16, n_centroids=256, metric=metric, n_iter=6, seed=7)
            pq.fit(vecs)
            pq_codes_t = np.ascontiguousarray(pq.encode(vecs).T)
            for q in queries[:3]:
                set_native_simd_enabled(False)
                d_np, _ = pq.compute_distances_adc(q, pq_codes_t, metric=metric)
                set_native_simd_enabled(True)
                d_simd, _ = pq.compute_distances_adc(q, pq_codes_t, metric=metric)
                np.testing.assert_allclose(d_simd, d_np, rtol=1e-4, atol=1e-3)

            ivfpq = IVFPQIndex(
                n_clusters=16, nprobe=4, n_subvectors=16, n_centroids=256, metric=metric, seed=7
            )
            ivfpq.train(vecs, n_iter=6)
            for q in queries[:3]:
                set_native_simd_enabled(False)
                idx_np, dist_np = ivfpq.search(q, k=10)
                set_native_simd_enabled(True)
                idx_simd, dist_simd = ivfpq.search(q, k=10)
                np.testing.assert_array_equal(idx_simd, idx_np)
                np.testing.assert_allclose(dist_simd, dist_np, rtol=1e-4, atol=1e-3)


class TestHighDimensionalLLMCollections(unittest.TestCase):
    """384D, 768D ve 1536D LLM embedding senaryolarında sıkıştırma, DiskIndex ve kalıcılık testi."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="mvdb_p6_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_high_dim_384_768_1536_quantization_and_disk_index(self):
        rng = np.random.default_rng(2026)

        # 1. 384D (MiniLM) + INT8 (Native SIMD fast-path)
        n = 400
        vecs_384 = rng.normal(size=(n, 384)).astype(np.float32)
        norms = np.linalg.norm(vecs_384, axis=1, keepdims=True)
        vecs_384 /= np.maximum(norms, 1e-8)

        db = MicroVectorDB(path=self.tmpdir)
        c384 = db.create_collection(
            "minilm_384", dimension=384, metric="cosine", quantization="int8", keep_raw_vectors=False
        )
        c384.add(ids=[f"d{i}" for i in range(n)], embeddings=vecs_384)
        res = c384.search(vecs_384[0], limit=5)
        self.assertEqual(res[0].id, "d0")
        st384 = c384.stats()
        self.assertEqual(st384["vector_ram_mb"], 0.0)
        self.assertLess(st384["quantized_ram_mb"], (vecs_384.nbytes / (1024 * 1024)) * 0.35)

        # 2. 768D (BGE/E5) + Binary + Rescore
        vecs_768 = rng.normal(size=(n, 768)).astype(np.float32)
        c768 = db.create_collection(
            "bge_768", dimension=768, metric="cosine", quantization="binary", keep_raw_vectors=True
        )
        c768.add(ids=[f"b{i}" for i in range(n)], embeddings=vecs_768)
        res768 = c768.search(vecs_768[17], limit=5)
        self.assertEqual(res768[0].id, "b17")

        # 3. 1536D (OpenAI) + DiskIndex (MMap streaming) + reload doğrulaması
        vecs_1536 = rng.normal(size=(300, 1536)).astype(np.float32)
        c1536 = db.create_collection(
            "openai_1536", dimension=1536, metric="cosine", index_type="disk", use_mmap=True
        )
        c1536.add(ids=[f"o{i}" for i in range(300)], embeddings=vecs_1536)
        c1536.build_index()
        res1536 = c1536.search(vecs_1536[42], limit=5)
        self.assertEqual(res1536[0].id, "o42")
        db.close()

        # Reload sonrası MMap sıfır vektör RAM doğrulaması
        db2 = MicroVectorDB(path=self.tmpdir)
        c1536_r = db2.get_collection("openai_1536")
        res1536_r = c1536_r.search(vecs_1536[42], limit=5)
        self.assertEqual(res1536_r[0].id, "o42")
        self.assertEqual(c1536_r.stats()["vector_ram_mb"], 0.0)
        db2.close()

    def test_version_and_public_exports(self):
        self.assertEqual(microvectordb.__version__, "1.0.0")
        self.assertTrue(callable(microvectordb.is_native_simd_available))
        self.assertTrue(callable(microvectordb.set_native_simd_enabled))
        self.assertTrue(callable(microvectordb.native_simd_path))


if __name__ == "__main__":
    unittest.main()
