"""
hisamdb.quantization.binary
---------------------------------
1-Bit Binary Kuantizasyon modülü (32x bellek tasarrufu).
Her boyutu eşik değerine (varsayılan: 0 veya boyut medyan/ortalaması) göre 1 bite sıkıştırır.

NumPy 2.0+ üzerinde 64-bit donanım POPCNT (`np.bitwise_count`) ile yüksek hızlı
Hamming mesafesi hesaplar; NumPy 1.x sürümlerinde 256-girdili LUT tablosuna
otomatik geri düşer.
"""

from typing import Optional, Tuple
import numpy as np

from hisamdb.metrics.native_simd import binary_hamming_u64_native

_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
_HAS_BITWISE_COUNT = hasattr(np, "bitwise_count")


class BinaryQuantizer:
    """
    1-Bit Binary Kuantizasyon: float32 -> paketlenmiş bit dizisi (32x RAM tasarrufu).

    Args:
        use_mean_threshold: True ise fit() sırasında boyut başına ortalama eşik öğrenilir.
                            False ise 0.0 eşiği kullanılır (sıfır merkezli embeddingler için ideal).
    """

    def __init__(self, use_mean_threshold: bool = True):
        self.use_mean_threshold = use_mean_threshold
        self.thresholds: Optional[np.ndarray] = None
        self.dimension: int = 0
        self._padded_bytes: int = 0
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, vectors: np.ndarray) -> "BinaryQuantizer":
        v32 = np.asarray(vectors, dtype=np.float32)
        if v32.ndim == 1:
            v32 = v32.reshape(1, -1)
        if v32.shape[0] == 0:
            raise ValueError("BinaryQuantizer.fit() boş vektör dizisi ile çağrılamaz.")

        self.dimension = v32.shape[1]
        if self.use_mean_threshold:
            self.thresholds = v32.mean(axis=0).astype(np.float32)
        else:
            self.thresholds = np.zeros(self.dimension, dtype=np.float32)

        n_bytes = (self.dimension + 7) // 8
        # uint64 view için 8 bayt katına hizala
        self._padded_bytes = ((n_bytes + 7) // 8) * 8
        self._fitted = True
        return self

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Float32 vektörleri 8-byte hizalı uint8 bit matrisine (N, padded_bytes) sıkıştırır."""
        if not self._fitted or self.thresholds is None:
            raise RuntimeError("BinaryQuantizer henüz eğitilmedi. Önce fit() çağırın.")

        v32 = np.asarray(vectors, dtype=np.float32)
        single = v32.ndim == 1
        if single:
            v32 = v32.reshape(1, -1)

        bits = (v32 > self.thresholds).astype(np.uint8)
        packed = np.packbits(bits, axis=1)
        if packed.shape[1] < self._padded_bytes:
            pad_width = self._padded_bytes - packed.shape[1]
            packed = np.pad(packed, ((0, 0), (0, pad_width)), mode="constant", constant_values=0)
        packed = np.ascontiguousarray(packed, dtype=np.uint8)
        return packed[0] if single else packed

    def hamming_distances(self, query_bits: np.ndarray, db_bits: np.ndarray) -> np.ndarray:
        """
        Tek sorgu bit vektörü (B,) ile db_bits (N, B) arasındaki Hamming mesafesini döndürür.
        """
        if db_bits.shape[0] == 0:
            return np.empty(0, dtype=np.int32)

        qb = query_bits.reshape(1, -1)
        if db_bits.shape[1] % 8 == 0:
            db64 = np.ascontiguousarray(db_bits).view(np.uint64)
            q64 = np.ascontiguousarray(qb).view(np.uint64)
            nat_hd = binary_hamming_u64_native(q64.ravel(), db64)
            if nat_hd is not None:
                return nat_hd.astype(np.int32, copy=False)
            if _HAS_BITWISE_COUNT:
                return np.bitwise_count(np.bitwise_xor(db64, q64)).sum(axis=1, dtype=np.int32)

        return _POPCOUNT_LUT[np.bitwise_xor(db_bits, qb)].sum(axis=1, dtype=np.int32)

    def compute_batch_distances(
        self,
        queries: np.ndarray,
        quantized_db: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Çoklu float32 sorgu matrisi için normalize edilmiş Hamming mesafesi [0, 1] ve
        benzerlik (1 - distance) döndürür.
        """
        q_arr = np.asarray(queries, dtype=np.float32)
        if q_arr.ndim == 1:
            q_arr = q_arr.reshape(1, -1)
        q_bits = self.encode(q_arr)
        q_cnt = q_arr.shape[0]
        n = quantized_db.shape[0]
        inv_dim = np.float32(1.0 / max(1, self.dimension))

        dists = np.empty((q_cnt, n), dtype=np.float32)
        for i in range(q_cnt):
            hd = self.hamming_distances(q_bits[i], quantized_db)
            dists[i] = hd.astype(np.float32) * inv_dim
        sims = np.float32(1.0) - dists
        return dists, sims

    @staticmethod
    def memory_stats(n_vectors: int, dimension: int) -> dict:
        f32 = n_vectors * dimension * 4
        n_bytes = (dimension + 7) // 8
        padded = ((n_bytes + 7) // 8) * 8
        b_bytes = n_vectors * padded + dimension * 4
        if f32 == 0:
            return {"float32_mb": 0.0, "binary_mb": 0.0, "savings_percent": 0.0, "compression_ratio": "0.0x"}
        return {
            "float32_mb": f32 / (1024 * 1024),
            "binary_mb": b_bytes / (1024 * 1024),
            "savings_percent": (1.0 - b_bytes / f32) * 100.0,
            "compression_ratio": f"{f32 / max(b_bytes, 1):.1f}x",
        }
