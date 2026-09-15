"""
microvectordb.quantization
--------------------------
Skalar Kuantizasyon modülü: float32 vektörleri int8'e sıkıştırarak
bellek tüketimini 4x azaltır ve CPU cache verimliliğini artırır.

Kullanım:
    quantizer = ScalarQuantizer()
    quantizer.fit(vectors)
    compressed = quantizer.encode(vectors)
    restored = quantizer.decode(compressed)

Not: Kuantizasyon yaklaşık bir işlemdir.
     Doğruluk kaybı genellikle %1-3 arasındadır ve çoğu uygulama için göz ardı edilebilir.
"""

import numpy as np
from typing import Optional, Tuple


class ScalarQuantizer:
    """
    Skalar Kuantizasyon: float32 → int8 dönüşümü.

    Her boyut için min/max değerleri öğrenir ve vektörleri
    [-128, 127] aralığına eşler. Bu sayede:
    - Bellek tüketimi 4x azalır (32-bit → 8-bit)
    - CPU cache'e daha fazla vektör sığar → daha hızlı arama
    - Mesafe hesaplamaları daha küçük veriyle yapılır
    """

    def __init__(self):
        self.min_vals: Optional[np.ndarray] = None
        self.max_vals: Optional[np.ndarray] = None
        self.range_vals: Optional[np.ndarray] = None
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        """Kuantizasyonun eğitilip eğitilmediğini döndürür."""
        return self._fitted

    def fit(self, vectors: np.ndarray) -> "ScalarQuantizer":
        """
        Her boyut için min/max değerleri öğrenir.

        Args:
            vectors: Eğitim vektörleri, şekli (N, D), dtype float32.

        Returns:
            self (zincirleme çağrı için).
        """
        self.min_vals = vectors.min(axis=0).astype(np.float32)
        self.max_vals = vectors.max(axis=0).astype(np.float32)

        # Sıfır bölmeyi engelle (sabit boyutlar)
        range_vals = self.max_vals - self.min_vals
        range_vals[range_vals == 0] = 1.0
        self.range_vals = range_vals

        self._fitted = True
        return self

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """
        Float32 vektörleri int8'e sıkıştırır.

        Args:
            vectors: Orijinal vektörler, şekli (N, D), dtype float32.

        Returns:
            Kuantize edilmiş vektörler, şekli (N, D), dtype int8.
        """
        if not self._fitted:
            raise RuntimeError("Kuantizasyon henüz eğitilmedi. Önce fit() çağırın.")

        # [0, 1] aralığına normalize et
        normalized = (vectors - self.min_vals) / self.range_vals
        # [0, 1] → [-128, 127] aralığına eşle
        quantized = np.clip(normalized * 255 - 128, -128, 127).astype(np.int8)
        return quantized

    def decode(self, quantized: np.ndarray) -> np.ndarray:
        """
        Int8 vektörleri yaklaşık float32'ye geri dönüştürür.

        Args:
            quantized: Kuantize edilmiş vektörler, şekli (N, D), dtype int8.

        Returns:
            Yaklaşık orijinal vektörler, şekli (N, D), dtype float32.
        """
        if not self._fitted:
            raise RuntimeError("Kuantizasyon henüz eğitilmedi. Önce fit() çağırın.")

        # [-128, 127] → [0, 1] → orijinal aralık
        normalized = (quantized.astype(np.float32) + 128) / 255.0
        restored = normalized * self.range_vals + self.min_vals
        return restored

    def compute_distances_quantized(
        self,
        query: np.ndarray,
        quantized_db: np.ndarray,
    ) -> np.ndarray:
        """
        Kuantize edilmiş vektörlerle yaklaşık L2 mesafe hesaplar.
        Sorgu vektörünü de int8'e dönüştürüp integer aritmetiği kullanır.

        Args:
            query: Sorgu vektörü (float32), şekli (D,) veya (1, D).
            quantized_db: Kuantize edilmiş DB vektörleri (int8), şekli (N, D).

        Returns:
            Yaklaşık L2 mesafeleri, şekli (N,).
        """
        if not self._fitted:
            raise RuntimeError("Kuantizasyon henüz eğitilmedi.")

        query = query.ravel()
        q_quantized = self.encode(query.reshape(1, -1)).ravel()

        # int16'ya genişlet (taşma önleme) ve mesafe hesapla
        q_int16 = q_quantized.astype(np.int16)
        db_int16 = quantized_db.astype(np.int16)

        diff = db_int16 - q_int16
        sq_dists = np.sum(diff ** 2, axis=1)

        return sq_dists.astype(np.float32)

    def memory_stats(self, n_vectors: int, dimension: int) -> dict:
        """
        Bellek tasarruf istatistiklerini döndürür.

        Returns:
            Dict ile float32 ve int8 boyutları ve tasarruf yüzdesi.
        """
        float32_bytes = n_vectors * dimension * 4
        int8_bytes = n_vectors * dimension * 1
        overhead_bytes = dimension * 4 * 3  # min_vals, max_vals, range_vals

        return {
            "float32_mb": float32_bytes / (1024 * 1024),
            "int8_mb": (int8_bytes + overhead_bytes) / (1024 * 1024),
            "savings_percent": (1 - (int8_bytes + overhead_bytes) / float32_bytes) * 100,
            "compression_ratio": f"{float32_bytes / (int8_bytes + overhead_bytes):.1f}x",
        }

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        dims = len(self.min_vals) if self._fitted else 0
        return f"<ScalarQuantizer(status={status}, dimensions={dims})>"
