"""
microvectordb.quantization.float16
----------------------------------
Float16 (IEEE 754 yarım hassasiyet) sıkıştırma modülü.
Float32'ye kıyasla %50 bellek tasarrufu (2x) ve >%99.9 Recall@10 sağlar.
"""

from typing import Optional, Tuple
import numpy as np
from microvectordb.metrics.kernels import compute_distances


class Float16Quantizer:
    """Float32 -> Float16 sıkıştırıcı (2x RAM tasarrufu, ihmal edilebilir doğruluk kaybı)."""

    def __init__(self) -> None:
        self._fitted = True

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, vectors: np.ndarray) -> "Float16Quantizer":
        self._fitted = True
        return self

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Float32 vektörleri float16'ya sıkıştırır (taşma için [-65504, 65504] kırpılır)."""
        v32 = np.asarray(vectors, dtype=np.float32)
        return np.clip(v32, np.float32(-65504.0), np.float32(65504.0)).astype(np.float16)

    def decode(self, quantized: np.ndarray) -> np.ndarray:
        return np.asarray(quantized, dtype=np.float32)

    def compute_batch_distances(
        self,
        queries: np.ndarray,
        quantized_db: np.ndarray,
        metric: str = "cosine",
        db_inv_norms: Optional[np.ndarray] = None,
        db_sq_norms: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return compute_distances(
            queries,
            quantized_db,
            metric=metric,  # type: ignore[arg-type]
            db_inv_norms=db_inv_norms,
            db_sq_norms=db_sq_norms,
        )

    @staticmethod
    def memory_stats(n_vectors: int, dimension: int) -> dict:
        f32 = n_vectors * dimension * 4
        f16 = n_vectors * dimension * 2
        if f32 == 0:
            return {"float32_mb": 0.0, "float16_mb": 0.0, "savings_percent": 0.0, "compression_ratio": "0.0x"}
        return {
            "float32_mb": f32 / (1024 * 1024),
            "float16_mb": f16 / (1024 * 1024),
            "savings_percent": 50.0,
            "compression_ratio": "2.0x",
        }
