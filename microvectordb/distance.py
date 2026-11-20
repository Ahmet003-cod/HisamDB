"""
microvectordb.distance
----------------------
Geriye dönük uyumluluk cephesi (facade).
Asıl yüksek performanslı çekirdekler `microvectordb.metrics.kernels` altındadır.
"""

from microvectordb.metrics.kernels import (
    VALID_METRICS,
    compute_distances,
    compute_inv_norms,
    compute_norms,
    compute_sq_norms,
    normalize_vectors,
    top_k_indices,
    validate_metric,
    validate_vectors,
)

__all__ = [
    "VALID_METRICS",
    "validate_metric",
    "validate_vectors",
    "compute_sq_norms",
    "compute_norms",
    "compute_inv_norms",
    "normalize_vectors",
    "compute_distances",
    "top_k_indices",
]
