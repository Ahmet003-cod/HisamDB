"""
microvectordb.distance
----------------------
Vektör benzerlik ve mesafe fonksiyonları.
NumPy ve BLAS (AVX2/SIMD) optimize edilmiş matris işlemleri kullanır.
"""

import numpy as np
from typing import Tuple
from microvectordb.types import InvalidMetricError, MetricType


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Vektörleri L2 normuna göre normalize eder (birim vektör yapar).
    Kosinüs benzerliğini hızlı matris çarpımına (dot product) dönüştürmek için kullanılır.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Sıfır bölme hatasını engelle
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


def compute_distances(
    query_vectors: np.ndarray,
    db_vectors: np.ndarray,
    metric: MetricType = "cosine",
    pre_normalized_db: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sorgu vektörleri ile veritabanındaki vektörler arasındaki mesafeyi ve benzerliği hesaplar.

    Args:
        query_vectors: Şekli (Q, D) olan float32 sorgu matrisi.
        db_vectors: Şekli (N, D) olan float32 veritabanı matrisi.
        metric: 'cosine', 'dot_product' veya 'euclidean'.
        pre_normalized_db: Önceden L2-normalize edilmiş DB vektörleri (cosine cache).
                           Verilmezse her sorguda yeniden hesaplanır.

    Returns:
        (distances, similarities):
            distances: Küçük olan daha yakın (sıralama için)
            similarities: Büyük olan daha benzer (anlamlı skor için)
    """
    if db_vectors.size == 0 or query_vectors.size == 0:
        return np.empty((query_vectors.shape[0], 0)), np.empty((query_vectors.shape[0], 0))

    if metric == "cosine":
        # 1. Sorgu vektörlerini normalize et (küçük — her zaman hızlı)
        q_norm = normalize_vectors(query_vectors)

        # 2. DB vektörleri: Cache varsa kullan, yoksa yeniden hesapla
        db_norm = pre_normalized_db if pre_normalized_db is not None else normalize_vectors(db_vectors)

        # 3. Hızlı BLAS Matris Çarpımı (Q x D) @ (D x N) -> (Q x N)
        # Cosine similarity -1 ile +1 arasındadır.
        similarities = np.dot(q_norm, db_norm.T)
        similarities = np.clip(similarities, -1.0, 1.0)

        # Mesafe: 0 (aynı) ile 2 (tam zıt) arasında
        distances = 1.0 - similarities
        return distances, similarities

    elif metric == "dot_product":
        # İç çarpım (özellikle gömülü vektörler önceden normalize edilmişse çok hızlıdır)
        similarities = np.dot(query_vectors, db_vectors.T)
        # Benzerlik ne kadar büyükse mesafe o kadar küçüktür
        distances = -similarities
        return distances, similarities

    elif metric == "euclidean":
        # Hızlı L2 mesafesi: ||q - d||^2 = ||q||^2 + ||d||^2 - 2 * (q . d)
        q_sq = np.sum(query_vectors ** 2, axis=1, keepdims=True)  # (Q, 1)
        db_sq = np.sum(db_vectors ** 2, axis=1, keepdims=True).T   # (1, N)
        dot = np.dot(query_vectors, db_vectors.T)                 # (Q, N)

        sq_dists = q_sq + db_sq - 2.0 * dot
        sq_dists = np.maximum(sq_dists, 0.0)  # Kayan nokta hassasiyeti için negatifleri sıfırla
        distances = np.sqrt(sq_dists)

        # Benzerlik skoru: 1 / (1 + distance) ile 0-1 aralığına dönüştür
        similarities = 1.0 / (1.0 + distances)
        return distances, similarities

    else:
        raise InvalidMetricError(
            f"Bilinmeyen metrik: '{metric}'. Desteklenenler: 'cosine', 'dot_product', 'euclidean'"
        )
