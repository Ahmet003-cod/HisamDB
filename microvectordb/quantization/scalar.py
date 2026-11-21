"""
microvectordb.quantization.scalar
---------------------------------
Taşmasız (overflow-free), ölçek-ağırlıklı ve L2-cache bloklu (tiled)
Skalar Kuantizasyon (float32 -> int8, 4x bellek tasarrufu).

v0.2.0'daki `int16` karesi taşma hatası (`diff ** 2` -> negatif değerler) ve
6x geçici bellek tahsisi tamamen giderilmiştir.

Matematiksel temel:
    x_hat = scale * q + bias   (scale = range / 255, bias = min + 128 * scale)
    ||y - x_hat||^2 = ||y - bias||^2 + ||scale * q||^2 - 2 * ((y - bias) * scale) . q
    y . x_hat       = (y . bias) + (y * scale) . q
"""

from typing import Optional, Tuple
import numpy as np

from microvectordb.metrics.native_simd import int8_dot_f32_native


_TILE_ROWS = 1024  # 1024 x D x 4B ~ 512KB-3MB -> CPU L2/L3 cache dostu


class ScalarQuantizer:
    """
    Skalar Kuantizasyon: float32 -> int8 dönüşümü (4x bellek tasarrufu).

    Args:
        percentile: İsteğe bağlı aykırı değer kırpma yüzdeliği (örn. 0.995).
                    None ise tam min/max aralığı kullanılır (v0.2.0 ile uyumlu).
    """

    def __init__(self, percentile: Optional[float] = None):
        if percentile is not None and not (0.5 < percentile <= 1.0):
            raise ValueError("percentile (0.5, 1.0] aralığında olmalıdır.")
        self.percentile = percentile
        self.min_vals: Optional[np.ndarray] = None
        self.max_vals: Optional[np.ndarray] = None
        self.range_vals: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray] = None
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        """Kuantizasyonun eğitilip eğitilmediğini döndürür."""
        return self._fitted

    def fit(self, vectors: np.ndarray) -> "ScalarQuantizer":
        """Her boyut için min/max (veya quantile) aralığını öğrenir."""
        v32 = np.asarray(vectors, dtype=np.float32)
        if v32.ndim == 1:
            v32 = v32.reshape(1, -1)
        if v32.shape[0] == 0:
            raise ValueError("ScalarQuantizer.fit() boş vektör dizisi ile çağrılamaz.")

        if self.percentile is not None and self.percentile < 1.0:
            low_q = (1.0 - self.percentile) * 0.5
            high_q = 1.0 - low_q
            self.min_vals = np.quantile(v32, low_q, axis=0).astype(np.float32)
            self.max_vals = np.quantile(v32, high_q, axis=0).astype(np.float32)
        else:
            self.min_vals = v32.min(axis=0).astype(np.float32)
            self.max_vals = v32.max(axis=0).astype(np.float32)

        range_vals = (self.max_vals - self.min_vals).astype(np.float32)
        range_vals[range_vals == np.float32(0.0)] = np.float32(1.0)
        self.range_vals = range_vals
        self.scale = (range_vals / np.float32(255.0)).astype(np.float32)
        self.bias = (self.min_vals + np.float32(128.0) * self.scale).astype(np.float32)

        self._fitted = True
        return self

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Float32 vektörleri int8'e ([-128, 127]) sıkıştırır."""
        if not self._fitted:
            raise RuntimeError("Kuantizasyon henüz eğitilmedi. Önce fit() çağırın.")

        v32 = np.asarray(vectors, dtype=np.float32)
        normalized = (v32 - self.min_vals) / self.range_vals
        quantized = np.rint(np.clip(normalized * np.float32(255.0) - np.float32(128.0), -128.0, 127.0)).astype(np.int8)
        return quantized

    def decode(self, quantized: np.ndarray) -> np.ndarray:
        """Int8 vektörleri yaklaşık float32'ye geri dönüştürür."""
        if not self._fitted:
            raise RuntimeError("Kuantizasyon henüz eğitilmedi. Önce fit() çağırın.")

        return (quantized.astype(np.float32) * self.scale + self.bias).astype(np.float32, copy=False)

    def precompute_db_norms(self, quantized_db: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kuantize DB vektörleri için iki yardımcı 1B dizi hesaplar (8 byte/vektör):
        - scaled_sq_norms: ||scale * q_i||^2 (L2 asimetrik mesafe için)
        - recon_inv_norms: 1 / ||x_hat_i|| (Kosinüs benzerliği için)
        """
        if not self._fitted:
            raise RuntimeError("Kuantizasyon henüz eğitilmedi.")

        n, d = quantized_db.shape
        scaled_sq = np.empty(n, dtype=np.float32)
        recon_inv = np.empty(n, dtype=np.float32)
        tile = min(_TILE_ROWS, max(1, n))
        scratch = np.empty((tile, d), dtype=np.float32)

        for start in range(0, n, tile):
            end = min(start + tile, n)
            sz = end - start
            np.copyto(scratch[:sz], quantized_db[start:end], casting="unsafe")
            scratch[:sz] *= self.scale
            scaled_sq[start:end] = np.einsum("ij,ij->i", scratch[:sz], scratch[:sz])
            scratch[:sz] += self.bias
            r_sq = np.einsum("ij,ij->i", scratch[:sz], scratch[:sz])
            r_norm = np.sqrt(r_sq)
            inv = np.zeros(sz, dtype=np.float32)
            nz = r_norm > np.float32(0.0)
            inv[nz] = np.float32(1.0) / r_norm[nz]
            recon_inv[start:end] = inv

        return scaled_sq, recon_inv

    def compute_distances_quantized(
        self,
        query: np.ndarray,
        quantized_db: np.ndarray,
        metric: str = "euclidean",
        precomputed_scaled_sq: Optional[np.ndarray] = None,
        precomputed_recon_inv: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Taşmasız, ölçek-ağırlıklı asimetrik mesafe hesaplar (sorgu float32 kalır,
        veritabanı int8'den L2-cache dostu küçük bloklarla okunur).

        Args:
            query: Sorgu vektörü (float32), şekli (D,) veya (1, D).
            quantized_db: Kuantize edilmiş DB vektörleri (int8), şekli (N, D).
            metric: 'euclidean' (varsayılan, geriye dönük uyumlu), 'cosine' veya 'dot_product'.
            precomputed_scaled_sq: Önceden hesaplanmış ||scale * q_i||^2 (opsiyonel).
            precomputed_recon_inv: Önceden hesaplanmış 1 / ||x_hat_i|| (opsiyonel).

        Returns:
            Mesafe vektörü, şekli (N,), dtype float32 (küçük olan daha yakın).
        """
        if not self._fitted:
            raise RuntimeError("Kuantizasyon henüz eğitilmedi.")

        n, d = quantized_db.shape
        if n == 0:
            return np.empty(0, dtype=np.float32)

        q32 = np.asarray(query, dtype=np.float32).ravel()
        tile = min(_TILE_ROWS, n)

        if metric == "euclidean":
            qp = q32 - self.bias
            w = qp * self.scale
            q_sq = np.float32(np.dot(qp, qp))

            if precomputed_scaled_sq is not None:
                nat_dots = int8_dot_f32_native(w, quantized_db)
                if nat_dots is not None:
                    out = nat_dots
                else:
                    scratch = np.empty((tile, d), dtype=np.float32)
                    out = np.empty(n, dtype=np.float32)
                    for start in range(0, n, tile):
                        end = min(start + tile, n)
                        sz = end - start
                        np.copyto(scratch[:sz], quantized_db[start:end], casting="unsafe")
                        out[start:end] = np.dot(scratch[:sz], w)
                out *= np.float32(-2.0)
                out += precomputed_scaled_sq
                out += q_sq
                np.maximum(out, np.float32(0.0), out=out)
                return out

            # precomputed_scaled_sq verilmediyse tek geçişte hem dot hem ||scale*q||^2 hesapla
            scratch = np.empty((tile, d), dtype=np.float32)
            out = np.empty(n, dtype=np.float32)
            scale_sq = (self.scale * self.scale).astype(np.float32)
            for start in range(0, n, tile):
                end = min(start + tile, n)
                sz = end - start
                np.copyto(scratch[:sz], quantized_db[start:end], casting="unsafe")
                dots = np.dot(scratch[:sz], w)
                # (q_i^2) . (scale^2) -> ayrı kopya olmadan yerinde karesi
                np.multiply(scratch[:sz], scratch[:sz], out=scratch[:sz])
                c_sq = np.dot(scratch[:sz], scale_sq)
                row = q_sq + c_sq - np.float32(2.0) * dots
                np.maximum(row, np.float32(0.0), out=row)
                out[start:end] = row
            return out

        if metric == "dot_product":
            w = q32 * self.scale
            const_bias = np.float32(np.dot(q32, self.bias))
            nat_dots = int8_dot_f32_native(w, quantized_db)
            if nat_dots is not None:
                out = nat_dots
            else:
                scratch = np.empty((tile, d), dtype=np.float32)
                out = np.empty(n, dtype=np.float32)
                for start in range(0, n, tile):
                    end = min(start + tile, n)
                    sz = end - start
                    np.copyto(scratch[:sz], quantized_db[start:end], casting="unsafe")
                    out[start:end] = np.dot(scratch[:sz], w)
            out += const_bias
            np.negative(out, out=out)  # distance = -dot
            return out

        if metric == "cosine":
            q_norm_val = np.float32(np.linalg.norm(q32))
            q_unit = q32 / q_norm_val if q_norm_val > np.float32(0.0) else q32
            w = q_unit * self.scale
            const_bias = np.float32(np.dot(q_unit, self.bias))

            if precomputed_recon_inv is None:
                _, precomputed_recon_inv = self.precompute_db_norms(quantized_db)

            nat_dots = int8_dot_f32_native(w, quantized_db)
            if nat_dots is not None:
                out = nat_dots
            else:
                scratch = np.empty((tile, d), dtype=np.float32)
                out = np.empty(n, dtype=np.float32)
                for start in range(0, n, tile):
                    end = min(start + tile, n)
                    sz = end - start
                    np.copyto(scratch[:sz], quantized_db[start:end], casting="unsafe")
                    out[start:end] = np.dot(scratch[:sz], w)
            out += const_bias
            out *= precomputed_recon_inv
            np.clip(out, np.float32(-1.0), np.float32(1.0), out=out)
            return (np.float32(1.0) - out).astype(np.float32, copy=False)

        raise ValueError(f"Desteklenmeyen metrik: '{metric}'")

    def compute_batch_distances(
        self,
        queries: np.ndarray,
        quantized_db: np.ndarray,
        metric: str = "cosine",
        precomputed_scaled_sq: Optional[np.ndarray] = None,
        precomputed_recon_inv: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Çoklu sorgu matrisi (Q, D) için (distances, similarities) döndürür."""
        q_arr = np.asarray(queries, dtype=np.float32)
        if q_arr.ndim == 1:
            q_arr = q_arr.reshape(1, -1)
        q_cnt = q_arr.shape[0]
        n = quantized_db.shape[0]
        dists = np.empty((q_cnt, n), dtype=np.float32)
        for i in range(q_cnt):
            d_row = self.compute_distances_quantized(
                q_arr[i],
                quantized_db,
                metric=metric,
                precomputed_scaled_sq=precomputed_scaled_sq,
                precomputed_recon_inv=precomputed_recon_inv,
            )
            if metric == "euclidean":
                np.sqrt(d_row, out=d_row)
            dists[i] = d_row

        if metric == "cosine":
            sims = np.float32(1.0) - dists
        elif metric == "dot_product":
            sims = -dists
        else:
            sims = np.float32(1.0) / (np.float32(1.0) + dists)
        return dists, sims

    def memory_stats(self, n_vectors: int, dimension: int) -> dict:
        """Bellek tasarruf istatistiklerini döndürür."""
        float32_bytes = n_vectors * dimension * 4
        int8_bytes = n_vectors * dimension * 1
        overhead_bytes = dimension * 4 * 5  # min, max, range, scale, bias
        total_int8 = int8_bytes + overhead_bytes
        if float32_bytes == 0:
            return {
                "float32_mb": 0.0,
                "int8_mb": total_int8 / (1024 * 1024),
                "savings_percent": 0.0,
                "compression_ratio": "0.0x",
            }
        return {
            "float32_mb": float32_bytes / (1024 * 1024),
            "int8_mb": total_int8 / (1024 * 1024),
            "savings_percent": (1.0 - total_int8 / float32_bytes) * 100.0,
            "compression_ratio": f"{float32_bytes / max(total_int8, 1):.1f}x",
        }

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        dims = len(self.min_vals) if self._fitted and self.min_vals is not None else 0
        return f"<ScalarQuantizer(status={status}, dimensions={dims}, percentile={self.percentile})>"
