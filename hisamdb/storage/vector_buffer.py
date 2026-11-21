"""
hisamdb.storage.vector_buffer
-----------------------------------
Amortized O(1) ekleme sağlayan, kapasite-ikileyen (capacity-doubling) bitişik
vektör tamponu ve hizalı 1B norm önbellekleri (`sq_norms`, `inv_norms`).

Avantajlar:
- `np.vstack` kaynaklı O(N^2 * D) yeniden kopyalamayı ortadan kaldırır.
- Kosinüs araması için (N, D) ikinci kopya yerine yalnızca (N,) `inv_norms` (4B/vektör)
  tutar -> %50'ye yakın RAM tasarrufu.
- Öklid araması için (N,) `sq_norms` (4B/vektör) tutar -> >10x hızlanma.
- `float32` ve `float16` depolama tiplerini ve salt-okunur `mmap` bağlamayı destekler.
"""

import threading
from typing import Optional, Sequence, Tuple, Union
import numpy as np

from hisamdb.metrics.kernels import compute_inv_norms


class VectorBuffer:
    """Bitişik (C-contiguous) dinamik vektör tamponu."""

    def __init__(
        self,
        dimension: Optional[int] = None,
        dtype: Union[str, np.dtype] = np.float32,
        initial_capacity: int = 0,
    ) -> None:
        self.dtype = np.dtype(dtype)
        self._dim: int = dimension or 0
        self._size: int = 0
        self._capacity: int = 0
        self._is_mmap: bool = False
        self._norms_ready: bool = True
        self._norm_lock = threading.Lock()

        if initial_capacity > 0 and self._dim > 0:
            self._allocate(initial_capacity)
        else:
            self._buf: np.ndarray = np.empty((0, self._dim), dtype=self.dtype)
            self._sq_norms: np.ndarray = np.empty(0, dtype=np.float32)
            self._inv_norms: np.ndarray = np.empty(0, dtype=np.float32)

    def _allocate(self, capacity: int) -> None:
        self._buf = np.empty((capacity, self._dim), dtype=self.dtype)
        self._sq_norms = np.empty(capacity, dtype=np.float32)
        self._inv_norms = np.empty(capacity, dtype=np.float32)
        self._capacity = capacity
        self._is_mmap = False
        self._norms_ready = True

    def _ensure_norms(self) -> None:
        """MMap modunda ertelenmiş normları thread-safe biçimde yalnızca ilk ihtiyaç anında hesaplar."""
        if self._norms_ready:
            return
        with self._norm_lock:
            if self._norms_ready:
                return
            if self._size == 0:
                self._sq_norms = np.empty(0, dtype=np.float32)
                self._inv_norms = np.empty(0, dtype=np.float32)
            else:
                sq, inv = compute_inv_norms(self._buf[: self._size])
                self._sq_norms = sq
                self._inv_norms = inv
            self._norms_ready = True

    def _ensure_writable(self, min_capacity: int) -> None:
        """Tampon mmap ise veya kapasite yetersizse büyütüp yazılabilir hale getirir."""
        if not self._is_mmap and min_capacity <= self._capacity:
            return

        if self._size > 0 and not self._norms_ready:
            self._ensure_norms()

        new_cap = max(min_capacity, self._capacity * 2 if self._capacity > 0 else 64)
        new_buf = np.empty((new_cap, self._dim), dtype=self.dtype)
        new_sq = np.empty(new_cap, dtype=np.float32)
        new_inv = np.empty(new_cap, dtype=np.float32)

        if self._size > 0:
            new_buf[: self._size] = self._buf[: self._size]
            new_sq[: self._size] = self._sq_norms[: self._size]
            new_inv[: self._size] = self._inv_norms[: self._size]

        self.release_mmap()
        self._buf = new_buf
        self._sq_norms = new_sq
        self._inv_norms = new_inv
        self._capacity = new_cap
        self._is_mmap = False
        self._norms_ready = True

    def release_mmap(self) -> None:
        """Windows'ta dosya kilitlenmesini önlemek için açık memmap tutamacını kapatır."""
        if self._is_mmap and isinstance(self._buf, np.memmap):
            mm = getattr(self._buf, "_mmap", None)
            if mm is not None:
                try:
                    mm.close()
                except Exception:
                    pass
            self._buf = np.empty((0, self._dim), dtype=self.dtype)
            self._is_mmap = False

    @property
    def size(self) -> int:
        return self._size

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def is_mmap(self) -> bool:
        return self._is_mmap

    @property
    def view(self) -> np.ndarray:
        """Sıfır-kopya (zero-copy) aktif vektör görünümü (N, D)."""
        return self._buf[: self._size]

    @property
    def sq_norms(self) -> np.ndarray:
        """Sıfır-kopya aktif L2 norm karesi görünümü (N,)."""
        self._ensure_norms()
        return self._sq_norms[: self._size]

    @property
    def inv_norms(self) -> np.ndarray:
        """Sıfır-kopya aktif 1/L2 norm görünümü (N,)."""
        self._ensure_norms()
        return self._inv_norms[: self._size]

    def load_array(self, arr: np.ndarray, is_mmap: bool = False) -> None:
        """Diskten yüklenen matrisi tampona bağlar. MMap modunda normları tembel hesaplar."""
        self.release_mmap()
        if arr.size == 0:
            dim = arr.shape[1] if arr.ndim == 2 else self._dim
            self._dim = dim
            self._size = 0
            self._capacity = 0
            self._buf = np.empty((0, dim), dtype=self.dtype)
            self._sq_norms = np.empty(0, dtype=np.float32)
            self._inv_norms = np.empty(0, dtype=np.float32)
            self._is_mmap = False
            self._norms_ready = True
            return

        self._dim = arr.shape[1]
        self._size = arr.shape[0]
        self._capacity = self._size
        self.dtype = arr.dtype
        self._is_mmap = is_mmap
        self._buf = arr if is_mmap else np.ascontiguousarray(arr)
        if is_mmap:
            # Sıfır disk sayfası okuma ile anında mmap açılışı!
            self._sq_norms = np.empty(0, dtype=np.float32)
            self._inv_norms = np.empty(0, dtype=np.float32)
            self._norms_ready = False
        else:
            sq, inv = compute_inv_norms(self._buf)
            self._sq_norms = sq
            self._inv_norms = inv
            self._norms_ready = True

    def append(self, vectors: np.ndarray) -> Tuple[int, int]:
        """
        Yeni vektör bloğunu amortized O(1) sürede ekler.
        Dönüş: (start_idx, end_idx)
        """
        m = vectors.shape[0]
        if m == 0:
            return self._size, self._size

        if self._dim == 0:
            self._dim = vectors.shape[1]

        self._ensure_writable(self._size + m)
        start = self._size
        end = start + m

        if self.dtype == np.float16:
            block = np.clip(vectors, np.float32(-65504.0), np.float32(65504.0)).astype(np.float16)
        else:
            block = np.asarray(vectors, dtype=self.dtype)

        self._buf[start:end] = block
        # Normları depolanan blok üzerinden hesapla (float16 hassasiyetiyle tutarlı)
        sq, inv = compute_inv_norms(self._buf[start:end])
        self._sq_norms[start:end] = sq
        self._inv_norms[start:end] = inv
        self._size = end
        return start, end

    def update_row(self, idx: int, vec: np.ndarray) -> None:
        """Tek satırı O(D) sürede yerinde günceller."""
        self._ensure_writable(self._capacity or self._size)
        if self.dtype == np.float16:
            v_store = np.clip(vec, np.float32(-65504.0), np.float32(65504.0)).astype(np.float16)
        else:
            v_store = np.asarray(vec, dtype=self.dtype)
        self._buf[idx] = v_store
        v32 = self._buf[idx].astype(np.float32, copy=False)
        sq = np.float32(np.dot(v32, v32))
        norm = np.sqrt(sq)
        self._sq_norms[idx] = sq
        self._inv_norms[idx] = np.float32(1.0) / norm if norm > np.float32(0.0) else np.float32(0.0)

    def truncate(self, new_size: int) -> None:
        """Hata geri alması (rollback) için boyutu O(1) sürede küçültür."""
        if 0 <= new_size <= self._size:
            self._size = new_size

    def keep_indices(self, indices: Sequence[int]) -> None:
        """Silme sonrası yalnızca tutulan indekslerle tamponu sıkıştırır."""
        n_keep = len(indices)
        if n_keep == 0:
            self.release_mmap()
            self._size = 0
            self._capacity = 0
            self._buf = np.empty((0, self._dim), dtype=self.dtype)
            self._sq_norms = np.empty(0, dtype=np.float32)
            self._inv_norms = np.empty(0, dtype=np.float32)
            return

        idx_arr = np.asarray(indices, dtype=np.int64)
        new_buf = np.ascontiguousarray(self._buf[idx_arr], dtype=self.dtype)
        if self._norms_ready:
            new_sq = np.ascontiguousarray(self._sq_norms[idx_arr], dtype=np.float32)
            new_inv = np.ascontiguousarray(self._inv_norms[idx_arr], dtype=np.float32)
        else:
            new_sq, new_inv = compute_inv_norms(new_buf)
        self.release_mmap()
        self._buf = new_buf
        self._sq_norms = new_sq
        self._inv_norms = new_inv
        self._size = n_keep
        self._capacity = n_keep
        self._norms_ready = True

    def memory_bytes(self) -> int:
        """Tamponun RAM'de tuttuğu gerçek bayt miktarı (mmap vektör sayfaları hariç)."""
        vec_bytes = 0 if self._is_mmap else int(self._buf.nbytes)
        norm_bytes = (
            int(self._sq_norms.nbytes) + int(self._inv_norms.nbytes)
            if self._norms_ready
            else 0
        )
        return vec_bytes + norm_bytes
