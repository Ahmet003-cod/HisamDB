"""
microvectordb.storage.binary_format
-----------------------------------
Sürümlü, 64-byte hizalı, CRC32 korumalı, atomik yazma ve O(M) artımlı ekleme
destekli ikili vektör depolama formatı (`.mvdbvec`).

Header Düzeni (64 byte — AVX-512 / cache-line / np.memmap hizalı):
  [0..8)   Magic:       b"MVDBVEC\\x00" (8 byte)
  [8..10)  Version:     uint16 = 1      (2 byte)
  [10..12) DType Code:  uint16          (0=float32, 1=float16, 2=int8, 3=uint8)
  [12..16) Dimension:   uint32          (4 byte)
  [16..24) Count:       uint64          (8 byte)
  [24..28) HeaderCRC32: uint32          (CRC32 of bytes [0..24))
  [28..32) DataCRC32:   uint32          (Full data CRC32; 0 if incremental before checkpoint)
  [32..64) Reserved:    32 zero bytes
  [64..)   Raw C-contiguous row-major vector bytes
"""

import os
import struct
import threading
import time
import zlib
from typing import Tuple
import numpy as np

from microvectordb.types import DataCorruptionError

MAGIC = b"MVDBVEC\x00"
FORMAT_VERSION = 1
HEADER_SIZE = 64
_HEADER_PREFIX_FMT = "<8sHHIQ"  # 24 bytes
_HEADER_FULL_FMT = "<8sHHIQII32s"  # 64 bytes

DTYPE_TO_CODE = {
    np.dtype(np.float32): 0,
    np.dtype(np.float16): 1,
    np.dtype(np.int8): 2,
    np.dtype(np.uint8): 3,
}
CODE_TO_DTYPE = {v: k for k, v in DTYPE_TO_CODE.items()}


def _pack_header(dtype: np.dtype, dim: int, count: int, data_crc: int = 0) -> bytes:
    dt = np.dtype(dtype)
    if dt not in DTYPE_TO_CODE:
        raise ValueError(f"Desteklenmeyen vektör veri tipi: {dt}")
    code = DTYPE_TO_CODE[dt]
    prefix = struct.pack(_HEADER_PREFIX_FMT, MAGIC, FORMAT_VERSION, code, int(dim), int(count))
    header_crc = zlib.crc32(prefix) & 0xFFFFFFFF
    return struct.pack(
        _HEADER_FULL_FMT,
        MAGIC,
        FORMAT_VERSION,
        code,
        int(dim),
        int(count),
        header_crc,
        int(data_crc) & 0xFFFFFFFF,
        b"\x00" * 32,
    )


def _unpack_header(raw: bytes, path: str) -> Tuple[np.dtype, int, int, int]:
    """Dönüş: (dtype, dim, count, data_crc)."""
    if len(raw) < HEADER_SIZE:
        raise DataCorruptionError(f"Vektör dosyası başlığı eksik/bozuk: '{path}'")
    magic, version, code, dim, count, header_crc, data_crc, _ = struct.unpack(
        _HEADER_FULL_FMT, raw[:HEADER_SIZE]
    )
    if magic != MAGIC:
        raise DataCorruptionError(f"Geçersiz vektör dosyası imzası (magic): '{path}'")
    if version > FORMAT_VERSION:
        raise DataCorruptionError(
            f"Desteklenmeyen vektör dosya sürümü ({version} > {FORMAT_VERSION}): '{path}'"
        )
    expected_crc = zlib.crc32(raw[:24]) & 0xFFFFFFFF
    if header_crc != expected_crc:
        raise DataCorruptionError(f"Vektör dosyası başlık CRC32 uyuşmazlığı: '{path}'")
    if code not in CODE_TO_DTYPE:
        raise DataCorruptionError(f"Bilinmeyen vektör veri tipi kodu ({code}): '{path}'")
    return CODE_TO_DTYPE[code], int(dim), int(count), int(data_crc)


def _compute_crc32_array(arr: np.ndarray) -> int:
    """Büyük diziler için bellek dostu bloklu CRC32 hesaplar."""
    if arr.size == 0:
        return 0
    arr_c = np.ascontiguousarray(arr)
    mv = memoryview(arr_c).cast("B")
    crc = 0
    chunk = 4 * 1024 * 1024  # 4 MB
    for offset in range(0, len(mv), chunk):
        crc = zlib.crc32(mv[offset : offset + chunk], crc)
    return crc & 0xFFFFFFFF


def write_vectors_atomic(
    path: str,
    vectors: np.ndarray,
    compute_data_crc: bool = True,
) -> None:
    """
    Vektör matrisini geçici dosyaya yazıp fsync sonrası os.replace() ile atomik değiştirir.
    Çökme anında mevcut dosya asla yarıda kesilmez.
    """
    arr = np.ascontiguousarray(vectors)
    if arr.ndim != 2:
        raise ValueError(f"2B matris bekleniyor, gelen şekil: {arr.shape}")
    count, dim = arr.shape
    data_crc = _compute_crc32_array(arr) if (compute_data_crc and arr.size > 0) else 0
    header = _pack_header(arr.dtype, dim, count, data_crc=data_crc)

    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}"

    try:
        with open(tmp_path, "wb") as f:
            f.write(header)
            if arr.size > 0:
                f.write(memoryview(arr).cast("B"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def append_vectors(
    path: str,
    new_vectors: np.ndarray,
    expected_prev_count: int,
) -> None:
    """
    Yeni eklenen M vektörü dosyanın sonuna O(M) sürede ekler.
    Dosya yoksa veya expected_prev_count == 0 ise atomik tam yazma yapar.
    Önceki yarım kalmış bir ekleme varsa (file_size > expected_end), üzerine yazıp temizler.
    """
    arr = np.ascontiguousarray(new_vectors)
    if arr.shape[0] == 0:
        return
    if expected_prev_count == 0 or not os.path.exists(path):
        write_vectors_atomic(path, arr, compute_data_crc=False)
        return

    with open(path, "r+b") as f:
        raw_header = f.read(HEADER_SIZE)
        dtype, dim, file_count, _ = _unpack_header(raw_header, path)
        if dim != arr.shape[1] or dtype != arr.dtype:
            raise DataCorruptionError(
                f"Vektör ekleme uyumsuzluğu: dosya=({dim}, {dtype}), gelen=({arr.shape[1]}, {arr.dtype})"
            )

        row_bytes = dim * dtype.itemsize
        expected_offset = HEADER_SIZE + expected_prev_count * row_bytes
        f.seek(0, os.SEEK_END)
        actual_size = f.tell()
        if actual_size < expected_offset or file_count < expected_prev_count:
            raise DataCorruptionError(
                f"Vektör dosyası beklenenden kısa: dosya_satir={file_count}, beklenen={expected_prev_count}"
            )

        # Yarım kalmış ekleme varsa tam commit edilmiş ofsetten devam et
        f.seek(expected_offset)
        f.write(memoryview(arr).cast("B"))
        new_count = expected_prev_count + arr.shape[0]
        new_end = HEADER_SIZE + new_count * row_bytes
        f.truncate(new_end)
        f.flush()

        # Başlığı güncelle (data_crc=0: checkpoint'te tam CRC hesaplanır)
        new_header = _pack_header(dtype, dim, new_count, data_crc=0)
        f.seek(0)
        f.write(new_header)
        f.flush()
        os.fsync(f.fileno())


def update_vector_row_in_file(path: str, row_idx: int, vec: np.ndarray) -> None:
    """Tek bir satırı diskte yerinde O(D) sürede günceller (upsert için)."""
    if not os.path.exists(path):
        return
    with open(path, "r+b") as f:
        raw_header = f.read(HEADER_SIZE)
        dtype, dim, count, _ = _unpack_header(raw_header, path)
        if row_idx < 0 or row_idx >= count:
            raise IndexError(f"Satır indeksi sınır dışı: {row_idx} / {count}")
        v = np.ascontiguousarray(vec, dtype=dtype).ravel()
        if v.shape[0] != dim:
            raise ValueError(f"Boyut uyumsuzluğu: beklenen {dim}, gelen {v.shape[0]}")

        row_bytes = dim * dtype.itemsize
        f.seek(HEADER_SIZE + row_idx * row_bytes)
        f.write(memoryview(v).cast("B"))
        # Data CRC'yi 0'a sıfırla (artık güncel değil)
        new_header = _pack_header(dtype, dim, count, data_crc=0)
        f.seek(0)
        f.write(new_header)
        f.flush()
        os.fsync(f.fileno())


def truncate_vectors_file(path: str, target_count: int) -> None:
    """
    Çökme kurtarmasında (SQLite'ta commit edilmemiş fazla eklemeleri temizlemek için)
    vektör dosyasını güvenli biçimde target_count satıra kırpar.
    """
    with open(path, "r+b") as f:
        raw_header = f.read(HEADER_SIZE)
        dtype, dim, file_count, _ = _unpack_header(raw_header, path)
        if target_count < 0 or target_count > file_count:
            raise DataCorruptionError(
                f"Kırpma hedefi geçersiz: hedef={target_count}, dosya={file_count}"
            )
        row_bytes = dim * dtype.itemsize
        f.truncate(HEADER_SIZE + target_count * row_bytes)
        new_header = _pack_header(dtype, dim, target_count, data_crc=0)
        f.seek(0)
        f.write(new_header)
        f.flush()
        os.fsync(f.fileno())


def load_vectors_file(
    path: str,
    use_mmap: bool = False,
    verify_crc: bool = True,
) -> np.ndarray:
    """
    Sürümlü `.mvdbvec` dosyasını doğrular ve yükler.
    `use_mmap=True` ise sıfır RAM kopyası ile salt-okunur `np.memmap` döndürür.
    """
    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        raw_header = f.read(HEADER_SIZE)
    dtype, dim, count, data_crc = _unpack_header(raw_header, path)

    row_bytes = dim * dtype.itemsize
    expected_size = HEADER_SIZE + count * row_bytes
    if file_size < expected_size:
        raise DataCorruptionError(
            f"Vektör dosyası kesilmiş/bozuk: boyut={file_size}, beklenen>={expected_size} ('{path}')"
        )

    if count == 0:
        return np.empty((0, dim), dtype=dtype)

    if use_mmap:
        return np.memmap(
            path,
            dtype=dtype,
            mode="r",
            offset=HEADER_SIZE,
            shape=(count, dim),
        )

    with open(path, "rb") as f:
        f.seek(HEADER_SIZE)
        buf = f.read(count * row_bytes)
    arr = np.frombuffer(buf, dtype=dtype).reshape((count, dim)).copy()

    if verify_crc and data_crc != 0:
        actual_crc = _compute_crc32_array(arr)
        if actual_crc != data_crc:
            raise DataCorruptionError(
                f"Vektör veri CRC32 doğrulaması başarısız (bozuk dosya): '{path}'"
            )
    return arr
