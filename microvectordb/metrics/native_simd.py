"""
microvectordb.metrics.native_simd
---------------------------------
Opsiyonel, sıfır bağımlılıklı yerel (Native Rust/C AVX2+FMA+POPCNT) SIMD hızlandırıcı yükleyicisi.

- `kernels_experimental/mvdb_simd.{dll,so,dylib}` derlenmişse `ctypes` ile otomatik yükler.
- Derlenmemişse hiçbir hata vermez; tüm çekirdek %100 saf NumPy yolundan çalışmaya devam eder.

Derleme (isteğe bağlı, Rust kuruluysa):
    Windows: rustc --crate-type cdylib -C opt-level=3 -C target-cpu=native kernels_experimental/mvdb_simd.rs -o kernels_experimental/mvdb_simd.dll
    Linux:   rustc --crate-type cdylib -C opt-level=3 -C target-cpu=native kernels_experimental/mvdb_simd.rs -o kernels_experimental/libmvdb_simd.so
    macOS:   rustc --crate-type cdylib -C opt-level=3 -C target-cpu=native kernels_experimental/mvdb_simd.rs -o kernels_experimental/libmvdb_simd.dylib
"""

import ctypes
import os
import sys
from typing import Optional
import numpy as np

_F32_P = ctypes.POINTER(ctypes.c_float)
_U8_P = ctypes.POINTER(ctypes.c_uint8)
_I8_P = ctypes.POINTER(ctypes.c_int8)
_U16_P = ctypes.POINTER(ctypes.c_uint16)
_U32_P = ctypes.POINTER(ctypes.c_uint32)
_U64_P = ctypes.POINTER(ctypes.c_uint64)

_LIB: Optional[ctypes.CDLL] = None
_LIB_PATH: Optional[str] = None
_NATIVE_ENABLED: bool = True


def _candidate_paths() -> list[str]:
    if sys.platform == "win32":
        names = ["mvdb_simd.dll"]
    elif sys.platform == "darwin":
        names = ["libmvdb_simd.dylib", "mvdb_simd.dylib"]
    else:
        names = ["libmvdb_simd.so", "mvdb_simd.so"]

    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    dirs = [
        here,
        os.path.join(repo_root, "kernels_experimental"),
    ]
    env_path = os.environ.get("MVDB_SIMD_LIB") or os.environ.get("MVDB_NATIVE_SIMD_PATH")
    out: list[str] = [env_path] if env_path else []
    for d in dirs:
        for nm in names:
            out.append(os.path.join(d, nm))
    return out


def _try_load() -> Optional[ctypes.CDLL]:
    global _LIB, _LIB_PATH
    if _LIB is not None:
        return _LIB
    for path in _candidate_paths():
        if not path or not os.path.exists(path):
            continue
        try:
            lib = ctypes.CDLL(path)
            lib.mvdb_pq_adc_u8.argtypes = [
                _F32_P, _U8_P, _F32_P, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float
            ]
            lib.mvdb_pq_adc_u8.restype = None

            lib.mvdb_pq_adc_u16.argtypes = [
                _F32_P, _U16_P, _F32_P, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float
            ]
            lib.mvdb_pq_adc_u16.restype = None

            lib.mvdb_int8_dot_f32.argtypes = [
                _F32_P, _I8_P, _F32_P, ctypes.c_size_t, ctypes.c_size_t
            ]
            lib.mvdb_int8_dot_f32.restype = None

            lib.mvdb_binary_hamming_u64.argtypes = [
                _U64_P, _U64_P, _U32_P, ctypes.c_size_t, ctypes.c_size_t
            ]
            lib.mvdb_binary_hamming_u64.restype = None

            _LIB = lib
            _LIB_PATH = path
            return _LIB
        except OSError:
            continue
    return None


# Modül yüklenirken sessizce kontrol et
_try_load()


def is_native_simd_available() -> bool:
    """Derlenmiş yerel AVX2/SIMD çekirdeğinin yüklü ve etkin olup olmadığını döndürür."""
    return _NATIVE_ENABLED and (_LIB is not None)


def set_native_simd_enabled(enabled: bool) -> None:
    """Benchmark ve karşılaştırma testleri için yerel SIMD hızlandırıcısını açıp kapatır."""
    global _NATIVE_ENABLED
    _NATIVE_ENABLED = bool(enabled)


def native_simd_path() -> Optional[str]:
    return _LIB_PATH if _LIB is not None else None


def pq_adc_u8_native(lut: np.ndarray, codes_t: np.ndarray, bias: float = 0.0) -> Optional[np.ndarray]:
    """`(M, K)` float32 LUT + `(M, N)` uint8 codes_t için yerel SIMD ADC toplamı."""
    if not is_native_simd_available() or _LIB is None:
        return None
    m_cnt, n = codes_t.shape
    if n == 0:
        return np.empty(0, dtype=np.float32)
    lut_c = np.ascontiguousarray(lut, dtype=np.float32)
    codes_c = np.ascontiguousarray(codes_t, dtype=np.uint8)
    out = np.empty(n, dtype=np.float32)
    _LIB.mvdb_pq_adc_u8(
        lut_c.ctypes.data_as(_F32_P),
        codes_c.ctypes.data_as(_U8_P),
        out.ctypes.data_as(_F32_P),
        m_cnt,
        lut_c.shape[1],
        n,
        float(bias),
    )
    return out


def pq_adc_u16_native(flat_lut: np.ndarray, codes_t: np.ndarray, bias: float = 0.0) -> Optional[np.ndarray]:
    """`(M*K,)` float32 düz LUT + `(M, N)` uint16 codes_t için yerel SIMD ADC toplamı."""
    if not is_native_simd_available() or _LIB is None:
        return None
    m_cnt, n = codes_t.shape
    if n == 0:
        return np.empty(0, dtype=np.float32)
    lut_c = np.ascontiguousarray(flat_lut, dtype=np.float32)
    codes_c = np.ascontiguousarray(codes_t, dtype=np.uint16)
    out = np.empty(n, dtype=np.float32)
    _LIB.mvdb_pq_adc_u16(
        lut_c.ctypes.data_as(_F32_P),
        codes_c.ctypes.data_as(_U16_P),
        out.ctypes.data_as(_F32_P),
        m_cnt,
        lut_c.size,
        n,
        float(bias),
    )
    return out


def int8_dot_f32_native(q_scaled: np.ndarray, db_i8: np.ndarray) -> Optional[np.ndarray]:
    """`(D,)` float32 sorgu ile `(N, D)` int8 veritabanı arasında sıfır-kopya AVX2 iç çarpım."""
    if not is_native_simd_available() or _LIB is None:
        return None
    n, d = db_i8.shape
    if n == 0:
        return np.empty(0, dtype=np.float32)
    q_c = np.ascontiguousarray(q_scaled, dtype=np.float32)
    db_c = np.ascontiguousarray(db_i8, dtype=np.int8)
    out = np.empty(n, dtype=np.float32)
    _LIB.mvdb_int8_dot_f32(
        q_c.ctypes.data_as(_F32_P),
        db_c.ctypes.data_as(_I8_P),
        out.ctypes.data_as(_F32_P),
        n,
        d,
    )
    return out


def binary_hamming_u64_native(q_words: np.ndarray, db_words: np.ndarray) -> Optional[np.ndarray]:
    """`(W,)` uint64 sorgu ile `(N, W)` uint64 veritabanı arasında donanım POPCNT Hamming."""
    if not is_native_simd_available() or _LIB is None:
        return None
    n, w = db_words.shape
    if n == 0:
        return np.empty(0, dtype=np.uint32)
    q_c = np.ascontiguousarray(q_words, dtype=np.uint64)
    db_c = np.ascontiguousarray(db_words, dtype=np.uint64)
    out = np.empty(n, dtype=np.uint32)
    _LIB.mvdb_binary_hamming_u64(
        q_c.ctypes.data_as(_U64_P),
        db_c.ctypes.data_as(_U64_P),
        out.ctypes.data_as(_U32_P),
        n,
        w,
    )
    return out
