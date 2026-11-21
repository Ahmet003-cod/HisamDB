# Donanım Hızlandırmalı Vektör Çekirdekleri (Rust SIMD / AVX2 / POPCNT)

Bu klasör, MicroVectorDB'nin saf NumPy çekirdeklerinin sınırına ulaştığı **PQ ADC lookup**, **asimetrik INT8 dot product** ve **64-bit POPCNT Binary Hamming** işlemlerini tek bir sıfır-bağımlılıklı native kütüphane (`mvdb_simd.rs`) üzerinden **makine kodu / SIMD (AVX2 + FMA3 + POPCNT)** hızına çıkaran opsiyonel çekirdekleri içerir.

---

## 1. Üretim Çekirdeği: `mvdb_simd.rs` (Rust `cdylib` — `v1.0.0`)

`microvectordb/metrics/native_simd.py` modülü, derlenmiş `mvdb_simd.dll` (Windows), `libmvdb_simd.so` (Linux) veya `libmvdb_simd.dylib` (macOS) kütüphanesini `ctypes` ile otomatik keşfeder ve:

- `ProductQuantizer.accumulate_adc()` & `IVFPQIndex.search()` (`mvdb_pq_adc_u8` / `mvdb_pq_adc_u16`) — ara `float32` matris kopyası oluşturmadan doğrudan LUT biriktirme,
- `ScalarQuantizer.compute_distances_asymmetric()` (`mvdb_int8_dot_f32`) — `int8` kodları `float32` kopyasına çevirmeden doğrudan SIMD dot-product,
- `BinaryQuantizer.hamming_distances()` (`mvdb_binary_hamming_u64`) — `u64::count_ones()` donanımsal `POPCNT` komutuyla tek adımda 64-bit Hamming mesafesi

işlemlerini otomatik olarak hızlandırır. Derlenmiş kütüphane yoksa sistem sıfır hata ile şeffaf biçimde **saf NumPy** yoluna döner.

### Derleme (Tek Komut, Harici Crate Yok)

#### Windows (PowerShell):
```powershell
rustc --crate-type cdylib -C opt-level=3 -C target-cpu=native kernels_experimental/mvdb_simd.rs -o kernels_experimental/mvdb_simd.dll
```

#### Linux:
```bash
rustc --crate-type cdylib -C opt-level=3 -C target-cpu=native kernels_experimental/mvdb_simd.rs -o kernels_experimental/libmvdb_simd.so
```

#### macOS (Apple Silicon / x86-64):
```bash
rustc --crate-type cdylib -C opt-level=3 -C target-cpu=native kernels_experimental/mvdb_simd.rs -o kernels_experimental/libmvdb_simd.dylib
```

### Python İçinden Kontrol & Açma/Kapatma

```python
import microvectordb

print("Native SIMD aktif mi?", microvectordb.is_native_simd_available())
print("Kütüphane yolu:", microvectordb.native_simd_path())

# Benchmark veya A/B doğrulama için çalışma zamanında aç/kapat:
microvectordb.set_native_simd_enabled(False)  # Saf NumPy yolu
microvectordb.set_native_simd_enabled(True)   # Native Rust SIMD yolu
```

> Ayrıca kütüphaneyi özel bir dizinde tutuyorsanız `MVDB_SIMD_LIB=/yol/mvdb_simd.so` ortam değişkenini verebilirsiniz.

---

## 2. Referans C / AVX2 Örneği: `avx2_dot_product.c`

Modern x86-64 işlemcilerde (Intel Core / AMD Ryzen) **AVX2 (Advanced Vector Extensions 2)** donanımı 256-bit genişliğinde YMM yazmaçları sunar:
- 1 adet float32 = 32 bit (4 byte)
- 256 bit / 32 bit = **8 adet float** tek komutla (`vfmadd231ps`) çarpılıp toplanır.

### GCC (MinGW / Linux):
```bash
gcc -O3 -mavx2 -mfma -shared -o avx2_kernel.dll avx2_dot_product.c
```

### MSVC (Visual Studio Geliştirici Komut İstemi):
```cmd
cl /O2 /arch:AVX2 /LD avx2_dot_product.c /Fe:avx2_kernel.dll
```

> **Not:** `microvectordb` çekirdeğinde `float32` yoğun (dense) matris çarpımları zaten OpenBLAS/MKL AVX2/AVX-512 çekirdeklerini kullanır. `mvdb_simd.rs` ise BLAS'ın desteklemediği **PQ lookup**, **asimetrik INT8** ve **64-bit POPCNT Binary Hamming** işlemlerindeki bellek bant genişliği darboğazını giderir.
