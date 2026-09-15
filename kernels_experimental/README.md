# Donanım Hızlandırmalı Vektör Çekirdekleri (AVX2 / SIMD & Assembly)

Bu klasör, sorduğunuz **"Bu yapıyı Assembly diliyle gerçek bir projeye çevirebilir miyiz?"** sorusunun pratik ve endüstri standardı cevabını içerir.

---

## 1. Neden Assembly / SIMD (AVX2)?

Normal bir Python veya C döngüsü, iki vektörü çarparken elemanları tek tek işler (Scalar Processing):
```c
for (int i = 0; i < 1024; i++) {
    sum += a[i] * b[i];  // 1024 adet tekil saat çevrimi
}
```

Modern x86-64 işlemcilerde (Intel Core i3/i5/i7/i9 ve AMD Ryzen) **AVX2 (Advanced Vector Extensions 2)** donanımı bulunur.
Bu donanımda **256-bit genişliğinde YMM yazmaçları (registers)** vardır:
- 1 adet float32 = 32 bit (4 byte)
- 256 bit / 32 bit = **8 adet float!**

`avx2_dot_product.c` dosyasındaki `_mm256_fmadd_ps` fonksiyonu, işlemciye doğrudan şu Assembly komutunu verdirir:
```nasm
vfmadd231ps ymm0, ymm1, ymm2
```
Bu tek bir komut, **8 adet sayıyı aynı anda çarpar ve toplar.** Dolayısıyla işlemci 8 kat daha az komut çalıştırarak aşırı bir hız kazanır.

---

## 2. Derleme (Compile to DLL / SO)

Eğer bilgisayarınızda GCC (MinGW) veya Microsoft Visual C++ (MSVC) kuruluysa:

### GCC (MinGW / Linux):
```bash
gcc -O3 -mavx2 -mfma -shared -o avx2_kernel.dll avx2_dot_product.c
```

### MSVC (Visual Studio Geliştirici Komut İstemi):
```cmd
cl /O2 /arch:AVX2 /LD avx2_dot_product.c /Fe:avx2_kernel.dll
```

---

## 3. Python'dan `ctypes` ile Çağırma Örneği

Derlenen `avx2_kernel.dll` dosyasını Python'da şu şekilde çağırabilirsiniz:

```python
import ctypes
import numpy as np

# 1. Derlenen DLL'i yükle
avx_lib = ctypes.CDLL("./avx2_kernel.dll")

# 2. Fonksiyon imzasını tanımla
avx_lib.avx2_dot_product.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t
]
avx_lib.avx2_dot_product.restype = ctypes.c_float

# 3. İki float32 vektör oluştur
a = np.random.randn(1024).astype(np.float32)
b = np.random.randn(1024).astype(np.float32)

# 4. Pointer'ları al ve Assembly hızında çalıştır
ptr_a = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
ptr_b = b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

result = avx_lib.avx2_dot_product(ptr_a, ptr_b, 1024)
print(f"AVX2 Dot Product Sonucu: {result:.4f}")
```

> **Not:** `microvectordb` çekirdeğinde kullandığımız NumPy kütüphanesi, kurulu olduğu sistemde zaten OpenBLAS veya Intel MKL ile derlenmiş olduğu için arka planda tam olarak bu AVX2/AVX-512 SIMD komutlarını çalıştırır.
