/**
 * avx2_dot_product.c
 * -------------------
 * x86-64 AVX2 / FMA SIMD Komut Seti ile Optimize Edilmiş Dot Product (İç Çarpım) Çekirdeği.
 *
 * Tek bir CPU saat çevriminde 8 adet 32-bit float sayıyı paralel olarak çarpar ve toplar.
 * Normal döngüye göre 4-8 kat daha yüksek işlemci verimi sağlar.
 *
 * Derleme Talimatları (Windows / Linux):
 *   GCC / MinGW (Windows / Linux):
 *     gcc -O3 -mavx2 -mfma -shared -o avx2_kernel.dll avx2_dot_product.c
 *
 *   MSVC (cl.exe):
 *     cl /O2 /arch:AVX2 /LD avx2_dot_product.c /Fe:avx2_kernel.dll
 */

#include <immintrin.h>
#include <stddef.h>

#if defined(_WIN32) || defined(__CYGWIN__)
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT __attribute__((visibility("default")))
#endif

/**
 * AVX2 FMA kullanarak iki float dizisi arasındaki iç çarpımı (dot product) hesaplar.
 *
 * @param a       İlk vektörün float pointer'ı (32-byte hizalı olması önerilir)
 * @param b       İkinci vektörün float pointer'ı
 * @param length  Vektör boyutu (örn: 1024)
 * @return        İç çarpım skoru (float)
 */
EXPORT float avx2_dot_product(const float* a, const float* b, size_t length) {
    // 256-bit YMM yazmacını sıfırla (8 adet float tutar)
    __m256 sum = _mm256_setzero_ps();

    size_t i = 0;
    // 8'er 8'er ilerle (256 bit = 8 x 32-bit float)
    size_t unrolled_len = length - (length % 8);

    for (i = 0; i < unrolled_len; i += 8) {
        // a ve b dizilerinden 8'er float yükle (vmovups komutu)
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);

        // FMA (Fused Multiply-Add): sum += va * vb (vfmadd231ps komutu)
        // Tek bir assembly komutunda hem çarpma hem toplama yapılır
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    // YMM yazmacındaki 8 float değeri toplayarak tek bir skaler float'a indirge
    // Yatay toplama (horizontal add)
    float buffer[8];
    _mm256_storeu_ps(buffer, sum);
    float total = buffer[0] + buffer[1] + buffer[2] + buffer[3] +
                  buffer[4] + buffer[5] + buffer[6] + buffer[7];

    // 8'e bölünmeyen kalan elemanları klasik yöntemle topla
    for (; i < length; ++i) {
        total += a[i] * b[i];
    }

    return total;
}
