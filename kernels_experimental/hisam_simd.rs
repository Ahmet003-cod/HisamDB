//! HisamDB Native SIMD Kernel (Zero-Dependency Rust -> C ABI cdylib)
//!
//! Derleme (Windows / Linux / macOS — harici crate gerektirmez!):
//!   Windows: rustc --crate-type cdylib -C opt-level=3 -C target-cpu=native kernels_experimental/hisam_simd.rs -o kernels_experimental/hisam_simd.dll
//!   Linux:   rustc --crate-type cdylib -C opt-level=3 -C target-cpu=native kernels_experimental/hisam_simd.rs -o kernels_experimental/libhisam_simd.so
//!   macOS:   rustc --crate-type cdylib -C opt-level=3 -C target-cpu=native kernels_experimental/hisam_simd.rs -o kernels_experimental/libhisam_simd.dylib

use std::slice;

/// Sütun-öncelikli `(M, N)` uint8 PQ kodları ve `(M, K)` float32 LUT üzerinden
/// sıfır ara-matris tahsisli, L1-cache bloklu ADC akümülasyonu.
#[no_mangle]
pub unsafe extern "C" fn mvdb_pq_adc_u8(
    lut_ptr: *const f32,
    codes_t_ptr: *const u8,
    out_ptr: *mut f32,
    m_cnt: usize,
    k_cnt: usize,
    n: usize,
    bias: f32,
) {
    if n == 0 || m_cnt == 0 {
        return;
    }
    let lut = slice::from_raw_parts(lut_ptr, m_cnt * k_cnt);
    let codes = slice::from_raw_parts(codes_t_ptr, m_cnt * n);
    let out = slice::from_raw_parts_mut(out_ptr, n);

    const TILE: usize = 256;
    let mut start = 0usize;
    while start < n {
        let end = (start + TILE).min(n);
        let len = end - start;
        let out_tile = &mut out[start..end];

        // İlk alt-vektör: doğrudan bias + lut[0, code]
        let lut0 = &lut[0..k_cnt];
        let c0 = &codes[start..end];
        for i in 0..len {
            out_tile[i] = bias + *lut0.get_unchecked(*c0.get_unchecked(i) as usize);
        }

        // Kalan alt-vektörler: L1 önbellek içi ekleme (sıfır bellek tahsisi!)
        for m in 1..m_cnt {
            let lut_m = &lut[m * k_cnt..(m + 1) * k_cnt];
            let c_m = &codes[m * n + start..m * n + end];
            for i in 0..len {
                *out_tile.get_unchecked_mut(i) +=
                    *lut_m.get_unchecked(*c_m.get_unchecked(i) as usize);
            }
        }
        start = end;
    }
}

/// Ön-kaydırmalı `(M, N)` uint16 PQ kodları ve düz `(M * K)` float32 LUT için
/// L1-cache bloklu ADC akümülasyonu (IVF-PQ sıcak yolu).
#[no_mangle]
pub unsafe extern "C" fn mvdb_pq_adc_u16(
    flat_lut_ptr: *const f32,
    codes_t_ptr: *const u16,
    out_ptr: *mut f32,
    m_cnt: usize,
    flat_lut_len: usize,
    n: usize,
    bias: f32,
) {
    if n == 0 || m_cnt == 0 {
        return;
    }
    let lut = slice::from_raw_parts(flat_lut_ptr, flat_lut_len);
    let codes = slice::from_raw_parts(codes_t_ptr, m_cnt * n);
    let out = slice::from_raw_parts_mut(out_ptr, n);

    const TILE: usize = 256;
    let mut start = 0usize;
    while start < n {
        let end = (start + TILE).min(n);
        let len = end - start;
        let out_tile = &mut out[start..end];

        let c0 = &codes[start..end];
        for i in 0..len {
            out_tile[i] = bias + *lut.get_unchecked(*c0.get_unchecked(i) as usize);
        }

        for m in 1..m_cnt {
            let c_m = &codes[m * n + start..m * n + end];
            for i in 0..len {
                *out_tile.get_unchecked_mut(i) +=
                    *lut.get_unchecked(*c_m.get_unchecked(i) as usize);
            }
        }
        start = end;
    }
}

/// INT8 kuantize vektör matrisi `(N, D)` ile ölçeklenmiş float32 sorgu vektörü `(D,)`
/// arasında doğrudan int8->f32 SIMD (AVX2 FMA) iç çarpımı (sıfır (N, D) float32 kopyası!).
#[no_mangle]
pub unsafe extern "C" fn mvdb_int8_dot_f32(
    q_scaled_ptr: *const f32,
    db_i8_ptr: *const i8,
    out_ptr: *mut f32,
    n: usize,
    d: usize,
) {
    if n == 0 || d == 0 {
        return;
    }
    let q = slice::from_raw_parts(q_scaled_ptr, d);
    let db = slice::from_raw_parts(db_i8_ptr, n * d);
    let out = slice::from_raw_parts_mut(out_ptr, n);

    for i in 0..n {
        let row = &db[i * d..(i + 1) * d];
        // 4'lü unroll: LLVM AVX2 FMA boru hattını doyurur
        let mut s0 = 0.0f32;
        let mut s1 = 0.0f32;
        let mut s2 = 0.0f32;
        let mut s3 = 0.0f32;
        let mut j = 0usize;
        while j + 4 <= d {
            s0 += *q.get_unchecked(j) * (*row.get_unchecked(j) as f32);
            s1 += *q.get_unchecked(j + 1) * (*row.get_unchecked(j + 1) as f32);
            s2 += *q.get_unchecked(j + 2) * (*row.get_unchecked(j + 2) as f32);
            s3 += *q.get_unchecked(j + 3) * (*row.get_unchecked(j + 3) as f32);
            j += 4;
        }
        let mut acc = (s0 + s1) + (s2 + s3);
        while j < d {
            acc += *q.get_unchecked(j) * (*row.get_unchecked(j) as f32);
            j += 1;
        }
        *out.get_unchecked_mut(i) = acc;
    }
}

/// Paketli uint64 ikili vektörler `(N, W)` üzerinde doğrudan donanım POPCNT Hamming mesafesi
/// (ara XOR matrisi tahsis etmez!).
#[no_mangle]
pub unsafe extern "C" fn mvdb_binary_hamming_u64(
    q_words_ptr: *const u64,
    db_words_ptr: *const u64,
    out_ptr: *mut u32,
    n: usize,
    n_words: usize,
) {
    if n == 0 || n_words == 0 {
        return;
    }
    let q = slice::from_raw_parts(q_words_ptr, n_words);
    let db = slice::from_raw_parts(db_words_ptr, n * n_words);
    let out = slice::from_raw_parts_mut(out_ptr, n);

    for i in 0..n {
        let row = &db[i * n_words..(i + 1) * n_words];
        let mut dist = 0u32;
        for w in 0..n_words {
            dist += (*q.get_unchecked(w) ^ *row.get_unchecked(w)).count_ones();
        }
        *out.get_unchecked_mut(i) = dist;
    }
}
