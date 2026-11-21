# HisamDB v1.0.0 — Büyük Ölçek, Native SIMD, GPU ve Dağıtık Mimari Araştırma Raporu

Bu belge, HisamDB `v1.0.0` Faz 6 kapsamında yürütülen **1.000.000 vektör ölçek testleri**, **384D / 768D / 1536D LLM embedding ölçümleri**, **opsiyonel Rust Native SIMD (AVX2 + FMA3 + POPCNT) çekirdeğinin saf NumPy ve C++ FAISS'e karşı bilimsel analizi**, **GPU/CUDA fizibilitesi** ve **`v2.x` dağıtık mimari tasarım taslağını** içerir.

> **Ölçüm Donanımı:** Windows x86-64, 12 mantıksal Intel çekirdeği (`AVX2`, `FMA3`, `F16C`, `POPCNT`), 15.73 GB RAM, Python 3.11.9, NumPy 2.0.2 (OpenBLAS 0.3.27), `rustc 1.96.0`, FAISS 1.13.2. Tüm veriler [`benchmarks/results/benchmark_v1.0.0.json`](../benchmarks/results/benchmark_v1.0.0.json) dosyasından alınmıştır.

---

## 1. Native SIMD (Rust `hisam_simd.rs`) vs Saf NumPy vs C++ FAISS Analizi

### 1.1. Neden Bazı İşlemlerde NumPy/OpenBLAS Yeterli, Bazılarında Değil?

Modern NumPy, `float32` matris çarpımlarını (`np.dot` / `SGEMV` / `SGEMM`) zaten **OpenBLAS AVX2/AVX-512 Assembly** çekirdeklerine devreder. Bu nedenle saf `float32` Flat ve IVF aramalarında harici C/Rust kodu yazmak kayda değer bir kazanım üretmez (HisamDB Flat Float32, FAISS `IndexFlatIP` ile neredeyse aynı hızdadır).

Ancak BLAS kütüphanelerinin **desteklemediği** üç kritik sıkıştırılmış arama çekirdeğinde saf NumPy ağır ara bellek kopyaları oluşturur:

1. **Product Quantization (PQ & IVF-PQ) ADC Toplaması:** NumPy `np.take` veya alt-vektör döngüsü her adımda geçici `(M, N)` `float32` dizisi tahsis edip RAM'e yazar ve geri okur (**bellek bant genişliği darboğazı**).
2. **Asimetrik INT8 Dot Product (`int8 × float32`):** NumPy `int8` diziyi BLAS'a vermeden önce `float32` çalışma tamponuna kopyalamak zorundadır (4x fazla bellek trafiği).
3. **64-bit Binary Hamming (`XOR + POPCNT`):** NumPy'da `XOR` ara dizisi bellek tahsis eder; oysa tek bir Rust döngüsü CPU'nun donanımsal `POPCNT` komutunu (`u64::count_ones()`) yazmaç içinde sıfır ara tahsisle çalıştırır.

### 1.2. Gerçek Ölçülmüş Kazanımlar (`100K` Vektör, Cosine)

| Çekirdek & Ölçek | Saf NumPy P50 | Rust Native SIMD P50 | Hızlanma (NumPy→SIMD) | C++ FAISS 1.13.2 P50 | MVDB vs FAISS Notu |
|---|---:|---:|---:|---:|---|
| **IVF-PQ (`100K × 128D`, C=128, np=8)** | 1.180 ms (773 QPS) | **0.620 ms (1,575 QPS)** | **1.90x (2.04x QPS)** | 0.573 ms (1,759 QPS) | MVDB P95 (`0.77 ms`) < FAISS (`0.94 ms`) ve **+%49.5 yüksek Recall** (`0.293` vs `0.196`)! |
| **Flat PQ (`100K × 128D`, M=16)** | 9.415 ms (101 QPS) | **1.948 ms (514 QPS)** | **4.83x (5.06x QPS)** | 1.098 ms (698 QPS) | Tek iş parçacıklı Rust ile P95 (`2.18 ms`) < FAISS (`2.22 ms`) |
| **INT8 Scalar (`100K × 128D`)** | 15.880 ms (62 QPS) | **9.269 ms (100 QPS)** | **1.71x** | 5.583 ms (181 QPS) | MVDB Recall (`0.979`) > FAISS `SQ8` (`0.972`) |
| **Binary POPCNT (`100K × 768D`)** | 13.098 ms (73 QPS) | **3.022 ms (320 QPS)** | **4.33x (4.39x QPS)** | — | 768D vektörler yalnızca **9.16 MB RAM** (32x sıkıştırma) |
| **INT8 (`50K × 768D` BGE/E5)** | 30.763 ms | **16.106 ms** | **1.91x** | — | Recall@10 = **0.968**, RAM **37.0 MB** (4x tasarruf) |
| **INT8 (`25K × 1536D` OpenAI)** | 26.025 ms | **14.096 ms** | **1.85x** | — | Recall@10 = **0.986**, RAM **36.8 MB** (4x tasarruf) |

**Mühendislik Kararı:** `kernels_experimental/hisam_simd.rs` (~140 satır, sıfır harici crate) tek `rustc` komutuyla derlenip `ctypes` üzerinden şeffaf biçimde devreye girer; derlenmediği ortamlarda HisamDB sıfır hata ile %100 saf NumPy yolunu korur. Böylece **sıfır zorunlu derleyici / sıfır ek bağımlılık** ilkesi bozulmadan C++ FAISS hız bandına erişilir.

---

## 2. Büyük Ölçek (`1M`) ve Yüksek Boyut (`384D–1536D`) Mimari Bulguları

### 2.1. 1.000.000 Vektörde Bellek Duvarı (Memory Wall)
Ham `1M × 128D` `float32` matrisi **488.3 MB**, `HNSW (M=16)` ise grafik bağlantılarıyla **625.6 MB** RAM tüketir (`1M × 1536D` OpenAI embedding'lerinde bu rakam **~6.2 GB RAM**'e çıkar!).

HisamDB `v1.0.0` ölçümleri, tek makinede RAM duvarını aşmanın iki pratik yolunu doğrulamıştır:
- **`DiskIndex (MMap)`:** `1,000,000 × 128D` vektörü yalnızca **15.52 MB RAM (-%96.8)** ile **9.56 ms P50** ve **%100.0 (`1.000`) Recall@10** ile arar! `25K × 1536D` OpenAI vektörlerinde ise **0.95 MB RAM (-%99.35)** ile **1.50 ms** (RAM'deki Flat Float32'den **4.2x daha hızlı!**).
- **`IVF-PQ (Native SIMD)`:** `1,000,000 × 128D` vektörü **42.71 MB RAM (-%91.3)** ile **2.36 ms (425 QPS)** hızında arar. `500,000` vektörde MVDB `IVF-PQ` (`1.55 ms`, R@10=`0.192`), C++ FAISS `IndexIVFPQ`'yu (`1.67 ms`, R@10=`0.186`) hem gecikme hem doğrulukta geride bırakmıştır.

### 2.2. Yüksek Boyutta Kuantizasyon Dürüst Analizi
- **Binary + Rescore (`oversample=20x`):** Boyut büyüdükçe ikili kuantizasyonun yön bilgisi dramatik biçimde güçlenir: `1536D` OpenAI vektörlerinde saf Binary `0.136` olan Recall@10, ilk 200 adayı `float32` ile yeniden puanlayarak yalnızca **`0.975 ms`**'de **`0.884` (%88.4)** düzeyine sıçrar (ikili indeks yalnızca `4.58 MB` RAM kullanır!).
- **Float16 x86-64 Gözlemi (Dürüst Uyarı):** `Float16` RAM'i tam yarıya indirir ve `Recall@10 = 0.998–1.000` verir; ancak AVX-512 FP16 donanımı olmayan x86-64 işlemcilerde OpenBLAS `float16` doğrudan GEMM desteklemediği için NumPy sorgu anında `float32`'ye yazılım dönüşümü yapar (`50K×768D`'de `61.4 ms`). Bu nedenle **RAM tasarrufu + yüksek hız** arayan kullanıcılara `Float16` yerine **`INT8 (Rust SIMD)`** veya **`DiskIndex (MMap)`** önerilir.

---

## 3. GPU / CUDA / CuPy Hızlandırma Fizibilite Analizi

"HisamDB'ye GPU/CUDA desteği eklemeli miyiz?" sorusu veri aktarım (PCIe) ve hesaplama maliyetine göre bilimsel olarak değerlendirilmiştir:

| İş Yükü Profili | CPU (AVX2 / HNSW) | GPU (CUDA / CuPy) Tahmini | Kazanan & Gerekçe |
|---|---|---|---|
| **Tekil Gerçek-Zamanlı Sorgu (`Q=1`, `N ≤ 1M`)** | **0.22 – 0.60 ms** (HNSW) | 0.8 – 2.5 ms *(PCIe + kernel launch)* | **CPU kazanır.** HNSW grafik gezintisi düzensiz bellek erişimlidir (branch-heavy); GPU warp verimsizliği ve PCIe gecikmesi hız kazancını yutar. |
| **Büyük Toplu Sorgu (`Q ≥ 128`, Flat / IVF-PQ, `N ≥ 500K`)** | 150 – 800 ms | **5 – 25 ms** (Tensor Core FP16/BF16 GEMM) | **GPU 10x–40x kazanır.** Matris-matris çarpımı (`(Q, D) @ (N, D)^T`) GPU bellek bant genişliğini (300–1000 GB/s) tam doyurur. |
| **Büyük K-Means / PQ / IVF Eğitimi (`N ≥ 1M`, `768D–1536D`)** | 20 – 180 sn | **1 – 6 sn** | **GPU 15x–30x kazanır.** `DiskIndex` ve `IVF-PQ` eğitim sürelerini dakikalardan saniyelere indirir. |

### Önerilen `v2.x` GPU Entegrasyon Stratejisi
- Çekirdek `numpy` bağımlılığı korunarak opsiyonel **`pip install "hisamdb[gpu]"` (`cupy-cuda12x` veya `torch`)** arka ucu tasarlanmalıdır.
- Yalnızca **K-Means/PQ eğitim motoru** ve **`batch_search(Q >= 64)` Flat/IVF çekirdekleri** `xp = cupy if gpu else numpy` dizi arayüzü (Array API Standard) ile GPU'ya yönlendirilmelidir; tekil HNSW sorguları CPU'da kalmalıdır.

---

## 4. `v2.x` Dağıtık Mimari Tasarım Taslağı (10M – 1B+ Vektör)

Tek düğümlü HisamDB `v1.0.0`, `DiskIndex` ve `IVF-PQ` sayesinde **1M–5M vektöre** kadar rahatça ölçeklenir. **10M–1B+ vektör** ve **yüksek erişilebilirlik (HA)** gerektiren kurumsal senaryolar için `v2.x` dağıtık mimarisi şu 4 bileşene dayanacaktır:

```text
                [ İstemciler: gRPC / REST SDK ]
                              │
                [ Coordinator / Query Router ]
                 (Consistent Hashing + RRF Merge)
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
        [ Shard 0 (L) ] [ Shard 1 (L) ] [ Shard 2 (L) ]
          │  WAL Akışı    │  WAL Akışı    │  WAL Akışı
          ▼               ▼               ▼
        [ Replica 0-R ] [ Replica 1-R ] [ Replica 2-R ]
```

1. **Yatay Bölümleme (Consistent-Hashing Sharding):**
   - Koleksiyonlar `shard_id = sha256(doc_id veya tenant_id) % num_shards` ile bağımsız HisamDB örneklerine dağıtılır.
   - Çok kiracılı (Multi-Tenant) iş yüklerinde `tenant_id` bazlı yönlendirme sayesinde bir kiracının sorgusu yalnızca kendi shard'ına gider (**0 scatter-gather maliyeti**).
2. **WAL Tabanlı Replikasyon ve Yüksek Erişilebilirlik:**
   - Faz 3'te oluşturduğumuz sıralı LSN'li ikili `WriteAheadLog` (`MWAL`), doğrudan **Leader → Read Replica** akış replikasyonunun temelidir.
   - Küme topolojisi ve shard lider seçimi **Raft** (veya harici `etcd`) üzerinde tutulur; yazmalar liderde WAL'a işlenip replikalara gönderilir, okumalar replikalar arasında yük dengelenir.
3. **Dağıtık Scatter-Gather ve Füzyon:**
   - Çapraz-shard sorgularda Coordinator paralel olarak tüm shard'lardan `top-k` ister ve yerel `fuse_rrf` / mesafe sıralaması ile birleştirir (mevcut `hisamdb/search/fusion.py` doğrudan yeniden kullanılabilir).
4. **gRPC + Protobuf İkili İletişim Protokolü:**
   - Faz 5 HTTP REST benchmarkımızda `~0.3 ms` olan çekirdek araması, JSON serileştirme + TCP HTTP yüzünden `3.94 ms`'ye çıkmıştır.
   - İç shard-arası haberleşme ve yüksek QPS istemciler için `float32` vektörleri JSON yerine ham `bytes` taşıyan **gRPC / HTTP2** katmanı ağ gecikmesini `<1 ms` bandına indirecektir.
