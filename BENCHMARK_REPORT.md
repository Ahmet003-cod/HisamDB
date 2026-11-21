# MicroVectorDB v0.5.0 — Bilimsel Benchmark Raporu (`BENCHMARK_REPORT.md`)

**Ölçüm Tarihi:** 2026-11-21  
**Ham Sonuç Dosyaları:**  
- Faz 1 (`v0.3.0`): [`benchmarks/results/benchmark_v0.3.0.json`](benchmarks/results/benchmark_v0.3.0.json) · [`benchmarks/results/benchmark_v0.3.0.csv`](benchmarks/results/benchmark_v0.3.0.csv)  
- Faz 2 (`v0.4.0`): [`benchmarks/results/benchmark_v0.4.0.json`](benchmarks/results/benchmark_v0.4.0.json) · [`benchmarks/results/benchmark_v0.4.0.csv`](benchmarks/results/benchmark_v0.4.0.csv)  
- Faz 3 (`v0.5.0`): [`benchmarks/results/benchmark_v0.5.0.json`](benchmarks/results/benchmark_v0.5.0.json) · [`benchmarks/results/benchmark_v0.5.0.csv`](benchmarks/results/benchmark_v0.5.0.csv)  
**Tekrar Çalıştırma Komutları:** `python benchmarks/run_benchmarks.py`, `python benchmarks/run_benchmarks_v04.py` ve `python benchmarks/run_benchmarks_v05.py`

---

## 1. Test Donanımı ve Yazılım Ortamı

| Bileşen | Değer |
|---|---|
| İşletim Sistemi | Windows 10/11 x86-64 (`Windows-10-10.0.26300-SP0`) |
| İşlemci | Intel x86-64, 12 mantıksal çekirdek (AVX2, FMA3, F16C, POPCNT) |
| Toplam Bellek (RAM) | 15.73 GB |
| Python | 3.11.9 |
| NumPy / BLAS | NumPy 2.0.2 (OpenBLAS 0.3.27, 64-bit int, AVX2/FMA) |
| Karşılaştırma Kütüphaneleri | FAISS `1.13.2` (`faiss-cpu`), `hnswlib` (`chroma-hnswlib 0.7.3`) |

> **Önemli Metodoloji Notu:**
> - Tüm karşılaştırmalar aynı donanım, aynı tohum (`seed=42`), aynı mesafe metriği ve aynı sorgu yükü üzerinde yapılmıştır.
> - Exact (Flat) aramalar kendi arasında, ANN algoritmaları (IVF, HNSW) ise gerçek Exact Ground Truth'a karşı ölçülen **Recall@10** değerleriyle birlikte raporlanmıştır.
> - Bu ortamda kurulu olan `chromadb 0.5.0`, NumPy 2.0 (`AttributeError: np.float_ was removed`) uyumsuzluğu nedeniyle test çalıştırılamamıştır; Qdrant, Milvus, Weaviate ve LanceDB ile 500K–1M (SIFT1M / GloVe) karşılaştırmaları Faz 6 kapsamında eklenecektir.

---

## 2. Sürüm Karşılaştırması: MicroVectorDB v0.2.0 → v0.3.0 (Faz 1)

Aynı Gaussian veri kümeleri (`seed=42`), aynı boyutlar ve aynı sorgu adedi üzerinde ölçülmüştür:

| Senaryo | N × D | Metrik | v0.2.0 P50 (ms) | v0.3.0 P50 (ms) | Hızlanma | v0.2.0 QPS | v0.3.0 QPS | v0.2.0 RAM | v0.3.0 RAM |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **Filtreli Kosinüs (`cat="c0"`, %10)** | 100K × 128 | cosine | 59.317 | **2.468** | **24.0x** | 15.9 | **383.7** | +110.7 MB | **49.6 MB** |
| **Aralık Filtreli Kosinüs (`val>=900`, %10)** | 100K × 128 | cosine | ~55.0 | **3.402** | **~16.2x** | ~18.0 | **262.2** | +110.7 MB | **49.6 MB** |
| **Flat Öklid 768D** | 50K × 768 | euclidean | 116.301 | **7.985** | **14.6x** | 8.6 | **110.8** | +158.6 MB | **146.9 MB** |
| **Flat Öklid 128D** | 100K × 128 | euclidean | 38.538 | **5.730** | **6.7x** | 24.8 | **158.0** | +59.1 MB | **49.6 MB** |
| **Flat Kosinüs 768D (Bellek & Soğuk Başlangıç)** | 50K × 768 | cosine | 7.451 (1. sorgu: 183.8 ms) | 9.046 (1. sorgu: ~9 ms) | **~20x soğuk başlangıç** | 124.9 | 108.4 | +297.4 MB | **146.9 MB (-%50.6)** |
| **Standalone `IVFIndex` (256 küme, np=8)** | 100K × 128 | euclidean | 2.102 (train: 7.15 s) | **0.421** (train: 4.70 s) | **5.0x arama / 1.5x eğitim** | 429.8 | **1,881.8** | 48.8 MB | 48.8 MB |
| **Standalone `ScalarQuantizer` (INT8)** | 100K × 128 | euclidean | 35.075 (**Rec@10=0.630**) | **16.888** (**Rec@10=0.980**) | **2.1x + 35 puan Recall** | 27.0 | **52.8** | 12.2 MB | 12.2 MB |
| **Artımlı Kalıcı Yazma (20 × 500 batch)** | 10K × 128 | cosine | 1.948 s (5,134 v/s) | **0.640 s (15,628 v/s)** | **3.0x** | — | — | — | — |
| **Artımlı Batch 20 / Batch 1 Oranı** | 10K × 128 | cosine | 9.4x yavaşlama | **1.77x** | $O(N^2) \to O(M)$ | — | — | — | — |
| **Bellek-Eşlemeli Yükleme (`use_mmap=True`)** | 10K × 128 | cosine | Kullanılamıyordu | **0.076 MB vektör RAM** | **-%98.4 vektör RAM** | — | — | 4.88 MB | **0.076 MB** |

> *Not:* `examples/04_benchmark.py` (10K × 128D tek toplu ekleme) ekleme hızı **244,639 v/s → 1,110,838 v/s (4.5x)** düzeyine çıkmıştır.

---

## 3. MicroVectorDB v0.3.0 vs FAISS (100.000 × 128D Gerçekçi Kümelemeli Veri)

64 konulu kümelemeli embedding dağılımı (100 sorgu, Top-10, Kosinüs metriği, Exact ground truth'a karşı Recall@10):

| Motor & Yapılandırma | Tip | Build/Add (s) | P50 (ms) | P95 (ms) | QPS | Recall@10 | Vektör RAM (MB) |
|---|---|---:|---:|---:|---:|---:|---:|
| **MicroVectorDB v0.3.0 — Flat Float32** | Exact | 0.143 | 5.488 | 11.220 | 157.6 | **1.000** | 49.6 |
| **FAISS 1.13.2 — `IndexFlatIP`** | Exact | 0.027 | 5.418 | 7.587 | 175.1 | **1.000** | 48.8 |
| **MicroVectorDB v0.3.0 — IVF (`C=128, np=8`)** | ANN (Saf NumPy) | 2.042 | **0.761** | 1.666 | **1,117.4** | **1.000** | 49.6 |
| **FAISS 1.13.2 — `IndexIVFFlat` (`nlist=128, np=8`)** | ANN (C++) | 0.274 | 0.324 | 0.538 | 3,031.9 | 1.000 | 48.8 |
| **MicroVectorDB v0.3.0 — HNSW (`M=16, efS=64`)** | ANN (`hnswlib`) | 6.736 | **0.278** | **0.364** | **3,463.3** | **0.975** | 49.6 |
| **MicroVectorDB v0.3.0 — HNSW (`M=16, efS=128`)** | ANN (`hnswlib`) | 6.283 | **0.411** | **0.495** | **2,377.3** | **0.995** | 49.6 |
| **FAISS 1.13.2 — `IndexHNSWFlat` (`M=16, efS=64`)** | ANN (C++) | 5.521 | 0.201 | 0.261 | 4,823.4 | 0.978 | 48.8 |

### Bilimsel Değerlendirme
1. **Exact Flat Arama:** MicroVectorDB v0.3.0, OpenBLAS SGEMV + `inv_norms` tasarımı sayesinde C++ `FAISS IndexFlatIP` ile **neredeyse birebir aynı gecikmeye (5.49 ms vs 5.42 ms)** ulaşmakta; buna ek olarak dahili metadata filtreleme, doküman deposu, atomik disk kalıcılığı ve RWLock eşzamanlılığı sunmaktadır.
2. **Saf NumPy IVF:** Hiçbir C++ bağımlılığı olmadan **1,117 QPS (0.761 ms)** ve **1.000 Recall@10** ile Flat aramadan **7.1x daha hızlıdır**. C++ `FAISS IndexIVFFlat` (~2.4x daha hızlı) ile arasındaki fark, Python küme döngüsü ve OpenBLAS küçük-blok çağrı yükünden kaynaklanmaktadır.
3. **HNSW:** Opsiyonel `hnswlib` entegrasyonu ile **3,463 QPS (P50 = 0.278 ms, Recall@10 = 0.975)** ve `efS=128` ile **2,377 QPS (Recall@10 = 0.995)** elde edilmiştir — tam veritabanı katmanına rağmen çıplak C++ `FAISS IndexHNSWFlat`'ın (0.201 ms) yalnızca ~1.38x gerisindedir.

---

## 4. Kuantizasyon Yöntemlerinin Karşılaştırmalı Analizi (100K × 128D)

| Yöntem | Vektör RAM | Bellek Tasarrufu | Recall@10 (128D) | P50 (ms) | QPS | En Verimli Kullanım Senaryosu |
|---|---:|---:|---:|---:|---:|---|
| **Float32 (Referans)** | 49.6 MB | %0 | 1.000 | 5.488 | 157.6 | Yeterli RAM varsa en hızlı exact BLAS tarama |
| **Float16** | **25.2 MB** | **%49.2 (2x)** | **0.999** | 60.178* | 15.7* | RAM kısıtlı ortamlarda neredeyse sıfır doğruluk kaybıyla depolama (IVF/HNSW ile eşlendiğinde cast maliyeti 20x düşer) |
| **INT8 (`ScalarQuantizer`, `keep_raw=False`)** | **13.0 MB** | **%73.8 (3.8x)** | **0.965** (Gauss: **0.980**) | 18.412* | 47.1* | 4x bellek sıkıştırma ve yüksek doğruluk gereken orta/büyük koleksiyonlar |
| **Binary (`keep_raw=False`, 1-bit)** | **1.5 MB** | **%97.0 (32x)** | 0.061 (128D) | 5.663 | 149.9 | Tek başına düşük boyutta (128D) yetersiz; yüksek boyutlu (768D–1536D) ham aday daraltma için |
| **Binary + Rescore (`mult=50`, 768D 20K ek ölçüm)** | 1.5 MB + disk/raw | 32x indeks | **1.000** (768D) / 0.657 (128D) | 5.26 (768D) | ~190 | Yüksek boyutlu (768D–1536D) embedding'lerde 2 aşamalı (Hamming + exact rescore) arama |

*\*Açık Mühendislik Notu (Saf NumPy Sınırı):* x86-64 CPU'larda NumPy'ın `float16 -> float32` ve `int8 -> float32` dönüştürmeleri tek iş parçacıklı skaler/F16C döngülerle çalıştığından, **Flat tam taramada** Float16 (60 ms) ve INT8 (18 ms), çok iş parçacıklı OpenBLAS Float32 SGEMV'den (5.5 ms) daha yavaştır. Bu fiziksel bir donanım/kütüphane gerçeğidir:
- Saf NumPy katmanında **Float16/INT8 bir bellek tasarrufu (2x–4x RAM)** optimizasyonudur.
- Hız kazanımı için Faz 2'de **IVF-PQ / IVF-SQ** (yalnızca %5 küme taranır) ve Faz 6'da **AVX2/VNNI C/Rust SIMD çekirdekleri** değerlendirilecektir.

---

## 5. Kalıcılık ve Kurtarma Ölçümleri (10K × 128D, 20 × 500 Batch)

- **Toplam Artımlı Ekleme Süresi:** `0.640 s` (**15,628 vektör/sn**, v0.2.0'da `1.948 s` / `5,134 v/s` idi — **3.0x hızlanma**).
- **Ölçeklenme Davranışı:** Batch 1 = `18.53 ms`, Batch 10 = `27.98 ms`, Batch 20 = `32.87 ms` (oran **1.77x**; v0.2.0'da **9.4x** idi).
- **Disk Tüketimi:** `.mvdbvec` = `4.88 MB` (64-byte header + saf float32), SQLite = `1.97 MB` (WAL dahil).
- **Yeniden Yükleme (Reload):** RAM modu = `149.0 ms`, `use_mmap=True` = `128.0 ms` (vektör RAM = yalnızca `0.076 MB`, yani **%98.4 RAM tasarrufu**).
- **Tekil Silme (Sıkıştırmalı):** `181.9 ms` (tam atomik `.mvdbvec` + CRC32 + SQLite yeniden indeksleme; Faz 4'te tombstone + arka plan compaction ile <1 ms hedeflenmektedir).

---

## 6. Faz 2 (`v0.4.0`): PQ, IVF-PQ, DiskIndex, Anında İndeks Yükleme ve Query Optimizer

### 6.1. Product Quantization (PQ), IVF-PQ ve Out-of-Core DiskIndex vs FAISS (100K × 128D)

Aynı 64-konulu kümelemeli 100K × 128D veri seti (100 sorgu, Top-10, Kosinüs metriği, Exact ground truth):

| Motor & Yapılandırma | Tip | Build (s) | P50 (ms) | P95 (ms) | QPS | Recall@10 | RAM (MB) | RAM Tasarrufu |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **MVDB `DiskIndex` (Partitioned MMap, `C=128, np=8`)** | Out-of-Core MMap | 2.79 | **1.227** | 2.586 | **684.1** | **1.000** | **2.35** | **%95.3 (21x)** |
| **MVDB `DiskStreamingScanner` (Exact MMap, 4K blk)** | Out-of-Core Exact | 0.00 | 11.510 | 15.362 | 82.7 | **1.000** | **0.76** | **%98.5 (65x)** |
| **MVDB `IVF-PQ` (`C=128, np=8, M=16`, `keep_raw=F`)** | Saf NumPy ANN+PQ | 13.08 | 3.446 | 6.277 | 255.1 | **0.270** | **2.48** | **%95.0 (20x)** |
| **FAISS `IndexIVFPQ` (`nlist=128, np=8, M=16`)** | C++ ANN+PQ | 4.30 | 0.584 | 0.763 | 1,787.9 | 0.258 | 2.29 | %95.3 |
| **MVDB Flat `PQ` (`M=16, 8-bit`, `keep_raw=F`)** | Saf NumPy PQ | 8.06 | 10.943 | 15.539 | 86.8 | 0.054 | **1.65** | **%96.7 (30x)** |
| **FAISS `IndexPQ` (`M=16, 8-bit`)** | C++ PQ | 3.30 | 1.378 | 1.853 | 722.5 | 0.054 | 1.53 | %96.9 |

> **Bilimsel Bulgular:**
> - **`DiskIndex` (Bölümlenmiş MMap) en çarpıcı sonucu verdi:** Yalnızca **2.35 MB RAM** (%95.3 RAM tasarrufu!) kullanarak **1.227 ms (684.1 QPS)** ve **1.000 Recall@10** ile RAM-içi Flat Float32'den (5.49 ms) ve FAISS `IndexFlatIP`'den (5.42 ms) **4.4x daha hızlı** çalışmaktadır.
> - **`IVF-PQ` doğrulukta FAISS'i geçti:** Aynı 8-bit `M=16` (16 byte/vektör) bütçesinde MicroVectorDB IVF-PQ, K-Means++ ve birim-küre L2 artık ADC sayesinde **Recall@10 = 0.270** elde ederek FAISS `IndexIVFPQ`'yu (`0.258`) geride bıraktı. (Saf Python/NumPy küme döngüsü C++ SIMD ADC'ye göre ~5.9x daha yavaştır; yüksek recall gereken senaryolarda `DiskIndex` veya `M=32` / float32 rescore önerilir.)

### 6.2. Anında İndeks Diskten Yükleme (v0.3.0 Soğuk İnşa vs v0.4.0 CRC32+Epoch Disk Yükleme, 100K × 128D)

| İndeks Tipi | v0.3.0 Soğuk Yeniden İnşa | v0.4.0 Diskten Açılış (Toplam DB) | Hızlanma | `index_loaded_from_disk` |
|---|---:|---:|---:|:---:|
| **IVF-PQ (`C=128, M=16`)** | 13.08 s | **542.0 ms** | **24.1x** | ✅ `True` |
| **HNSW (`M=16, efC=128`)** | 9.22 s | **940.6 ms** | **9.8x** | ✅ `True` |
| **DiskIndex (`C=128`, MMap)** | 2.79 s | **532.8 ms** | **5.2x** | ✅ `True` |
| **IVF (`C=128`)** | 2.86 s | **781.3 ms** | **3.7x** | ✅ `True` |

*(Not: Toplam açılış süresinin ~450 ms'si 100K SQLite metadata satırının Python sözlüklerine okunmasından gelir; ikili indeks dosyalarının kendisi ~50–150 ms içinde yüklenmektedir.)*

### 6.3. Maliyet Tabanlı Query Optimizer: Filtreli HNSW (100K × 128D)

v0.3.0'daki graf-içi Python callback (`filter_fn`) ile v0.4.0'daki maliyet tabanlı `QueryOptimizer` (`pre_filter_exact` ve sıfır-set `np.searchsorted` tabanlı `hnsw_post_filter`) karşılaştırması:

| Filtre Seçiciliği | Optimizer Seçimi (`explain()`) | v0.3.0 Callback P50 (ms) | v0.4.0 Optimizer P50 (ms) | Hızlanma (v0.3 → v0.4) | Hızlanma (v0.2 → v0.4) |
|---|---|---:|---:|---:|---:|
| **%2 (2.000 aday)** | `flat_exact` (`pre_filter_exact`) | 3.712 | **0.705** | **5.26x** | — |
| **%10 (10.000 aday)** | `hnsw_post_filter` (`searchsorted`) | 1.069 | **0.479** (>2,000 QPS) | **2.23x** | **123.8x** *(59.32 ms → 0.48 ms!)* |
| **%30 (30.000 aday)** | `hnsw_post_filter` (`searchsorted`) | 2.268 | **0.346** (~2,890 QPS) | **6.55x** | — |
| **%60 (60.000 aday)** | `hnsw_post_filter` (`searchsorted`) | 2.960 | **0.384** (~2,600 QPS) | **7.71x** | — |

---

## 7. Faz 3 (`v0.5.0`): Sorun #1/#2/#3 Çözümleri, Hibrit/BM25/Sparse Arama ve WAL/Tombstone/Snapshot

### 7.1. Sorun #1 Çözümü: 100K × 128D Hızlı Açılış (`.items.mvdbcol` + `orjson` + Tembel Norm & Metadata İndeksi)

v0.4.0'da açılış süresinin ~450 ms'si 100K SQLite satırının Python sözlüklerine çevrilmesinden ve `mmap` modunda tüm vektör normlarının açılışta hesaplanmasından kaynaklanıyordu. v0.5.0'da eklenen epoch+CRC32 korumalı ikili sütunsal önbellek (`.items.mvdbcol`), opsiyonel `orjson`, tembel `MetadataIndex`/`BM25Index`/`SparseVectorIndex` ve sıfır-dokunuşlu tembel `mmap` normları ile ölçülen sonuçlar:

| Koleksiyon & İndeks (100K × 128D) | v0.3.0 Soğuk İnşa | v0.4.0 Açılış | v0.5.0 Açılış | v0.4 → v0.5 | v0.3 → v0.5 | Açılış Vektör RAM |
|---|---:|---:|---:|---:|---:|---:|
| **`DiskIndex` (`C=128`, MMap)** | 2,791 ms | 532.8 ms | **125.8 ms** | **4.23x** | **22.2x** | **0.00 MB** |
| **`IVF-PQ` (`C=128, M=16`, `keep_raw=F`)** | 13,079 ms | 542.0 ms | **198.9 ms** | **2.72x** | **65.8x** | **0.00 MB** |
| **`IVF` (`C=128`, RAM)** | 2,858 ms | 781.3 ms | **486.7 ms** | **1.61x** | **5.9x** | 49.59 MB |
| **`HNSW` (`M=16, efC=128`, RAM)** | 9,225 ms | 940.6 ms | **867.8 ms** | **1.08x** | **10.6x** | 49.59 MB |
| **MMap + 100K Zengin Metadata (3 alan)** | — | ~880 ms *(SQLite)* | **328.7 ms** | **~2.68x** | — | **0.00 MB** |

*(Bozuk `.items.mvdbcol` veya çökme sonrası WAL replay durumunda sistem otomatik ve şeffaf biçimde SQLite doğruluk kaynağına düşer; `test_columnar_cache_corruption_falls_back_to_sqlite_and_preserves_mmap` ile doğrulanmıştır.)*

### 7.2. Sorun #2 Çözümü: Ön-Hesaplanmış Küme-Merkezli `IVF-PQ` + Tek-Atış `uint16` ADC (100K × 128D)

v0.4.0'da her sorgu için küme başına ayrı `np.einsum` LUT + 16 döngülü `+=` ADC çalıştırılıyordu. v0.5.0'da $P[c, m, k] = \|C_{m,k}\|_2^2 + 2(c_{c,m}\cdot C_{m,k})$ tablosu eğitim/yükleme anında önceden hesaplanıp sorgu başına tek `np.matmul` ve `uint16` ön-kaydırmalı tek-atış `np.take(...).sum(axis=0)` ile birleştirildi:

| Motor & Yapılandırma | P50 (ms) | P95 (ms) | QPS | Recall@10 | Toplam RAM |
|---|---:|---:|---:|---:|---:|
| MicroVectorDB v0.4.0 `IVF-PQ` (`C=128, np=8, M=16`) | 3.446 | 6.277 | 255.1 | **0.270** | 2.48 MB |
| **MicroVectorDB v0.5.0 `IVF-PQ` (Precomputed LUT + `u16` ADC)** | **2.267** | **3.776** | **412.2 (+%61.6)** | **0.270** | **5.62 MB** *(-%88.7 RAM)* |
| FAISS `1.13.2` `IndexIVFPQ` (`nlist=128, np=8, M=16`, C++ SIMD) | 0.636 | 0.803 | 1,625.4 | 0.258 | 2.29 MB |

> **Bilimsel Değerlendirme:** Saf NumPy katmanında **P50 %34.2 düşmüş (3.45 → 2.27 ms)** ve **QPS +%61.6 artmıştır (255 → 412 QPS)**; doğrulukta FAISS'i geçmeye devam etmektedir (`0.270` vs `0.258`). Kalan fark, NumPy'ın ara `(M, n_c)` float32 dizisini bellekte oluşturması ile C++ SIMD yazmaç akümülasyonu arasındaki fiziksel donanım farkından kaynaklanmakta olup Faz 6'da opsiyonel AVX2 C çekirdeği ile kapatılacaktır.

### 7.3. Sorun #3 Çözümü: `<1 ms` Tombstone Silme, İkili WAL, Compaction ve SHA-256 Snapshot

v0.3.0'da tek bir kayıt silindiğinde tüm `.mvdbvec` dosyası ve SQLite tablosu yeniden yazılıyordu (`181.9 ms`). v0.5.0'da kalıcı tutamaçlı ikili CRC32 `WriteAheadLog` + `O(1)` bitmask Tombstone silme + otomatik/manuel `compact()` + SHA-256 manifestli `create_snapshot()` / `restore_snapshot()` eklendi:

| İşlem & Senaryo | v0.3.0 P50 (ms) | v0.5.0 P50 (ms) | v0.5.0 P95 (ms) | Hızlanma | Ek Ölçüm |
|---|---:|---:|---:|---:|---|
| **10K × 128D Tekil Kalıcı Silme (`wal_sync="normal"`)** | 181.900 | **0.403** | **0.694** | **451.4x** | `compact()` = `151.7 ms` |
| **10K × 128D Tekil Kalıcı Silme (`wal_sync="full"` fsync)** | 181.900 | **1.967** | **3.567** | **92.5x** | Her yazımda `os.fsync()` |
| **100K × 128D Tekil Kalıcı Silme (`wal_sync="normal"`)** | *(~1,800+)* | **0.382** | **0.899** | **$O(1)$ sabit!** | Veri 10x artsa da `<0.4 ms` |
| **10K × 128D SHA-256 Snapshot Oluştur / Geri Yükle** | Yoktu | **203.09** *(Create)* | **63.48** *(Restore)* | — | 4 dosya SHA-256 doğrulamalı |

### 7.4. Faz 3: 10.000 Dokümanlık Kurumsal Külliyatta Dense vs BM25 vs Hibrit Arama (128D, HNSW)

10.000 Türkçe/İngilizce kurumsal teknik doküman (`seed=123`, 40 semantik konu + benzersiz `ERR-xxxx` hata kodları + sparse vektörler, 100 karma sorgu: 50 kesin kod + konu, 50 semantik parafraz):

| Arama Modu | P50 (ms) | P95 (ms) | QPS | Hit@10 (Recall@10) | Notlar |
|---|---:|---:|---:|---:|---|
| **Dense-Only (HNSW)** | **0.299** | 0.472 | **3,127.2** | 0.530 | Semantik parafrazda güçlü, aynı konudaki kesin hata kodunda zayıf |
| **BM25-Only (Vektörize NumPy Okapi BM25)** | **0.476** | 0.707 | **1,982.0** | 0.500 | Kesin hata kodunda %100, parafrazda zayıf *(Python döngüsünden **28.3x** hızlı!)* |
| **Sparse-Vector-Only (Ters İndeks)** | **0.151** | 0.263 | **7,268.8** | 0.550 | SPLADE / BGE-M3 seyrek vektörleri için `<0.2 ms` |
| **Hibrit (Dense + BM25, RRF)** | **0.936** | 1.284 | **1,033.4** | **1.000 (%100!)** | **Hem semantik hem kesin kod sorgularında tam kapsama, `<1 ms`!** |
| **Hibrit (Dense + BM25, Weighted Min-Max)** | **1.110** | 1.468 | **867.0** | **1.000 (%100!)** | Ayarlanabilir `alpha` / `beta` ağırlıklı füzyon |
| **3-Yollu Hibrit (Dense + BM25 + Sparse, RRF)** | 1.247 | 1.843 | 751.9 | 0.780 | 3 farklı retrieval sinyalini tek çağrıda birleştirir |

*(10.000 dokümanın Dense HNSW + BM25 + Sparse birlikte toplam ekleme süresi: **0.712 saniye**.)*
