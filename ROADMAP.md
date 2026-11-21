# HisamDB — Mühendislik Yol Haritası (`ROADMAP.md`)

**Vizyon:** *High Performance + Low Memory + Fast Retrieval + Reliable Storage + Developer Friendly*  
**Temel İlke:** Her faz; bilimsel ölçüm, geriye dönük uyumluluk, kapsamlı testler ve kanıtlanmış benchmark sonuçları ile tamamlanır.

---

## Faz 1 — Çekirdek Optimizasyon, Güvenlik ve Veri Bütünlüğü (`v0.3.0`)
**Durum:** ✅ Tamamlandı (`52/52` test, `BENCHMARK_REPORT.md` doğrulandı)

### Hedefler
1. **Güvenlik & Doğruluk Düzeltmeleri (P0):**
   - Path traversal koruması ve SHA-256 tabanlı çakışmasız SQLite tablo/dosya adlandırması (eski v0.2 veritabanlarına otomatik uyumluluk ile).
   - `NaN`/`Inf`, batch içi tekrarlanan ID, geçersiz metrik, sorgu boyut uyuşmazlığı ve boş batch doğrulamaları.
   - `IVFIndex` `argpartition` sınır çökmesinin ve `ScalarQuantizer` `int16` taşma hatasının giderilmesi.
   - Karma veri tipli metadata filtrelerinde `TypeError` çökmesinin giderilmesi.
2. **Bellek ve Mesafe Motoru Optimizasyonu:**
   - Kosinüs benzerliğinde tam `(N, D)` normalizasyon kopyası yerine `(N,)` `inv_norms` vektörü ile **~%45–50 RAM tasarrufu** ve soğuk başlangıç gecikmesinde >15x düşüş.
   - Öklid mesafesinde önbellekli `sq_norms` ile **>10x sorgu hızlanması**.
   - NumPy 1.x / 2.x uyumlu katı `float32` koruması.
3. **Depolama Motoru v2 & Dinamik Tampon:**
   - Amortized $O(1)$ ekleme için kapasite-ikileyen (capacity-doubling) bitişik `VectorBuffer`.
   - Sürümlü binary vektör formatı (`.mvdbvec`: 64-byte header, magic, version, dtype, dim, count, CRC32) + artımlı $O(M)$ append + atomik yazma + `use_mmap=True` desteği + eski `.npy` otomatik göçü.
   - SQLite WAL modu + yalnızca değişen kayıtları yazan delta kalıcılık + chunked toplu silme + `EphemeralClient` sıfır-SQLite hızlı yolu + `close()` / context manager.
4. **Ters Metadata İndeksi ve Zengin Filtre Motoru:**
   - Eşitlik (`$eq`, `$in`), aralık (`$gt`, `$gte`, `$lt`, `$lte`), olumsuzlama (`$ne`, `$nin`), mantıksal (`$and`, `$or`, `$not`) ve iç içe (`nested`) dot-path filtreleri.
   - 100K filtreli sorguda **>20x–50x hızlanma** hedefi.
5. **İndeks ve Kuantizasyon İyileştirmeleri + Koleksiyon Entegrasyonu:**
   - `IVFIndex`: K-Means++, vektörize merkez güncelleme, bitişik küme vektör blokları, `cosine`/`dot_product`/`euclidean` desteği ve BLAS hızlandırma.
   - `ScalarQuantizer` (taşmasız, ölçek-ağırlıklı asimetrik mesafe + opsiyonel quantile clipping), `Float16Quantizer` (2x RAM tasarrufu), `BinaryQuantizer` (32x RAM tasarrufu, LUT popcount).
   - `Collection` içine `index_type="flat"|"ivf"|"hnsw"`, `quantization=None|"float16"|"int8"|"binary"`, `use_mmap`, `score_threshold`, `search()` ve `RWLock` eşzamanlılık güvenliği entegrasyonu.
   - `HisamDB` üst düzey kolaylık istemcisi (`PersistentClient` / `EphemeralClient` ile %100 uyumlu).
6. **Bilimsel Benchmark & Test Altyapısı:**
   - Otomatik JSON/CSV/Markdown çıktı üreten benchmark paketi (`benchmarks/`).
   - Birim, eşzamanlılık, sınır durum, kalıcılık, v0.2 geriye dönük uyumluluk ve kurtarma testleri.

### Başarı Kriterleri (Exit Criteria)
- Mevcut 27/27 testin tamamı değişiklik gerektirmeden geçer + tüm yeni testler (%100 yeşil).
- 100K×128D filtreli (%10) sorgu gecikmesi: 59.3 ms → **< 3 ms** (>20x iyileşme).
- 50K×768D Öklid P50 gecikmesi: 116.3 ms → **< 12 ms** (>10x iyileşme).
- 50K×768D Kosinüs RAM artışı: +297 MB → **< 175 MB** (>%40 RAM tasarrufu).
- 20×500 kalıcı artımlı ekleme: 1.95 s (5.1K v/s) → **< 0.25 s** (>40K v/s, >8x iyileşme).
- `ScalarQuantizer` 100K×128D Recall@10: 0.630 → **> 0.95** ve hızda >5x artış.

---

## Faz 2 — İleri İndeksleme, Kuantizasyon ve Sorgu Optimizasyonu (`v0.4.0`)
**Durum:** ✅ Tamamlandı (`63/63` test, `benchmarks/results/benchmark_v0.4.0.json` doğrulandı)
- **HNSW, IVF, IVF-PQ & DiskIndex Disk Kalıcılığı:** CRC32 + epoch doğrulamalı atomik kaydetme/yükleme (IVF-PQ 24.1x, HNSW 9.8x, DiskIndex 5.2x, IVF 3.7x hızlı açılış) ve bozulma anında güvenli yeniden inşa.
- **Product Quantization (PQ) & IVF-PQ:** Vektörize `np.einsum` ADC arama tabloları ve birim-küre L2 kosinüs düzeltmesi ile 20x–30x RAM sıkıştırma (FAISS `IndexIVFPQ` ile eşdeğer/yüksek Recall@10).
- **Query Optimizer (`collection.explain()`):** Maliyet tabanlı otomatik Flat / IVF / IVF-PQ / HNSW / DiskIndex / DiskStream seçimi; sıfır-set `np.searchsorted` HNSW post-filter oversampling ile 100K filtreli sorguda **0.35–0.48 ms** (v0.3 callback'ten 2.2x–7.7x, v0.2'den **123.8x** hızlı).
- **Out-of-Core Disk Tabanlı Arama (`DiskIndex` & `DiskStreamingScanner`):** 100K×128D'de **2.35 MB RAM (%95.3 tasarruf)** ile **1.23 ms (684 QPS, Recall@10=1.000)** ve sabit-RAM akış taramasında **0.76 MB RAM (%98.5 tasarruf)**.

---

## Faz 3 — Gelişmiş Arama (Sparse, Hybrid), RAG Optimizasyonu ve WAL/Tombstone (`v0.5.0`)
**Durum:** ✅ Tamamlandı (`75/75` test, `benchmarks/results/benchmark_v0.5.0.json` doğrulandı)
- **Vektörize Okapi BM25 & Sparse Vector Motoru:** Türkçe/Unicode uyumlu tokenizer (`İ -> i`, `I -> ı`), NumPy derlenmiş BM25 (**0.476 ms / 1,982 QPS**, 10K doküman) ve kalıcı `SparseVectorIndex` (**0.151 ms / 7,269 QPS**).
- **Hybrid Search (`hybrid_search`):** 2-yollu ve 3-yollu (Dense + BM25 + Sparse) arama; Reciprocal Rank Fusion (`rrf`), `min_max` ve `z_score` normalizasyonlu ağırlıklı füzyon; 10K külliyatta **0.936 ms (1,033 QPS)** ve **Hit@10 = 1.000** (Dense-Only `0.530`, BM25-Only `0.500`).
- **RAG İlkelleri & Multi-Tenant:** `chunk_text()`, `add_documents()`, `retrieve_with_context()` (Parent-Child bağlam penceresi), `multi_vector_search()`, doküman gruplama (`group_by`), tekilleştirme (`deduplicate`), çekirdekten bağımsız harici `reranker` arayüzü ve `TenantCollection` izolasyonu.
- **Sorun #1, #2, #3 Çözümleri (Çekirdek Güvenilirlik & Hız):**
  - `.items.mvdbcol` sütunsal önbellek + `orjson` + tembel indeks/normlarla 100K `DiskIndex` açılışı **532.8 ms → 125.8 ms (4.23x)**, `IVF-PQ` **542 ms → 198.9 ms (2.72x)**.
  - Ön-hesaplanmış küme-merkezli `IVF-PQ` + `uint16` tek-atış ADC ile **3.45 ms → 2.27 ms (+%61.6 QPS)**.
  - İkili CRC32 `WriteAheadLog` (torn-write recovery & crash replay), `<1 ms` Tombstone silme (**181.9 ms → 0.403 ms, 451x hızlanma**), otomatik/manuel `compact()` ve SHA-256 doğrulamalı `create_snapshot()` / `restore_snapshot()`.

---

## Faz 4 — İleri Eşzamanlılık, Async İstemci ve Arka Plan İşçileri (`v0.6.0`)
**Durum:** ✅ Tamamlandı (`81/81` test, `benchmarks/results/benchmark_v0.6.0.json` ve `docs/RELIABILITY.md` doğrulandı)
- **Paralel Toplu Sorgu Yürütücüsü (`BatchQueryExecutor`, `batch_search`, `batch_hybrid_search`):** 100K×128D HNSW üzerinde **18,765 QPS (2.45x)**; ölçümlenmiş GIL/OpenBLAS sınırlarına göre otomatik işçi yönlendirmesi.
- **`asyncio` İstemci Sarmalayıcısı (`AsyncHisamDB`, `AsyncCollection`, `AsyncTenantCollection`):** Sıfır ek bağımlılıkla `asyncio.gather()` fan-out (**3,287 QPS** @ 100K HNSW).
- **Bloklamayan Arka Plan Bakım (`BackgroundTaskManager`):** Snapshot-and-Swap `rebuild_index_async()` (50K IVF inşası süresince **2,212 canlı okuma, P50 = 0.363 ms, Max = 3.99 ms**) + coalesced `compact_async()` + `auto_compact_async=True` (**49.89 ms → 0.821 ms, 60.8x düşük silme kuyruk gecikmesi**).
- **Açık Dayanıklılık & Eşzamanlılık Sözleşmesi (`docs/RELIABILITY.md`)** + double-checked thread-safe lazy init + `HNSWIndex` `_ef_lock` + 12 iş parçacıklı kaos/stres test paketi.

---

## Faz 5 — Üretim Sunucusu, Güvenlik, Gözlemlenebilirlik ve HTTP SDK (`v0.7.0`)
**Durum:** ✅ Tamamlandı (`86/86` test, `benchmarks/results/benchmark_v0.7.0.json` ve `docs/API_SERVER.md` doğrulandı)
- **FastAPI + Uvicorn REST Sunucusu (`hisamdb-server`):** Koleksiyon yönetimi, CRUD, toplu işlemler, hibrit/RAG/filtreli arama, `/health`, `/ready`, sıfır-bağımlılıklı Prometheus `/metrics`. Canlı 50K HNSW üzerinde `POST /search` **3.94 ms**, `POST /batch_search(64)` **1,610 QPS**.
- **Güvenlik:** SHA-256 hash'lenmiş API anahtarları (`hmac.compare_digest`), kapsam (`read`/`write`/`admin`) + sunucu tarafı katı `tenant_id` yalıtımı, Pydantic v2 sonlu-vektör/batch doğrulama, `TokenBucketRateLimiter` (`429`), snapshot path-traversal koruması ve `SanitizingLogFilter`.
- **Dağıtım & SDK:** Çok aşamalı non-root `Dockerfile`, `docker-compose.yml` ve sıfır-bağımlılıklı Keep-Alive Python HTTP SDK'sı (`RemoteHisamDB`, `RemoteCollection`).

---

## Faz 6 — Büyük Ölçekli Araştırma, Native SIMD, 1M/1536D Benchmarklar ve Görsel Dokümantasyon (`v1.0.0`)
**Durum:** ✅ Tamamlandı (`90/90` test, `benchmarks/results/benchmark_v1.0.0.json`, `docs/ARCHITECTURE_SCALE_RESEARCH.md` ve `docs/assets/*.png` doğrulandı)
- **Opsiyonel Sıfır-Bağımlılıklı Rust Native SIMD (`kernels_experimental/hisam_simd.rs`):** AVX2 + FMA3 + 64-bit POPCNT çekirdeği; `Flat PQ` **4.83x**, `768D Binary` **4.33x**, `IVF-PQ` **2.04x QPS**, `INT8` **1.71x–1.91x** hızlanma (derlenmemiş ortamlarda şeffaf saf NumPy fallback).
- **1.000.000 Vektör & 384D / 768D / 1536D Benchmarkları (`benchmarks/run_benchmarks_v10.py`):**
  - `1M × 128D`: `DiskIndex (MMap)` **15.52 MB RAM (-%96.8)** ile **9.56 ms, Recall=1.000**; `HNSW` **0.605 ms (1,640 QPS)**; `IVF-PQ` `500K`'da FAISS `IndexIVFPQ`'yu hem gecikme hem doğrulukta geçti.
  - `OpenAI-1536D (25K)`: `DiskIndex` **0.95 MB RAM (-%99.4), 1.50 ms, Recall=1.000**; `Binary+Rescore` **0.975 ms, Recall=0.884**.
- **GPU/CUDA ve `v2.x` Dağıtık Mimari Araştırması (`docs/ARCHITECTURE_SCALE_RESEARCH.md`):** PCIe-vs-compute başa-baş analizi, opsiyonel `cupy` stratejisi ve Consistent-Hashing Sharding + WAL Replikasyon + gRPC taslağı.
- **Yayın Kalitesinde Görseller & Erişilebilir `README.md`:** Gerçek JSON verilerinden üretilen 6 PNG grafik (`docs/assets/*.png`) ve teknik terimleri günlük benzetmelerle açıklayan kapsamlı sözlük.

---

## Riskler ve Azaltma Stratejileri

| Risk | Etki | Azaltma Stratejisi |
|---|---|---|
| Geriye dönük uyumluluk kırılması | Mevcut kullanıcı kodları ve testler bozulur | Tüm mevcut sınıflar, imza adları ve `.npy` v0.2 disk dosyaları otomatik uyumlulukla korunur; tüm fazlarda regresyon testi çalıştırılır. |
| Aşırı bağımlılık şişmesi | HisamDB'nin hafiflik avantajı kaybolur | Çekirdek **yalnızca `numpy`** gerektirir; `hnswlib`, `fastapi`, `uvicorn` vb. kesinlikle opsiyonel eklentiler olarak tutulur. |
| Windows dosya kilitleme / `mmap` sorunları | Windows'ta açık `memmap` dosyası üzerine yazma/silme `PermissionError` verir | `close()` ile açık tutamaç kapatma, atomik değiştirme öncesi mmap serbest bırakma ve Windows üzerinde kapsamlı test. |
| Python GIL darboğazı | Çok iş parçacıklı Python kodları ölçeklenmez | Ağır hesaplamalar OpenBLAS/NumPy C döngülerinde (GIL serbest) çalıştırılır; kritik filtre/indeks yolları vektörize edilir. |
