# MicroVectorDB — Mühendislik Yol Haritası (`ROADMAP.md`)

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
   - `MicroVectorDB` üst düzey kolaylık istemcisi (`PersistentClient` / `EphemeralClient` ile %100 uyumlu).
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

## Faz 3 — Gelişmiş Arama (Sparse, Hybrid) ve RAG Optimizasyonu (`v0.5.0`)
**Durum:** Sıradaki Aşama
- **Sparse Vector & BM25 Motoru:** Ters indeksli anahtar kelime ve sparse embedding (`{token_id: weight}`) araması.
- **Hybrid Search:** Reciprocal Rank Fusion (RRF), Min-Max / Z-Score skor normalizasyonu ve ayarlanabilir ağırlıklı füzyon ($\alpha \cdot \text{Dense} + \beta \cdot \text{Sparse}$).
- **RAG İlkelleri:** Doküman/chunk yaşam döngüsü, Parent-Child retrieval, doküman gruplama, sonuç tekilleştirme (deduplication), bağlam penceresi birleştirme, harici reranker kanca arayüzü (CrossEncoder/Cohere/API bağımsız), çoklu vektör (multi-vector) ve multi-tenant koleksiyon izolasyonu.

---

## Faz 4 — Üretim Güvenilirliği: WAL, Recovery, Snapshot ve Eşzamanlılık (`v0.6.0`)
- **Write-Ahead Logging (WAL):** Sıra numaralı (LSN), CRC32 sağlama toplamlı ikili yazma günlüğü; beklenmedik kapanmada otomatik WAL replay ve kısmi yazma budaması.
- **Checkpoint, Snapshot, Backup & Restore:** Atomik nokta-zaman anlık görüntüleri, sıkıştırılmış yedekleme ve doğrulamalı geri yükleme.
- **Açık İşlem Garantileri:** Batch atomikliği ve hata anında eksiksiz bellek+disk geri alması (rollback); belgelenmiş izolasyon/dayanıklılık sözleşmesi (kanıtlanmamış ACID iddiası olmadan).
- **Yüksek Eşzamanlılık:** Paralel batch sorgu yürütme, `asyncio` istemci sarmalayıcısı, arka plan indeks oluşturma/sıkıştırma (background compaction).

---

## Faz 5 — Üretim Sunucusu ve REST/gRPC API (`v0.7.0`)
- **FastAPI Sunucusu:** Koleksiyon yönetimi, CRUD, toplu işlemler, hibrit/filtreli arama, `/health`, `/ready`, Prometheus `/metrics`.
- **Güvenlik:** Hash'lenmiş API anahtarları, kapsam/tenant bazlı yetkilendirme, Pydantic girdi doğrulama, hız sınırlama (rate limiting), hassas verileri maskeleyen loglama.
- **Dağıtım:** Çok aşamalı `Dockerfile`, `docker-compose.yml` ve istemci SDK'sı.

---

## Faz 6 — Büyük Ölçekli Araştırma, SIMD/Native ve Rakip Benchmark (`v1.0.0`)
- **Standart Veri Kümeleri:** SIFT1M, GloVe, BEIR üzerinde 10K / 100K / 500K / 1M ölçekte ve 128 / 384 / 768 / 1024 / 1536 boyutlarda Qdrant, FAISS, ChromaDB, LanceDB ve Milvus ile eşit donanım & eşit Recall@10 benchmarkları (`BENCHMARK_REPORT.md` + grafikler).
- **Native SIMD / C / Rust Uzantısı Değerlendirmesi:** `kernels_experimental/avx2_dot_product.c` tabanlı toplu SIMD / INT8 VNNI / Binary POPCNT çekirdeklerinin NumPy/OpenBLAS'a karşı gerçek kazanım analizi.
- **GPU ve Dağıtık Mimari Araştırması:** Sharding, replikasyon ve opsiyonel CUDA/CuPy fizibilite raporu.

---

## Riskler ve Azaltma Stratejileri

| Risk | Etki | Azaltma Stratejisi |
|---|---|---|
| Geriye dönük uyumluluk kırılması | Mevcut kullanıcı kodları ve testler bozulur | Tüm mevcut sınıflar, imza adları ve `.npy` v0.2 disk dosyaları otomatik uyumlulukla korunur; tüm fazlarda regresyon testi çalıştırılır. |
| Aşırı bağımlılık şişmesi | MicroVectorDB'nin hafiflik avantajı kaybolur | Çekirdek **yalnızca `numpy`** gerektirir; `hnswlib`, `fastapi`, `uvicorn` vb. kesinlikle opsiyonel eklentiler olarak tutulur. |
| Windows dosya kilitleme / `mmap` sorunları | Windows'ta açık `memmap` dosyası üzerine yazma/silme `PermissionError` verir | `close()` ile açık tutamaç kapatma, atomik değiştirme öncesi mmap serbest bırakma ve Windows üzerinde kapsamlı test. |
| Python GIL darboğazı | Çok iş parçacıklı Python kodları ölçeklenmez | Ağır hesaplamalar OpenBLAS/NumPy C döngülerinde (GIL serbest) çalıştırılır; kritik filtre/indeks yolları vektörize edilir. |
