# Changelog

Bu dosya MicroVectorDB projesindeki tüm önemli değişiklikleri belgeler.

## [0.7.0] - 2026-11-21 (Faz 5 — Üretim FastAPI REST Sunucusu, Güvenlik, Prometheus `/metrics`, Python SDK & Docker)

### Eklenenler
- **Üretim FastAPI REST Sunucusu (`microvectordb/api/server.py`, `microvectordb-server` CLI):**
  - Lifespan yönetimli `AsyncMicroVectorDB` çekirdeği, opsiyonel `pip install ".[server]"` eklentisi (çekirdek hâlâ yalnızca `numpy>=1.20`!).
  - Tam `/v1/collections` REST API: koleksiyon yönetimi, `add`, `documents` (RAG), `upsert`, `get`, `delete`, `search`, `batch_search`, `sparse_search`, `hybrid_search`, `batch_hybrid_search`, `retrieve_context`, `explain`, `compact`, `rebuild`, `checkpoint` ve path-traversal korumalı `/v1/admin/snapshot`.
  - Canlı Uvicorn TCP loopback benchmarkı (Bearer Auth etkin, 50K HNSW): tekil `POST /search` **3.94 ms (233 QPS)**, `POST /batch_search(64)` **1,610 QPS**, `POST /hybrid_search` (10K) **5.68 ms**, `POST /delete` **5.54 ms**.
- **Güvenlik & Hız Sınırlama (`microvectordb/api/security.py`, `schemas.py`):**
  - SHA-256 hash'lenmiş API anahtarları (`sha256:<hex>` desteği + sabit zamanlı `hmac.compare_digest`), kapsam (`read`/`write`/`admin`) ve sunucu tarafı **katı Multi-Tenant anahtar yalıtımı** (çapraz kiracı okuma/silme/upsert saldırılarına karşı korumalı).
  - Kimlik/IP bazlı thread-safe **`TokenBucketRateLimiter`** (`HTTP 429` + `Retry-After` / `X-RateLimit-Remaining`).
  - Katı Pydantic v2 şema doğrulaması (sonlu float / `NaN`/`Inf` reddi, `max_batch_size` `413` koruması) ve Bearer/anahtar maskeleyen **`SanitizingLogFilter`**.
- **Gözlemlenebilirlik (`microvectordb/api/metrics.py`):**
  - `/health`, `/ready` probları ve sıfır bağımlılıklı Prometheus metin dışa aktarımı **`/metrics`** (`mvdb_http_requests_total`, gecikme histogramı, `mvdb_rate_limited_total`, `mvdb_auth_failures_total`, `mvdb_vectors_total`, `mvdb_tombstones_total`, `mvdb_vector_ram_bytes`).
- **Sıfır-Bağımlılıklı Python HTTP SDK (`RemoteMicroVectorDB`, `RemoteCollection`):**
  - Standart kütüphane `http.client` tabanlı thread-safe **HTTP/1.1 Keep-Alive** istemcisi (`requests`/`httpx` gerekmez).
- **Konteyner & Dokümantasyon:** Çok aşamalı non-root `Dockerfile` (dahili `HEALTHCHECK`), `docker-compose.yml`, `.dockerignore`, [`docs/API_SERVER.md`](docs/API_SERVER.md), `examples/07_rest_server_and_sdk.py`, `benchmarks/run_benchmarks_v07.py` ve 5 yeni kapsamlı REST/güvenlik/canlı-sunucu testi (toplam **86/86 test**).

## [0.6.0] - 2026-11-21 (Faz 4 — Yüksek Eşzamanlılık, Async İstemci, Paralel Batch ve Bloklamayan Arka Plan İşçileri)

### Eklenenler
- **Paralel Toplu Sorgu Yürütücüsü (`BatchQueryExecutor`, `batch_search`, `batch_hybrid_search`):**
  - Paylaşımlı `filter` veya sorgu başına ayrı `filters` listesiyle paralel toplu arama; `100K × 128D` HNSW üzerinde `workers=8` ile **18,764.8 QPS (2.45x hızlanma)**.
  - Ölçümlenmiş Python GIL / OpenBLAS sınırlarına göre otomatik işçi yönlendirmesi (HNSW ve harici reranker için çok çekirdekli havuz; kısa NumPy/OpenBLAS çekirdeklerinde sıfır çekişmeli tek-işçi yolu, `max_workers` ile geçersiz kılınabilir).
- **Standart `asyncio` İstemcisi (`AsyncMicroVectorDB`, `AsyncCollection`, `AsyncTenantCollection`):**
  - Sıfır ek bağımlılıkla `async with AsyncMicroVectorDB(...) as db:` ve `asyncio.gather()` desteği (`100K HNSW` `gather(256)` → **3,286.8 QPS**).
- **Bloklamayan Arka Plan Bakım Yöneticisi (`BackgroundTaskManager`):**
  - `collection.rebuild_index_async()`: Snapshot-and-Swap mimarisi ile ağır K-Means/PQ/HNSW eğitimini **tamamen kilit dışında** çalıştırır ve inşa sırasında gelen yeni vektör deltalarını otomatik birleştirir (`50K IVF` 1.05s inşa süresince **2,212 canlı okuma**, okuyucu **P50 = 0.363 ms, Max = 3.99 ms — 262x düşük azami gecikme**).
  - `collection.compact_async()` (görev birleştirmeli / coalesced) ve `index_params={"auto_compact_async": True}` (yoğun silme eşiğinde çağıran gecikmesi **49.89 ms → 0.821 ms, 60.8x düşük kuyruk gecikmesi**).
  - `wait_for_background()` ve `close()` anında temiz havuz/görev kapatma.
- **Açık Dayanıklılık & Eşzamanlılık Sözleşmesi (`docs/RELIABILITY.md`):** Kanıtlanmamış ACID iddiası olmadan gerçek Atomicity, Consistency, Isolation ve `wal_sync="normal"|"full"` Durability matrisi.
- Yeni örnek (`examples/06_async_and_parallel.py`), `benchmarks/run_benchmarks_v06.py` ve 12 iş parçacıklı stres/yarış testleri dahil 6 yeni test grubu (toplam **81/81 test**).

### İyileştirilenler & Sertleştirilenler (Thread-Safety)
- `Collection` tembel indeks başlatıcıları (`_ensure_meta_index_locked`, `_ensure_bm25_index_locked`, `_ensure_sparse_index_locked`), `VectorBuffer._ensure_norms()` ve `BM25Index._ensure_compiled()` soğuk açılış sonrası eşzamanlı okuyucu yarışlarını engellemek için double-checked `threading.Lock` ile korundu.
- `HNSWIndex.search()` içindeki dinamik `ef_search` değişikliği dahili `RWLock` ile korundu (varsayılan `ef_search`'te sıfır bekleme, özel `ef_search`'te yarışmasız koruma).
- Yazar bilgisi `LICENSE` dosyasıyla uyumlu biçimde `Ahmet GÜN` olarak güncellendi.

## [0.5.0] - 2026-11-21 (Faz 3 — Hibrit/BM25/Sparse Arama, RAG İlkelleri, WAL, `<1 ms` Tombstone Silme & Hızlı Açılış)

### Çözülen 3 Kritik Performans & Güvenilirlik Sorunu
- **Sorun #1 — 100K Hızlı Açılış (4.23x'e Kadar Hızlanma):**
  - Epoch + CRC32 doğrulamalı sütunsal ikili önbellek (`.items.mvdbcol`), ham tuple SQLite yükleyici, opsiyonel `orjson` hızlandırıcısı (`pip install ".[fast]"`), tembel (lazy) `MetadataIndex` / `BM25Index` / `SparseVectorIndex` kurulumu ve `mmap` modunda tembel norm hesaplama (`VectorBuffer`) eklendi.
  - `use_mmap=True` ve `keep_raw_vectors=False` ayarları koleksiyon metaverisinde kalıcı hale getirildi.
  - 100K×128D açılış süresi: `DiskIndex` **532.8 ms → 125.8 ms (4.23x, 0.00 MB vektör RAM)**, `IVF-PQ (keep_raw=F)` **542.0 ms → 198.9 ms (2.72x, 0.00 MB vektör RAM)**, `IVF` **781.3 ms → 486.7 ms (1.61x)**.
- **Sorun #2 — Ön-Hesaplanmış `IVF-PQ` + Tek-Atış `uint16` ADC (+%61.6 QPS):**
  - `ProductQuantizer.compute_distance_table()` içindeki `np.einsum` yükü `np.matmul` ile değiştirildi.
  - `IVFPQIndex` için $P[c, m, k] = \|C_{m,k}\|_2^2 + 2(c_{c,m}\cdot C_{m,k})$ küme-artık ön-hesaplama tablosu, sorgu başına tek batched `matmul` ve ön-kaydırmalı `uint16` tek-atış `np.take(...).sum(axis=0)` ADC motoru eklendi.
  - 100K×128D `IVF-PQ`: **3.446 ms (255.1 QPS) → 2.267 ms (412.2 QPS)**, Recall@10 = **0.270** (FAISS `IndexIVFPQ`'nun `0.258` değerinin üzerinde).
- **Sorun #3 — `<1 ms` Tombstone Silme (451x Hızlanma) + İkili WAL + SHA-256 Snapshot:**
  - `WriteAheadLog` (`microvectordb/storage/wal.py`): 21-byte başlıklı (`MWAL`), LSN + OpType + CRC32 korumalı ikili WAL, kalıcı dosya tutamacı, elektrik/süreç kesintisi sonrası yarım çerçeve budaması (torn-write recovery), `replay()` ve `wal_sync="normal" | "full"` dayanıklılık seçenekleri.
  - `Collection.delete()` artık tüm dosyayı yeniden yazmak yerine $O(\text{silinen})$ bitmask Tombstone işaretleme + WAL + tek atomik SQLite `DELETE` kullanır: 10K tekil silme **181.9 ms → 0.403 ms (451.4x hızlanma)**, 100K tekil silme **0.382 ms** (`wal_sync="full"` fsync ile **1.967 ms — 92.5x**).
  - Otomatik (varsayılan ≥%25 ve ≥64 tombstone) ve manuel `collection.compact()` sıkıştırması.
  - `db.create_snapshot(dir)` ve `MicroVectorDB.restore_snapshot(src, dst)`: Online SQLite backup + tüm ikili/indeks dosyaları + SHA-256 manifest doğrulaması ve bozulma/kurcalama tespiti (`DataCorruptionError`).

### Eklenenler (Faz 3 — Gelişmiş Arama & RAG Motoru)
- **Vektörize Okapi BM25 (`BM25Index`) & Türkçe/Unicode Tokenizer (`tokenize_text`):** `İ -> i`, `I -> ı` dönüşümlü tokenizer ve derlenmiş NumPy terim ağırlık dizileriyle 10K dokümanda **0.476 ms (1,982 QPS)** anahtar kelime araması (Python döngüsüne göre **28.3x hızlanma**).
- **Seyrek Vektör Motoru (`SparseVectorIndex`):** `{token_id: weight}` (SPLADE / BGE-M3 / TF-IDF) kalıcı SQLite + ters indeks depolaması; 10K dokümanda **0.151 ms (7,269 QPS)**.
- **Hibrit Arama (`collection.hybrid_search()`):** Dense + BM25 + Sparse 2-yollu ve 3-yollu arama; **Reciprocal Rank Fusion (`rrf`)**, **Min-Max (`weighted`)** ve **Z-Score (`z_score`)** füzyon yöntemleri, ayarlanabilir `alpha` / `beta` / `sparse_weight` ağırlıkları. 10K dokümanda **0.936 ms (1,033 QPS)** ve **Hit@10 = 1.000** (Dense-Only `0.530`, BM25-Only `0.500`).
- **RAG İlkelleri (`microvectordb/search/rag.py` & `Collection`):**
  - `chunk_text()` örtüşmeli doküman parçalayıcı ve `collection.add_documents()` (`_doc_id`, `_parent_id`, `_chunk_index`, `_tenant_id`).
  - `collection.retrieve_with_context(window=1)`: Parent-Child komşu parça bağlam penceresi birleştirme.
  - `collection.multi_vector_search()`: Çoklu sorgu vektörü için `rrf`, `max`, `mean` füzyonu.
  - Harici Reranker kancası (`reranker=` — çekirdeği ML modellerine bağlamaz, kilit dışında çalışır), sonuç tekilleştirme (`deduplicate=True`) ve doküman gruplama (`group_by="_doc_id"`, `max_per_group`).
  - **Multi-Tenant İzolasyon (`db.get_tenant_collection(name, tenant_id)` / `TenantCollection`):** Tüm ekleme, arama ve silme işlemlerinde katı kiracı yalıtımı.
- Yeni uçtan uca örnek (`examples/05_hybrid_rag_and_reliability.py`), `benchmarks/run_benchmarks_v05.py` ve 12 yeni birim testi (toplam **75/75 test**).

## [0.4.0] - 2026-11-20 (Faz 2 — İleri İndeksleme, PQ/IVF-PQ, DiskIndex ve Query Optimizer)

### Eklenenler
- **Product Quantization (`ProductQuantizer`):** K-Means++ alt-uzay kod kitapları, vektörize `np.einsum` ADC arama tabloları, sütun-bitişik `(M, N)` önbellek dostu akümülasyon, bölünmeyen boyutlarda şeffaf sıfır-dolgu, birim-küre L2 kosinüs düzeltmesi ve CRC32 ikili serileştirme (`to_bytes`/`from_bytes`). 100K×128D'de **1.65 MB RAM (%96.7 tasarruf)**.
- **IVF-PQ (`IVFPQIndex`, `index_type="ivf_pq"`):** Kaba IVF kümeleme + artık vektörler üzerinde Product Quantization. 100K×128D'de **2.48 MB RAM (%95 tasarruf)** ve aynı 8-bit `M=16` bütçesinde FAISS `IndexIVFPQ`'dan daha yüksek Recall@10 (`0.270` vs `0.258`).
- **Out-of-Core Disk Tabanlı İndeks (`DiskIndex`, `index_type="disk"`) ve `DiskStreamingScanner`:**
  - `DiskIndex`: Küme sırasına dizilmiş `.mvdbdisk` mmap dosyası; 100K×128D'de yalnızca **2.35 MB RAM (%95.3 tasarruf)** ile **1.23 ms (684 QPS)** ve **1.000 Recall@10**.
  - `DiskStreamingScanner`: Sabit RAM bütçeli (4K blok) exact mmap akış taraması; 100K×128D'de yalnızca **0.76 MB RAM (%98.5 tasarruf)** ile **1.000 Recall@10**.
- **CRC32 + Epoch Doğrulamalı Anında İndeks Kalıcılığı:** `HNSWIndex`, `IVFIndex`, `IVFPQIndex` ve `DiskIndex` artık `checkpoint()` ve `close()` anında atomik olarak diske kaydedilir ve açılışta yeniden inşa edilmeden yüklenir (**IVF-PQ 24.1x**, **HNSW 9.8x**, **DiskIndex 5.2x**, **IVF 3.7x** daha hızlı açılış). Bozuk/bayat indekslerde otomatik güvenli yeniden inşa yapılır.
- **Maliyet Tabanlı Query Optimizer (`QueryOptimizer` & `collection.explain()`):**
  - Filtre seçiciliğine göre otomatik `pre_filter_exact` (`s < %5` veya `M <= 2048`), sıfır-Python-set `np.searchsorted` tabanlı `hnsw_post_filter` (eksik sonuçta otomatik callback fallback), uyarlamalı `nprobe` artışı, `disk_stream` seçimi ve uzman kullanıcılar için manuel `strategy` / `filter_strategy` parametreleri.
  - 100K×128D filtreli HNSW'de v0.3.0 callback'e göre **2.2x–7.7x**, v0.2.0'a göre %10 filtrede **123.8x hızlanma** (59.32 ms → **0.479 ms**, >2,000 QPS).

### İyileştirilenler
- `HNSWIndex.load_with_meta()` içindeki `hnswlib` çift başlatma uyarısı `_skip_init=True` ile temizlendi.
- `StorageEngine` şeması `data_epoch` ve `index_params_json` sütunlarıyla geriye dönük uyumlu biçimde genişletildi; koleksiyon silinirken tüm yan indeks dosyaları (`.mvdbidx`, `.hnsw.bin`, `.mvdbdisk` vb.) otomatik temizlenir.

## [0.3.0] - 2026-11-20 (Faz 1 — Çekirdek Optimizasyon, Güvenlik ve Veri Bütünlüğü)

### Eklenenler
- **Modüler Paket Mimarisi:** `metrics/`, `storage/`, `filtering/`, `quantization/`, `indexes/`, `concurrency/` paketleri (tüm v0.2.0 import yolları geriye dönük uyumlu).
- **Birleşik İstemci ve Ergonomik Arama:** `MicroVectorDB(path=...)`, `collection.search(...) -> List[SearchResultItem]`, `score_threshold`, `collection.stats()`, `collection.checkpoint()`, `client.close()` ve `with client:` context manager desteği.
- **Ters Metadata İndeksi (`MetadataIndex`):** Eşitlik kovaları + önbellekli sayısal sütun dizileri; `$and`, `$or`, `$not`, `$exists`, `$contains` ve nokta-yollu iç içe (`"author.country"`) filtre desteği; 100K filtreli sorguda **24x hızlanma** (59.3 ms → 2.47 ms).
- **Sürümlü İkili Depolama Formatı (`.mvdbvec`):** 64-byte hizalı başlık, CRC32 sağlama toplamı, atomik yazma (`fsync` + `os.replace`), $O(M)$ artımlı ekleme, gerçek `use_mmap=True` desteği, çökme kurtarma budaması ve eski v0.2.0 `.npy` dosyalarından otomatik göç.
- **Koleksiyon İçi İndeks ve Kuantizasyon Entegrasyonu:** `index_type="flat" | "ivf" | "hnsw" | "auto"`, `quantization="none" | "float16" | "int8" | "binary"`, `storage_dtype="float32" | "float16"`, `keep_raw_vectors=True|False`.
- **Yeni Kuantizasyon Motorları:** `Float16Quantizer` (2x RAM tasarrufu) ve `BinaryQuantizer` (32x RAM tasarrufu, NumPy 2.0 `uint64 bitwise_count` + NumPy 1.x LUT fallback).
- **Opsiyonel HNSW Desteği:** `HNSWIndex` (`hnswlib` adaptörü; dinamik yeniden boyutlandırma, filtreli arama, `cosine`/`dot_product`/`euclidean`).
- **Eşzamanlılık Koruması (`RWLock`):** Yazıcı-öncelikli Reader-Writer Lock ve SQLite WAL + dahili `RLock`.
- **Bilimsel Denetim ve Benchmark Altyapısı:** `ARCHITECTURE_AUDIT.md`, `ROADMAP.md`, `BENCHMARK_REPORT.md` ve JSON/CSV çıktılı `benchmarks/run_benchmarks.py`.

### İyileştirilenler
- **Kosinüs Benzerliği Bellek Kullanımı (%50.6 Tasarruf):** Tam `(N, D)` normalizasyon kopyası yerine `(N,)` `inv_norms` vektörü (50K×768D: +297.4 MB → 146.9 MB; soğuk ilk sorgu 183.8 ms → ~9 ms).
- **Öklid Mesafesi (14.6x Hızlanma):** Önbellekli `sq_norms` ve yerinde BLAS işlemleri (50K×768D: 116.3 ms → 7.99 ms).
- **Dinamik `VectorBuffer`:** Kapasite-ikileyen bitişik tampon ile `np.vstack` $O(N^2 D)$ kopyalaması kaldırıldı.
- **Artımlı Kalıcılık (3.0x Hızlanma):** Yalnızca değişen kayıtları yazan delta SQLite + `.mvdbvec` append; `EphemeralClient` için sıfır-SQLite hızlı bellek yolu (10K ekleme: 244K v/s → 1.11M v/s).
- **`IVFIndex`:** Gerçek K-Means++, vektörize centroid güncellemesi, bitişik küme blokları, BLAS mesafe hesaplama, tüm 3 metrik desteği (100K Gaussian araması 2.10 ms → 0.42 ms, 5.0x hızlanma).
- **`Collection.get(ids=...)`:** $O(N \cdot |ids|)$ taramadan $O(|ids|)$ doğrudan hash erişimine geçirildi.

### Düzeltilenler (Güvenlik & Doğruluk)
- **Güvenlik (Path Traversal):** Koleksiyon adları doğrulandı ve tüm dosya yolları SHA-256 hash'li güvenli tekil adlarla `persist_dir` içine hapsedildi.
- **Veri Bütünlüğü (Tablo Çakışması):** `"my-docs"` ve `"my_docs"` artık SHA-256 tabanlı ayrı SQLite tabloları kullanır.
- **`ScalarQuantizer` `int16` Taşma Hatası:** `diff ** 2` taşması giderilerek ölçek-ağırlıklı asimetrik BLAS hesabına geçildi (100K Gaussian Recall@10: 0.630 → 0.980, 2.1x daha hızlı).
- **`IVFIndex` `argpartition` Çökmesi:** `nprobe >= n_clusters` durumunda oluşan `ValueError: kth out of bounds` giderildi.
- **Girdi Doğrulaması & Rollback:** `NaN`/`Inf` reddi (`InvalidVectorError`), batch içi tekrarlanan ID tespiti, oluşturma anında metrik doğrulaması, sorgu boyut doğrulaması (`DimensionMismatchError`), karma tip metadata filtre güvenliği, 500'lü chunked SQLite silme ve disk yazım hatasında bellek geri alması (rollback).

## [0.2.0] - 2026-09-15
- İlk halka açık sürüm: `PersistentClient`, `EphemeralClient`, `Collection`, `argpartition` Top-K, normalizasyon önbelleği, bağımsız `IVFIndex` ve `ScalarQuantizer`, 27 birim testi.
