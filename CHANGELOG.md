# Changelog

Bu dosya MicroVectorDB projesindeki tüm önemli değişiklikleri belgeler.

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
