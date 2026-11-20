# Changelog

Bu dosya MicroVectorDB projesindeki tüm önemli değişiklikleri belgeler.

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
