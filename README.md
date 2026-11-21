# MicroVectorDB 🚀

[![Version](https://img.shields.io/badge/version-0.7.0-blue.svg)](CHANGELOG.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-brightgreen.svg)](https://www.python.org/)
[![Core Dependencies: NumPy](https://img.shields.io/badge/core%20dependencies-NumPy%20only-blueviolet.svg)](requirements.txt)

**MicroVectorDB**, yüksek performanslı, düşük bellek tüketimli, hibrit arama ve RAG yeteneklerine sahip, asenkron/çok çekirdekli, WAL korumalı, gömülü veya üretim FastAPI REST sunucusu olarak çalışabilen ve yalnızca `numpy` temel bağımlılığı ile çalışan açık kaynaklı bir Python Vektör Veritabanıdır.

> **Felsefe:** *High Performance + Low Memory + Fast Retrieval + Reliable Storage + Developer Friendly*

---

## ✨ Neler Sunar? (v0.7.0 — Doğrulanmış Özellikler)

- **Sıfır Ağır Çekirdek Bağımlılığı:** Çekirdek yalnızca `numpy>=1.20` gerektirir (NumPy 1.x ve 2.x tam uyumlu). Opsiyonel eklentiler: HNSW için `pip install ".[hnsw]"`, hızlı JSON/metadata açılışı için `pip install ".[fast]"` (`orjson`), üretim REST sunucusu için `pip install ".[server]"` (`fastapi` + `uvicorn` + `pydantic`) veya hepsi için `pip install ".[all]"`.
- **Üretim REST Sunucusu, Güvenlik, Gözlemlenebilirlik & Python SDK (`v0.7.0`):**
  - **FastAPI + Uvicorn Sunucusu (`microvectordb-server`):** Tam `/v1/collections` CRUD, Dense/Batch/Sparse/Hybrid/RAG Context, `explain`, `compact`, `rebuild`, `checkpoint`, güvenli `/v1/admin/snapshot`, `/health`, `/ready` ve sıfır bağımlılıklı Prometheus **`/metrics`**.
  - **Güvenlik & Rate Limiting:** SHA-256 hash'lenmiş API anahtarları (`hmac.compare_digest`), kapsam (`read`/`write`/`admin`) + sunucu tarafı katı **`tenant_id` yalıtımı**, kimlik/IP bazlı `TokenBucketRateLimiter` (`HTTP 429`), Pydantic v2 sonlu-vektör doğrulaması ve gizli anahtar maskeleyen loglama.
  - **Sıfır-Bağımlılıklı Python HTTP SDK (`RemoteMicroVectorDB`):** Standart `http.client` Keep-Alive istemcisi (canlı 50K HNSW üzerinde tekil `POST /search` **3.94 ms**, `POST /batch_search(64)` **1,610 QPS**).
  - **Docker Hazır:** Çok aşamalı non-root `Dockerfile` (`HEALTHCHECK` dahil) ve `docker-compose.yml`. Tam rehber: [`docs/API_SERVER.md`](docs/API_SERVER.md).
- **Yüksek Eşzamanlılık, Async İstemci & Bloklamayan Arka Plan İşçileri (`v0.6.0`):**
  - **Çok Çekirdekli Paralel Toplu Arama (`batch_search` & `batch_hybrid_search`):** 100K×128D HNSW üzerinde `workers=8` ile **18,765 QPS (2.45x hızlanma)**; GIL/OpenBLAS ölçümlerine dayalı otomatik işçi yönlendirmesi.
  - **Standart `asyncio` İstemcisi (`AsyncMicroVectorDB`, `AsyncCollection`, `AsyncTenantCollection`):** Sıfır ek bağımlılıkla `async with` ve `asyncio.gather()` fan-out (**3,287 QPS** @ 100K HNSW).
  - **Bloklamayan Arka Plan İndeks İnşası (`rebuild_index_async()`):** Snapshot-and-Swap mimarisi ile 50K IVF yeniden inşası (`1.05 s`) sürerken okuyucular **0.363 ms P50 (Max 3.99 ms)** ile kesintisiz hizmet verir ve inşa sırasında gelen yeni eklemeler otomatik birleştirilir.
  - **Arka Plan Sıkıştırma (`compact_async()` & `auto_compact_async=True`):** Yoğun silme eşiğinde kuyruk gecikmesini **49.89 ms → 0.821 ms (60.8x)** düşürür.
  - **Sertleştirilmiş Thread-Safety:** Yazıcı-öncelikli `RWLock`, double-checked tembel indeks/norm başlatma ve HNSW `_ef_lock` koruması. Detaylı ACID/dayanıklılık sözleşmesi: [`docs/RELIABILITY.md`](docs/RELIABILITY.md).
- **Hibrit Arama Motoru (`Dense + BM25 + Sparse`):**
  - **Vektörize Okapi BM25 (`BM25Index`) + Türkçe/Unicode Tokenizer:** `İ → i`, `I → ı` uyumlu; derlenmiş NumPy ağırlıklarıyla 10K dokümanda **0.476 ms (1,982 QPS)**.
  - **Seyrek Vektör İndeksi (`SparseVectorIndex`):** SPLADE / BGE-M3 / TF-IDF `{token: weight}` vektörleri için SQLite kalıcılığı ve **0.151 ms (7,269 QPS)**.
  - **Akıllı Skor Füzyonu (`hybrid_search`):** Reciprocal Rank Fusion (**RRF**), **Min-Max** ve **Z-Score** normalizasyonu, ayarlanabilir `alpha` / `beta` / `sparse_weight` ağırlıkları; 10K külliyatta **0.936 ms (1,033 QPS)** ve **Hit@10 = 1.000**.
- **RAG & Multi-Tenant İlkelleri:**
  - `chunk_text()`, `add_documents()` (`_doc_id`, `_parent_id`, `_chunk_index`, `_tenant_id`), `retrieve_with_context()` (Parent-Child bağlam penceresi), `multi_vector_search()`, sonuç tekilleştirme (`deduplicate=True`), doküman gruplama (`group_by`, `max_per_group`), çekirdekten bağımsız harici `reranker` kancası (kilit dışında çalışır) ve katı kiracı yalıtımlı **`TenantCollection` (`db.get_tenant_collection(...)`)**.
- **Çoklu İndeks Motoru + Anında Disk Kalıcılığı (CRC32 + Epoch):**
  - **Optimize Flat (Exact):** `inv_norms` (%50 daha az Kosinüs RAM), `sq_norms` (~15x hızlı Öklid) ve OpenBLAS SGEMM/SGEMV.
  - **`IVFIndex` & Ön-Hesaplanmış `IVFPQIndex` (Saf NumPy):** K-Means++, artık Product Quantization (**-%88.7 RAM**), sorgu başına tek batched `matmul` + `uint16` tek-atış ADC (**2.27 ms / 412 QPS**, FAISS `IndexIVFPQ`'dan daha yüksek Recall@10).
  - **`DiskIndex` (`index_type="disk"`) & `DiskStreamingScanner`:** Out-of-core bölümlenmiş `mmap`; 100K×128D'de **2.35 MB RAM (%95.3 tasarruf)** ile **1.23 ms (684 QPS, Recall@10=1.000)** ve **125.8 ms** açılış (`0.00 MB` vektör RAM).
  - **`HNSWIndex` (Opsiyonel `hnswlib`):** 100K×128D'de **3,463 QPS (P50 = 0.278 ms, Recall@10 = 0.975)**.
- **Maliyet Tabanlı Query Optimizer (`collection.explain()`):**
  - Seçicilik bazlı otomatik `pre_filter_exact`, sıfır-set `np.searchsorted` tabanlı `hnsw_post_filter` (100K %10 filtreli sorguda **0.479 ms — v0.2.0'a göre 123.8x hızlanma**), uyarlamalı `nprobe` ve sabit-RAM `disk_stream`.
- **Entegre Kuantizasyon:** `float16` (2x RAM), `int8` (taşmasız `ScalarQuantizer`, 4x RAM, Recall@10 ≈ 0.965–0.980), `pq` (Product Quantization, 30x RAM), `binary` (32x RAM, SIMD `bitwise_count` + isteğe bağlı float32 rescore).
- **Üretim Düzeyi Güvenilirlik (`WAL` + `<1 ms` Tombstone + `Snapshot`):**
  - 21-byte başlıklı CRC32 ikili **`WriteAheadLog`** (`MWAL`, yarım yazma budaması / torn-write recovery, çökme sonrası otomatik `replay()`, `wal_sync="normal" | "full"`).
  - $O(1)$ bitmask **Tombstone silme (`0.38–0.40 ms`!)** + otomatik/manuel `compact()`.
  - **SHA-256 doğrulamalı `create_snapshot()` / `restore_snapshot()`**, atomik `.mvdbvec`, `.items.mvdbcol` sütunsal önbellek, path-traversal koruması, v0.2.0 `.npy` otomatik göçü ve yazıcı-öncelikli `RWLock`.

---

## 🔧 v0.5.0'da Çözülen 3 Kritik Performans ve Güvenilirlik Sorunu

v0.4.0 benchmark denetiminde tespit edilen ve üretim kullanımını etkileyen **3 temel darboğaz**, v0.5.0'da bilimsel profilleme ile kök nedeninden çözülmüştür:

### 1. Sorun #1: 100K Koleksiyon Açılış (Reload) Süresi (~450 ms SQLite Yükü)
- **Kök Neden:** v0.4.0'da ikili indeksler 50–150 ms'de yüklense de 100.000 satırın SQLite'tan tek tek Python sözlüklerine dönüştürülmesi ve `mmap` modunda tüm vektör normlarının açılışta hevesle hesaplanması toplam açılışı 533–941 ms'ye çıkarıyordu.
- **Nasıl Çözüldü?**
  1. SQLite doğruluk kaynağının yanında epoch + CRC32 doğrulamalı ikili sütunsal önbellek (**`.items.mvdbcol`**) eklendi (herhangi bir bozulma veya çökme sonrası WAL replay durumunda şeffaf biçimde SQLite'a düşer).
  2. Ham tuple SQLite okuyucu + opsiyonel **`orjson`** hızlandırması (`pip install ".[fast]"`).
  3. **`MetadataIndex`, `BM25Index`, `SparseVectorIndex` ve `mmap` normları tembel (lazy)** hale getirildi — ilk filtreli/metin sorgusu gelene kadar açılışta sıfır gereksiz CPU ve **0.00 MB** vektör RAM harcar.
  4. `use_mmap=True` ve `keep_raw_vectors=False` ayarları koleksiyon metaverisinde kalıcılaştırıldı.
- **Ölçülen Sonuç (100K × 128D):**
  - `DiskIndex`: **532.8 ms → 125.8 ms (4.23x hızlanma**, soğuk inşaya göre **22.2x**, `0.00 MB` vektör RAM).
  - `IVF-PQ (keep_raw=F)`: **542.0 ms → 198.9 ms (2.72x hızlanma**, soğuk inşaya göre **65.8x**, `0.00 MB` vektör RAM).
  - `IVF`: **781.3 ms → 486.7 ms (1.61x hızlanma)**.
  - `MMap + 100K Zengin Metadata (3 alan)`: **~880 ms → 328.7 ms (2.68x hızlanma**, `0.00 MB` vektör RAM).

### 2. Sorun #2: Saf NumPy `IVF-PQ` Sorgu Hızı
- **Kök Neden:** v0.4.0'da her sorguda ziyaret edilen her küme için ayrı `np.einsum` mesafe tablosu (LUT) oluşturuluyor ve 16 alt-vektör için Python döngüsünde `+=` toplama yapılıyordu (`3.446 ms`, `255.1 QPS`).
- **Nasıl Çözüldü?**
  1. Küme-merkezli artık terimi $P[c, m, k] = \|C_{m,k}\|_2^2 + 2(c_{c,m} \cdot C_{m,k})$ eğitim ve yükleme anında tek seferlik ön-hesaplandı (`_precomp_flat_lut`).
  2. Sorgu başına yalnızca **1 kez** batched `np.matmul` (`-2 q · C`) hesaplanarak `np.einsum` Python yükü tamamen kaldırıldı.
  3. Küme içi kodlar bellek içinde ön-kaydırmalı `uint16` olarak tutularak 16 Python döngüsü yerine **tek-atış `np.take(lut_f, codes_t).sum(axis=0)`** ile hızlandırıldı (diskte hâlâ kompakt `uint8` saklanır).
- **Ölçülen Sonuç (100K × 128D, `C=128, np=8, M=16`):**
  - **P50: 3.446 ms → 2.267 ms** · **P95: 6.277 ms → 3.776 ms** · **QPS: 255.1 → 412.2 (+%61.6)** · **Recall@10: 0.270** *(FAISS `IndexIVFPQ`'nun `0.258` değerinin üzerinde, toplam RAM 5.62 MB = -%88.7 RAM)*.

### 3. Sorun #3: Tekil Kalıcı Silme Gecikmesi (181.9 ms) ve Çökme/Yedekleme Güvenilirliği
- **Kök Neden:** v0.3.0'da tek bir ID silindiğinde bile tüm `.mvdbvec` dosyası ve SQLite tablosu sıfırdan yeniden yazılıyordu (`181.9 ms`, 100K'da >1.8 saniye).
- **Nasıl Çözüldü?**
  1. **İkili CRC32 `WriteAheadLog` (`microvectordb/storage/wal.py`):** 21-byte başlıklı (`MWAL` + LSN + OpType + CRC32), kalıcı dosya tutamaçlı WAL eklendi. Yarım kalan yazmalar (torn writes) açılışta CRC32 ile tespit edilip temizce budanır ve uygulanmamış silmeler SQLite'a geri oynatılır (`replay()`). Varsayılan `wal_sync="normal"` (süreç çökmesine dayanıklı OS flush); elektrik kesintisi düzeyinde katı dayanıklılık için `index_params={"wal_sync": "full"}` (`os.fsync()`) seçilebilir.
  2. **`<1 ms` Bitmask Tombstone Silme + Compaction:** `delete()` yalnızca silinen slotları bellek bitmask'inde işaretler, WAL'a yazar ve SQLite'tan tek atomik sorguyla siler; indeks epoch'u bozulmadan korunur. Tombstone oranı eşiği aştığında (varsayılan ≥%25 ve ≥64 kayıt) otomatik veya istendiğinde `collection.compact()` ile fiziksel sıkıştırma yapılır.
  3. **SHA-256 Doğrulamalı Snapshot / Restore:** `db.create_snapshot(dir)` tutarlı SQLite online backup + tüm ikili/indeks dosyalarını SHA-256 manifestiyle paketler; `MicroVectorDB.restore_snapshot(src, dst)` tek bir bit bozulsa bile `DataCorruptionError` fırlatarak engeller.
- **Ölçülen Sonuç:**
  - 10K × 128D Tekil Kalıcı Silme (`wal_sync="normal"`): **181.9 ms → 0.403 ms (451.4x hızlanma!)**
  - 10K × 128D Tekil Kalıcı Silme (`wal_sync="full"` fsync): **181.9 ms → 1.967 ms (92.5x hızlanma!)**
  - 100K × 128D Tekil Kalıcı Silme (`wal_sync="normal"`): **0.382 ms** *(O(1) sabit gecikme!)*
  - 10K × 128D SHA-256 Snapshot Oluştur / Geri Yükle: **203.1 ms / 63.5 ms**.

---

## 📊 Doğrulanmış Performans Sonuçları (v0.7.0)

*Ölçüm ortamı: Windows x86-64, 12 mantıksal Intel çekirdeği (AVX2/FMA), 15.73 GB RAM, Python 3.11.9, NumPy 2.0.2 (OpenBLAS 0.3.27), FAISS 1.13.2. Tüm ham veriler için [`BENCHMARK_REPORT.md`](BENCHMARK_REPORT.md) dosyasına bakın.*

### v0.2.0 → v0.7.0 Evrim Özeti

| Senaryo | Eski (v0.2 / v0.3 / v0.4) | v0.7.0 | İyileşme |
|---|---:|---:|---:|
| **100K×128D Paralel `batch_search` (HNSW, `w=8`)** | 7,659 QPS *(tek işçi)* | **18,765 QPS (13.64 ms / 256 sorgu)** | **2.45x paralel ölçeklenme** |
| **50K×128D Canlı HTTP REST (`POST /search` / `/batch_search`)** | Yoktu | **3.94 ms (233 QPS) / 1,610 QPS (batch 64)** | **Bearer Auth + Prometheus dahil** |
| **50K×128D Arka Plan `rebuild_index_async()` Okuyucu Max** | 1,048 ms tam bloklama | **P50=0.363 ms · Max=3.99 ms (2,212 okuma)** | **262x düşük kesinti (sıfır duraklama)** |
| **10K×128D Tekil / Yoğun Silme (`auto_compact_async`)** | 181.90 ms *(v0.3)* / 49.89 ms *(sync)* | **0.403 ms tekil / 0.821 ms eşik silme** | **451.4x / 60.8x hızlı** |
| **100K×128D %10 Filtreli Kosinüs (Flat / HNSW+Opt)** | 59.32 ms *(v0.2)* | **2.47 ms / 0.48 ms** | **24.0x / 123.8x hızlı** |
| **100K×128D `DiskIndex` Açılış / Arama (Recall=1.0)** | 2,791 ms inşa / 532.8 ms *(v0.4)* | **125.8 ms açılış / 1.23 ms (684 QPS)** | **22.2x / 4.23x, %95.3 az RAM** |
| **100K×128D `IVF-PQ` Açılış / Arama (Saf NumPy)** | 542 ms / 3.45 ms *(v0.4)* | **198.9 ms / 2.27 ms (412 QPS)** | **2.72x açılış / +%61.6 QPS** |
| **10K Doküman Hibrit Arama (Dense + BM25, RRF)** | Yoktu | **0.936 ms (1,033 QPS), Hit@10=1.000** | **<1 ms Hibrit RAG** |
| **50K×768D Flat Öklid (P50) & Kosinüs RAM** | 116.3 ms / +297.4 MB *(v0.2)* | **7.99 ms / 146.9 MB** | **14.6x hızlı / -%50.6 RAM** |

### 100K × 128D Kümelemeli Veri: MicroVectorDB v0.5.0 vs FAISS 1.13.2

| Sistem & İndeks | P50 (ms) | QPS | Recall@10 | RAM |
|---|---:|---:|---:|---:|
| MicroVectorDB Flat Float32 | 5.49 | 157.6 | 1.000 | 49.6 MB |
| FAISS `IndexFlatIP` | 5.42 | 175.1 | 1.000 | 48.8 MB |
| **MicroVectorDB `DiskIndex` (MMap, C=128, np=8)** | **1.23** | **684.1** | **1.000** | **2.35 MB** |
| MicroVectorDB IVF (`C=128, np=8`, saf NumPy) | **0.76** | **1,117.4** | **1.000** | 49.6 MB |
| MicroVectorDB HNSW (`M=16, efS=64`) | **0.28** | **3,463.3** | **0.975** | 49.6 MB |
| MicroVectorDB HNSW (`M=16, efS=128`) | **0.41** | **2,377.3** | **0.995** | 49.6 MB |
| FAISS `IndexHNSWFlat` (`M=16, efS=64`) | 0.20 | 4,823.4 | 0.978 | 48.8 MB |
| MicroVectorDB v0.5 `IVF-PQ` (Precomputed LUT, `M=16`) | **2.27** | **412.2** | **0.270** | **5.62 MB** |
| FAISS `IndexIVFPQ` (`nlist=128, np=8, M=16`, C++ SIMD) | 0.64 | 1,625.4 | 0.258 | 2.29 MB |

### 10.000 Dokümanlık Kurumsal Külliyat: Dense vs BM25 vs Hibrit (128D, HNSW)

| Arama Modu | P50 (ms) | QPS | Hit@10 (Recall@10) |
|---|---:|---:|---:|
| Dense-Only (HNSW) | **0.299** | 3,127.2 | 0.530 |
| BM25-Only (Vektörize NumPy Okapi BM25) | **0.476** | 1,982.0 | 0.500 |
| Sparse-Vector-Only (Ters İndeks) | **0.151** | 7,268.8 | 0.550 |
| **Hibrit (Dense + BM25, RRF)** | **0.936** | **1,033.4** | **1.000 (%100!)** |
| **Hibrit (Dense + BM25, Weighted Min-Max)** | **1.110** | **867.0** | **1.000 (%100!)** |

---

## 📦 Kurulum

```bash
# Temel kurulum (yalnızca numpy):
pip install .

# Opsiyonel HNSW desteği ile:
pip install ".[hnsw]"

# Opsiyonel hızlı orjson metadata yükleyici ile:
pip install ".[fast]"

# Opsiyonel üretim REST sunucusu (FastAPI + Uvicorn + Pydantic):
pip install ".[server]"

# Tüm performans ve sunucu eklentileri (hnswlib + orjson + server):
pip install ".[all]"

# Geliştirme ve benchmark araçları ile:
pip install -e ".[all,dev,bench]"
```

---

## ⚡ Hızlı Başlangıç

### 1. Hibrit Arama (Dense + BM25 + RRF) ve RAG Pipeline

```python
from microvectordb import MicroVectorDB, chunk_text

with MicroVectorDB("./rag_db") as db:
    coll = db.get_or_create_collection(
        name="knowledge_base",
        dimension=4,
        metric="cosine",
        index_type="flat",     # "flat", "ivf", "ivf_pq", "hnsw", "disk", "auto"
        quantization="none",   # "none", "float16", "int8", "binary", "pq"
    )

    # Dokümanları otomatik _doc_id / _parent_id / _chunk_index ile ekle
    coll.add_documents(
        documents=[
            "İstanbul veri merkezi HNSW vektör veritabanı kurulum rehberi.",
            "Sunucu bağlantı hatası ERR-9042 için WAL kurtarma ve checkpoint prosedürü.",
            "Yedekleme sonrası SHA-256 doğrulamalı snapshot geri yükleme adımları.",
        ],
        embeddings=[
            [0.95, 0.10, 0.05, 0.00],
            [0.45, 0.88, 0.05, 0.00],
            [0.10, 0.15, 0.96, 0.00],
        ],
        doc_ids=["ops_manual", "ops_manual", "ops_manual"],
        metadatas=[{"category": "ops", "year": 2026}] * 3,
    )

    # Dense + Okapi BM25 Hibrit Arama (RRF füzyon + tekilleştirme + gruplama)
    hits = coll.hybrid_search(
        query_vector=[0.80, 0.50, 0.05, 0.00],
        query_text="ERR-9042 WAL kurtarma",
        limit=3,
        alpha=0.5,
        beta=0.5,
        fusion="rrf",          # "rrf", "weighted", "z_score"
        deduplicate=True,
        group_by="_doc_id",
        max_per_group=2,
        filter={"year": {"$gte": 2025}},
    )
    for item in hits:
        print(item.id, round(item.score, 4), item.document)

    # Parent-Child Bağlam Penceresi (komşu parçaları otomatik birleştirir)
    ctx = coll.retrieve_with_context(
        query_vector=[0.45, 0.88, 0.05, 0.00],
        query_text="ERR-9042",
        limit=1,
        window=1,
    )
    print("Birleştirilmiş Bağlam:", ctx[0]["context"])
```

### 2. Multi-Tenant İzolasyon, `<1 ms` Tombstone Silme ve SHA-256 Snapshot

```python
from microvectordb import MicroVectorDB

with MicroVectorDB("./prod_db") as db:
    db.get_or_create_collection("saas", dimension=4, metric="cosine")
    acme = db.get_tenant_collection("saas", "acme_corp")
    beta = db.get_tenant_collection("saas", "beta_ltd")

    acme.add(ids=["a1", "a2"], vectors=[[1, 0, 0, 0], [0.9, 0.1, 0, 0]], documents=["ACME gizli", "ACME rapor"])
    beta.add(ids=["b1"], vectors=[[1, 0, 0, 0]], documents=["Beta gizli"])

    # Beta, ACME'nin verisini ne görebilir ne de silebilir:
    assert beta.delete(ids=["a1"]) == 0

    # <1 ms Tombstone silme (WAL korumalı) + isteğe bağlı fiziksel sıkıştırma
    acme.delete(ids=["a2"])
    db.get_collection("saas").compact()

    # SHA-256 doğrulamalı tutarlı yedekleme (snapshot)
    db.create_snapshot("./backups/snap_2026_11_21")

# Bozulma tespitli geri yükleme (tek bit bile bozulsa DataCorruptionError fırlatır)
MicroVectorDB.restore_snapshot("./backups/snap_2026_11_21", "./restored_db")
```

### 3. Asenkron İstemci (`AsyncMicroVectorDB`), Paralel Batch ve Arka Plan Bakım (`v0.6.0`)

```python
import asyncio
from microvectordb import AsyncMicroVectorDB

async def main():
    async with AsyncMicroVectorDB("./async_db", max_workers=8) as db:
        coll = await db.get_or_create_collection(
            "docs", dimension=4, metric="cosine", index_type="hnsw",
            index_params={"auto_compact_async": True},
        )
        await coll.add(ids=["1", "2"], vectors=[[1, 0, 0, 0], [0, 1, 0, 0]], documents=["İstanbul", "Ankara"])

        # Eşzamanlı asyncio.gather() + çok çekirdekli batch_search()
        hits = await asyncio.gather(
            coll.hybrid_search(query_vector=[1, 0, 0, 0], query_text="İstanbul", limit=2),
            coll.batch_search([[1, 0, 0, 0], [0, 1, 0, 0]], limit=2),
        )
        # Okuyucuları bloklamadan arka planda indeks yenileme
        await coll.rebuild_index_async()

asyncio.run(main())
```

### 4. Üretim REST Sunucusu, Docker ve Sıfır-Bağımlılıklı Python SDK (`v0.7.0`)

```bash
# Sunucuyu başlat (veya: docker compose up -d --build):
MVDB_API_KEYS="mvdb_admin_key:admin:read,write,admin" microvectordb-server --host 0.0.0.0 --port 8000
```

```python
from microvectordb import RemoteMicroVectorDB

with RemoteMicroVectorDB("http://127.0.0.1:8000", api_key="mvdb_admin_key") as db:
    coll = db.get_or_create_collection("docs", dimension=4, metric="cosine")
    coll.add(ids=["1"], vectors=[[1, 0, 0, 0]], documents=["İstanbul sunucu kılavuzu"])
    hits = coll.hybrid_search(query_vector=[1, 0, 0, 0], query_text="İstanbul", limit=3)
    print(hits[0].id, round(hits[0].score, 4))
```

### 5. Geriye Dönük Uyumlu ChromaDB Stili API (v0.2.0)

```python
import microvectordb

client = microvectordb.PersistentClient(path="./my_vectordb", use_mmap=False)
collection = client.get_or_create_collection(name="makaleler", metric="cosine")
collection.add(ids=["doc1"], embeddings=[[0.9, 0.1, 0.0]], documents=["Merhaba"], metadatas=[{"tag": "v1"}])
res = collection.query(query_embeddings=[[0.9, 0.1, 0.0]], n_results=1, where={"tag": "v1"})
client.close()
```

---

## 📁 Proje Mimarisi

```text
MicroVectorDB/
├── microvectordb/
│   ├── __init__.py             # Ana paket arayüzü (v0.7.0)
│   ├── client.py               # MicroVectorDB, PersistentClient, EphemeralClient, Snapshot & Multi-Tenant
│   ├── collection.py           # Collection koordinatörü (RWLock, Optimizer, Hybrid/RAG, Tombstone, WAL, BG)
│   ├── types.py                # QueryPlan, QueryResult, SearchResultItem ve özel hatalar
│   ├── metrics/                # inv_norms Kosinüs, sq_norms Öklid, Dot, bloklu GEMM
│   ├── storage/                # VectorBuffer (lazy mmap), .mvdbvec, .items.mvdbcol, ikili WAL, SQLite + Snapshot
│   ├── filtering/              # Tembel ters metadata indeksi ($eq..$nin, $and/$or/$not, nested)
│   ├── quantization/           # Scalar (INT8), Float16, Binary (POPCNT), ProductQuantizer (PQ)
│   ├── indexes/                # IVF, Precomputed IVF-PQ, DiskIndex + Streaming, HNSW, QueryOptimizer
│   ├── search/                 # Vektörize Okapi BM25, SparseVectorIndex, RRF/Weighted Füzyon, RAG & Multi-Tenant
│   ├── concurrency/            # RWLock, BatchQueryExecutor, BackgroundTaskManager, AsyncMicroVectorDB
│   ├── api/                    # FastAPI REST sunucusu, SHA-256 Auth/Tenant, Rate Limiter, Prometheus /metrics, HTTP SDK
│   ├── distance.py             # v0.2.0 uyumluluk cephesi -> metrics/
│   └── ivf_index.py            # v0.2.0 uyumluluk cephesi -> indexes/ivf.py
├── benchmarks/
│   ├── run_benchmarks.py       # Faz 1 çekirdek benchmark paketi
│   ├── run_benchmarks_v04.py   # Faz 2 PQ / IVF-PQ / DiskIndex / Reload / Optimizer benchmarkı
│   ├── run_benchmarks_v05.py   # Faz 3 Sorun #1/#2/#3 + Hibrit/BM25/WAL/Tombstone/Snapshot benchmarkı
│   ├── run_benchmarks_v06.py   # Faz 4 Paralel Batch / Async / Bloklamayan Arka Plan Bakım benchmarkı
│   ├── run_benchmarks_v07.py   # Faz 5 Canlı Uvicorn HTTP REST & Keep-Alive SDK benchmarkı
│   └── results/                # Kayıtlı JSON ve CSV ölçüm çıktıları
├── docs/
│   ├── API_SERVER.md           # REST API, güvenlik, Prometheus, SDK ve Docker dağıtım rehberi
│   └── RELIABILITY.md          # Açık ACID, WAL dayanıklılık ve eşzamanlılık sözleşmesi
├── tests/                      # 86/86 birim, bütünlük, WAL, hibrit/RAG, async, stres ve REST API testi
├── examples/                   # 7 çalışan örnek (quickstart..async..REST sunucu + SDK)
├── Dockerfile                  # Çok aşamalı non-root üretim imajı (HEALTHCHECK dahil)
├── docker-compose.yml          # Kalıcı hacimli ve ortam yapılandırmalı Compose tanımı
├── kernels_experimental/       # Deneysel AVX2 C/SIMD çekirdeği
├── ARCHITECTURE_AUDIT.md       # v0.2.0 derinlemesine mimari ve performans denetimi
├── BENCHMARK_REPORT.md         # Bilimsel benchmark raporu ve rakiplerle kıyas
├── ROADMAP.md                  # 6 fazlı mühendislik yol haritası
└── CHANGELOG.md                # Sürüm geçmişi
```

---

## 🗺️ Dokümantasyon ve Yol Haritası

- [`ARCHITECTURE_AUDIT.md`](ARCHITECTURE_AUDIT.md) — Mevcut mimari analizi, 10 performans darboğazı ve güvenlik denetimi
- [`BENCHMARK_REPORT.md`](BENCHMARK_REPORT.md) — Ölçülmüş QPS, P50/P95, Recall@10, RAM, FAISS, Hibrit RAG, Eşzamanlılık ve HTTP REST kıyasları
- [`docs/API_SERVER.md`](docs/API_SERVER.md) — FastAPI REST sunucusu, API anahtarı/kiracı güvenliği, `/metrics`, Python SDK ve Docker
- [`docs/RELIABILITY.md`](docs/RELIABILITY.md) — Açık ACID garanti matrisi, `wal_sync` modları, çökme kurtarma ve eşzamanlılık sözleşmesi
- [`ROADMAP.md`](ROADMAP.md) — Faz 1 ✅ → Faz 2 ✅ → Faz 3 ✅ → Faz 4 ✅ → Faz 5 ✅ → Faz 6 (1M SIFT/GloVe & Native SIMD)
- [`CHANGELOG.md`](CHANGELOG.md) — Detaylı sürüm notları

---

## 🧪 Test ve Benchmark Çalıştırma

```bash
# Tüm testler (86/86):
python -m unittest discover -s tests -v

# Bilimsel benchmark paketleri (JSON + CSV üretir):
python benchmarks/run_benchmarks.py
python benchmarks/run_benchmarks_v04.py
python benchmarks/run_benchmarks_v05.py
python benchmarks/run_benchmarks_v06.py
python benchmarks/run_benchmarks_v07.py

# Uçtan uca demolar:
python examples/05_hybrid_rag_and_reliability.py
python examples/06_async_and_parallel.py
python examples/07_rest_server_and_sdk.py
```

---

## 👨‍💻 Geliştirici & Lisans

**Ahmet GÜN** · GitHub: [@Ahmet003-cod](https://github.com/Ahmet003-cod)  
Bu proje **[MIT Lisansı](LICENSE)** ile lisanslanmıştır.
