# MicroVectorDB 🚀

[![Version](https://img.shields.io/badge/version-0.3.0-blue.svg)](CHANGELOG.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-brightgreen.svg)](https://www.python.org/)
[![Core Dependencies: NumPy](https://img.shields.io/badge/core%20dependencies-NumPy%20only-blueviolet.svg)](requirements.txt)

**MicroVectorDB**, yüksek performanslı, düşük bellek tüketimli, güvenilir kalıcılığa sahip, yalnızca `numpy` temel bağımlılığı ile çalışan açık kaynaklı bir Python Vektör Veritabanıdır.

> **Felsefe:** *High Performance + Low Memory + Fast Retrieval + Reliable Storage + Developer Friendly*

---

## ✨ Neler Sunar? (v0.3.0 — Doğrulanmış Özellikler)

- **Sıfır Ağır Çekirdek Bağımlılığı:** Yalnızca `numpy>=1.20` (NumPy 1.x ve 2.x tam uyumlu). İsteğe bağlı `hnswlib` ile graf tabanlı HNSW desteği (`pip install ".[hnsw]"`).
- **Yüksek Performanslı Arama ve İndeks Motoru:**
  - **Optimize Flat (Exact):** Kosinüs için `(N,)` `inv_norms` (%50 daha az RAM), Öklid için önbellekli `sq_norms` (~15x hızlanma) ve OpenBLAS SGEMM/SGEMV.
  - **Optimize `IVFIndex` (Saf NumPy):** K-Means++, bitişik küme blokları ve tüm metrikler (`cosine`, `dot_product`, `euclidean`).
  - **`HNSWIndex` (Opsiyonel `hnswlib`):** 100K×128D'de **3,463 QPS (P50 = 0.278 ms, Recall@10 = 0.975)**.
  - **Otomatik İndeks Modu (`index_type="auto"`):** Küçük veride exact Flat, büyük veride otomatik HNSW/IVF.
- **Entegre Kuantizasyon & Düşük Bellek Modları:**
  - `quantization="float16"` (~2x RAM tasarrufu, Recall@10 ≈ 0.999)
  - `quantization="int8"` (Taşmasız ölçek-ağırlıklı `ScalarQuantizer`, `keep_raw_vectors=False` ile ~4x RAM tasarrufu, Recall@10 ≈ 0.965–0.980)
  - `quantization="binary"` (1-bit paketlenmiş Hamming + `uint64` SIMD `bitwise_count`, 32x sıkıştırma, isteğe bağlı float32 yeniden skorlama)
  - Gerçek `use_mmap=True` bellek-eşlemeli disk okuma (**%98+ vektör RAM tasarrufu**).
- **Ters İndeksli Gelişmiş Metadata Filtreleme:**
  - `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$exists`, `$contains`, `$and`, `$or`, `$not` ve iç içe (`"author.country"`) filtreler.
  - 100K vektörde %10 seçicilikli filtreli sorgu: **2.47 ms (383.7 QPS)** — v0.2.0'a göre **24x daha hızlı**.
- **Güvenilir Depolama & Eşzamanlılık:**
  - Sürümlü 64-byte hizalı `.mvdbvec` binary formatı, CRC32 doğrulama, atomik yazma (`fsync` + `os.replace`), $O(M)$ artımlı ekleme, çökme kurtarma, disk hatasında bellek geri alması (rollback).
  - SQLite WAL modu, path-traversal koruması, SHA-256 çakışmasız tablo adlandırması, v0.2.0 `.npy` otomatik göçü.
  - Yazıcı-öncelikli `RWLock` ile çok iş parçacıklı güvenli eşzamanlı okuma/yazma.
- **Geliştirici Dostu & Geriye Dönük Uyumlu API:**
  - Yeni `MicroVectorDB` / `collection.search()` / `score_threshold` / `stats()` / `checkpoint()` + tüm v0.2.0 `PersistentClient` / `EphemeralClient` / `collection.query()` imzaları %100 korunur.

---

## 📊 Doğrulanmış Performans Sonuçları (v0.3.0)

*Ölçüm ortamı: Windows x86-64, 12 mantıksal Intel çekirdeği (AVX2/FMA), 15.73 GB RAM, Python 3.11.9, NumPy 2.0.2 (OpenBLAS 0.3.27), FAISS 1.13.2. Tüm ham veriler ve metodoloji için [`BENCHMARK_REPORT.md`](BENCHMARK_REPORT.md) dosyasına bakın.*

### v0.2.0 → v0.3.0 İyileştirme Özeti

| Senaryo | v0.2.0 | v0.3.0 | İyileşme |
|---|---:|---:|---:|
| **100K×128D Filtreli Kosinüs (`eq` %10)** | 59.32 ms (15.9 QPS) | **2.47 ms (383.7 QPS)** | **24.0x hızlı** |
| **50K×768D Flat Öklid (P50)** | 116.30 ms (8.6 QPS) | **7.99 ms (110.8 QPS)** | **14.6x hızlı** |
| **50K×768D Flat Kosinüs RAM** | +297.4 MB | **146.9 MB** | **%50.6 az RAM** |
| **100K×128D `ScalarQuantizer` (Recall@10 / P50)** | 0.630 / 35.08 ms | **0.980 / 16.89 ms** | **+35 puan Recall & 2.1x hızlı** |
| **100K×128D `IVFIndex` (Gaussian P50)** | 2.10 ms (429.8 QPS) | **0.42 ms (1,881.8 QPS)** | **5.0x hızlı** |
| **10K×128D Artımlı Kalıcı Yazma (20×500)** | 1.95 s (5,134 v/s) | **0.64 s (15,628 v/s)** | **3.0x hızlı** |
| **10K×128D Tek Toplu Ekleme (`04_benchmark`)** | 244,639 v/s | **1,110,838 v/s** | **4.5x hızlı** |

### 100K × 128D Kümelemeli Veri: MicroVectorDB v0.3.0 vs FAISS 1.13.2

| Sistem & İndeks | P50 (ms) | QPS | Recall@10 | Vektör RAM |
|---|---:|---:|---:|---:|
| MicroVectorDB Flat Float32 | 5.49 | 157.6 | 1.000 | 49.6 MB |
| FAISS `IndexFlatIP` | 5.42 | 175.1 | 1.000 | 48.8 MB |
| MicroVectorDB IVF (`C=128, np=8`, saf NumPy) | **0.76** | **1,117.4** | **1.000** | 49.6 MB |
| MicroVectorDB HNSW (`M=16, efS=64`) | **0.28** | **3,463.3** | **0.975** | 49.6 MB |
| MicroVectorDB HNSW (`M=16, efS=128`) | **0.41** | **2,377.3** | **0.995** | 49.6 MB |
| FAISS `IndexHNSWFlat` (`M=16, efS=64`) | 0.20 | 4,823.4 | 0.978 | 48.8 MB |

---

## 📦 Kurulum

```bash
# Temel kurulum (yalnızca numpy):
pip install .

# Opsiyonel HNSW desteği ile:
pip install ".[hnsw]"

# Geliştirme ve benchmark araçları ile:
pip install -e ".[hnsw,dev,bench]"
```

---

## ⚡ Hızlı Başlangıç

### Yeni Birleşik API (`MicroVectorDB` + `search()`)

```python
from microvectordb import MicroVectorDB

with MicroVectorDB("./database") as db:
    collection = db.get_or_create_collection(
        name="knowledge_base",
        dimension=4,
        metric="cosine",
        index_type="flat",     # "flat", "ivf", "hnsw", "auto"
        quantization="none",   # "none", "float16", "int8", "binary"
    )

    collection.add(
        ids=["1", "2", "3"],
        vectors=[
            [0.95, 0.10, 0.05, 0.02],
            [0.10, 0.92, 0.20, 0.05],
            [0.85, 0.15, 0.10, 0.01],
        ],
        documents=["RAG mimarisi", "Veritabanı güvenliği", "AI Agent hafızası"],
        metadata=[
            {"category": "AI", "year": 2025, "tags": ["rag", "llm"]},
            {"category": "Security", "year": 2024, "tags": ["auth"]},
            {"category": "AI", "year": 2026, "tags": ["agent", "rag"]},
        ],
    )

    results = collection.search(
        query_vector=[0.90, 0.12, 0.06, 0.02],
        limit=5,
        filter={
            "$and": [
                {"category": "AI"},
                {"year": {"$gte": 2025}},
                {"tags": {"$contains": "rag"}},
            ]
        },
        score_threshold=0.70,
    )

    for item in results:
        print(item.id, round(item.score, 4), item.document)
```

### Geriye Dönük Uyumlu ChromaDB Stili API (v0.2.0)

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
│   ├── __init__.py             # Ana paket arayüzü (v0.3.0)
│   ├── client.py               # MicroVectorDB, PersistentClient, EphemeralClient
│   ├── collection.py           # Collection koordinatörü (RWLock, indeks/kuantizasyon, delta kalıcılık)
│   ├── types.py                # Veri modelleri ve özel hata sınıfları
│   ├── metrics/                # inv_norms Kosinüs, sq_norms Öklid, Dot, bloklu GEMM
│   ├── storage/                # VectorBuffer, sürümlü .mvdbvec (CRC32+mmap), güvenli SQLite WAL
│   ├── filtering/              # Ters metadata indeksi ($eq..$nin, $and/$or/$not, nested)
│   ├── quantization/           # Taşmasız INT8 ScalarQuantizer, Float16, Binary (SIMD POPCNT)
│   ├── indexes/                # K-Means++ bitişik IVFIndex, opsiyonel HNSWIndex
│   ├── concurrency/            # Yazıcı-öncelikli RWLock
│   ├── distance.py             # v0.2.0 uyumluluk cephesi -> metrics/
│   └── ivf_index.py            # v0.2.0 uyumluluk cephesi -> indexes/ivf.py
├── benchmarks/
│   ├── run_benchmarks.py       # Otomatik JSON/CSV benchmark paketi
│   └── results/                # Kayıtlı ölçüm çıktıları
├── tests/                      # 52/52 birim, bütünlük, kurtarma ve eşzamanlılık testi
├── examples/                   # Çalışan hızlı başlangıç, filtre, RAG ve hız örnekleri
├── kernels_experimental/       # Deneysel AVX2 C/SIMD çekirdeği
├── ARCHITECTURE_AUDIT.md       # v0.2.0 derinlemesine mimari ve performans denetimi
├── BENCHMARK_REPORT.md         # Bilimsel benchmark raporu ve rakiplerle kıyas
├── ROADMAP.md                  # 6 fazlı mühendislik yol haritası
└── CHANGELOG.md                # Sürüm geçmişi
```

---

## 🗺️ Dokümantasyon ve Yol Haritası

- [`ARCHITECTURE_AUDIT.md`](ARCHITECTURE_AUDIT.md) — Mevcut mimari analizi, 10 performans darboğazı ve güvenlik denetimi
- [`BENCHMARK_REPORT.md`](BENCHMARK_REPORT.md) — Ölçülmüş QPS, P50/P95, Recall@10, RAM ve FAISS karşılaştırması
- [`ROADMAP.md`](ROADMAP.md) — Faz 1 (Tamamlandı) → Faz 2 (PQ, Query Optimizer, Disk İndeks) → Faz 3 (Sparse/Hybrid/RRF/RAG) → Faz 4 (WAL/Snapshot) → Faz 5 (FastAPI Sunucu) → Faz 6 (1M SIFT/GloVe & SIMD)
- [`CHANGELOG.md`](CHANGELOG.md) — Detaylı sürüm notları

---

## 🧪 Test ve Benchmark Çalıştırma

```bash
# Tüm testler (52/52):
python -m unittest discover -s tests -v

# Bilimsel benchmark paketi (JSON + CSV üretir):
python benchmarks/run_benchmarks.py
```

---

## 👨‍💻 Geliştirici & Lisans

**Huneyn Kaya** · GitHub: [@Ahmet003-cod](https://github.com/Ahmet003-cod)  
Bu proje **[MIT Lisansı](LICENSE)** ile lisanslanmıştır.
