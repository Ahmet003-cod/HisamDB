# MicroVectorDB 🚀

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/Ahmet003-cod/MicroVectorDB/blob/main/CHANGELOG.md)
[![PyPI](https://img.shields.io/pypi/v/microvectordb.svg)](https://pypi.org/project/microvectordb/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Ahmet003-cod/MicroVectorDB/blob/main/LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-brightgreen.svg)](https://www.python.org/)
[![Core Dependencies: NumPy](https://img.shields.io/badge/core%20dependencies-NumPy%20only-blueviolet.svg)](https://github.com/Ahmet003-cod/MicroVectorDB/blob/main/requirements.txt)
[![Tests: 90/90 Passing](https://img.shields.io/badge/tests-90%2F90%20passing-success.svg)](https://github.com/Ahmet003-cod/MicroVectorDB/tree/main/tests)

**MicroVectorDB**, yapay zekâ uygulamaları (RAG, anlamsal arama, sohbet botları, öneri sistemleri) için geliştirilmiş; **yüksek performanslı**, **ultra düşük bellek (RAM) tüketimli**, **hibrit arama (Vektör + Metin)** destekli, **çökme korumalı (WAL)** ve çekirdeğinde **yalnızca `numpy`** gerektiren açık kaynaklı bir Python Vektör Veritabanıdır.

Gömülü Python kütüphanesi olarak tek satırda başlayabilir, `asyncio` ile binlerce eşzamanlı sorgu çalıştırabilir veya yerleşik **FastAPI + Docker REST sunucusu** ile üretim ortamına çıkabilirsiniz.

> **Felsefe:** *High Performance + Low Memory + Fast Retrieval + Reliable Storage + Developer Friendly*

![MicroVectorDB v1.0.0 Katmanlı Sistem Mimarisi](https://raw.githubusercontent.com/Ahmet003-cod/MicroVectorDB/main/docs/assets/architecture_overview.png)

---

## 🧠 Teknik Kavramlar Sözlüğü (Herkes İçin Basit Anlatım)

Vektör veritabanlarına yeni başlıyorsanız aşağıdaki kısa sözlük, bu README'deki tüm grafikleri ve ayarları **hiç ön bilgi gerektirmeden** anlamanızı sağlar:

### 1. Vektör (Embedding) Nedir?
Bilgisayarlar *"kedi"* ile *"yavru kedi"* kelimelerinin anlamca yakın olduğunu doğrudan bilemez. Yapay zekâ modelleri (OpenAI, BGE, MiniLM vb.) her metni, resmi veya belgeyi örneğin **384, 768 veya 1536 sayıdan oluşan bir koordinata** (vektöre) çevirir. Tıpkı bir şehir haritasındaki GPS koordinatları gibi, **anlamı benzer cümlelerin koordinatları birbirine yakın** olur. MicroVectorDB bu milyonlarca koordinat içinde en yakınları milisaniyeler içinde bulur.

### 2. Benzerlik Metrikleri (`Cosine`, `Euclidean`, `Dot Product`) Ne Demek?
İki vektörün ne kadar benzediğini ölçmenin 3 yolu vardır:
- **`cosine` (Kosinüs Benzerliği):** İki okun arasındaki **açıya** bakar (uzunluktan bağımsızdır). Metin ve RAG aramalarında en yaygın standarttır.
- **`euclidean` (Öklid / L2 Mesafesi):** Haritadaki **kuş uçuşu düz çizgi mesafesidir**. Görüntü ve sensör verilerinde sık kullanılır.
- **`dot_product` (İç Çarpım):** Hem yönü hem büyüklüğü birlikte ölçer. Öneri sistemleri ve önceden normalize edilmiş OpenAI/BGE vektörlerinde çok hızlıdır.

### 3. İndeks Türleri (`Flat`, `IVF`, `IVF-PQ`, `HNSW`, `DiskIndex`) Arasındaki Fark Nedir?
1 milyon kitaplık bir kütüphanede aradığınız kitabı nasıl bulursunuz?
- **`flat` (Tam Tarama):** Her rafa tek tek bakar. **%100 kesindir**, 10–50 bin kayda kadar çok hızlıdır; milyonlarca kayıtta yavaşlar.
- **`ivf` (Kümeleme / Inverted File):** Kitapları önce konularına göre örneğin 256 bölüme (kümeye) ayırır. Sorgu gelince tüm kütüphaneyi değil, yalnızca en ilgili 8–16 bölümü tarar.
- **`ivf_pq` (Kümeleme + Product Quantization):** Hem konulara ayırır hem de her vektörü parçalayıp **16–32 kat sıkıştırılmış özet kodlara** dönüştürür. 1 milyon vektör yalnızca ~43 MB RAM'e sığar!
- **`hnsw` (Çok Katlı Graf Ağı):** Tıpkı otoyol → ana cadde → sokak haritası gibi çok katmanlı bir kısayol ağı kurar. **1 milyon vektörde bile 0.60 milisaniyede** (saniyede 1.600+ sorgu) sonuç verir!
- **`disk` (`DiskIndex` + `MMap`):** Vektörleri RAM'e yüklemek yerine küme sırasına göre diskte düzenli tutar ve işletim sisteminin bellek eşleme (`mmap`) teknolojisiyle yalnızca ilgili parçayı okur. **1 milyon vektörde yalnızca 15.5 MB RAM** harcar ve **%100 doğruluk** verir!

### 4. Kuantizasyon / Sıkıştırma (`Float32`, `Float16`, `INT8`, `Binary + Rescore`) Nedir?
Vektörlerdeki her sayı normalde 32 bit (`float32` = 4 bayt) yer kaplar. Tıpkı RAW fotoğrafı JPEG'e çevirmek gibi:
- **`float16`:** Sayıları yarı hassasiyete indirir → **2 kat daha az RAM**, neredeyse %100 doğruluk.
- **`int8` (`ScalarQuantizer`):** Her sayıyı `-128..+127` arası tek bayta sıkıştırır → **4 kat daha az RAM**, **%97–%99 doğruluk**.
- **`binary` + `rescore`:** Her sayıyı yalnızca `0` veya `1` (tek bit!) yapar → **32 kat daha az RAM**! Önce 1 bitlik parmak iziyle en yakın 200 adayı milisaniye altında süzer, ardından bu 200 adayı gerçek `float32` vektörlerle yeniden puanlayarak (`rescore`) doğruluğu (örneğin OpenAI 1536D'de) **%88+'e** çıkarır.

### 5. Hibrit Arama (`Dense + BM25 + Sparse`) ve `RRF` Nedir?
- **Yalnızca Vektör (Dense):** *"Sunucu çöktü"* deyince *"sistem kesintisi"* belgesini harika bulur, ama **`ERR-9042`** gibi kesin bir hata kodunu veya fatura numarasını kaçırabilir.
- **Yalnızca Metin (`Okapi BM25`):** `ERR-9042` kodunu tam bulur, ama eş anlamlı cümleleri anlayamaz.
- **Hibrit Arama + `RRF` (Reciprocal Rank Fusion):** İki motoru aynı anda çalıştırır ve sonuç listelerini **sıralamalarına göre adil biçimde birleştirir** (`1 / (k + sıra)`). Sonuç: Aşağıdaki grafikte göreceğiniz gibi doğruluk **%53'ten %100'e** fırlar!
- **Sparse Vektör:** SPLADE veya BGE-M3 gibi modern yapay zekâ modellerinin ürettiği ağırlıklı kelime vektörleridir (`{kelime_id: ağırlık}`).

### 6. RAG ve Parent-Child Bağlam Penceresi Nedir?
**RAG (Retrieval-Augmented Generation)**, ChatGPT/Claude/Gemini gibi bir LLM'e cevap vermeden önce kendi belgelerinizden ilgili parçaları bulup vermektir. MicroVectorDB uzun PDF/Word dokümanlarını örtüşmeli parçalara (`chunk_text`) böler; arama bir cümleyi bulduğunda `retrieve_with_context(window=1)` ile o parçanın **bir önceki ve bir sonraki komşu parçalarını da** otomatik getirerek LLM'in bağlamı kaybetmesini önler.

### 7. Güvenilirlik: `WAL`, `Tombstone`, `Compaction` ve `SHA-256 Snapshot` Nedir?
- **`WAL` (Write-Ahead Log):** Uçağın kara kutusu gibidir. Bir ekleme veya silme işlemi ana dosyalara işlenmeden önce CRC32 korumalı günlüğe yazılır; elektrik kesilse veya program çökse bile açılışta yarım kalan kayıtlar temizlenip veri kurtarılır.
- **`Tombstone` Silme & `Compaction`:** 100.000 vektörlü bir dosyadan 1 kayıt silmek için tüm dosyayı yeniden yazmak yerine (182 ms), o satıra bellekte *"silindi"* işareti (tombstone) konur (**0.40 ms — 451x daha hızlı!**). Silinenlerin oranı %25'i geçince `Compaction` arka planda boşlukları temizleyip dosyayı sıkıştırır.
- **`SHA-256 Snapshot`:** Veritabanının o anki tutarlı yedeğini alır ve her dosyanın kriptografik parmak izini (`SHA-256`) kaydeder. Geri yüklerken tek bir bit bile bozulmuşsa bunu yakalayıp uyarır.

### 8. Grafiklerdeki Ölçüm Birimleri (`Recall@10`, `P50`, `P95`, `QPS`, `SIMD`) Ne Demek?
- **`Recall@10` / `Hit@10`:** Sistemin bulduğu ilk 10 sonucun, gerçekten en yakın olan 10 sonuçla ne kadar örtüştüğüdür (`1.00` = **%100 kusursuz**, `0.98` = 10 sonuçtan ortalama 9.8'i doğru).
- **`P50 (Medyan)` ve `P95` (ms):** Sorguların yarısının tamamlandığı tipik süre (`P50`) ve en yavaş %5'lik dilimin sınırı (`P95`). **Düşük olan daha iyidir** (`1 ms` = saniyenin binde biri!).
- **`QPS` (Queries Per Second):** Saniyede cevaplanan sorgu sayısıdır. **Yüksek olan daha iyidir.**
- **`SIMD` (`AVX2` / `FMA` / `POPCNT`):** Modern işlemcilerin tek bir komutta 1 sayı yerine **8 sayıyı aynı anda** çarpıp toplayabilmesi (`AVX2`) veya 64 bit içindeki `1` sayısını tek saat çevriminde sayabilmesidir (`POPCNT`). MicroVectorDB'deki opsiyonel **Rust Native SIMD** çekirdeği tam olarak bu donanım gücünü kullanır.

---

## 🧭 Hangi Ayarı Seçmeliyim? (Hızlı Karar Rehberi)

| Kullanım Senaryosu | Önerilen Ayar | Beklenen Performans (Ölçülmüş) |
|---|---|---|
| **Küçük/Orta Proje (< 50K vektör)** | `index_type="flat"` *(varsayılan)* | `%100` kesin doğruluk, `0.2–3.5 ms` |
| **En Yüksek Hız & 100K–1M+ Vektör** | `index_type="hnsw"` (`pip install ".[hnsw]"`) | `1M` vektörde **`0.60 ms` (1,640 QPS)**, paralel batch **`18,765 QPS`** |
| **Çok Düşük RAM (1M Vektör veya 1536D OpenAI)** | `index_type="disk", use_mmap=True` | `1M` vektör → **`15.5 MB RAM`**, `25K×1536D` → **`0.95 MB RAM`**, **`%100` Recall!** |
| **Aşırı Sıkıştırılmış RAM İçi Arama** | `index_type="ivf_pq", keep_raw_vectors=False` | `1M` vektör → **`42.7 MB RAM` (-%91)**, **`2.36 ms`** *(Rust SIMD ile)* |
| **Kurumsal RAG / Doküman & Hata Kodu Arama** | `index_type="hnsw"` + `hybrid_search(fusion="rrf")` | **`0.94 ms` (1,033 QPS)**, **`Hit@10 = %100`** (Dense-Only `%53` idi!) |
| **Çok Kullanıcılı SaaS / Mikroservis** | `microvectordb-server` + `RemoteMicroVectorDB` | `POST /search` **`3.94 ms`**, `/batch_search(64)` **`1,610 QPS`**, SHA-256 Tenant İzolasyonu |

---

## 📊 Doğrulanmış Performans Sonuçları ve Grafikler (`v1.0.0`)

*Tüm grafikler ve tablolar kanıtlanmamış tahmini rakam içermez; doğrudan bu depodaki [`benchmarks/results/`](https://github.com/Ahmet003-cod/MicroVectorDB/tree/main/benchmarks/results) JSON kayıtlarından üretilmiştir. Donanım: Windows x86-64, 12 mantıksal Intel çekirdeği (AVX2/FMA/POPCNT), 15.73 GB RAM, Python 3.11.9, NumPy 2.0.2 (OpenBLAS), `rustc 1.96.0`, FAISS 1.13.2. Tam bilimsel rapor: [`BENCHMARK_REPORT.md`](https://github.com/Ahmet003-cod/MicroVectorDB/blob/main/BENCHMARK_REPORT.md).*

### 1. `v0.2.0 → v1.0.0` Evrim Özeti (Gerçek Hızlanma Çarpanları)

![MicroVectorDB Evrimi v0.2.0 - v1.0.0 Hızlanma Grafiği](https://raw.githubusercontent.com/Ahmet003-cod/MicroVectorDB/main/docs/assets/evolution_speedup.png)

| Optimizasyon Alanı | Eski Durum | MicroVectorDB `v1.0.0` | Kazanım |
|---|---|---|---|
| **Tekil Kalıcı Silme (10K / 100K)** | `181.90 ms` *(v0.3)* | **`0.403 ms` / `0.382 ms`** *(Tombstone+WAL)* | **451.4x daha hızlı** |
| **Arka Plan İndeks İnşası Okuyucu Kesintisi** | `1,048 ms` tam bloklama | **P50=`0.363 ms` · Max=`3.99 ms`** *(2,212 canlı okuma)* | **262.6x düşük kesinti** |
| **100K %10 Filtreli Kosinüs Arama** | `59.32 ms` *(v0.2)* | **`0.479 ms`** *(HNSW + QueryOptimizer)* | **123.8x daha hızlı** |
| **Yoğun Silme Kuyruk Gecikmesi** | `49.89 ms` *(senkron)* | **`0.821 ms`** *(`auto_compact_async=True`)* | **60.8x düşük gecikme** |
| **50K × 768D Flat Öklid & Kosinüs RAM** | `116.3 ms` / `+297 MB` | **`7.99 ms` / `146.9 MB`** | **14.6x hızlı / -%50.6 RAM** |
| **100K Paralel Toplu Arama (HNSW, `w=8`)** | `7,659 QPS` *(1 işçi)* | **`18,765 QPS`** *(13.64 ms / 256 sorgu)* | **2.45x çok çekirdekli hız** |

---

### 2. Büyük Ölçek Performansı: `10.000 → 1.000.000` Vektör (128D vs C++ FAISS)

![10K'dan 1 Milyon Vektöre Ölçek ve RAM Karşılaştırması](https://raw.githubusercontent.com/Ahmet003-cod/MicroVectorDB/main/docs/assets/scale_1m_benchmark.png)

- **1.000.000 Vektörde `DiskIndex (MMap)`:** Ham veri `488.3 MB` iken yalnızca **`15.52 MB RAM` (-%96.8 tasarruf)** kullanarak **`9.56 ms`** gecikmeyle **`%100.0` (`1.000`) Recall@10** sağlar!
- **1.000.000 Vektörde `HNSW`:** **`0.605 ms` (`1,640 QPS`)** ile milisaniye-altı arama yapar ve `FAISS IndexHNSWFlat`'ten daha yüksek Recall@10 (`0.744` vs `0.736`) elde eder.
- **500.000 Vektörde `IVF-PQ (Rust SIMD)`:** **`1.548 ms` (`638 QPS`, Recall `0.192`)** ile C++ `FAISS IndexIVFPQ`'yu (`1.667 ms`, `604 QPS`, Recall `0.186`) hem hız hem doğrulukta geride bırakır!

---

### 3. Yüksek Boyutlu LLM Embedding'leri (`384D MiniLM`, `768D BGE/E5`, `1536D OpenAI`)

![Yüksek Boyutlu LLM Embedding Performansı: RAM, Gecikme ve Doğruluk](https://raw.githubusercontent.com/Ahmet003-cod/MicroVectorDB/main/docs/assets/memory_vs_recall.png)

- **OpenAI `1536D` (`25K` vektör, ham `146.5 MB`):**
  - **`DiskIndex (MMap)`:** Yalnızca **`0.95 MB RAM` (-%99.4 tasarruf!)** ile **`1.50 ms` (`526 QPS`)** ve **`%100.0` Recall@10** — RAM'deki Flat Float32'den (`6.30 ms`) **4.2x daha hızlı!**
  - **`Binary + Rescore (20x)`:** İkili indeks yalnızca **`4.58 MB RAM`** kaplar; **`0.98 ms` (`933 QPS`)** sürede **`%88.4` Recall@10** verir.
  - **`INT8 (Rust SIMD)`:** **`36.8 MB RAM`** (4x tasarruf) ile **`%98.6` Recall@10** (`14.10 ms`, saf NumPy'dan `1.85x` hızlı).
  - **`HNSW`:** **`0.405 ms` (`2,193 QPS`)** ve **`%99.2` Recall@10**.

---

### 4. Opsiyonel Rust Native SIMD (`AVX2 + FMA3 + POPCNT`) vs Saf NumPy vs C++ FAISS

![Rust Native SIMD vs Saf NumPy vs C++ FAISS Karşılaştırması](https://raw.githubusercontent.com/Ahmet003-cod/MicroVectorDB/main/docs/assets/simd_vs_numpy_vs_faiss.png)

Çekirdekte hiçbir zorunlu C/Rust derleyicisi olmadan %100 saf NumPy ile çalışan MicroVectorDB, isteyen kullanıcılar için tek komutla derlenen sıfır-bağımlılıklı Rust SIMD çekirdeği (`kernels_experimental/mvdb_simd.rs`) sunar:
- **Flat PQ (`100K × 128D`):** `9.41 ms → 1.95 ms` (**4.83x hızlanma**, P95 `2.18 ms` < FAISS `2.22 ms`).
- **Binary 64-bit POPCNT (`100K × 768D`):** `13.10 ms → 3.02 ms` (**4.33x hızlanma**, yalnızca `9.16 MB RAM`).
- **IVF-PQ (`100K × 128D`):** `1.18 ms → 0.62 ms` (**2.04x QPS**, P95 `0.77 ms` < FAISS `0.94 ms` ve **+%49.5 daha yüksek Recall@10**!).
- **INT8 Scalar (`128D / 768D / 1536D`):** **1.71x – 1.91x hızlanma** (`Recall@10 = %96.8 – %98.6`).

---

### 5. Neden Hibrit Arama? — Kurumsal RAG Doğruluk & Hız Kıyaslaması

![Dense vs BM25 vs Hibrit RAG Doğruluk ve Hız Grafiği](https://raw.githubusercontent.com/Ahmet003-cod/MicroVectorDB/main/docs/assets/hybrid_rag_accuracy.png)

10.000 kurumsal dokümanda (hem anlamsal sorular hem kesin `ERR-XXXX` hata kodu / terim sorguları):
- Yalnızca vektör (`Dense-Only`) kesin kodları kaçırarak **%53 Hit@10**, yalnızca metin (`BM25-Only`) eş anlamlı ifadeleri kaçırarak **%50 Hit@10** alırken;
- **MicroVectorDB Hibrit Arama (`Dense + BM25, RRF`)** yalnızca **`0.94 ms` (`1,033 QPS`)** sürede **`%100` (`1.000`) Hit@10** doğruluğuna ulaşır!

---

## 📦 Kurulum

```bash
# 1. PyPI'dan temel kurulum (yalnızca numpy — sıfır ağır bağımlılık!):
pip install microvectordb

# 2. Opsiyonel HNSW graf indeksi desteği ile:
pip install "microvectordb[hnsw]"

# 3. Opsiyonel hızlı orjson metadata yükleyici ile:
pip install "microvectordb[fast]"

# 4. Opsiyonel üretim FastAPI REST sunucusu (FastAPI + Uvicorn + Pydantic):
pip install "microvectordb[server]"

# 5. Tüm performans ve sunucu eklentileri (hnswlib + orjson + server):
pip install "microvectordb[all]"

# 6. Kaynak koddan geliştirme ve benchmark araçlarıyla kurulum:
pip install -e ".[all,dev,bench]"
```

### Opsiyonel: Rust Native SIMD (`AVX2 + FMA3 + POPCNT`) Hızlandırması (İsteğe Bağlı)

Bilgisayarınızda Rust (`rustc`) varsa tek komutla yerel SIMD hızlandırıcısını derleyebilirsiniz — MicroVectorDB bunu açılışta otomatik algılar (derlemezseniz hiçbir hata vermeden saf NumPy ile çalışır):

```bash
# Windows (PowerShell):
rustc --crate-type cdylib -C opt-level=3 -C target-cpu=native kernels_experimental/mvdb_simd.rs -o kernels_experimental/mvdb_simd.dll

# Linux:
rustc --crate-type cdylib -C opt-level=3 -C target-cpu=native kernels_experimental/mvdb_simd.rs -o kernels_experimental/libmvdb_simd.so

# macOS:
rustc --crate-type cdylib -C opt-level=3 -C target-cpu=native kernels_experimental/mvdb_simd.rs -o kernels_experimental/libmvdb_simd.dylib
```

---

## ⚡ Hızlı Başlangıç

### 1. Hibrit Arama (Dense + BM25 + RRF) ve RAG Bağlam Penceresi

```python
from microvectordb import MicroVectorDB

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

    # Dense + Türkçe Okapi BM25 Hibrit Arama (RRF füzyon + tekilleştirme + filtre)
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

    # Parent-Child Bağlam Penceresi (önceki/sonraki komşu parçaları da getirir)
    ctx = coll.retrieve_with_context(
        query_vector=[0.45, 0.88, 0.05, 0.00],
        query_text="ERR-9042",
        limit=1,
        window=1,
    )
    print("Birleştirilmiş Bağlam:", ctx[0]["context"])
```

### 2. Ultra Düşük RAM (`DiskIndex` / `MMap`), Multi-Tenant ve SHA-256 Snapshot

```python
from microvectordb import MicroVectorDB

with MicroVectorDB("./prod_db", use_mmap=True) as db:
    # 1M vektörde ~15.5 MB RAM ile %100 Recall:
    db.get_or_create_collection("docs_large", dimension=1536, metric="cosine", index_type="disk", use_mmap=True)

    # Katı Multi-Tenant İzolasyonu (SaaS uygulamaları için):
    db.get_or_create_collection("saas", dimension=4, metric="cosine")
    acme = db.get_tenant_collection("saas", "acme_corp")
    beta = db.get_tenant_collection("saas", "beta_ltd")

    acme.add(ids=["a1", "a2"], vectors=[[1, 0, 0, 0], [0.9, 0.1, 0, 0]], documents=["ACME gizli", "ACME rapor"])
    beta.add(ids=["b1"], vectors=[[1, 0, 0, 0]], documents=["Beta gizli"])

    # Beta, ACME'nin verisini ne görebilir ne de silebilir:
    assert beta.delete(ids=["a1"]) == 0

    # <1 ms Tombstone silme (WAL korumalı) + SHA-256 doğrulamalı yedekleme:
    acme.delete(ids=["a2"])
    db.create_snapshot("./backups/snap_v1")

# Tek bir bit bile bozulsa DataCorruptionError fırlatarak güvenli geri yükleme:
MicroVectorDB.restore_snapshot("./backups/snap_v1", "./restored_db")
```

### 3. Asenkron İstemci (`AsyncMicroVectorDB`), Paralel Batch & Bloklamayan Arka Plan Bakım

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
        # Okuyucuları bloklamadan arka planda indeks yenileme (Snapshot-and-Swap)
        await coll.rebuild_index_async()

asyncio.run(main())
```

### 4. Üretim FastAPI REST Sunucusu, Docker ve Sıfır-Bağımlılıklı Python SDK

```bash
# Sunucuyu başlat (veya: docker compose up -d --build):
MVDB_API_KEYS="mvdb_admin_key:admin:read,write,admin" microvectordb-server --host 0.0.0.0 --port 8000
```

```python
from microvectordb import RemoteMicroVectorDB

# Harici requests/httpx gerektirmez — standart kütüphane Keep-Alive HTTP/1.1 SDK!
with RemoteMicroVectorDB("http://127.0.0.1:8000", api_key="mvdb_admin_key") as db:
    coll = db.get_or_create_collection("docs", dimension=4, metric="cosine")
    coll.add(ids=["1"], vectors=[[1, 0, 0, 0]], documents=["İstanbul sunucu kılavuzu"])
    hits = coll.hybrid_search(query_vector=[1, 0, 0, 0], query_text="İstanbul", limit=3)
    print(hits[0].id, round(hits[0].score, 4))
```

### 5. Geriye Dönük Uyumlu ChromaDB Stili API (`v0.2.0`)

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
│   ├── __init__.py             # Ana paket arayüzü (v1.0.0)
│   ├── client.py               # MicroVectorDB, PersistentClient, EphemeralClient, Snapshot & Multi-Tenant
│   ├── collection.py           # Collection koordinatörü (RWLock, Optimizer, Hybrid/RAG, Tombstone, WAL, BG)
│   ├── types.py                # QueryPlan, QueryResult, SearchResultItem ve özel hatalar
│   ├── metrics/                # inv_norms Kosinüs, sq_norms Öklid, Dot + opsiyonel native_simd yükleyici
│   ├── storage/                # VectorBuffer (lazy mmap), .mvdbvec, .items.mvdbcol, ikili WAL, SQLite + Snapshot
│   ├── filtering/              # Tembel ters metadata indeksi ($eq..$nin, $and/$or/$not, nested)
│   ├── quantization/           # Scalar (INT8), Float16, Binary (POPCNT), ProductQuantizer (PQ)
│   ├── indexes/                # IVF, Precomputed IVF-PQ, DiskIndex + Streaming, HNSW, QueryOptimizer
│   ├── search/                 # Vektörize Okapi BM25, SparseVectorIndex, RRF/Weighted Füzyon, RAG & Multi-Tenant
│   ├── concurrency/            # RWLock, BatchQueryExecutor, BackgroundTaskManager, AsyncMicroVectorDB
│   ├── api/                    # FastAPI REST sunucusu, SHA-256 Auth/Tenant, Rate Limiter, Prometheus /metrics, HTTP SDK
│   ├── distance.py             # v0.2.0 uyumluluk cephesi -> metrics/
│   └── ivf_index.py            # v0.2.0 uyumluluk cephesi -> indexes/ivf.py
├── kernels_experimental/
│   ├── mvdb_simd.rs            # Opsiyonel sıfır-bağımlılıklı Rust AVX2+FMA3+POPCNT çekirdeği
│   ├── avx2_dot_product.c      # Referans C/AVX2 SIMD örneği
│   └── README.md               # Tek komutla derleme & Python kullanımı
├── benchmarks/
│   ├── run_benchmarks*.py      # Faz 1..6 (10K → 1M, 384D–1536D, Hybrid, Async, HTTP REST, Native SIMD)
│   ├── generate_charts.py      # Gerçek JSON sonuçlarından 6 görsel PNG üreticisi
│   └── results/                # Ham JSON ve CSV ölçüm kayıtları
├── docs/
│   ├── assets/                 # README ve rapor görselleri (6 yüksek çözünürlüklü PNG)
│   ├── API_SERVER.md           # REST API, güvenlik, Prometheus, SDK ve Docker dağıtım rehberi
│   ├── RELIABILITY.md          # Açık ACID garanti matrisi, WAL ve eşzamanlılık sözleşmesi
│   └── ARCHITECTURE_SCALE_RESEARCH.md  # Native SIMD, GPU/CUDA ve v2.x Dağıtık Mimari araştırma raporu
├── tests/                      # 90/90 birim, bütünlük, WAL, hibrit/RAG, async, stres, REST ve SIMD parite testi
├── examples/                   # 7 çalışan örnek (quickstart..async..REST sunucu + SDK)
├── Dockerfile                  # Çok aşamalı non-root üretim imajı (HEALTHCHECK dahil)
├── docker-compose.yml          # Kalıcı hacimli ve ortam yapılandırmalı Compose tanımı
├── ARCHITECTURE_AUDIT.md       # v0.2.0 derinlemesine mimari ve performans denetimi
├── BENCHMARK_REPORT.md         # Tam bilimsel benchmark raporu (Faz 1..6)
├── ROADMAP.md                  # 6/6 fazı tamamlanmış mühendislik yol haritası
└── CHANGELOG.md                # Sürüm geçmişi
```

---

## 🗺️ Dokümantasyon

- [`BENCHMARK_REPORT.md`](https://github.com/Ahmet003-cod/MicroVectorDB/blob/main/BENCHMARK_REPORT.md) — Tüm fazların tam ölçüm tabloları (`10K → 1M`, `128D–1536D`, SIMD, Hibrit RAG, Async, HTTP REST)
- [`docs/ARCHITECTURE_SCALE_RESEARCH.md`](https://github.com/Ahmet003-cod/MicroVectorDB/blob/main/docs/ARCHITECTURE_SCALE_RESEARCH.md) — Native Rust SIMD analizi, GPU/CUDA fizibilitesi ve `v2.x` dağıtık sharding/Raft/gRPC taslağı
- [`docs/API_SERVER.md`](https://github.com/Ahmet003-cod/MicroVectorDB/blob/main/docs/API_SERVER.md) — FastAPI REST sunucusu, SHA-256 Auth/Tenant güvenliği, Prometheus `/metrics`, Python SDK ve Docker
- [`docs/RELIABILITY.md`](https://github.com/Ahmet003-cod/MicroVectorDB/blob/main/docs/RELIABILITY.md) — Açık ACID garanti matrisi, `wal_sync` modları, çökme kurtarma ve eşzamanlılık sözleşmesi
- [`ARCHITECTURE_AUDIT.md`](https://github.com/Ahmet003-cod/MicroVectorDB/blob/main/ARCHITECTURE_AUDIT.md) — Projenin `v0.2.0` başlangıç denetimi ve çözülen darboğazların kökeni
- [`ROADMAP.md`](https://github.com/Ahmet003-cod/MicroVectorDB/blob/main/ROADMAP.md) & [`CHANGELOG.md`](https://github.com/Ahmet003-cod/MicroVectorDB/blob/main/CHANGELOG.md) — Tamamlanan 6 fazın yol haritası ve sürüm notları

---

## 🧪 Test, Benchmark ve Grafik Üretimi

```bash
# Tüm testler (90/90):
python -m unittest discover -s tests -v

# Faz 6 1M / 1536D / Native SIMD benchmark paketi:
python benchmarks/run_benchmarks_v10.py

# Gerçek JSON sonuçlarından tüm grafikleri yeniden üret:
python benchmarks/generate_charts.py

# Uçtan uca demolar:
python examples/05_hybrid_rag_and_reliability.py
python examples/06_async_and_parallel.py
python examples/07_rest_server_and_sdk.py
```

---

## 👨‍💻 Geliştirici & Lisans

**Ahmet GÜN** · GitHub: [@Ahmet003-cod](https://github.com/Ahmet003-cod)  
Bu proje **[MIT Lisansı](https://github.com/Ahmet003-cod/MicroVectorDB/blob/main/LICENSE)** ile lisanslanmıştır.
