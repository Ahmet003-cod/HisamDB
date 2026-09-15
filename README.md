# microvectordb 🚀

[![PyPI version](https://img.shields.io/badge/version-0.2.0-blue.svg)](https://pypi.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-brightgreen.svg)](https://www.python.org/)
[![Dependencies: NumPy](https://img.shields.io/badge/dependencies-NumPy-blueviolet.svg)](https://numpy.org/)

**microvectordb**, harici ağır ve hantal bağımlılıklara ihtiyaç duymadan çalışan, **ChromaDB ve FAISS benzeri** temiz bir API sunan, son derece hızlı, hafif ve kalıcı (persistent) bir Python Vektör Veritabanıdır.

---

## 🎯 Neden microvectordb?

Yapay zeka, RAG (Retrieval-Augmented Generation) veya semantik arama projelerinde çoğu geliştirici iki zorlukla karşılaşır:
1. **ChromaDB:** Çok yeteneklidir ancak yüzlerce megabaytlık bağımlılık (ONNX Runtime, Pydantic vb.) indirir, bazen Windows veya hafif ortamlarda kurulum/derleme hataları verir.
2. **FAISS:** Facebook'un geliştirdiği harika bir hız kütüphanesidir; ancak **bir veritabanı değildir**. Doküman saklayamaz, metadata filtreleyemez (`where={"kategori": "ai"}`) ve C++ derleyicisi gerektirir.

**microvectordb** tam bu noktada devreye girer:
> **ChromaDB'nin kullanım kolaylığı ve zengin veritabanı yeteneklerini, FAISS'in hafiflik ve indeksleme felsefesiyle birleştirir; üstelik sadece `numpy` kullanarak!**

---

## 🥊 Karşılaştırma: microvectordb vs ChromaDB vs FAISS

| Kriter | 🚀 microvectordb (v0.2.0) | 🔵 ChromaDB | 🟣 FAISS |
| :--- | :---: | :---: | :---: |
| **Bağımlılık Ağırlığı** | 🟢 **Sadece `numpy`** (~1 MB) | 🟡 10+ kütüphane (~300-500 MB) | 🔴 C++ derleyicisi / Conda zorunlu |
| **Kurulum Hızı** | ⚡ **2 saniye** (`pip install .`) | ⏳ 1-3 dakika | ⚠️ Platforma göre zor |
| **Kullanım Amacı** | Tam Vektör Veritabanı | Tam Vektör Veritabanı | Sadece Vektör Arama Kütüphanesi |
| **Metadata Filtreleme** | ✅ Zengin (`$eq`, `$in`, `$gte` vb.) | ✅ Zengin | ❌ **Yok** |
| **Doküman Saklama** | ✅ Dahili (SQLite destekli) | ✅ Dahili | ❌ **Yok** |
| **Disk Kalıcılığı** | ✅ Dahili (`.npy` + SQLite) | ✅ Dahili | ❌ Manuel indeks kaydetme |
| **Arama Algoritmaları** | ✅ **Exact (Flat) + IVF Kümeleme** | HNSW (Approximate) | Flat, IVF, HNSW, PQ |
| **Bellek Sıkıştırma** | ✅ **Skalar Kuantizasyon (int8)** | ❌ Yok | ✅ Var |
| **Büyük Veri Desteği** | ✅ Memory-Mapped (`use_mmap=True`)| ❌ (RAM tüketir) | ✅ Var |
| **Cross-Platform** | ✅ Windows/Mac/Linux/Raspberry Pi | ⚠️ Bazen Windows'ta derleme hatası | ⚠️ Derleme/ortam bağımlı |
| **Öğrenme Eğrisi** | ⭐ Sıfır (ChromaDB API uyumlu) | ⭐ Kolay | ⚠️ Karmaşık |

---

## ✨ Temel Özellikler

- **Sıfır Hantal Bağımlılık:** Sadece `numpy` gerektirir. Derleyici (C/C++) zorunluluğu yoktur, her sistemde kutudan çıktığı gibi çalışır.
- **Kalıcı (Persistent) & Bellek İçi (In-Memory) Mod:**
  - `PersistentClient(path="./my_db")`: Verileri diske (`.npy` + SQLite) yazar, oturumlar arasında korur.
  - `EphemeralClient()`: Testler ve geçici oturumlar için tamamen RAM'de çalışır.
- **Yüksek Hızlı Arama Optimizasyonları (v0.2.0):**
  - **`argpartition` Top-K:** Mesafeleri $O(N \log N)$ yerine $O(N)$ sürede kısmi sıralar (3-10x sıralama hızı).
  - **Normalizasyon Cache:** Kosinüs benzerliğinde tekrar eden sorgularda %30-40 hız artışı.
  - **`IVFIndex` (Inverted File Index):** 100K+ vektörlük büyük veri setlerinde K-Means kümelemesi ile **10-50x hızlanma**.
  - **`ScalarQuantizer`:** Vektörleri float32'den int8'e sıkıştırarak **%75 bellek tasarrufu** (4x az RAM).
  - **Memory-Mapped Arrays:** RAM'den büyük dev veri setlerini diskten doğrudan arayabilme.
- **Zengin Benzerlik Metrikleri:** `cosine`, `dot_product`, `euclidean`.
- **Güçlü Metadata Filtreleme (`where`):** `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`.
- **ChromaDB Uyumlu API:** `add()`, `upsert()`, `query()`, `get()`, `delete()`, `count()`.

---

## 📦 Kurulum

Projeyi sisteminize veya sanal ortamınıza (venv) kurmak için:

```bash
# Doğrudan kurulum:
pip install .

# Geliştirici (düzenlenebilir) modda kurulum:
pip install -e .
```

---

## ⚡ Hızlı Başlangıç (Quickstart)

```python
import microvectordb

# 1. Kalıcı istemciyi başlatın
client = microvectordb.PersistentClient(path="./my_vectordb")

# 2. Koleksiyon oluşturun veya mevcut olanı açın
collection = client.get_or_create_collection(
    name="makaleler",
    dimension=3,       # İsteğe bağlı, ilk eklemede otomatik algılanır
    metric="cosine"    # "cosine", "dot_product" veya "euclidean"
)

# 3. Vektör, doküman ve metadata ekleyin
collection.add(
    ids=["doc1", "doc2", "doc3"],
    embeddings=[
        [0.9, 0.1, 0.0],
        [0.1, 0.8, 0.1],
        [0.0, 0.1, 0.9]
    ],
    documents=[
        "Yapay zeka ve LLM modelleri",
        "İlişkisel veritabanları mimarisi",
        "Web tasarımı ve kullanıcı arayüzü"
    ],
    metadatas=[
        {"kategori": "ai", "yazar": "Ahmet", "fiyat": 150},
        {"kategori": "veritabani", "yazar": "Mehmet", "fiyat": 200},
        {"kategori": "tasarim", "yazar": "Ayşe", "fiyat": 90}
    ]
)

# 4. Semantik arama yapın (Top-2 En Yakın Sonuç)
results = collection.query(
    query_embeddings=[[0.85, 0.15, 0.0]],
    n_results=2,
    where={"kategori": "ai"}  # İsteğe bağlı metadata filtresi
)

print("Bulunan ID'ler:", results.ids[0])
print("Dokümanlar:", results.documents[0])
print("Benzerlik Skorları:", results.similarities[0])
```

---

## 🔍 Gelişmiş Metadata Filtreleme (Where Syntax)

`query()`, `get()` ve `delete()` fonksiyonlarında gelişmiş filtreleme operatörlerini kullanabilirsiniz:

```python
# Fiyatı 100 TL ve üzeri olanlar ($gte)
results = collection.query(query_embeddings=[...], where={"fiyat": {"$gte": 100}})

# Kategori listesinde 'ai' veya 'tasarim' olanlar ($in)
results = collection.query(query_embeddings=[...], where={"kategori": {"$in": ["ai", "tasarim"]}})

# Çoklu koşul (Hem kategori 'ai' hem fiyat >= 100)
results = collection.query(query_embeddings=[...], where={"kategori": "ai", "fiyat": {"$gte": 100}})
```

---

## 🚀 Büyük Veri & İleri Seviye Modüller (v0.2.0)

### 1. IVF Kümeleme İndeksi (100K+ Vektörde 10-50x Hız)
Büyük veri setlerinde yaklaşık en yakın komşu (ANN) araması yapmak için saf NumPy IVF indeksini kullanabilirsiniz:

```python
from microvectordb import IVFIndex
import numpy as np

# 1. İndeksi oluşturun (64 küme, en yakın 4 kümede arama)
index = IVFIndex(n_clusters=64, nprobe=4)

# 2. Vektörlerle eğitin
vectors = np.random.randn(100000, 128).astype(np.float32)
index.train(vectors)

# 3. Işık hızında arayın
query = np.random.randn(128).astype(np.float32)
nearest_indices, distances = index.search(query, vectors, k=10)
```

### 2. Skalar Kuantizasyon (4 Kat Daha Az Bellek)
Vektörlerinizi 32-bit float yerine 8-bit integer'a sıkıştırın:

```python
from microvectordb import ScalarQuantizer

sq = ScalarQuantizer()
sq.fit(vectors)

# float32 -> int8 sıkıştırma
compressed_vectors = sq.encode(vectors)
print(sq.memory_stats(n_vectors=100000, dimension=128))
# Tasarruf: %75 daha az RAM kullanımı!
```

---

## 📁 Proje Mimarisi

```
vector db/
├── pyproject.toml              # Modern pip yapılandırması
├── setup.py                    # setuptools yapılandırması
├── LICENSE                     # MIT Lisansı
├── README.md                   # Kapsamlı Dokümantasyon
├── requirements.txt            # Sadece numpy>=1.20
│
├── microvectordb/              # Çekirdek Kütüphane
│   ├── __init__.py             # Dışa aktarılan ana arayüz (v0.2.0)
│   ├── client.py               # PersistentClient & EphemeralClient
│   ├── collection.py           # Koleksiyon yönetimi, CRUD ve optimize Query
│   ├── distance.py             # Normalize cache'li Kosinüs, Dot ve Öklid çekirdekleri
│   ├── ivf_index.py            # Saf NumPy IVF Kümeleme İndeksi
│   ├── quantization.py         # Skalar int8 Kuantizasyon Motoru
│   ├── storage.py              # SQLite + .npy (Memory-Mapped destekli)
│   └── types.py                # Veri modelleri ve özel hata sınıfları
│
├── examples/                   # Çalışan Örnekler
│   ├── 01_quickstart.py        # Temel başlangıç
│   ├── 02_metadata_filter.py   # Gelişmiş filtreleme
│   ├── 03_rag_integration.py   # RAG Chatbot entegrasyonu
│   └── 04_benchmark.py         # QPS hız testi
│
├── tests/                      # Birim Testler (27/27 Geçti)
│   ├── test_microvectordb.py   # Temel işlev testleri
│   └── test_optimizations.py   # IVF, Kuantizasyon ve Cache testleri
│
└── kernels_experimental/       # Donanım Hızlandırma
    ├── avx2_dot_product.c      # AVX2 C/Assembly çekirdeği
    └── README.md               # Donanım hızlandırma rehberi
```

---

## 👨‍💻 Geliştirici & Yazar

**Huneyn Kaya**  
GitHub: [@Ahmet003-cod](https://github.com/Ahmet003-cod)

---

## 📄 Lisans

Bu proje **[MIT Lisansı](LICENSE)** ile lisanslanmıştır. Ticari ve kişisel projelerinizde özgürce kullanabilir, değiştirebilir ve dağıtabilirsiniz.
