# HisamDB `v1.1.0` — Resmi LangChain & LlamaIndex Entegrasyon Rehberi

HisamDB, çekirdeğinin **`numpy`-only** hafifliğini bozmadan **LangChain (`HisamDBVectorStore`)** ve **LlamaIndex (`HisamDBLlamaIndexStore`)** için üretim düzeyinde optimize edilmiş resmi vektör deposu adaptörleri sunar.

```bash
# Yalnızca LangChain eklentisi ile:
pip install "hisamdb[langchain]"

# Yalnızca LlamaIndex eklentisi ile:
pip install "hisamdb[llamaindex]"

# HNSW + orjson + LangChain + LlamaIndex:
pip install "hisamdb[hnsw,fast,langchain,llamaindex]"
```

---

## ⚡ Neden HisamDB Adaptörleri? (Diğerlerinden 6 Temel Farkımız)

| Özellik | Klasik Chroma / FAISS Adaptörleri | **HisamDB (`HisamDBVectorStore` / `HisamDBLlamaIndexStore`)** |
|---|---|---|
| **Varsayılan Arama** | Yalnızca Dense vektör *(hata kodlarını kaçırır, ~%53 Hit@10)* | **Yerleşik Hibrit Arama (Dense + Türkçe BM25 + RRF, `%100` Hit@10)** — sıfır harici Elasticsearch/BM25! |
| **MMR (Çeşitlilik) Hızı** | Python iç döngüsü + çoğu adaptörde yeniden-gömme yükü | **Vektörize BLAS MMR + Sıfır Yeniden-Gömme (`11.3x – 29.9x` daha hızlı, `0.997 ms` uçtan uca!)** |
| **Tekrarlı Belge Maliyeti** | Aynı metinler yeniden OpenAI/Gemini API'ye gönderilir | **`deduplicate_on_add=True`** — SHA-256 içerik özetiyle mevcut metinler embed **edilmeden** elenir (**%100 API tasarrufu**) |
| **Parent-Child Bağlam** | Ayrı `ParentDocumentRetriever` + harici Redis/DocStore gerekir | **Tek parametre: `context_window=1`** — eşleşen parçanın komşularını otomatik birleştirir |
| **Bellek & Ölçek** | Tüm `float32` vektörler RAM'de şişer | **`index_type="disk"` (1M vektörde `15.5 MB RAM`)**, `INT8`, `Binary+Rescore`, `IVF-PQ` tek satırda! |
| **Gömülü ↔ Sunucu Geçişi** | Ayrı istemci/kod değişikliği gerekir | Aynı sınıf hem yerel `path="./db"` hem uzak **`RemoteHisamDB` REST sunucusuyla** çalışır + **Katı `tenant_id` izolasyonu** |

---

## 📊 Gerçek Ölçülmüş Performans (Vektörize BLAS MMR & Uçtan Uca VectorStore)

*Ölçüm betiği: [`benchmarks/run_benchmarks_integrations.py`](../benchmarks/run_benchmarks_integrations.py) · Ham veri: [`benchmark_v1.1.0_integrations.json`](../benchmarks/results/benchmark_v1.1.0_integrations.json)*

| Senaryo | Standart Python MMR | **HisamDB BLAS MMR** | Hızlanma |
|---|---:|---:|---:|
| **768D (BGE/E5), `fetch_k=50, k=10`** | 1.054 ms | **0.093 ms** | **11.32x hızlı** |
| **768D (BGE/E5), `fetch_k=200, k=15`** | 7.513 ms | **0.252 ms** | **29.85x hızlı** |
| **768D (BGE/E5), `fetch_k=500, k=20`** | 24.780 ms | **1.798 ms** | **13.78x hızlı** |
| **1536D (OpenAI), `fetch_k=50, k=10`** | 1.393 ms | **0.123 ms** | **11.35x hızlı** |
| **1536D (OpenAI), `fetch_k=200, k=15`** | 39.653 ms | **2.775 ms** | **14.29x hızlı** |
| **1536D (OpenAI), `fetch_k=500, k=20`** | 120.514 ms | **8.314 ms** | **14.50x hızlı** |

- **Uçtan Uca `HisamDBVectorStore` (`10K × 128D` HNSW):**
  - **Zero-Re-Embed MMR (`50 → 10`):** **`0.997 ms`** / sorgu
  - **Çok Çekirdekli `batch_similarity_search(64)`:** **`7.19 ms`** (**`8,897.5 QPS`**)

---

## 🦜 LangChain Kullanımı (`HisamDBVectorStore`)

```python
from hisamdb import HisamDBVectorStore
# Herhangi bir LangChain Embeddings sınıfı ile çalışır (OpenAIEmbeddings, HuggingFaceEmbeddings, OllamaEmbeddings...):
# from langchain_openai import OpenAIEmbeddings

with HisamDBVectorStore(
    embedding=embeddings,
    collection_name="company_kb",
    path="./rag_db",
    index_type="hnsw",          # 1M+ düşük RAM için: index_type="disk", use_mmap=True
    search_mode="hybrid",       # Dense + Türkçe Okapi BM25 + RRF (varsayılan!)
    deduplicate_on_add=True,    # Aynı metinleri tekrar OpenAI'a göndermez
    context_window=1,           # Komşu parçaları otomatik bağlam olarak ekler
    tenant_id="acme_corp",      # Opsiyonel: katı kiracı yalıtımı
) as vectorstore:
    vectorstore.add_texts(
        texts=[
            "İstanbul veri merkezi bakım prosedürü.",
            "Hata kodu ERR-9042 için WAL kurtarma adımları.",
            "Kurtarma sonrası SHA-256 snapshot doğrulaması yapılır.",
        ],
        metadatas=[{"category": "ops", "year": 2026}] * 3,
        doc_ids=["ops_manual"] * 3,
    )

    # 1. Yerleşik Hibrit Arama + Parent-Child Bağlam:
    docs = vectorstore.similarity_search("kesinti ERR-9042", k=2, filter={"year": {"$gte": 2025}})

    # 2. Sıfır Yeniden-Gömme Vektörize BLAS MMR (11x–30x daha hızlı):
    diverse_docs = vectorstore.max_marginal_relevance_search("veri merkezi bakım", k=3, fetch_k=20, lambda_mult=0.5)

    # 3. Resmi LangChain LCEL Retriever:
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    # chain = retriever | format_docs | prompt | llm

    # 4. Çok çekirdekli toplu arama & Async:
    batch_hits = vectorstore.batch_similarity_search(["ERR-9042", "snapshot"], k=2)
    # async_hits = await vectorstore.asimilarity_search("ERR-9042", k=2)
```

### Uzak HisamDB REST Sunucusuyla LangChain:
```python
from hisamdb import RemoteHisamDB, HisamDBVectorStore

remote = RemoteHisamDB("http://127.0.0.1:8000", api_key="hisam_admin_key")
vs = HisamDBVectorStore(embedding=embeddings, collection_name="prod_kb", client=remote)
```

---

## 🦙 LlamaIndex Kullanımı (`HisamDBLlamaIndexStore`)

```python
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters, VectorStoreQueryMode
from hisamdb import HisamDBLlamaIndexStore

vector_store = HisamDBLlamaIndexStore(
    collection_name="llama_kb",
    path="./llama_db",
    index_type="disk",          # 1M vektörde ~15.5 MB RAM!
    use_mmap=True,
    deduplicate_on_add=True,
    context_window=1,
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# 1. Yerleşik Hibrit (Dense + Türkçe BM25 + RRF) Query Engine:
query_engine = index.as_query_engine(
    vector_store_query_mode=VectorStoreQueryMode.HYBRID,
    alpha=0.5,
    similarity_top_k=5,
    filters=MetadataFilters(
        filters=[MetadataFilter(key="year", value=2025, operator=">=")]
    ),
)
response = query_engine.query("ERR-9042 için WAL kurtarma adımları nelerdir?")

# 2. Vektörize BLAS MMR Modu:
mmr_engine = index.as_query_engine(
    vector_store_query_mode=VectorStoreQueryMode.MMR,
    vector_store_kwargs={"mmr_threshold": 0.5},
)
```
