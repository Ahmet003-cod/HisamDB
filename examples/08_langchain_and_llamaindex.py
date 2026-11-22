"""
Örnek 08 — Resmi LangChain & LlamaIndex Entegrasyonları (HisamDB v1.1.0)
========================================================================

Bu örnek harici bir LLM API anahtarına ihtiyaç duymadan:
1. LangChain `HisamDBVectorStore`: Yerleşik Hibrit (Dense + Türkçe BM25 + RRF),
   API maliyeti koruyan `deduplicate_on_add=True`, sıfır yeniden-gömme Vektörize BLAS MMR,
   Parent-Child bağlam penceresi ve LCEL `as_retriever()` kullanımını;
2. LlamaIndex `HisamDBLlamaIndexStore`: `DiskIndex (MMap)` ile `HYBRID`, `MMR` ve
   `MetadataFilters` sorgularını uçtan uca gösterir.
"""

import shutil
import tempfile
import numpy as np

from hisamdb import HisamDBLlamaIndexStore, HisamDBVectorStore, is_langchain_available
from hisamdb.integrations import HisamTextNode, VectorStoreQuery, VectorStoreQueryMode


class DemoEmbeddings:
    """Örnek amaçlı hafif deterministik embedding sağlayıcısı (gerçekte OpenAI/HuggingFace/Ollama verilir)."""

    def __init__(self) -> None:
        self.api_calls = 0

    @staticmethod
    def _encode(text: str) -> list[float]:
        low = text.lower()
        v = np.array([
            1.0 if any(w in low for w in ("sunucu", "kesinti", "kurtarma", "wal")) else 0.0,
            1.0 if any(w in low for w in ("istanbul", "veri merkezi", "hnsw")) else 0.0,
            1.0 if any(w in low for w in ("snapshot", "yedek", "sha")) else 0.0,
            1.0 if any(w in low for w in ("fatura", "ödeme", "bütçe")) else 0.0,
            0.05,
        ], dtype=np.float32)
        return (v / np.linalg.norm(v)).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.api_calls += len(texts)
        return [self._encode(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._encode(text)


def main() -> None:
    tmp = tempfile.mkdtemp(prefix="hisam_demo_int_")
    try:
        emb = DemoEmbeddings()
        print("=== 1. LangChain HisamDBVectorStore (Hibrit + Dedup + BLAS MMR + Bağlam) ===")
        with HisamDBVectorStore(
            embedding=emb,
            collection_name="ops_kb",
            path=tmp,
            index_type="flat",
            search_mode="hybrid",
            deduplicate_on_add=True,
        ) as vs:
            docs = [
                "Sunucu genel kurulum kontrol listesi.",
                "Sunucu kesintisi hata kodu ERR-9042 için WAL kurtarma prosedürü.",
                "Kurtarma sonrası SHA-256 doğrulamalı snapshot yedeği alınır.",
                "İstanbul veri merkezi HNSW performans rehberi.",
            ]
            vs.add_texts(docs, metadatas=[{"year": 2026}] * 4, doc_ids=["manual"] * 3 + ["guide"])
            print(f"İlk eklemede embedding çağrısı: {emb.api_calls}")

            # Aynı metinleri tekrar ekle -> deduplicate_on_add sayesinde 0 yeni embedding çağrısı!
            vs.add_texts(docs, metadatas=[{"year": 2026}] * 4, doc_ids=["manual"] * 3 + ["guide"])
            print(f"Tekrar eklemede embedding çağrısı (değişmemeli!): {emb.api_calls}")

            hits = vs.similarity_search("kesinti ERR-9042", k=2)
            print("Hibrit 1. sonuç:", hits[0].page_content)

            ctx = vs.similarity_search_with_context("kesinti ERR-9042", k=1, window=1)
            print("Parent-Child Bağlam:\n ", ctx[0].page_content.replace("\n", " | "))

            mmr_hits = vs.max_marginal_relevance_search("sunucu", k=2, fetch_k=4, lambda_mult=0.4)
            print("Vektörize MMR sonuçları:", [d.page_content[:35] + "..." for d in mmr_hits])

            if is_langchain_available():
                retriever = vs.as_retriever(search_kwargs={"k": 1})
                print("LangChain Retriever .invoke():", retriever.invoke("kesinti ERR-9042")[0].page_content)

        print("\n=== 2. LlamaIndex HisamDBLlamaIndexStore (DiskIndex + HYBRID + MMR) ===")
        with HisamDBLlamaIndexStore(
            collection_name="llama_kb",
            path=tmp,
            index_type="disk",
            use_mmap=True,
        ) as store:
            nodes = [
                HisamTextNode(id_=f"n{i}", text=t, embedding=emb.embed_query(t), metadata={"year": 2026}, ref_doc_id="ops")
                for i, t in enumerate(docs)
            ]
            store.add(nodes)
            q = VectorStoreQuery(
                query_embedding=emb.embed_query("kesinti ERR-9042"),
                query_str="kesinti ERR-9042",
                similarity_top_k=2,
                mode=VectorStoreQueryMode.HYBRID,
                alpha=0.5,
            )
            res = store.query(q)
            print("LlamaIndex Hybrid 1. sonuç:", res.ids[0], "|", res.nodes[0].text)
            print("LlamaIndex MMR sonuç sayısı:", len(store.query(
                VectorStoreQuery(query_embedding=emb.embed_query("sunucu"), similarity_top_k=2, mode=VectorStoreQueryMode.MMR)
            ).nodes))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
