"""
HisamDB v1.1.0 — Resmi LangChain & LlamaIndex Entegrasyon Test Paketi
=====================================================================

Test edilen fark yaratan optimizasyonlar:
1. LangChain `HisamDBVectorStore` (gerçek `langchain_core.vectorstores.VectorStore` kalıtımı +
   `as_retriever().invoke(...)` + `Document` uyumluluğu)
2. Varsayılan yerleşik Hibrit Arama (Dense + Türkçe BM25 + RRF): Dense-Only'nin kaçırdığı
   `ERR-9042` hata kodunu tek satırda bulma
3. API Maliyeti Koruyan İçerik-Hash Tekilleştirme (`deduplicate_on_add=True`): özdeş metinlerin
   `embed_documents()` çağrısını 0'a indirme kanıtı!
4. Sıfır Yeniden-Gömme Vektörize BLAS MMR (`max_marginal_relevance_search`):
   hem sayısal doğruluğu hem `embed_documents()` çağrılmadığını kanıtlama!
5. Parent-Child Bağlam Penceresi (`context_window=1` / `similarity_search_with_context`)
6. Multi-Tenant izolasyonu, `batch_similarity_search`, `delete`, `get_by_ids`, `explain`,
   harici `reranker` ve gerçek `asyncio` (`aadd_texts`, `asimilarity_search`, `amax_marginal_relevance_search`)
7. LlamaIndex `HisamDBLlamaIndexStore`: `DEFAULT`, `TEXT_SEARCH`/`SPARSE`, `HYBRID`, `MMR` modları,
   özyinelemeli `MetadataFilters` -> `MetadataIndex` AST çevirisi (`EQ`, `NE`, `GTE`, `LTE`, `IN`,
   `NIN`, `CONTAINS`, `AND`, `OR`, `NOT`), `ref_doc_id` silme, `context_window`, `tenant_id`, `async`!
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import unittest
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

import hisamdb
from hisamdb import (
    HisamDBLlamaIndexStore,
    HisamDBVectorStore,
    is_langchain_available,
)
from hisamdb.integrations import (
    HisamTextNode,
    VectorStoreQuery,
    VectorStoreQueryMode,
    translate_llamaindex_filters,
    vectorized_mmr,
)


class CountingDeterministicEmbeddings:
    """
    Test için deterministik 8-boyutlu embedding sağlayıcısı + çağrı sayacı.
    `embed_docs_calls` ile sıfır tekrar-gömme (dedup & zero-re-embed MMR) garantisi kanıtlanır.
    """

    def __init__(self) -> None:
        self.embed_docs_calls = 0
        self.embed_docs_texts = 0
        self.embed_query_calls = 0

    @staticmethod
    def _vec(text: str) -> List[float]:
        low = text.lower()
        v = np.zeros(8, dtype=np.float32)
        # Konu bazlı kaba semantik boyutlar (hata kodlarını bilerek görmezden gelir!)
        if any(w in low for w in ("sunucu", "server", "ağ", "bağlantı", "kesinti", "kurtarma")):
            v[0] = 1.0
        if any(w in low for w in ("istanbul", "veri merkezi", "datacenter")):
            v[1] = 1.0
        if any(w in low for w in ("fatura", "ödeme", "muhasebe", "vergi")):
            v[2] = 1.0
        if any(w in low for w in ("yapay zeka", "vektör", "rag", "llm", "hnsw")):
            v[3] = 1.0
        if any(w in low for w in ("yedekleme", "snapshot", "güvenlik", "sha")):
            v[4] = 1.0
        if any(w in low for w in ("python", "kod", "derleme", "simd")):
            v[5] = 1.0
        if np.linalg.norm(v) == 0.0:
            v[6] = 1.0
        v[7] = 0.05
        v /= np.linalg.norm(v)
        return v.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.embed_docs_calls += 1
        self.embed_docs_texts += len(texts)
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        self.embed_query_calls += 1
        return self._vec(text)


# LlamaIndex filtre nesnesi taklidi (llama-index-core kurulu olmasa da tam AST çevirisi test edilir)
@dataclass
class _MockFilter:
    key: str
    value: Any
    operator: str = "=="


@dataclass
class _MockFilters:
    filters: List[Any]
    condition: str = "and"


class TestVectorizedMMRAndFilters(unittest.TestCase):
    def test_vectorized_mmr_selects_diverse_results(self):
        # 0 ve 1 neredeyse özdeş (sorguya en yakın); 2 sorguya yakın ama farklı yönde!
        q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        cands = np.array([
            [0.99, 0.14, 0.0],   # 0: en yakın
            [0.985, 0.17, 0.0],  # 1: 0 ile neredeyse kopya!
            [0.75, 0.0, 0.66],   # 2: çeşitli ve hâlâ alakalı
            [0.0, 0.0, 1.0],     # 3: alakasız
        ], dtype=np.float32)

        # lambda=1.0 -> saf ilgililik: [0, 1]
        pure_rel = vectorized_mmr(q, cands, k=2, lambda_mult=1.0)
        self.assertEqual(pure_rel, [0, 1])

        # lambda=0.5 -> çeşitlilik cezası 1 yerine 2'yi seçmelidir!
        diverse = vectorized_mmr(q, cands, k=2, lambda_mult=0.5)
        self.assertEqual(diverse[0], 0)
        self.assertEqual(diverse[1], 2)

    def test_llamaindex_recursive_filter_translation(self):
        flt = _MockFilters(
            condition="and",
            filters=[
                _MockFilter("year", 2025, ">="),
                _MockFilter("category", ["ops", "rag"], "in"),
                _MockFilters(
                    condition="or",
                    filters=[
                        _MockFilter("lang", "tr", "=="),
                        _MockFilter("priority", 1, "!="),
                    ],
                ),
            ],
        )
        ast = translate_llamaindex_filters(flt)
        self.assertEqual(
            ast,
            {
                "$and": [
                    {"year": {"$gte": 2025}},
                    {"category": {"$in": ["ops", "rag"]}},
                    {"$or": [{"lang": {"$eq": "tr"}}, {"priority": {"$ne": 1}}]},
                ]
            },
        )


class TestLangChainIntegration(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="hisam_lc_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_langchain_vectorstore_hybrid_dedup_mmr_context_and_retriever(self):
        emb = CountingDeterministicEmbeddings()
        texts = [
            "Sunucu kurulum ve bakım takvimi notu 1.",
            "Sunucu izleme paneli genel ayarları 2.",
            "Sunucu kesintisi hata kodu ERR-9042 için WAL kurtarma adımları.",
            "Yedekleme sonrası SHA-256 snapshot doğrulaması yapılır.",
            "İstanbul veri merkezi HNSW vektör veritabanı kurulum rehberi.",
            "Fatura ve muhasebe ödeme takvimi 2026.",
        ]
        metas = [
            {"cat": "ops", "year": 2026},
            {"cat": "ops", "year": 2026},
            {"cat": "ops", "year": 2026},
            {"cat": "ops", "year": 2026},
            {"cat": "ai", "year": 2026},
            {"cat": "fin", "year": 2025},
        ]
        doc_ids = ["manual_1", "manual_1", "manual_1", "manual_1", "ai_guide", "fin_doc"]

        with HisamDBVectorStore(
            embedding=emb,
            collection_name="lc_docs",
            path=self.tmpdir,
            index_type="flat",
            search_mode="hybrid",
            deduplicate_on_add=True,
        ) as vs:
            # 1. Resmi langchain_core.vectorstores.VectorStore alt sınıfı doğrulaması
            if is_langchain_available():
                from langchain_core.vectorstores import VectorStore
                self.assertIsInstance(vs, VectorStore)

            ids1 = vs.add_texts(texts, metadatas=metas, doc_ids=doc_ids)
            self.assertEqual(len(ids1), 6)
            self.assertEqual(emb.embed_docs_texts, 6)

            # 2. Maliyet koruyan dedup: aynı 6 metni tekrar eklerken embed_documents() ÇAĞRILMAMALI!
            ids2 = vs.add_texts(texts, metadatas=metas, doc_ids=doc_ids)
            self.assertEqual(ids1, ids2)
            self.assertEqual(emb.embed_docs_texts, 6)  # 0 ekstra embedding çağrısı!

            # 3. Yerleşik Hibrit Arama: Dense 'kesinti' -> tüm sunucu dokümanlarında eşitlik verir,
            #    ancak yerleşik Türkçe BM25 + RRF sayesinde ERR-9042 kesin olarak 1. sırada gelir!
            hits = vs.similarity_search("kesinti ERR-9042", k=2)
            self.assertIn("ERR-9042", hits[0].page_content)

            # 4. Resmi LangChain Retriever (LCEL .invoke() uyumluluğu)
            if is_langchain_available():
                retriever = vs.as_retriever(search_kwargs={"k": 2, "filter": {"cat": "ops"}})
                ret_docs = retriever.invoke("kesinti ERR-9042")
                self.assertTrue(len(ret_docs) >= 1)
                self.assertIn("ERR-9042", ret_docs[0].page_content)

            # 5. Relevance scores [0, 1] aralığında olmalı
            rel = vs.similarity_search_with_relevance_scores("İstanbul HNSW vektör", k=3)
            self.assertTrue(all(0.0 <= s <= 1.0 for _, s in rel))

            # 6. Sıfır Yeniden-Gömme Vektörize BLAS MMR: embed_docs_calls ARTMAMALI!
            calls_before = emb.embed_docs_calls
            mmr_docs = vs.max_marginal_relevance_search("sunucu kesintisi", k=3, fetch_k=6, lambda_mult=0.4)
            self.assertEqual(len(mmr_docs), 3)
            self.assertEqual(emb.embed_docs_calls, calls_before)

            # 7. Parent-Child Bağlam Penceresi: ERR-9042 parçasının komşu parçaları da gelmeli!
            ctx_docs = vs.similarity_search_with_context("kesinti ERR-9042", k=1, window=1)
            self.assertIn("ERR-9042", ctx_docs[0].page_content)
            self.assertIn("SHA-256 snapshot", ctx_docs[0].page_content)

            # 8. Paralel batch_similarity_search + get_by_ids + explain + delete
            batch_res = vs.batch_similarity_search(["ERR-9042", "fatura ödeme"], k=1)
            self.assertEqual(len(batch_res), 2)
            self.assertIn("ERR-9042", batch_res[0][0].page_content)
            self.assertIn("Fatura", batch_res[1][0].page_content)

            fetched = vs.get_by_ids([ids1[0], ids1[2]])
            self.assertEqual(len(fetched), 2)
            self.assertTrue(hasattr(vs.explain(k=3), "execution_path"))

            # 9. Async API doğrulaması
            async def _run_async():
                a_hits = await vs.asimilarity_search("ERR-9042", k=2)
                self.assertIn("ERR-9042", a_hits[0].page_content)
                a_mmr = await vs.amax_marginal_relevance_search("sunucu", k=2, fetch_k=5)
                self.assertEqual(len(a_mmr), 2)
                new_ids = await vs.aadd_texts(["Python SIMD derleme kılavuzu"], metadatas=[{"cat": "dev", "year": 2026}])
                got = await vs.aget_by_ids(new_ids)
                self.assertEqual(len(got), 1)
                self.assertTrue(await vs.adelete(ids=new_ids))

            asyncio.run(_run_async())

    def test_langchain_tenant_isolation_and_external_reranker(self):
        emb = CountingDeterministicEmbeddings()
        with hisamdb.HisamDB(self.tmpdir) as db:
            acme = HisamDBVectorStore(
                embedding=emb, collection_name="saas_lc", client=db, tenant_id="acme", search_mode="hybrid"
            )
            beta = HisamDBVectorStore(
                embedding=emb, collection_name="saas_lc", client=db, tenant_id="beta", search_mode="hybrid"
            )
            a_ids = acme.add_texts(["ACME gizli bütçe 2026", "ACME sunucu planı"])
            beta.add_texts(["Beta genel duyuru"])

            # Beta ACME'yi ne görebilir ne silebilir
            b_hits = beta.similarity_search("ACME gizli bütçe", k=5)
            self.assertTrue(all("ACME" not in d.page_content for d in b_hits))
            self.assertFalse(beta.delete(ids=a_ids))
            self.assertEqual(len(acme.get_by_ids(a_ids)), 2)

            # Harici reranker doğrulaması (özel skorlama ile 2. kaydı 1. sıraya taşı)
            def prefer_plan_reranker(_q, items):
                return [10.0 if "planı" in (it.document or "") else 1.0 for it in items]

            r_reranked = acme.similarity_search("ACME", k=2, reranker=prefer_plan_reranker)
            self.assertIn("planı", r_reranked[0].page_content)


class TestLlamaIndexIntegration(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="hisam_li_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_llamaindex_hybrid_mmr_filters_context_delete_and_async(self):
        emb = CountingDeterministicEmbeddings()
        raw_items = [
            ("n1", "Sunucu bakım çizelgesi genel notu.", {"cat": "ops", "year": 2024}, "doc_ops"),
            ("n2", "Sunucu kesintisi ERR-7711 WAL kurtarma rehberi.", {"cat": "ops", "year": 2026}, "doc_ops"),
            ("n3", "Snapshot SHA-256 bütünlük doğrulaması.", {"cat": "ops", "year": 2026}, "doc_ops"),
            ("n4", "İstanbul HNSW yapay zeka vektör arama.", {"cat": "ai", "year": 2026}, "doc_ai"),
            ("n5", "Fatura muhasebe ödeme raporu.", {"cat": "fin", "year": 2025}, "doc_fin"),
        ]
        nodes = [
            HisamTextNode(id_=nid, text=txt, embedding=emb.embed_query(txt), metadata=meta, ref_doc_id=ref)
            for nid, txt, meta, ref in raw_items
        ]

        with HisamDBLlamaIndexStore(
            collection_name="li_docs",
            path=self.tmpdir,
            index_type="disk",
            use_mmap=True,
        ) as store:
            added = store.add(nodes)
            self.assertEqual(len(added), 5)

            # 1. HYBRID mod: Dense + Türkçe BM25 + RRF -> ERR-7711 1. sırada!
            q_hyb = VectorStoreQuery(
                query_embedding=emb.embed_query("kesinti ERR-7711"),
                query_str="kesinti ERR-7711",
                similarity_top_k=2,
                mode=VectorStoreQueryMode.HYBRID,
                alpha=0.5,
            )
            res_hyb = store.query(q_hyb)
            self.assertEqual(res_hyb.ids[0], "n2")
            self.assertTrue(all(0.0 <= s <= 1.0 for s in res_hyb.similarities))

            # 2. TEXT_SEARCH / SPARSE (BM25) + Karmaşık MetadataFilters!
            flt = _MockFilters(
                condition="and",
                filters=[
                    _MockFilter("cat", "ops", "=="),
                    _MockFilter("year", 2025, ">="),
                ],
            )
            q_txt = VectorStoreQuery(
                query_str="sunucu kesintisi",
                similarity_top_k=5,
                mode=VectorStoreQueryMode.TEXT_SEARCH,
                filters=flt,
            )
            res_txt = store.query(q_txt)
            # n1 year=2024 olduğu için filtrelenmeli; n2 dönmeli!
            self.assertNotIn("n1", res_txt.ids)
            self.assertIn("n2", res_txt.ids)

            # 3. MMR modu (sıfır yeniden-gömme vektörize BLAS MMR!)
            q_mmr = VectorStoreQuery(
                query_embedding=emb.embed_query("sunucu kesintisi"),
                similarity_top_k=3,
                mode=VectorStoreQueryMode.MMR,
                mmr_threshold=0.4,
            )
            res_mmr = store.query(q_mmr)
            self.assertEqual(len(res_mmr.nodes), 3)

            # 4. Parent-Child Bağlam Penceresi (context_window=1)
            res_ctx = store.query(q_hyb, context_window=1)
            self.assertIn("ERR-7711", res_ctx.nodes[0].text)
            self.assertIn("Snapshot SHA-256", res_ctx.nodes[0].text)

            # 5. Async API + ref_doc_id silme (<1 ms tombstone!)
            async def _run_async():
                ares = await store.aquery(q_hyb)
                self.assertEqual(ares.ids[0], "n2")
                got = await store.aget_nodes(node_ids=["n4"])
                self.assertEqual(len(got), 1)
                await store.adelete("doc_ops")  # n1, n2, n3 silinmeli
                rem = await store.aget_nodes()
                self.assertEqual(sorted(n.node_id for n in rem), ["n4", "n5"])

            asyncio.run(_run_async())
            store.persist()


if __name__ == "__main__":
    unittest.main()
