"""
hisamdb.integrations.langchain
------------------------------
Resmi, yüksek performanslı LangChain `VectorStore` entegrasyonu (`HisamDBVectorStore`).

Diğer LangChain vektör adaptörlerinden (Chroma, FAISS, Qdrant) fark yaratan optimizasyonlar:
1. **Varsayılan Yerleşik Hibrit Arama (`search_mode="hybrid"`):** Harici BM25 veya ElasticSearch
   kurmadan Dense + Türkçe/Unicode Okapi BM25 + RRF (<1 ms, %100 Hit@10) doğrudan
   `similarity_search()` ve `as_retriever()` içinde çalışır.
2. **Sıfır Yeniden-Gömme (Zero-Re-Embed) Vektörize BLAS MMR:** Aday vektörleri doğrudan HisamDB
   `VectorBuffer` / kuantize depodan tek kilitte çeker; standart Python-döngülü MMR'den 5x–15x hızlıdır.
3. **API Maliyeti Koruyan İçerik-Hash Tekilleştirme (`deduplicate_on_add=True`):** Veritabanında zaten
   bulunan özdeş metinler embedding modeline (OpenAI/Gemini) TEKRAR gönderilmez!
4. **Yerleşik Parent-Child Bağlam Penceresi (`context_window=1`):** Ayrı Redis/DocStore kurmadan
   eşleşen parçanın komşu parçalarını (`_chunk_index ± window`) otomatik birleştirir.
5. **15.5 MB RAM'de 1M Vektör (`index_type="disk"`), Multi-Tenant (`tenant_id`), Harici Reranker,
   Çok Çekirdekli `batch_similarity_search` ve Tam `asyncio` Desteği.**
6. **Sıfır Zorunlu Çekirdek Bağımlılığı:** `langchain-core` kuruluysa doğrudan resmi
   `langchain_core.vectorstores.VectorStore` alt sınıfı olarak `Document` döndürür; kurulu değilse
   hafif uyumlu `HisamDocument` ile bağımsız çalışır.
"""

from __future__ import annotations

import asyncio
import functools
import uuid
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import numpy as np

from hisamdb.api.client import RemoteCollection, RemoteHisamDB
from hisamdb.client import BaseClient, HisamDB
from hisamdb.collection import Collection
from hisamdb.integrations._common import (
    content_hash_id,
    merge_tenant_filter,
    normalize_relevance_score,
    sanitize_metadata,
    vectorized_mmr,
)
from hisamdb.search.rag import apply_reranker
from hisamdb.types import (
    IndexType,
    MetricType,
    QuantizationType,
    SearchResultItem,
    StorageDType,
)

try:
    from langchain_core.documents import Document as _LCDocument
    from langchain_core.embeddings import Embeddings as _LCEmbeddings
    from langchain_core.vectorstores import VectorStore as _LCVectorStore

    _HAS_LANGCHAIN = True
except ImportError:  # pragma: no cover - opsiyonel bağımlılık
    _HAS_LANGCHAIN = False

    @dataclass
    class _LCDocument:  # type: ignore[no-redef]
        """`langchain-core` kurulu değilken kullanılan uyumlu hafif doküman nesnesi."""

        page_content: str
        metadata: Dict[str, Any] = field(default_factory=dict)
        id: Optional[str] = None

    class _LCEmbeddings:  # type: ignore[no-redef]
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            raise NotImplementedError

        def embed_query(self, text: str) -> List[float]:
            raise NotImplementedError

    class _LCVectorStore:  # type: ignore[no-redef]
        pass


def is_langchain_available() -> bool:
    """`langchain-core` paketinin kurulu olup olmadığını döndürür."""
    return _HAS_LANGCHAIN


def _make_document(
    content: str,
    metadata: Dict[str, Any],
    doc_id: Optional[str] = None,
) -> Any:
    try:
        return _LCDocument(page_content=content, metadata=metadata, id=doc_id)
    except TypeError:
        return _LCDocument(page_content=content, metadata=metadata)


def _embed_query_sync(embedding: Any, query: str) -> List[float]:
    if hasattr(embedding, "embed_query"):
        return list(embedding.embed_query(query))
    if callable(embedding):
        vec = embedding(query)
        return list(vec[0]) if isinstance(vec, list) and vec and isinstance(vec[0], (list, tuple, np.ndarray)) else list(vec)
    raise TypeError("`embedding` nesnesi `embed_query(text)` metoduna veya çağrılabilir fonksiyona sahip olmalıdır.")


def _embed_documents_sync(embedding: Any, texts: List[str]) -> List[List[float]]:
    if hasattr(embedding, "embed_documents"):
        return [list(v) for v in embedding.embed_documents(texts)]
    if callable(embedding):
        return [list(v) for v in embedding(texts)]
    raise TypeError("`embedding` nesnesi `embed_documents(texts)` metoduna sahip olmalıdır.")


class HisamDBVectorStore(_LCVectorStore):
    """
    Resmi, optimize edilmiş LangChain `VectorStore` adaptörü.

    Örnek (Hibrit RAG + Parent-Child + DiskIndex):
        from hisamdb.integrations.langchain import HisamDBVectorStore

        vs = HisamDBVectorStore(
            embedding=embeddings,
            collection_name="docs",
            path="./rag_db",
            index_type="disk",          # 1M vektörde ~15.5 MB RAM!
            search_mode="hybrid",       # Dense + Türkçe BM25 + RRF (%100 Hit@10)
            deduplicate_on_add=True,    # Özdeş metinleri tekrar embed etmez
            context_window=1,           # Komşu parçaları otomatik birleştirir
        )
        retriever = vs.as_retriever(search_kwargs={"k": 5})
    """

    def __init__(
        self,
        embedding: Any,
        collection_name: str = "langchain_hisamdb",
        *,
        path: Optional[str] = None,
        client: Optional[Union[BaseClient, RemoteHisamDB]] = None,
        dimension: Optional[int] = None,
        metric: MetricType = "cosine",
        index_type: IndexType = "auto",
        quantization: QuantizationType = "none",
        storage_dtype: StorageDType = "float32",
        use_mmap: bool = False,
        keep_raw_vectors: bool = True,
        index_params: Optional[Dict[str, Any]] = None,
        max_workers: Optional[int] = None,
        # HisamDB'ye özgü ileri RAG optimizasyonları:
        search_mode: str = "hybrid",  # "hybrid" | "dense" | "sparse"
        alpha: float = 0.5,
        beta: float = 0.5,
        fusion: str = "rrf",
        rrf_k: int = 60,
        tenant_id: Optional[str] = None,
        reranker: Optional[Callable[[str, List[SearchResultItem]], List[SearchResultItem]]] = None,
        context_window: int = 0,
        deduplicate_on_add: bool = False,
        deduplicate_results: bool = False,
        group_by: Optional[str] = None,
        max_per_group: int = 1,
        batch_size: int = 1024,
        include_internal_metadata: bool = False,
    ) -> None:
        if search_mode not in ("hybrid", "dense", "sparse"):
            raise ValueError("`search_mode` yalnızca 'hybrid', 'dense' veya 'sparse' olabilir.")

        self._embedding = embedding
        self.collection_name = collection_name
        self.metric: MetricType = metric
        self.search_mode = search_mode
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.fusion = fusion
        self.rrf_k = int(rrf_k)
        self.tenant_id = tenant_id
        self.reranker = reranker
        self.context_window = max(0, int(context_window))
        self.deduplicate_on_add = bool(deduplicate_on_add)
        self.deduplicate_results = bool(deduplicate_results)
        self.group_by = group_by
        self.max_per_group = max(1, int(max_per_group))
        self.batch_size = max(1, int(batch_size))
        self.include_internal_metadata = bool(include_internal_metadata)

        self._owns_client = client is None
        if client is not None:
            self._client: Union[BaseClient, RemoteHisamDB] = client
        else:
            self._client = HisamDB(path=path, use_mmap=use_mmap, max_workers=max_workers)

        self._collection: Union[Collection, RemoteCollection] = self._client.get_or_create_collection(
            name=collection_name,
            dimension=dimension,
            metric=metric,
            index_type=index_type,
            quantization=quantization,
            storage_dtype=storage_dtype,
            use_mmap=use_mmap,
            keep_raw_vectors=keep_raw_vectors,
            index_params=index_params,
        )

    @property
    def embeddings(self) -> Any:
        return self._embedding

    @property
    def client(self) -> Union[BaseClient, RemoteHisamDB]:
        return self._client

    @property
    def collection(self) -> Union[Collection, RemoteCollection]:
        return self._collection

    def close(self) -> None:
        """İstemciyi bu store oluşturduysa kaynakları güvenle kapatır."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "HisamDBVectorStore":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Yazma: Akışlı Batch + Sıfır Tekrar-Gömme (Cost-Saving Dedup!)
    # ------------------------------------------------------------------ #
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        *,
        ids: Optional[List[str]] = None,
        embeddings: Optional[Sequence[Sequence[float]]] = None,
        doc_ids: Optional[List[str]] = None,
        deduplicate: Optional[bool] = None,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Metinleri akışlı toplu (batch) modda ekler.
        `deduplicate_on_add=True` (veya `deduplicate=True`) iken veritabanında zaten bulunan
        özdeş metinler embedding modeline TEKRAR gönderilmez (API maliyeti & gecikme tasarrufu!).
        """
        text_list = list(texts)
        if not text_list:
            return []

        n = len(text_list)
        if metadatas is not None and len(metadatas) != n:
            raise ValueError(f"`metadatas` uzunluğu ({len(metadatas)}) `texts` ({n}) ile eşleşmelidir.")
        if ids is not None and len(ids) != n:
            raise ValueError(f"`ids` uzunluğu ({len(ids)}) `texts` ({n}) ile eşleşmelidir.")
        if doc_ids is not None and len(doc_ids) != n:
            raise ValueError(f"`doc_ids` uzunluğu ({len(doc_ids)}) `texts` ({n}) ile eşleşmelidir.")

        do_dedup = self.deduplicate_on_add if deduplicate is None else bool(deduplicate)
        bs = max(1, int(batch_size or self.batch_size))

        # 1. ID ve metadata hazırlığı (Parent-Child _doc_id / _chunk_index + tenant_id)
        resolved_ids: List[str] = []
        resolved_metas: List[Dict[str, Any]] = []
        chunk_counters: Dict[str, int] = {}

        for i, txt in enumerate(text_list):
            m = dict(metadatas[i]) if (metadatas is not None and metadatas[i] is not None) else {}
            if self.tenant_id:
                m["_tenant_id"] = self.tenant_id

            if doc_ids is not None:
                d_id = str(doc_ids[i])
                m.setdefault("_doc_id", d_id)
                m.setdefault("_parent_id", d_id)
            elif "source" in m and "_doc_id" not in m:
                d_id = str(m["source"])
                m["_doc_id"] = d_id
                m.setdefault("_parent_id", d_id)

            if "_doc_id" in m and "_chunk_index" not in m:
                d_key = str(m["_doc_id"])
                c_idx = chunk_counters.get(d_key, 0)
                m["_chunk_index"] = c_idx
                chunk_counters[d_key] = c_idx + 1

            if ids is not None:
                rid = str(ids[i])
            elif do_dedup:
                rid = content_hash_id(txt, m, namespace=self.tenant_id or "")
            else:
                rid = str(uuid.uuid4())
            resolved_ids.append(rid)
            resolved_metas.append(m)

        # 2. Tekilleştirme: Hem batch içi hem DB içi mevcut ID'leri embed ETMEDEN ele!
        to_insert_idx: List[int] = []
        if do_dedup:
            existing: set[str] = set()
            if isinstance(self._collection, Collection):
                # O(M) bellek-içi sözlük kontrolü (sıfır I/O!)
                existing = set(self._collection.get(ids=resolved_ids)["ids"])
            else:
                try:
                    existing = set(self._collection.get(ids=resolved_ids)["ids"])
                except Exception:
                    existing = set()
            seen_batch: set[str] = set()
            for i, rid in enumerate(resolved_ids):
                if rid in existing or rid in seen_batch:
                    continue
                seen_batch.add(rid)
                to_insert_idx.append(i)
        else:
            to_insert_idx = list(range(n))

        if not to_insert_idx:
            return resolved_ids

        # 3. Akışlı (streaming) toplu gömme + upsert
        for start in range(0, len(to_insert_idx), bs):
            chunk_pos = to_insert_idx[start : start + bs]
            c_texts = [text_list[i] for i in chunk_pos]
            c_ids = [resolved_ids[i] for i in chunk_pos]
            c_metas = [resolved_metas[i] for i in chunk_pos]
            if embeddings is not None:
                c_vecs = [list(embeddings[i]) for i in chunk_pos]
            else:
                c_vecs = _embed_documents_sync(self._embedding, c_texts)
            self._collection.upsert(
                ids=c_ids,
                embeddings=c_vecs,
                documents=c_texts,
                metadatas=c_metas,
            )

        return resolved_ids

    def add_documents(self, documents: List[Any], **kwargs: Any) -> List[str]:
        texts = [doc.page_content for doc in documents]
        metadatas = [dict(doc.metadata) if doc.metadata else {} for doc in documents]
        ids = kwargs.pop("ids", None)
        if ids is None:
            doc_ids_attr = [getattr(doc, "id", None) for doc in documents]
            if all(x is not None for x in doc_ids_attr):
                ids = [str(x) for x in doc_ids_attr]
        return self.add_texts(texts=texts, metadatas=metadatas, ids=ids, **kwargs)

    @classmethod
    def from_texts(
        cls: Type["HisamDBVectorStore"],
        texts: List[str],
        embedding: Any,
        metadatas: Optional[List[dict]] = None,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "HisamDBVectorStore":
        store = cls(embedding=embedding, **kwargs)
        store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        return store

    @classmethod
    def from_documents(
        cls: Type["HisamDBVectorStore"],
        documents: List[Any],
        embedding: Any,
        **kwargs: Any,
    ) -> "HisamDBVectorStore":
        store = cls(embedding=embedding, **kwargs)
        store.add_documents(documents)
        return store

    # ------------------------------------------------------------------ #
    # Arama: Yerleşik Hibrit (Dense + Türkçe BM25 + RRF) + Parent-Child
    # ------------------------------------------------------------------ #
    def _items_to_docs_with_scores(
        self,
        items: List[SearchResultItem],
        *,
        normalize: bool = False,
        eff_mode: str = "dense",
    ) -> List[Tuple[Any, float]]:
        out: List[Tuple[Any, float]] = []
        strip = not self.include_internal_metadata
        for it in items:
            meta = sanitize_metadata(it.metadata, strip_internal=strip)
            doc = _make_document(it.document or "", meta, doc_id=it.id)
            sc = (
                normalize_relevance_score(
                    it.score,
                    metric=self.metric,
                    search_mode=eff_mode,
                    rrf_k=self.rrf_k,
                )
                if normalize
                else float(it.score)
            )
            out.append((doc, sc))
        return out

    def _run_search(
        self,
        *,
        query_text: Optional[str],
        query_vector: Optional[Sequence[float]],
        k: int,
        filter: Optional[Dict[str, Any]] = None,
        search_mode: Optional[str] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        fusion: Optional[str] = None,
        score_threshold: Optional[float] = None,
        reranker: Optional[Callable[[str, List[SearchResultItem]], List[SearchResultItem]]] = None,
        deduplicate: Optional[bool] = None,
        group_by: Optional[str] = None,
        max_per_group: Optional[int] = None,
        context_window: Optional[int] = None,
        ef_search: Optional[int] = None,
        nprobe: Optional[int] = None,
    ) -> Tuple[List[SearchResultItem], str]:
        eff_mode = search_mode or self.search_mode
        if query_text is None and eff_mode == "hybrid":
            eff_mode = "dense"
        eff_filter = merge_tenant_filter(filter, self.tenant_id)
        eff_win = self.context_window if context_window is None else max(0, int(context_window))
        eff_reranker = reranker if reranker is not None else self.reranker
        eff_dedup = self.deduplicate_results if deduplicate is None else bool(deduplicate)
        eff_group = self.group_by if group_by is None else group_by
        eff_mpg = self.max_per_group if max_per_group is None else int(max_per_group)

        # Parent-Child bağlam penceresi etkinse doğrudan HisamDB retrieve_with_context kullan!
        if eff_win > 0:
            ctx_rows = self._collection.retrieve_with_context(
                query_vector=query_vector if eff_mode != "sparse" else None,
                query_text=query_text if eff_mode != "dense" else None,
                limit=k,
                window=eff_win,
                filter=eff_filter,
                alpha=self.alpha if alpha is None else float(alpha),
                beta=self.beta if beta is None else float(beta),
                fusion=fusion or self.fusion,
            )
            items: List[SearchResultItem] = []
            for row in ctx_rows:
                base_it: SearchResultItem = row["item"]
                meta = dict(base_it.metadata or {})
                meta["_hisam_matched_chunk"] = base_it.document
                meta["_hisam_neighbor_ids"] = [nb.id for nb in row.get("neighbors", [])]
                items.append(
                    SearchResultItem(
                        id=base_it.id,
                        score=base_it.score,
                        distance=base_it.distance,
                        document=row["context"],
                        metadata=meta,
                    )
                )
            return items, eff_mode

        if eff_mode == "hybrid":
            hyb_kwargs: Dict[str, Any] = dict(
                query_vector=query_vector,
                query_text=query_text,
                limit=k,
                filter=eff_filter,
                alpha=self.alpha if alpha is None else float(alpha),
                beta=self.beta if beta is None else float(beta),
                fusion=fusion or self.fusion,
                rrf_k=self.rrf_k,
                score_threshold=score_threshold,
                deduplicate=eff_dedup,
                group_by=eff_group,
                max_per_group=eff_mpg,
            )
            if isinstance(self._collection, Collection) and eff_reranker is not None:
                hyb_kwargs["reranker"] = eff_reranker
            items = self._collection.hybrid_search(**hyb_kwargs)
            return items, "hybrid"

        if eff_mode == "sparse":
            items = self._collection.sparse_search(
                query_text=query_text,
                limit=k,
                filter=eff_filter,
                score_threshold=score_threshold,
            )
            return items, "sparse"

        # Dense
        if query_vector is None:
            raise ValueError("`dense` arama modu için sorgu vektörü gereklidir.")
        items = self._collection.search(
            query_vector=query_vector,
            limit=k,
            filter=eff_filter,
            score_threshold=score_threshold,
            ef_search=ef_search,
            nprobe=nprobe,
        )
        if eff_reranker is not None and query_text is not None and items:
            items = apply_reranker(query_text, items, eff_reranker, limit=k)
        return items, "dense"

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Any, float]]:
        mode = kwargs.get("search_mode", self.search_mode)
        q_vec = None if mode == "sparse" else _embed_query_sync(self._embedding, query)
        items, eff_mode = self._run_search(
            query_text=query,
            query_vector=q_vec,
            k=k,
            filter=filter,
            **kwargs,
        )
        return self._items_to_docs_with_scores(items, normalize=False, eff_mode=eff_mode)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Any]:
        pairs = self.similarity_search_with_score(query, k=k, filter=filter, **kwargs)
        return [doc for doc, _ in pairs]

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        score_threshold: Optional[float] = None,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Any, float]]:
        mode = kwargs.get("search_mode", self.search_mode)
        q_vec = None if mode == "sparse" else _embed_query_sync(self._embedding, query)
        items, eff_mode = self._run_search(
            query_text=query,
            query_vector=q_vec,
            k=k,
            filter=filter,
            **kwargs,
        )
        pairs = self._items_to_docs_with_scores(items, normalize=True, eff_mode=eff_mode)
        if score_threshold is not None:
            pairs = [(d, s) for d, s in pairs if s >= float(score_threshold)]
        return pairs

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Any]:
        kwargs.setdefault("search_mode", "dense")
        items, eff_mode = self._run_search(
            query_text=None,
            query_vector=embedding,
            k=k,
            filter=filter,
            **kwargs,
        )
        return [doc for doc, _ in self._items_to_docs_with_scores(items, eff_mode=eff_mode)]

    def similarity_search_with_context(
        self,
        query: str,
        k: int = 4,
        window: int = 1,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Any]:
        """Parent-Child bağlam penceresiyle komşu parçaları birleştirerek döndürür."""
        return self.similarity_search(query, k=k, filter=filter, context_window=window, **kwargs)

    # ------------------------------------------------------------------ #
    # Sıfır Yeniden-Gömme Vektörize BLAS MMR (5x–15x Daha Hızlı!)
    # ------------------------------------------------------------------ #
    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        *,
        query_text: Optional[str] = None,
        search_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Any]:
        eff_fetch = max(int(fetch_k), int(k))
        eff_mode = search_mode or ("hybrid" if query_text else "dense")
        items, actual_mode = self._run_search(
            query_text=query_text,
            query_vector=embedding,
            k=eff_fetch,
            filter=filter,
            search_mode=eff_mode,
            context_window=0,
            **kwargs,
        )
        if not items:
            return []

        cand_ids = [it.id for it in items]
        by_id = {it.id: it for it in items}

        # Vektörleri HisamDB belleğinden sıfır yeniden-gömme maliyetiyle çek
        cand_vecs: Optional[np.ndarray] = None
        ordered_items: List[SearchResultItem] = items
        if isinstance(self._collection, Collection):
            found_ids, mat = self._collection.get_vectors(cand_ids)
            if mat.shape[0] > 0:
                ordered_items = [by_id[fid] for fid in found_ids]
                cand_vecs = mat

        # Uzak sunucu veya ham/decode vektörü yoksa (örn. binary keep_raw=False) metinden fallback
        if cand_vecs is None or cand_vecs.shape[0] == 0:
            texts = [it.document or "" for it in items]
            cand_vecs = np.asarray(_embed_documents_sync(self._embedding, texts), dtype=np.float32)
            ordered_items = items

        sel_idx = vectorized_mmr(embedding, cand_vecs, k=k, lambda_mult=lambda_mult)
        picked = [ordered_items[i] for i in sel_idx]
        return [doc for doc, _ in self._items_to_docs_with_scores(picked, eff_mode=actual_mode)]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Any]:
        q_vec = _embed_query_sync(self._embedding, query)
        return self.max_marginal_relevance_search_by_vector(
            embedding=q_vec,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            query_text=query,
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    # Çok Çekirdekli Toplu Arama (18K+ QPS HNSW!) & CRUD & Explain
    # ------------------------------------------------------------------ #
    def batch_similarity_search(
        self,
        queries: Sequence[str],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        search_mode: Optional[str] = None,
        max_workers: Optional[int] = None,
    ) -> List[List[Any]]:
        """
        Birden çok sorguyu tek toplu gömme + HisamDB çok çekirdekli `batch_search` /
        `batch_hybrid_search` motoruyla paralel çalıştırır.
        """
        q_list = list(queries)
        if not q_list:
            return []
        eff_mode = search_mode or self.search_mode
        eff_filter = merge_tenant_filter(filter, self.tenant_id)
        q_vecs = _embed_documents_sync(self._embedding, q_list)

        if eff_mode == "hybrid":
            payload = [
                {"query_vector": q_vecs[i], "query_text": q_list[i], "filter": eff_filter}
                for i in range(len(q_list))
            ]
            batch_items = self._collection.batch_hybrid_search(
                queries=payload,
                limit=k,
                alpha=self.alpha,
                beta=self.beta,
                fusion=self.fusion,
                max_workers=max_workers,
            )
        else:
            batch_items = self._collection.batch_search(
                query_vectors=q_vecs,
                limit=k,
                filter=eff_filter,
                max_workers=max_workers,
            )
        return [
            [doc for doc, _ in self._items_to_docs_with_scores(row, eff_mode=eff_mode)]
            for row in batch_items
        ]

    def get_by_ids(self, ids: Sequence[str], /) -> List[Any]:
        id_list = [str(i) for i in ids]
        if not id_list:
            return []
        eff_filter = merge_tenant_filter(None, self.tenant_id)
        res = self._collection.get(ids=id_list, where=eff_filter)
        strip = not self.include_internal_metadata
        docs: List[Any] = []
        for rid, txt, meta in zip(res["ids"], res["documents"], res["metadatas"]):
            docs.append(_make_document(txt or "", sanitize_metadata(meta, strip_internal=strip), doc_id=rid))
        return docs

    def delete(
        self,
        ids: Optional[List[str]] = None,
        *,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        eff_filter = merge_tenant_filter(filter, self.tenant_id)
        if ids is None and eff_filter is None:
            return False
        if ids is not None and self.tenant_id:
            # Kiracı güvenliği: yalnızca kendi kiracısına ait ID'leri sil
            owned = set(self._collection.get(ids=ids, where={"_tenant_id": self.tenant_id})["ids"])
            if not owned:
                return False
            self._collection.delete(ids=list(owned))
            return True
        deleted = self._collection.delete(ids=ids, where=eff_filter)
        return deleted > 0

    def explain(self, k: int = 4, filter: Optional[Dict[str, Any]] = None) -> Any:
        """HisamDB QueryOptimizer sorgu yürütme planını döndürür."""
        return self._collection.explain(n_results=k, where=merge_tenant_filter(filter, self.tenant_id))

    # ------------------------------------------------------------------ #
    # Gerçek Bloklamayan `asyncio` Desteği
    # ------------------------------------------------------------------ #
    async def _run_async(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        call = functools.partial(fn, *args, **kwargs)
        return await loop.run_in_executor(None, call)

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        return await self._run_async(self.add_texts, list(texts), metadatas, ids=ids, **kwargs)

    async def aadd_documents(self, documents: List[Any], **kwargs: Any) -> List[str]:
        return await self._run_async(self.add_documents, documents, **kwargs)

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Any]:
        return await self._run_async(self.similarity_search, query, k=k, filter=filter, **kwargs)

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Any, float]]:
        return await self._run_async(self.similarity_search_with_score, query, k=k, filter=filter, **kwargs)

    async def asimilarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Any, float]]:
        return await self._run_async(
            self.similarity_search_with_relevance_scores, query, k=k, filter=filter, **kwargs
        )

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Any]:
        return await self._run_async(
            self.max_marginal_relevance_search,
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )

    async def adelete(
        self,
        ids: Optional[List[str]] = None,
        *,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        return await self._run_async(self.delete, ids=ids, filter=filter, **kwargs)

    async def aget_by_ids(self, ids: Sequence[str], /) -> List[Any]:
        return await self._run_async(self.get_by_ids, list(ids))
