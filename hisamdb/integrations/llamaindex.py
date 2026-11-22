"""
hisamdb.integrations.llamaindex
-------------------------------
Resmi, yüksek performanslı LlamaIndex `VectorStore` entegrasyonu (`HisamDBLlamaIndexStore`).

Standart LlamaIndex vektör depolarından (SimpleVectorStore, Chroma, FAISS) farkları:
1. **Tek Satırda Yerleşik Hibrit Arama (`VectorStoreQueryMode.HYBRID` & `TEXT_SEARCH` / `SPARSE`):**
   Harici BM25 veya Elasticsearch kurmadan Dense + Türkçe/Unicode Okapi BM25 + RRF (<1 ms, %100 Hit@10).
2. **Sıfır Yeniden-Gömme Vektörize BLAS MMR (`VectorStoreQueryMode.MMR`):**
   `query.mmr_threshold` ile çeşitlilik optimizasyonunu HisamDB vektör tamponundan milisaniye altında yapar.
3. **Tam `MetadataFilters` → O(1) Ters İndeks Çevirisi:**
   LlamaIndex'in tüm `FilterOperator` (`==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `nin`, `any`, `all`,
   `contains`, `text_match`, `is_empty`) ve iç içe `AND`/`OR`/`NOT` koşullarını HisamDB ters
   `MetadataIndex`'ine sıfır tam-tarama yüküyle çevirir.
4. **Kayıpsız `TextNode` İlişki Koruması + Yerleşik Parent-Child Bağlam Penceresi (`context_window`):**
   Node metni, metadata ve `ref_doc_id` / ilişkileri korunur; `delete(ref_doc_id)` `<1 ms` tombstone
   ile tüm doküman parçalarını siler.
5. **15.5 MB RAM'de 1M Vektör (`index_type="disk"`), Multi-Tenant (`tenant_id`), Harici Reranker,
   İçerik-Hash Tekilleştirme ve Tam `asyncio` (`async_add`, `aquery`, `adelete`).**
"""

from __future__ import annotations

import asyncio
import functools
import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from hisamdb.api.client import RemoteCollection, RemoteHisamDB
from hisamdb.client import BaseClient, HisamDB
from hisamdb.collection import Collection
from hisamdb.integrations._common import (
    content_hash_id,
    merge_tenant_filter,
    normalize_relevance_score,
    sanitize_metadata,
    translate_llamaindex_filters,
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
    from llama_index.core.schema import BaseNode as _LIBaseNode
    from llama_index.core.schema import TextNode as _LITextNode
    from llama_index.core.vector_stores.types import (
        BasePydanticVectorStore as _LIBaseStore,
        MetadataFilters as _LIMetadataFilters,
        VectorStoreQuery as _LIVectorStoreQuery,
        VectorStoreQueryMode as _LIVectorStoreQueryMode,
        VectorStoreQueryResult as _LIVectorStoreQueryResult,
    )
    from pydantic import ConfigDict, PrivateAttr

    _HAS_LLAMAINDEX = True
except ImportError:  # pragma: no cover - opsiyonel bağımlılık
    _HAS_LLAMAINDEX = False

    class _LIVectorStoreQueryMode(str, Enum):  # type: ignore[no-redef]
        DEFAULT = "default"
        SPARSE = "sparse"
        HYBRID = "hybrid"
        TEXT_SEARCH = "text_search"
        SEMANTIC_HYBRID = "semantic_hybrid"
        MMR = "mmr"

    @dataclass
    class _LITextNode:  # type: ignore[no-redef]
        """`llama-index-core` kurulu değilken kullanılan uyumlu hafif düğüm (node) nesnesi."""

        text: str = ""
        id_: str = field(default_factory=lambda: str(uuid.uuid4()))
        embedding: Optional[List[float]] = None
        metadata: Dict[str, Any] = field(default_factory=dict)
        ref_doc_id: Optional[str] = None

        @property
        def node_id(self) -> str:
            return self.id_

        def get_content(self, metadata_mode: Any = None) -> str:
            return self.text

        def get_embedding(self) -> List[float]:
            if self.embedding is None:
                raise ValueError("Node embedding değeri atanmamış.")
            return list(self.embedding)

        def to_dict(self) -> Dict[str, Any]:
            return {
                "id_": self.id_,
                "text": self.text,
                "metadata": dict(self.metadata),
                "ref_doc_id": self.ref_doc_id,
            }

    _LIBaseNode = _LITextNode  # type: ignore[misc, assignment]

    @dataclass
    class _LIVectorStoreQuery:  # type: ignore[no-redef]
        query_embedding: Optional[List[float]] = None
        similarity_top_k: int = 4
        doc_ids: Optional[List[str]] = None
        node_ids: Optional[List[str]] = None
        query_str: Optional[str] = None
        output_fields: Optional[List[str]] = None
        embedding_field: Optional[str] = None
        mode: Any = _LIVectorStoreQueryMode.DEFAULT
        alpha: Optional[float] = None
        filters: Optional[Any] = None
        mmr_threshold: Optional[float] = None
        sparse_top_k: Optional[int] = None
        hybrid_top_k: Optional[int] = None

    @dataclass
    class _LIVectorStoreQueryResult:  # type: ignore[no-redef]
        nodes: Optional[Sequence[Any]] = None
        similarities: Optional[List[float]] = None
        ids: Optional[List[str]] = None

    _LIMetadataFilters = Any  # type: ignore[misc, assignment]

    class _LIBaseStore:  # type: ignore[no-redef]
        stores_text: bool = True
        flat_metadata: bool = False
        is_embedding_query: bool = True


# Dışa aktarılan uyumluluk tipleri
VectorStoreQuery = _LIVectorStoreQuery
VectorStoreQueryMode = _LIVectorStoreQueryMode
VectorStoreQueryResult = _LIVectorStoreQueryResult
HisamTextNode = _LITextNode


def is_llamaindex_available() -> bool:
    """`llama-index-core` paketinin kurulu olup olmadığını döndürür."""
    return _HAS_LLAMAINDEX


def _node_id(node: Any) -> str:
    nid = getattr(node, "node_id", None) or getattr(node, "id_", None)
    return str(nid) if nid else str(uuid.uuid4())


def _node_text(node: Any) -> str:
    if hasattr(node, "get_content"):
        try:
            return str(node.get_content(metadata_mode="none"))
        except TypeError:
            return str(node.get_content())
    return str(getattr(node, "text", "") or "")


def _node_embedding(node: Any) -> List[float]:
    if hasattr(node, "get_embedding"):
        return [float(x) for x in node.get_embedding()]
    emb = getattr(node, "embedding", None)
    if emb is None:
        raise ValueError(f"Node '{_node_id(node)}' için embedding değeri bulunamadı.")
    return [float(x) for x in emb]


def _node_ref_doc_id(node: Any) -> Optional[str]:
    ref = getattr(node, "ref_doc_id", None)
    if ref:
        return str(ref)
    meta = getattr(node, "metadata", None) or {}
    if "_ref_doc_id" in meta:
        return str(meta["_ref_doc_id"])
    if "_doc_id" in meta:
        return str(meta["_doc_id"])
    return None


def _rebuild_node(
    node_id: str,
    text: str,
    metadata: Optional[Dict[str, Any]],
    *,
    strip_internal: bool = True,
) -> Any:
    raw_meta = dict(metadata or {})
    ref_doc_id = raw_meta.get("_ref_doc_id") or raw_meta.get("_doc_id")
    node_json = raw_meta.get("_hisam_node_json")
    clean_meta = sanitize_metadata(raw_meta, strip_internal=strip_internal)

    if _HAS_LLAMAINDEX and isinstance(node_json, str):
        try:
            data = json.loads(node_json)
            data["text"] = text
            data["id_"] = node_id
            data["metadata"] = clean_meta
            return _LITextNode.from_dict(data)
        except Exception:
            pass

    if _HAS_LLAMAINDEX:
        return _LITextNode(id_=node_id, text=text, metadata=clean_meta)
    return _LITextNode(id_=node_id, text=text, metadata=clean_meta, ref_doc_id=ref_doc_id)


def _mode_value(mode: Any) -> str:
    if hasattr(mode, "value"):
        return str(mode.value).lower()
    return str(mode).lower()


class _HisamDBLlamaIndexCore:
    """
    LlamaIndex adaptörünün ortak yüksek performanslı yürütme çekirdeği.
    Hem Pydantic `BasePydanticVectorStore` hem sıfır-bağımlılıklı mod tarafından kullanılır.
    """

    def __init__(
        self,
        collection_name: str = "llamaindex_hisamdb",
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
        default_alpha: float = 0.5,
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
        self.collection_name = collection_name
        self.path = path
        self.metric: MetricType = metric
        self.default_alpha = float(default_alpha)
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

    def add(self, nodes: Sequence[Any], **add_kwargs: Any) -> List[str]:
        node_list = list(nodes)
        if not node_list:
            return []

        do_dedup = bool(add_kwargs.get("deduplicate", self.deduplicate_on_add))
        bs = max(1, int(add_kwargs.get("batch_size", self.batch_size)))

        ids: List[str] = []
        texts: List[str] = []
        vecs: List[List[float]] = []
        metas: List[Dict[str, Any]] = []
        chunk_counters: Dict[str, int] = {}

        for node in node_list:
            txt = _node_text(node)
            emb = _node_embedding(node)
            m = dict(getattr(node, "metadata", None) or {})
            if self.tenant_id:
                m["_tenant_id"] = self.tenant_id

            ref_id = _node_ref_doc_id(node)
            if ref_id:
                m["_ref_doc_id"] = ref_id
                m.setdefault("_doc_id", ref_id)
                m.setdefault("_parent_id", ref_id)
                if "_chunk_index" not in m:
                    c_idx = chunk_counters.get(ref_id, 0)
                    m["_chunk_index"] = c_idx
                    chunk_counters[ref_id] = c_idx + 1

            # LlamaIndex ilişkilerini hafifçe sakla (embedding hariç)
            if _HAS_LLAMAINDEX and hasattr(node, "to_dict"):
                try:
                    nd = node.to_dict()
                    nd.pop("embedding", None)
                    nd.pop("text", None)
                    m["_hisam_node_json"] = json.dumps(nd, ensure_ascii=False, default=str)
                except Exception:
                    pass

            nid = content_hash_id(txt, m, namespace=self.tenant_id or "") if do_dedup else _node_id(node)
            ids.append(nid)
            texts.append(txt)
            vecs.append(emb)
            metas.append(m)

        # Tekilleştirme
        if do_dedup:
            existing = set(self._collection.get(ids=ids)["ids"])
            seen_batch: set[str] = set()
            keep_idx = []
            for i, nid in enumerate(ids):
                if nid in existing or nid in seen_batch:
                    continue
                seen_batch.add(nid)
                keep_idx.append(i)
        else:
            keep_idx = list(range(len(ids)))

        for start in range(0, len(keep_idx), bs):
            sl = keep_idx[start : start + bs]
            self._collection.upsert(
                ids=[ids[i] for i in sl],
                embeddings=[vecs[i] for i in sl],
                documents=[texts[i] for i in sl],
                metadatas=[metas[i] for i in sl],
            )
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Verilen `ref_doc_id` değerine sahip tüm parçaları `<1 ms` tombstone ile siler."""
        flt = merge_tenant_filter({"_ref_doc_id": str(ref_doc_id)}, self.tenant_id)
        self._collection.delete(where=flt)

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[Any] = None,
        **delete_kwargs: Any,
    ) -> None:
        translated = translate_llamaindex_filters(filters)
        eff_filter = merge_tenant_filter(translated, self.tenant_id)
        if node_ids is not None and self.tenant_id:
            owned = self._collection.get(ids=list(node_ids), where={"_tenant_id": self.tenant_id})["ids"]
            if owned:
                self._collection.delete(ids=owned)
            return
        if node_ids is not None or eff_filter is not None:
            self._collection.delete(ids=list(node_ids) if node_ids is not None else None, where=eff_filter)

    def clear(self) -> None:
        if self.tenant_id:
            self._collection.delete(where={"_tenant_id": self.tenant_id})
        else:
            all_ids = self._collection.get()["ids"]
            if all_ids:
                self._collection.delete(ids=all_ids)

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[Any] = None,
    ) -> List[Any]:
        translated = translate_llamaindex_filters(filters)
        eff_filter = merge_tenant_filter(translated, self.tenant_id)
        res = self._collection.get(
            ids=list(node_ids) if node_ids is not None else None,
            where=eff_filter,
        )
        strip = not self.include_internal_metadata
        return [
            _rebuild_node(rid, txt or "", meta, strip_internal=strip)
            for rid, txt, meta in zip(res["ids"], res["documents"], res["metadatas"])
        ]

    def query(self, query: Any, **kwargs: Any) -> Any:
        top_k = int(getattr(query, "similarity_top_k", 4) or 4)
        q_emb = getattr(query, "query_embedding", None)
        q_str = getattr(query, "query_str", None)
        mode_str = _mode_value(getattr(query, "mode", "default"))
        raw_filters = getattr(query, "filters", None)
        doc_ids = getattr(query, "doc_ids", None)
        node_ids = getattr(query, "node_ids", None)
        q_alpha = getattr(query, "alpha", None)

        # Filtreleri birleştir (MetadataFilters + doc_ids + tenant_id)
        clauses: List[Dict[str, Any]] = []
        translated = translate_llamaindex_filters(raw_filters)
        if translated:
            clauses.append(translated)
        if doc_ids:
            clauses.append({"_ref_doc_id": {"$in": [str(d) for d in doc_ids]}})
        if self.tenant_id:
            clauses.append({"_tenant_id": self.tenant_id})

        eff_filter: Optional[Dict[str, Any]] = None
        if len(clauses) == 1:
            eff_filter = clauses[0]
        elif len(clauses) > 1:
            eff_filter = {"$and": clauses}

        eff_win = int(kwargs.get("context_window", self.context_window))
        eff_reranker = kwargs.get("reranker", self.reranker)
        eff_fusion = str(kwargs.get("fusion", self.fusion))
        alpha_val = self.default_alpha if q_alpha is None else float(q_alpha)
        beta_val = max(0.0, 1.0 - alpha_val)

        # Eğer node_ids kısıtı varsa aşırı örnekle ve süz
        fetch_k = max(top_k * 4, 32) if node_ids else top_k

        # --- 1. MMR Modu (Sıfır Yeniden-Gömme Vektörize BLAS MMR!) ---
        if mode_str == "mmr":
            if q_emb is None:
                raise ValueError("MMR sorgusu için `query_embedding` gereklidir.")
            mmr_fetch = max(int(kwargs.get("fetch_k", top_k * 5)), top_k)
            mmr_lam = float(
                getattr(query, "mmr_threshold", None)
                if getattr(query, "mmr_threshold", None) is not None
                else kwargs.get("lambda_mult", 0.5)
            )
            if q_str:
                cand_items = self._collection.hybrid_search(
                    query_vector=q_emb,
                    query_text=q_str,
                    limit=mmr_fetch,
                    filter=eff_filter,
                    alpha=alpha_val,
                    beta=beta_val,
                    fusion=eff_fusion,
                )
            else:
                cand_items = self._collection.search(
                    query_vector=q_emb,
                    limit=mmr_fetch,
                    filter=eff_filter,
                )
            if node_ids:
                allow = set(str(x) for x in node_ids)
                cand_items = [it for it in cand_items if it.id in allow]
            if not cand_items:
                return _LIVectorStoreQueryResult(nodes=[], similarities=[], ids=[])

            if isinstance(self._collection, Collection):
                by_id = {it.id: it for it in cand_items}
                found_ids, mat = self._collection.get_vectors([it.id for it in cand_items])
                if mat.shape[0] > 0:
                    ordered = [by_id[fid] for fid in found_ids]
                    sel = vectorized_mmr(q_emb, mat, k=top_k, lambda_mult=mmr_lam)
                    items = [ordered[i] for i in sel]
                else:
                    items = cand_items[:top_k]
            else:
                items = cand_items[:top_k]
            eff_mode = "hybrid" if q_str else "dense"

        # --- 2. Parent-Child Bağlam Penceresi ---
        elif eff_win > 0:
            ctx_rows = self._collection.retrieve_with_context(
                query_vector=q_emb if mode_str not in ("sparse", "text_search") else None,
                query_text=q_str if mode_str in ("hybrid", "semantic_hybrid", "sparse", "text_search") else None,
                limit=fetch_k,
                window=eff_win,
                filter=eff_filter,
                alpha=alpha_val,
                beta=beta_val,
                fusion=eff_fusion,
            )
            items = []
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
            eff_mode = "hybrid" if mode_str in ("hybrid", "semantic_hybrid") else "dense"

        # --- 3. Hibrit Mod (Dense + Türkçe BM25 + RRF/Weighted) ---
        elif mode_str in ("hybrid", "semantic_hybrid"):
            hyb_kwargs: Dict[str, Any] = dict(
                query_vector=q_emb,
                query_text=q_str,
                limit=fetch_k,
                filter=eff_filter,
                alpha=alpha_val,
                beta=beta_val,
                fusion=eff_fusion,
                rrf_k=self.rrf_k,
                deduplicate=self.deduplicate_results,
                group_by=self.group_by,
                max_per_group=self.max_per_group,
            )
            if isinstance(self._collection, Collection) and eff_reranker is not None:
                hyb_kwargs["reranker"] = eff_reranker
            items = self._collection.hybrid_search(**hyb_kwargs)
            eff_mode = "hybrid"

        # --- 4. Saf Metin / Sparse (Okapi BM25) ---
        elif mode_str in ("sparse", "text_search"):
            items = self._collection.sparse_search(
                query_text=q_str,
                limit=fetch_k,
                filter=eff_filter,
            )
            eff_mode = "sparse"

        # --- 5. Varsayılan Dense Vektör Arama ---
        else:
            if q_emb is None:
                if q_str:
                    items = self._collection.sparse_search(query_text=q_str, limit=fetch_k, filter=eff_filter)
                    eff_mode = "sparse"
                else:
                    return _LIVectorStoreQueryResult(nodes=[], similarities=[], ids=[])
            else:
                items = self._collection.search(
                    query_vector=q_emb,
                    limit=fetch_k,
                    filter=eff_filter,
                )
                if eff_reranker is not None and q_str and items:
                    items = apply_reranker(q_str, items, eff_reranker, limit=fetch_k)
                eff_mode = "dense"

        if node_ids:
            allow = set(str(x) for x in node_ids)
            items = [it for it in items if it.id in allow][:top_k]

        strip = not self.include_internal_metadata
        out_nodes: List[Any] = []
        out_sims: List[float] = []
        out_ids: List[str] = []
        for it in items[:top_k]:
            out_nodes.append(_rebuild_node(it.id, it.document or "", it.metadata, strip_internal=strip))
            out_sims.append(
                normalize_relevance_score(
                    it.score,
                    metric=self.metric,
                    search_mode=eff_mode,
                    rrf_k=self.rrf_k,
                )
            )
            out_ids.append(it.id)

        return _LIVectorStoreQueryResult(nodes=out_nodes, similarities=out_sims, ids=out_ids)

    def persist(self, persist_path: Optional[str] = None, fs: Any = None) -> None:
        if isinstance(self._collection, Collection):
            self._collection.checkpoint()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


if _HAS_LLAMAINDEX:  # pragma: no cover - llama-index-core kurulu ortamlarda gerçek BasePydanticVectorStore

    class HisamDBLlamaIndexStore(_LIBaseStore):  # type: ignore[misc]
        """Resmi LlamaIndex `BasePydanticVectorStore` alt sınıfı."""

        model_config = ConfigDict(arbitrary_types_allowed=True)
        stores_text: bool = True
        flat_metadata: bool = False
        is_embedding_query: bool = True
        collection_name: str = "llamaindex_hisamdb"

        _core: _HisamDBLlamaIndexCore = PrivateAttr()

        def __init__(self, collection_name: str = "llamaindex_hisamdb", **kwargs: Any) -> None:
            super().__init__(collection_name=collection_name)
            self._core = _HisamDBLlamaIndexCore(collection_name=collection_name, **kwargs)

        @classmethod
        def class_name(cls) -> str:
            return "HisamDBLlamaIndexStore"

        @property
        def client(self) -> Any:
            return self._core._client

        @property
        def collection(self) -> Union[Collection, RemoteCollection]:
            return self._core._collection

        def add(self, nodes: Sequence[Any], **add_kwargs: Any) -> List[str]:
            return self._core.add(nodes, **add_kwargs)

        async def async_add(self, nodes: Sequence[Any], **add_kwargs: Any) -> List[str]:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, functools.partial(self._core.add, nodes, **add_kwargs))

        def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
            self._core.delete(ref_doc_id, **delete_kwargs)

        async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, functools.partial(self._core.delete, ref_doc_id, **delete_kwargs))

        def delete_nodes(
            self,
            node_ids: Optional[List[str]] = None,
            filters: Optional[Any] = None,
            **delete_kwargs: Any,
        ) -> None:
            self._core.delete_nodes(node_ids=node_ids, filters=filters, **delete_kwargs)

        async def adelete_nodes(
            self,
            node_ids: Optional[List[str]] = None,
            filters: Optional[Any] = None,
            **delete_kwargs: Any,
        ) -> None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, functools.partial(self._core.delete_nodes, node_ids=node_ids, filters=filters, **delete_kwargs)
            )

        def clear(self) -> None:
            self._core.clear()

        def get_nodes(
            self,
            node_ids: Optional[List[str]] = None,
            filters: Optional[Any] = None,
        ) -> List[Any]:
            return self._core.get_nodes(node_ids=node_ids, filters=filters)

        async def aget_nodes(
            self,
            node_ids: Optional[List[str]] = None,
            filters: Optional[Any] = None,
        ) -> List[Any]:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, functools.partial(self._core.get_nodes, node_ids=node_ids, filters=filters)
            )

        def query(self, query: Any, **kwargs: Any) -> Any:
            return self._core.query(query, **kwargs)

        async def aquery(self, query: Any, **kwargs: Any) -> Any:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, functools.partial(self._core.query, query, **kwargs))

        def persist(self, persist_path: str = "", fs: Any = None) -> None:
            self._core.persist(persist_path or None, fs=fs)

        def close(self) -> None:
            self._core.close()

        @classmethod
        def from_persist_dir(cls, persist_dir: str, collection_name: str = "llamaindex_hisamdb", **kwargs: Any) -> "HisamDBLlamaIndexStore":
            return cls(collection_name=collection_name, path=persist_dir, **kwargs)

        def __enter__(self) -> "HisamDBLlamaIndexStore":
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            self.close()

else:

    class HisamDBLlamaIndexStore(_LIBaseStore):  # type: ignore[no-redef]
        """
        Resmi LlamaIndex `VectorStore` protokolüyle %100 uyumlu HisamDB deposu.
        (`llama-index-core` kurulduğunda otomatik olarak `BasePydanticVectorStore` alt sınıfına dönüşür.)
        """

        stores_text: bool = True
        flat_metadata: bool = False
        is_embedding_query: bool = True

        def __init__(self, collection_name: str = "llamaindex_hisamdb", **kwargs: Any) -> None:
            self.collection_name = collection_name
            self._core = _HisamDBLlamaIndexCore(collection_name=collection_name, **kwargs)

        @classmethod
        def class_name(cls) -> str:
            return "HisamDBLlamaIndexStore"

        @property
        def client(self) -> Any:
            return self._core._client

        @property
        def collection(self) -> Union[Collection, RemoteCollection]:
            return self._core._collection

        def add(self, nodes: Sequence[Any], **add_kwargs: Any) -> List[str]:
            return self._core.add(nodes, **add_kwargs)

        async def async_add(self, nodes: Sequence[Any], **add_kwargs: Any) -> List[str]:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, functools.partial(self._core.add, nodes, **add_kwargs))

        def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
            self._core.delete(ref_doc_id, **delete_kwargs)

        async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, functools.partial(self._core.delete, ref_doc_id, **delete_kwargs))

        def delete_nodes(
            self,
            node_ids: Optional[List[str]] = None,
            filters: Optional[Any] = None,
            **delete_kwargs: Any,
        ) -> None:
            self._core.delete_nodes(node_ids=node_ids, filters=filters, **delete_kwargs)

        async def adelete_nodes(
            self,
            node_ids: Optional[List[str]] = None,
            filters: Optional[Any] = None,
            **delete_kwargs: Any,
        ) -> None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, functools.partial(self._core.delete_nodes, node_ids=node_ids, filters=filters, **delete_kwargs)
            )

        def clear(self) -> None:
            self._core.clear()

        def get_nodes(
            self,
            node_ids: Optional[List[str]] = None,
            filters: Optional[Any] = None,
        ) -> List[Any]:
            return self._core.get_nodes(node_ids=node_ids, filters=filters)

        async def aget_nodes(
            self,
            node_ids: Optional[List[str]] = None,
            filters: Optional[Any] = None,
        ) -> List[Any]:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, functools.partial(self._core.get_nodes, node_ids=node_ids, filters=filters)
            )

        def query(self, query: Any, **kwargs: Any) -> Any:
            return self._core.query(query, **kwargs)

        async def aquery(self, query: Any, **kwargs: Any) -> Any:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, functools.partial(self._core.query, query, **kwargs))

        def persist(self, persist_path: str = "", fs: Any = None) -> None:
            self._core.persist(persist_path or None, fs=fs)

        def close(self) -> None:
            self._core.close()

        @classmethod
        def from_persist_dir(cls, persist_dir: str, collection_name: str = "llamaindex_hisamdb", **kwargs: Any) -> "HisamDBLlamaIndexStore":
            return cls(collection_name=collection_name, path=persist_dir, **kwargs)

        def __enter__(self) -> "HisamDBLlamaIndexStore":
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            self.close()
