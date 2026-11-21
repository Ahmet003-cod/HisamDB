"""
hisamdb.search.rag
------------------------
Retrieval-Augmented Generation (RAG) ve Çok Kiracılı (Multi-Tenant) Yardımcıları:
- Metin parçalama (`chunk_text`)
- Harici Reranker entegrasyon protokolü (`apply_reranker` — çekirdeği ML modellerine bağlamaz!)
- Sonuç tekilleştirme (`deduplicate_results`: metin hash + kosinüs benzerlik eşiği)
- Doküman gruplama (`group_results_by_field`: tek bir PDF'in tüm Top-K'yı doldurmasını engeller)
- Çok kiracılı (Multi-Tenant) güvenli koleksiyon sarmalayıcısı (`TenantCollection`)
"""

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from hisamdb.types import (
    BatchVectors,
    Metadata,
    QueryResult,
    SearchResultItem,
    VectorLike,
)


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    separator: Optional[str] = None,
) -> List[str]:
    """
    Uzun bir dokümanı örtüşmeli (overlapping) RAG parçalarına (chunk) böler.
    """
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size > 0 olmalıdır.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap [0, chunk_size) aralığında olmalıdır.")

    if separator:
        parts = [p.strip() for p in text.split(separator) if p.strip()]
        chunks: List[str] = []
        current = ""
        for p in parts:
            candidate = f"{current}{separator}{p}" if current else p
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = p
        if current:
            chunks.append(current)
        return chunks

    chunks = []
    step = chunk_size - chunk_overlap
    n = len(text)
    for start in range(0, n, step):
        piece = text[start : start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        if start + chunk_size >= n:
            break
    return chunks


def apply_reranker(
    query_text: str,
    items: List[SearchResultItem],
    reranker: Any,
    limit: Optional[int] = None,
) -> List[SearchResultItem]:
    """
    Harici bir reranker modeli veya fonksiyonu ile aday sonuçları yeniden sıralar.
    Çekirdek veritabanını hiçbir harici ML kütüphanesine bağımlı kılmaz.

    Desteklenen reranker arayüzleri:
    1. Callable: `fn(query: str, items: List[SearchResultItem]) -> List[float] | List[Tuple[int, float]]`
    2. Nesne metodu: `obj.rerank(query: str, documents: List[str]) -> List[float] | List[Tuple[int, float]]`
    """
    if not items or reranker is None:
        return items[:limit] if limit is not None else items

    if hasattr(reranker, "rerank") and callable(reranker.rerank):
        docs = [it.document or "" for it in items]
        raw_out = reranker.rerank(query_text, docs)
    elif callable(reranker):
        raw_out = reranker(query_text, items)
    else:
        raise TypeError("reranker çağrılabilir bir fonksiyon veya .rerank(query, docs) metoduna sahip nesne olmalıdır.")

    if not raw_out:
        return []

    first = raw_out[0]
    scored_items: List[SearchResultItem] = []

    if isinstance(first, (tuple, list)) and len(first) == 2:
        # [(orig_idx, new_score), ...]
        for idx, new_score in raw_out:
            it = items[int(idx)]
            scored_items.append(
                SearchResultItem(
                    id=it.id,
                    score=float(new_score),
                    distance=it.distance,
                    document=it.document,
                    metadata=it.metadata,
                )
            )
    else:
        # [score_0, score_1, ...]
        for it, new_score in zip(items, raw_out):
            scored_items.append(
                SearchResultItem(
                    id=it.id,
                    score=float(new_score),
                    distance=it.distance,
                    document=it.document,
                    metadata=it.metadata,
                )
            )

    scored_items.sort(key=lambda x: x.score, reverse=True)
    return scored_items[:limit] if limit is not None else scored_items


def deduplicate_results(
    items: List[SearchResultItem],
    vectors_by_id: Optional[Dict[str, np.ndarray]] = None,
    similarity_threshold: float = 0.95,
    limit: Optional[int] = None,
) -> List[SearchResultItem]:
    """
    Birebir aynı metinleri ve (vektörler verildiyse) kosinüs benzerliği
    `similarity_threshold` üzerinde olan neredeyse özdeş sonuçları tekilleştirir.
    """
    if not items:
        return []

    kept: List[SearchResultItem] = []
    seen_texts = set()
    kept_unit_vecs: List[np.ndarray] = []

    for it in items:
        norm_text = " ".join(it.document.lower().split()) if it.document else None
        if norm_text is not None and norm_text in seen_texts:
            continue

        unit_v: Optional[np.ndarray] = None
        if vectors_by_id is not None and it.id in vectors_by_id:
            v = np.asarray(vectors_by_id[it.id], dtype=np.float32).ravel()
            nrm = float(np.linalg.norm(v))
            if nrm > 0.0:
                unit_v = v / np.float32(nrm)
                if any(float(np.dot(unit_v, prev)) >= similarity_threshold for prev in kept_unit_vecs):
                    continue

        if norm_text is not None:
            seen_texts.add(norm_text)
        if unit_v is not None:
            kept_unit_vecs.append(unit_v)
        kept.append(it)
        if limit is not None and len(kept) >= limit:
            break

    return kept


def group_results_by_field(
    items: List[SearchResultItem],
    group_by: str,
    max_per_group: int = 1,
    limit: Optional[int] = None,
) -> List[SearchResultItem]:
    """
    Sonuçları belirli bir metadata alanına (ör. `doc_id` veya `_doc_id`) göre gruplayarak
    her dokümandan en fazla `max_per_group` parça döndürür.
    """
    if not items or max_per_group <= 0:
        return []
    counts: Dict[Any, int] = defaultdict(int)
    grouped: List[SearchResultItem] = []

    for it in items:
        gval = it.metadata.get(group_by) if it.metadata else None
        if gval is not None:
            if counts[gval] >= max_per_group:
                continue
            counts[gval] += 1
        grouped.append(it)
        if limit is not None and len(grouped) >= limit:
            break
    return grouped


class TenantCollection:
    """
    Çok kiracılı (Multi-Tenant) güvenli koleksiyon sarmalayıcısı.
    Tüm yazma, okuma, arama ve silme işlemlerini otomatik olarak `_tenant_id`
    filtresiyle yalıtır — Kiracı A, Kiracı B'nin verisini asla göremez veya silemez.
    """

    TENANT_FIELD = "_tenant_id"

    def __init__(self, collection: Any, tenant_id: str) -> None:
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("tenant_id boş olmayan bir metin (str) olmalıdır.")
        self._collection = collection
        self.tenant_id = tenant_id

    def _scope_filter(self, flt: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        tenant_cond = {self.TENANT_FIELD: self.tenant_id}
        if not flt:
            return tenant_cond
        return {"$and": [tenant_cond, flt]}

    def _tag_metadatas(self, count: int, metadatas: Optional[List[Metadata]]) -> List[Metadata]:
        out: List[Metadata] = []
        for i in range(count):
            base = dict(metadatas[i]) if (metadatas is not None and metadatas[i] is not None) else {}
            base[self.TENANT_FIELD] = self.tenant_id
            out.append(base)
        return out

    def add(
        self,
        ids: List[str],
        embeddings: Optional[BatchVectors] = None,
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Metadata]] = None,
        *,
        vectors: Optional[BatchVectors] = None,
        metadata: Optional[List[Metadata]] = None,
        sparse_vectors: Optional[List[Optional[Dict[Union[int, str], float]]]] = None,
    ) -> None:
        raw_metas = metadatas if metadatas is not None else metadata
        tagged = self._tag_metadatas(len(ids), raw_metas)
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=tagged,
            vectors=vectors,
            sparse_vectors=sparse_vectors,
        )

    def add_documents(
        self,
        documents: List[str],
        embeddings: BatchVectors,
        metadatas: Optional[List[Metadata]] = None,
        ids: Optional[List[str]] = None,
        doc_ids: Optional[List[str]] = None,
        parent_ids: Optional[List[str]] = None,
        sparse_vectors: Optional[List[Optional[Dict[Union[int, str], float]]]] = None,
    ) -> List[str]:
        return self._collection.add_documents(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
            doc_ids=doc_ids,
            parent_ids=parent_ids,
            sparse_vectors=sparse_vectors,
            tenant_id=self.tenant_id,
        )

    def query(
        self,
        query_embeddings: Optional[Union[VectorLike, BatchVectors]] = None,
        query_embedding: Optional[Union[VectorLike, BatchVectors]] = None,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> QueryResult:
        return self._collection.query(
            query_embeddings=query_embeddings,
            query_embedding=query_embedding,
            n_results=n_results,
            where=self._scope_filter(where),
            **kwargs,
        )

    def search(
        self,
        query_vector: VectorLike,
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[SearchResultItem]:
        return self._collection.search(
            query_vector=query_vector,
            limit=limit,
            filter=self._scope_filter(filter),
            **kwargs,
        )

    def hybrid_search(
        self,
        query_vector: Optional[VectorLike] = None,
        query_text: Optional[str] = None,
        query_sparse: Optional[Dict[Union[int, str], float]] = None,
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[SearchResultItem]:
        return self._collection.hybrid_search(
            query_vector=query_vector,
            query_text=query_text,
            query_sparse=query_sparse,
            limit=limit,
            filter=self._scope_filter(filter),
            **kwargs,
        )

    def get(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self._collection.get(ids=ids, where=self._scope_filter(where), limit=limit)

    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> int:
        if ids is not None:
            # Yalnızca bu kiracıya ait olan ID'lerin silinmesine izin ver!
            owned = set(self.get(ids=ids)["ids"])
            if not owned:
                return 0
            ids = list(owned)
        if where is not None:
            where = self._scope_filter(where)
        elif ids is None:
            raise ValueError("Silme için ya 'ids' ya da 'where' filtresi belirtilmelidir.")
        return self._collection.delete(ids=ids, where=where)

    def count(self) -> int:
        return len(self.get()["ids"])

    def __repr__(self) -> str:
        return f"<TenantCollection(collection='{self._collection.name}', tenant='{self.tenant_id}')>"
