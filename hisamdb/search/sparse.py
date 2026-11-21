"""
hisamdb.search.sparse
---------------------------
Sıfır ek bağımlılıklı Sparse Vector ve Okapi BM25 ters indeks motoru.

Özellikler:
- Türkçe ve Unicode uyumlu hızlı tokenizer (`İ -> i`, `I -> ı`, aksan/noktalama temizliği).
- Okapi BM25 (`k1=1.5`, `b=0.75`): Artımlı `add_document`, `update_document`,
  O(terim_sayısı) `remove_document` ve `allowed_mask` filtreli arama.
- Yerel Sparse Embedding indeksi (`Dict[Union[int, str], float]`): SPLADE, BGE-M3
  veya TF-IDF seyrek vektörleri için hızlı ters indeksli iç çarpım (dot product) araması.
"""

import math
import re
import threading
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
import numpy as np

SparseVector = Dict[Union[int, str], float]

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_TR_LOWER_MAP = str.maketrans({"İ": "i", "I": "ı", "Ş": "ş", "Ğ": "ğ", "Ü": "ü", "Ö": "ö", "Ç": "ç"})


def tokenize_text(
    text: str,
    stopwords: Optional[Set[str]] = None,
    min_len: int = 1,
) -> List[str]:
    """Türkçe ve Unicode karakterleri doğru küçük harfe çevirip terim listesi döndürür."""
    if not text:
        return []
    lowered = text.translate(_TR_LOWER_MAP).lower()
    tokens = _TOKEN_RE.findall(lowered)
    if min_len > 1:
        tokens = [t for t in tokens if len(t) >= min_len]
    if stopwords:
        tokens = [t for t in tokens if t not in stopwords]
    return tokens


class BM25Index:
    """
    Bellek içi artımlı Okapi BM25 ters indeksi.
    Yalnızca sorgu terimlerini içeren dokümanlara dokunur -> O(eşleşen posting).
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        stopwords: Optional[Iterable[str]] = None,
        min_token_len: int = 1,
    ) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.stopwords: Optional[Set[str]] = set(stopwords) if stopwords else None
        self.min_token_len = max(1, int(min_token_len))

        # term -> {slot_idx: tf}
        self._postings: Dict[str, Dict[int, int]] = defaultdict(dict)
        # slot_idx -> token_count
        self._doc_lengths: Dict[int, int] = {}
        # slot_idx -> list[term] (O(doc_terms) silme/güncelleme için)
        self._doc_terms: Dict[int, List[str]] = {}
        self._total_len: int = 0

        # Tembel vektörize NumPy BM25 önbelleği (25x-40x hızlanma, double-checked thread-safe!)
        self._compiled: bool = False
        self._compile_lock = threading.Lock()
        self._max_slot: int = -1
        self._compiled_postings: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    @property
    def num_docs(self) -> int:
        return len(self._doc_lengths)

    @property
    def vocab_size(self) -> int:
        return len(self._postings)

    def add_document(self, slot_idx: int, text: Optional[str]) -> None:
        if slot_idx in self._doc_lengths:
            self.remove_document(slot_idx)
        if not text:
            return
        tokens = tokenize_text(text, stopwords=self.stopwords, min_len=self.min_token_len)
        if not tokens:
            return
        tf_map = Counter(tokens)
        self._doc_lengths[slot_idx] = len(tokens)
        self._total_len += len(tokens)
        self._doc_terms[slot_idx] = list(tf_map.keys())
        for term, tf in tf_map.items():
            self._postings[term][slot_idx] = tf
        self._compiled = False

    def update_document(self, slot_idx: int, text: Optional[str]) -> None:
        self.remove_document(slot_idx)
        self.add_document(slot_idx, text)

    def remove_document(self, slot_idx: int) -> None:
        old_len = self._doc_lengths.pop(slot_idx, None)
        if old_len is None:
            return
        self._total_len -= old_len
        terms = self._doc_terms.pop(slot_idx, ())
        for term in terms:
            bucket = self._postings.get(term)
            if bucket is not None:
                bucket.pop(slot_idx, None)
                if not bucket:
                    del self._postings[term]
        self._compiled = False

    def rebuild(
        self,
        documents: Sequence[Optional[str]],
        deleted_mask: Optional[np.ndarray] = None,
    ) -> None:
        self._postings.clear()
        self._doc_lengths.clear()
        self._doc_terms.clear()
        self._total_len = 0
        self._compiled = False
        self._compiled_postings.clear()
        self._max_slot = -1
        for slot_idx, doc in enumerate(documents):
            if deleted_mask is not None and slot_idx < len(deleted_mask) and deleted_mask[slot_idx]:
                continue
            if doc:
                self.add_document(slot_idx, doc)

    def _ensure_compiled(self) -> None:
        """
        Okapi BM25 terim ağırlıklarını (sorgudan bağımsız w_{t,d} = IDF * TF*(k1+1)/(TF + dl_norm))
        thread-safe biçimde bir kez vektörize NumPy dizilerine derler.
        """
        if self._compiled:
            return
        with self._compile_lock:
            if self._compiled:
                return
            self._compiled_postings.clear()
            n_docs = len(self._doc_lengths)
            if n_docs == 0:
                self._max_slot = -1
                self._compiled = True
                return

            max_slot = max(self._doc_lengths.keys())
            self._max_slot = max_slot
            avgdl = self._total_len / max(1, n_docs)
            k1 = self.k1
            b = self.b
            k1_p1 = np.float32(k1 + 1.0)

            dl_norm = np.empty(max_slot + 1, dtype=np.float32)
            slots_all = np.fromiter(self._doc_lengths.keys(), dtype=np.int32, count=n_docs)
            dls_all = np.fromiter(self._doc_lengths.values(), dtype=np.float32, count=n_docs)
            dl_norm[slots_all] = np.float32(k1) * (
                np.float32(1.0 - b) + np.float32(b / avgdl) * dls_all
            )

            compiled: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            for term, postings in self._postings.items():
                df = len(postings)
                if df == 0:
                    continue
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                if idf <= 0.0:
                    continue
                p_slots = np.fromiter(postings.keys(), dtype=np.int32, count=df)
                p_tfs = np.fromiter(postings.values(), dtype=np.float32, count=df)
                p_weights = (np.float32(idf) * k1_p1 * p_tfs) / (p_tfs + dl_norm[p_slots])
                compiled[term] = (p_slots, p_weights.astype(np.float32, copy=False))

            self._compiled_postings = compiled
            self._compiled = True

    def search(
        self,
        query_text: str,
        k: int = 10,
        allowed_set: Optional[Set[int]] = None,
        deleted_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vektörize Okapi BM25 araması (<0.5 ms / 10K doküman).
        Dönüş: (slot_indices (int64), bm25_scores (float32)) — yüksek skor daha alakalı.
        """
        n_docs = len(self._doc_lengths)
        if k <= 0 or n_docs == 0 or not query_text:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        q_terms = tokenize_text(query_text, stopwords=self.stopwords, min_len=self.min_token_len)
        if not q_terms:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        self._ensure_compiled()
        if self._max_slot < 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        scores = np.zeros(self._max_slot + 1, dtype=np.float32)
        compiled = self._compiled_postings
        hit_any = False

        q_counts = Counter(q_terms)
        for term, q_tf in q_counts.items():
            entry = compiled.get(term)
            if entry is None:
                continue
            p_slots, p_weights = entry
            hit_any = True
            if q_tf == 1:
                scores[p_slots] += p_weights
            else:
                scores[p_slots] += p_weights * np.float32(q_tf)

        if not hit_any:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        if deleted_mask is not None and len(deleted_mask) > 0:
            n_m = min(len(deleted_mask), len(scores))
            scores[:n_m][deleted_mask[:n_m]] = np.float32(0.0)

        if allowed_set is not None:
            if not allowed_set:
                return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
            allow_arr = np.fromiter(allowed_set, dtype=np.int64, count=len(allowed_set))
            allow_arr = allow_arr[allow_arr < len(scores)]
            if allow_arr.size == 0:
                return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
            cand_vals = scores[allow_arr]
            pos_mask = cand_vals > np.float32(0.0)
            nz_slots = allow_arr[pos_mask]
            nz_vals = cand_vals[pos_mask]
        else:
            nz_slots = np.flatnonzero(scores > np.float32(0.0)).astype(np.int64, copy=False)
            nz_vals = scores[nz_slots]

        n_hits = nz_slots.size
        if n_hits == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        actual_k = min(k, n_hits)
        if actual_k < n_hits:
            part = np.argpartition(-nz_vals, actual_k - 1)[:actual_k]
            top_slots = nz_slots[part]
            top_vals = nz_vals[part]
        else:
            top_slots = nz_slots
            top_vals = nz_vals

        order = np.argsort(-top_vals, kind="stable")
        return top_slots[order].astype(np.int64, copy=False), top_vals[order].astype(np.float32, copy=False)


class SparseVectorIndex:
    """
    Seyrek vektör (Sparse Embedding) ters indeks motoru.
    Her kayıt `{token_id: weight}` sözlüğü olarak saklanır ve ters indeksli
    iç çarpım (dot product) ile aranır.
    """

    def __init__(self) -> None:
        # key_str -> {slot_idx: weight}
        self._postings: Dict[str, Dict[int, float]] = defaultdict(dict)
        self._doc_keys: Dict[int, List[str]] = {}

    @property
    def num_vectors(self) -> int:
        return len(self._doc_keys)

    @staticmethod
    def _norm_key(k: Union[int, str]) -> str:
        return str(k)

    def add(self, slot_idx: int, sparse_vec: Optional[SparseVector]) -> None:
        if slot_idx in self._doc_keys:
            self.remove(slot_idx)
        if not sparse_vec:
            return
        keys: List[str] = []
        for raw_k, w in sparse_vec.items():
            wf = float(w)
            if wf == 0.0 or not math.isfinite(wf):
                continue
            ks = self._norm_key(raw_k)
            self._postings[ks][slot_idx] = wf
            keys.append(ks)
        if keys:
            self._doc_keys[slot_idx] = keys

    def update(self, slot_idx: int, sparse_vec: Optional[SparseVector]) -> None:
        self.remove(slot_idx)
        self.add(slot_idx, sparse_vec)

    def remove(self, slot_idx: int) -> None:
        keys = self._doc_keys.pop(slot_idx, ())
        for ks in keys:
            bucket = self._postings.get(ks)
            if bucket is not None:
                bucket.pop(slot_idx, None)
                if not bucket:
                    del self._postings[ks]

    def rebuild(
        self,
        sparse_vectors: Sequence[Optional[SparseVector]],
        deleted_mask: Optional[np.ndarray] = None,
    ) -> None:
        self._postings.clear()
        self._doc_keys.clear()
        for slot_idx, sv in enumerate(sparse_vectors):
            if deleted_mask is not None and slot_idx < len(deleted_mask) and deleted_mask[slot_idx]:
                continue
            if sv:
                self.add(slot_idx, sv)

    def search(
        self,
        query_sparse: SparseVector,
        k: int = 10,
        allowed_set: Optional[Set[int]] = None,
        deleted_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Seyrek vektör iç çarpım araması. Dönüş: (slot_indices, scores)."""
        if k <= 0 or not query_sparse or not self._doc_keys:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        scores: Dict[int, float] = defaultdict(float)
        for raw_k, qw in query_sparse.items():
            qwf = float(qw)
            if qwf == 0.0 or not math.isfinite(qwf):
                continue
            postings = self._postings.get(self._norm_key(raw_k))
            if not postings:
                continue
            for slot_idx, dw in postings.items():
                if deleted_mask is not None and deleted_mask[slot_idx]:
                    continue
                if allowed_set is not None and slot_idx not in allowed_set:
                    continue
                scores[slot_idx] += qwf * dw

        if not scores:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        idxs = np.array([r[0] for r in ranked], dtype=np.int64)
        vals = np.array([r[1] for r in ranked], dtype=np.float32)
        return idxs, vals
