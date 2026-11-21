"""
microvectordb.filtering.metadata_index
--------------------------------------
Ters indeks (Inverted Index) + tembel (lazy) sayısal sütun önbellekli
yüksek hızlı metadata filtre motoru.

Desteklenen operatörler:
- Doğrudan eşitlik: {"kategori": "ai"}
- Karşılaştırma: $eq, $ne, $gt, $gte, $lt, $lte
- Küme üyeliği:  $in, $nin, $contains
- Varlık:        $exists
- Mantıksal:     $and, $or, $not
- İç içe (Nested) anahtarlar: {"yazar.ulke": "TR"} veya {"yazar": {"ulke": "TR"}}

Karma tip güvenliği:
Sayısal karşılaştırmalarda ($gt, $gte, $lt, $lte) uyumsuz tipler (ör. str vs int)
TypeError fırlatmak yerine güvenli biçimde False döndürür.
"""

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import numpy as np

from microvectordb.types import Metadata

_OPERATORS = frozenset(
    {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin", "$exists", "$contains"}
)
_LOGICAL_OPS = frozenset({"$and", "$or", "$not"})


def _is_number(val: Any) -> bool:
    """bool hariç int veya float olup olmadığını kontrol eder."""
    return isinstance(val, (int, float)) and not isinstance(val, bool)


def _canonical_key(val: Any) -> Optional[Tuple[str, Any]]:
    """
    Python'da True == 1 ve False == 0 çakışmasını engellemek için
    tür-etiketli hashlenebilir anahtar üretir.
    """
    if val is None:
        return ("0", None)
    if isinstance(val, bool):
        return ("b", val)
    if isinstance(val, (int, float)):
        # int ve float eşitliği korunsun (ör. 10 == 10.0)
        return ("n", float(val))
    if isinstance(val, str):
        return ("s", val)
    if isinstance(val, tuple):
        try:
            hash(val)
            return ("t", val)
        except TypeError:
            return None
    return None


def _flatten_metadata(meta: Dict[str, Any], prefix: str = "") -> Iterable[Tuple[str, Any]]:
    """İç içe sözlükleri nokta-ayrılmış yollara ('a.b') düzleştirir."""
    for k, v in meta.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            yield path, v
            yield from _flatten_metadata(v, path)
        else:
            yield path, v


def _resolve_field(meta: Optional[Metadata], key: str) -> Tuple[bool, Any]:
    """
    Metadata sözlüğünden düz veya nokta-ayrılmış ('author.country') alan değerini çözer.
    Dönüş: (found, value)
    """
    if meta is None:
        return False, None
    if key in meta:
        return True, meta[key]
    if "." in key:
        cur: Any = meta
        for part in key.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return False, None
        return True, cur
    return False, None


def _safe_compare(val: Any, target: Any, op: str) -> bool:
    """Karma tiplerde çökmeden güvenli sıralama karşılaştırması yapar."""
    if val is None or target is None:
        return False
    # bool ile sayısal karşılaştırmayı engelle
    if isinstance(val, bool) != isinstance(target, bool):
        return False
    if _is_number(val) and _is_number(target):
        pass
    elif type(val) is not type(target):
        return False

    try:
        if op == "$gt":
            return bool(val > target)
        if op == "$gte":
            return bool(val >= target)
        if op == "$lt":
            return bool(val < target)
        if op == "$lte":
            return bool(val <= target)
    except TypeError:
        return False
    return False


def _values_equal(a: Any, b: Any) -> bool:
    """bool vs int karışmasını önleyen güvenli eşitlik."""
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    return bool(a == b)


def _match_field_condition(found: bool, val: Any, condition: Any) -> bool:
    """Tek bir alan için koşulu değerlendirir."""
    if isinstance(condition, dict) and any(k in _OPERATORS for k in condition):
        for op, target in condition.items():
            if op == "$exists":
                if bool(target) != (found and val is not None):
                    return False
            elif op == "$eq":
                if not found or not _values_equal(val, target):
                    return False
            elif op == "$ne":
                if found and _values_equal(val, target):
                    return False
            elif op in ("$gt", "$gte", "$lt", "$lte"):
                if not found or not _safe_compare(val, target, op):
                    return False
            elif op == "$in":
                if not found or not isinstance(target, (list, tuple, set, frozenset)):
                    return False
                if not any(_values_equal(val, t) for t in target):
                    return False
            elif op == "$nin":
                if isinstance(target, (list, tuple, set, frozenset)):
                    if found and any(_values_equal(val, t) for t in target):
                        return False
            elif op == "$contains":
                if not found or val is None:
                    return False
                if isinstance(val, str):
                    if not isinstance(target, str) or target not in val:
                        return False
                elif isinstance(val, (list, tuple, set)):
                    if not any(_values_equal(elem, target) for elem in val):
                        return False
                else:
                    return False
            else:
                raise ValueError(f"Desteklenmeyen filtre operatörü: '{op}'")
        return True

    # İç içe sözlük filtresi: {"author": {"country": "TR"}}
    if isinstance(condition, dict):
        if not found or not isinstance(val, dict):
            return False
        return matches_filter(val, condition)

    # Doğrudan eşitlik
    return found and _values_equal(val, condition)


def matches_filter(item_meta: Optional[Metadata], where: Optional[Dict[str, Any]]) -> bool:
    """
    Tek bir metadata sözlüğünün 'where' filtresine uyup uymadığını kontrol eder.
    """
    if not where:
        return True

    for key, condition in where.items():
        if key == "$and":
            if not isinstance(condition, list):
                raise ValueError("'$and' operatörü liste bekler.")
            if not all(matches_filter(item_meta, sub) for sub in condition):
                return False
        elif key == "$or":
            if not isinstance(condition, list):
                raise ValueError("'$or' operatörü liste bekler.")
            if not any(matches_filter(item_meta, sub) for sub in condition):
                return False
        elif key == "$not":
            if not isinstance(condition, dict):
                raise ValueError("'$not' operatörü sözlük bekler.")
            if item_meta is None:
                return False
            if matches_filter(item_meta, condition):
                return False
        else:
            if item_meta is None:
                # Yalnızca $exists: False durumunda None metadata eşleşebilir
                if isinstance(condition, dict) and list(condition.keys()) == ["$exists"] and condition["$exists"] is False:
                    continue
                return False
            found, val = _resolve_field(item_meta, key)
            if not _match_field_condition(found, val, condition):
                return False

    return True


class MetadataIndex:
    """
    Koleksiyon metadata'sı için bellek içi ters indeks + tembel sayısal sütun önbelleği.

    Zaman karmaşıklığı:
    - Eşitlik ($eq, $in):  O(1) kova erişimi + önbellekli sıralı NumPy dizi dönüşü
    - Sayısal aralık ($gt/$gte/$lt/$lte): NumPy vektörize maske (~0.15ms / 100K)
    - Mantıksal ($and/$or/$not): Kümeler veya sıralı NumPy indeks kesişim/birleşimleri
    """

    def __init__(self) -> None:
        self._size: int = 0
        self._metadatas: List[Optional[Metadata]] = []
        # field -> canonical_key -> set[row_idx]
        self._eq_index: Dict[str, Dict[Tuple[str, Any], Set[int]]] = defaultdict(lambda: defaultdict(set))
        # field -> set[row_idx] (alanı mevcut ve None olmayan satırlar)
        self._field_present: Dict[str, Set[int]] = defaultdict(set)
        # field -> canonical_elem_key -> set[row_idx] (liste üyeliği için $contains)
        self._list_index: Dict[str, Dict[Tuple[str, Any], Set[int]]] = defaultdict(lambda: defaultdict(set))
        # metadata'sı None olan satırlar ve tombstone (silinmiş) slotlar
        self._null_meta: Set[int] = set()
        self._deleted_slots: Set[int] = set()
        # Önbellekler (yazma anında ilgili alan temizlenir)
        self._posting_cache: Dict[Tuple[str, Tuple[str, Any]], np.ndarray] = {}
        self._numeric_col_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def _invalidate_field_caches(self, field: Optional[str] = None) -> None:
        if field is None:
            self._posting_cache.clear()
            self._numeric_col_cache.clear()
        else:
            self._numeric_col_cache.pop(field, None)
            if self._posting_cache:
                stale = [k for k in self._posting_cache if k[0] == field]
                for k in stale:
                    del self._posting_cache[k]

    def _index_row(self, idx: int, meta: Optional[Metadata]) -> None:
        if meta is None:
            self._null_meta.add(idx)
            return
        self._null_meta.discard(idx)
        for field, val in _flatten_metadata(meta):
            if val is not None:
                self._field_present[field].add(idx)
            ck = _canonical_key(val)
            if ck is not None:
                self._eq_index[field][ck].add(idx)
            elif isinstance(val, (list, tuple, set)):
                for elem in val:
                    eck = _canonical_key(elem)
                    if eck is not None:
                        self._list_index[field][eck].add(idx)

    def _deindex_row(self, idx: int, old_meta: Optional[Metadata]) -> None:
        if old_meta is None:
            self._null_meta.discard(idx)
            return
        for field, val in _flatten_metadata(old_meta):
            self._invalidate_field_caches(field)
            if val is not None:
                present = self._field_present.get(field)
                if present is not None:
                    present.discard(idx)
            ck = _canonical_key(val)
            if ck is not None:
                buckets = self._eq_index.get(field)
                if buckets is not None:
                    b = buckets.get(ck)
                    if b is not None:
                        b.discard(idx)
                        if not b:
                            del buckets[ck]
            elif isinstance(val, (list, tuple, set)):
                lbuckets = self._list_index.get(field)
                if lbuckets is not None:
                    for elem in val:
                        eck = _canonical_key(elem)
                        if eck is not None:
                            lb = lbuckets.get(eck)
                            if lb is not None:
                                lb.discard(idx)
                                if not lb:
                                    del lbuckets[eck]

    def add_batch(self, metadatas: List[Optional[Metadata]]) -> None:
        """Yeni eklenen satırları artımlı olarak indeksler."""
        start = self._size
        touched_fields: Set[str] = set()
        for offset, meta in enumerate(metadatas):
            idx = start + offset
            self._metadatas.append(meta)
            self._index_row(idx, meta)
            if meta is not None:
                for field, _ in _flatten_metadata(meta):
                    touched_fields.add(field)
        self._size += len(metadatas)
        # Sadece dokunulan alanların sayısal sütun & posting önbelleğini temizle
        # (Not: sayısal sütun boyutları arttığı için tüm numeric_col_cache temizlenir)
        self._numeric_col_cache.clear()
        if touched_fields and self._posting_cache:
            stale = [k for k in self._posting_cache if k[0] in touched_fields]
            for k in stale:
                del self._posting_cache[k]

    def update(self, idx: int, new_meta: Optional[Metadata]) -> None:
        """Tek bir satırın metadata indeksini O(alan_sayısı) sürede günceller."""
        old_meta = self._metadatas[idx]
        self._deindex_row(idx, old_meta)
        self._deleted_slots.discard(idx)
        self._metadatas[idx] = new_meta
        self._index_row(idx, new_meta)
        if new_meta is not None:
            for field, _ in _flatten_metadata(new_meta):
                self._invalidate_field_caches(field)

    def remove(self, idx: int) -> None:
        """Tek bir silinmiş (tombstone) slotu O(alan_sayısı) sürede indeksten çıkarır."""
        if 0 <= idx < self._size:
            old_meta = self._metadatas[idx]
            self._deindex_row(idx, old_meta)
            self._metadatas[idx] = None
            self._null_meta.discard(idx)
            self._deleted_slots.add(idx)

    def rebuild(
        self,
        metadatas: List[Optional[Metadata]],
        deleted_slots: Optional[Set[int]] = None,
    ) -> None:
        """Silme/sıkıştırma sonrası indeksi sıfırdan kurar."""
        self._eq_index.clear()
        self._field_present.clear()
        self._list_index.clear()
        self._null_meta.clear()
        self._deleted_slots = set(deleted_slots) if deleted_slots else set()
        self._posting_cache.clear()
        self._numeric_col_cache.clear()
        self._metadatas = list(metadatas)
        self._size = len(metadatas)
        for idx, meta in enumerate(metadatas):
            if idx in self._deleted_slots:
                continue
            self._index_row(idx, meta)

    def _get_numeric_column(self, field: str) -> Tuple[np.ndarray, np.ndarray]:
        """Bir sayısal alan için (values_f64, valid_mask) önbellekli sütun dizisi döndürür."""
        cached = self._numeric_col_cache.get(field)
        if cached is not None:
            return cached

        vals = np.empty(self._size, dtype=np.float64)
        valid = np.zeros(self._size, dtype=bool)
        present = self._field_present.get(field)
        if present:
            for idx in present:
                meta = self._metadatas[idx]
                found, v = _resolve_field(meta, field)
                if found and _is_number(v):
                    vals[idx] = float(v)
                    valid[idx] = True

        self._numeric_col_cache[field] = (vals, valid)
        return vals, valid

    def _get_posting_array(self, field: str, ck: Tuple[str, Any]) -> np.ndarray:
        """Eşitlik kovası için önbellekli sıralı int64 NumPy dizisi döndürür."""
        cache_key = (field, ck)
        cached = self._posting_cache.get(cache_key)
        if cached is not None:
            return cached

        buckets = self._eq_index.get(field)
        if not buckets or ck not in buckets:
            return np.empty(0, dtype=np.int64)
        s = buckets[ck]
        arr = np.fromiter(s, dtype=np.int64, count=len(s))
        arr.sort()
        self._posting_cache[cache_key] = arr
        return arr

    def _eval_to_set(self, where: Dict[str, Any]) -> Set[int]:
        """Filtreyi eşleşen satır indekslerinin kümesi (Set[int]) olarak değerlendirir."""
        if not where:
            return set(range(self._size)) - self._deleted_slots if self._deleted_slots else set(range(self._size))

        all_non_null: Optional[Set[int]] = None

        def get_all_non_null() -> Set[int]:
            nonlocal all_non_null
            if all_non_null is None:
                if not self._null_meta and not self._deleted_slots:
                    all_non_null = set(range(self._size))
                else:
                    all_non_null = set(range(self._size)) - self._null_meta - self._deleted_slots
            return all_non_null

        result: Optional[Set[int]] = None

        for key, condition in where.items():
            part: Set[int]

            if key == "$and":
                if not isinstance(condition, list):
                    raise ValueError("'$and' operatörü liste bekler.")
                if not condition:
                    part = set(range(self._size)) - self._deleted_slots if self._deleted_slots else set(range(self._size))
                else:
                    sub_sets = [self._eval_to_set(sub) for sub in condition]
                    sub_sets.sort(key=len)
                    part = set(sub_sets[0])
                    for s in sub_sets[1:]:
                        part &= s
                        if not part:
                            break
            elif key == "$or":
                if not isinstance(condition, list):
                    raise ValueError("'$or' operatörü liste bekler.")
                part = set()
                for sub in condition:
                    part |= self._eval_to_set(sub)
            elif key == "$not":
                if not isinstance(condition, dict):
                    raise ValueError("'$not' operatörü sözlük bekler.")
                sub_match = self._eval_to_set(condition)
                part = get_all_non_null() - sub_match
            else:
                part = self._eval_field_to_set(key, condition, get_all_non_null)

            if result is None:
                result = part
            else:
                result = result & part
            if not result:
                return set()

        return result if result is not None else set(range(self._size))

    def _eval_field_to_set(
        self,
        field: str,
        condition: Any,
        get_all_non_null: Any,
    ) -> Set[int]:
        # Operatörlü koşul sözlüğü
        if isinstance(condition, dict) and any(k in _OPERATORS for k in condition):
            part: Optional[Set[int]] = None
            for op, target in condition.items():
                op_set: Set[int]
                if op == "$eq":
                    ck = _canonical_key(target)
                    if ck is not None:
                        op_set = set(self._eq_index.get(field, {}).get(ck, ()))
                    else:
                        op_set = {
                            idx for idx in get_all_non_null()
                            if _match_field_condition(*_resolve_field(self._metadatas[idx], field), {"$eq": target})
                        }
                elif op == "$ne":
                    ck = _canonical_key(target)
                    if ck is not None:
                        excl = self._eq_index.get(field, {}).get(ck, set())
                        op_set = get_all_non_null() - excl
                    else:
                        op_set = {
                            idx for idx in get_all_non_null()
                            if _match_field_condition(*_resolve_field(self._metadatas[idx], field), {"$ne": target})
                        }
                elif op == "$in":
                    if not isinstance(target, (list, tuple, set, frozenset)):
                        return set()
                    op_set = set()
                    buckets = self._eq_index.get(field, {})
                    unhashable = []
                    for t in target:
                        ck = _canonical_key(t)
                        if ck is not None:
                            b = buckets.get(ck)
                            if b:
                                op_set |= b
                        else:
                            unhashable.append(t)
                    if unhashable:
                        for idx in get_all_non_null():
                            if _match_field_condition(*_resolve_field(self._metadatas[idx], field), {"$in": unhashable}):
                                op_set.add(idx)
                elif op == "$nin":
                    if not isinstance(target, (list, tuple, set, frozenset)):
                        op_set = set(get_all_non_null())
                    else:
                        excl = set()
                        buckets = self._eq_index.get(field, {})
                        for t in target:
                            ck = _canonical_key(t)
                            if ck is not None:
                                b = buckets.get(ck)
                                if b:
                                    excl |= b
                        op_set = get_all_non_null() - excl
                elif op in ("$gt", "$gte", "$lt", "$lte"):
                    if _is_number(target):
                        vals, valid = self._get_numeric_column(field)
                        t_f = float(target)
                        if op == "$gt":
                            mask = valid & (vals > t_f)
                        elif op == "$gte":
                            mask = valid & (vals >= t_f)
                        elif op == "$lt":
                            mask = valid & (vals < t_f)
                        else:
                            mask = valid & (vals <= t_f)
                        op_set = set(np.flatnonzero(mask).tolist())
                    else:
                        # String vb. kıyaslanabilir tipler için yalnızca alanı dolu satırlarda ara
                        present = self._field_present.get(field, set())
                        op_set = {
                            idx for idx in present
                            if _safe_compare(_resolve_field(self._metadatas[idx], field)[1], target, op)
                        }
                elif op == "$exists":
                    present = self._field_present.get(field, set())
                    if bool(target):
                        op_set = set(present)
                    else:
                        op_set = set(range(self._size)) - present - self._deleted_slots
                elif op == "$contains":
                    eck = _canonical_key(target)
                    list_matches = set(self._list_index.get(field, {}).get(eck, ())) if eck is not None else set()
                    if isinstance(target, str):
                        # String alt-dize eşleşmeleri için alanı dolu satırları tara
                        present = self._field_present.get(field, set())
                        str_matches = {
                            idx for idx in present
                            if isinstance(_resolve_field(self._metadatas[idx], field)[1], str)
                            and target in _resolve_field(self._metadatas[idx], field)[1]
                        }
                        op_set = list_matches | str_matches
                    else:
                        op_set = list_matches
                else:
                    raise ValueError(f"Desteklenmeyen filtre operatörü: '{op}'")

                if part is None:
                    part = op_set
                else:
                    part &= op_set
                if not part:
                    return set()
            return part if part is not None else set()

        # İç içe sözlük filtresi: {"author": {"country": "TR"}} -> dot-path olarak değerlendir
        if isinstance(condition, dict):
            sub_where = {f"{field}.{k}": v for k, v in condition.items()}
            return self._eval_to_set(sub_where)

        # Doğrudan eşitlik
        ck = _canonical_key(condition)
        if ck is not None:
            return set(self._eq_index.get(field, {}).get(ck, ()))
        return {
            idx for idx in get_all_non_null()
            if _match_field_condition(*_resolve_field(self._metadatas[idx], field), condition)
        }

    def query_indices(self, where: Optional[Dict[str, Any]]) -> np.ndarray:
        """
        Filtreye uyan tüm satır indekslerini sıralı `int64` NumPy dizisi olarak döndürür.
        Tek-alan basit eşitlik ve tek-alan sayısal aralık sorgularında sıfır Python küme
        kopyası yapan hızlı yol (fast path) kullanır.
        """
        if not where or self._size == 0:
            if not self._deleted_slots:
                return np.arange(self._size, dtype=np.int64)
            return np.array([i for i in range(self._size) if i not in self._deleted_slots], dtype=np.int64)

        # HIZLI YOL 1: Tek alanlı basit eşitlik {"cat": "c0"} veya {"cat": {"$eq": "c0"}}
        if len(where) == 1:
            field, cond = next(iter(where.items()))
            if field not in _LOGICAL_OPS:
                if not isinstance(cond, dict):
                    ck = _canonical_key(cond)
                    if ck is not None:
                        return self._get_posting_array(field, ck)
                elif len(cond) == 1:
                    op, target = next(iter(cond.items()))
                    if op == "$eq":
                        ck = _canonical_key(target)
                        if ck is not None:
                            return self._get_posting_array(field, ck)
                    # HIZLI YOL 2: Tek alanlı sayısal aralık {"price": {"$gte": 500}}
                    elif op in ("$gt", "$gte", "$lt", "$lte") and _is_number(target):
                        vals, valid = self._get_numeric_column(field)
                        t_f = float(target)
                        if op == "$gt":
                            return np.flatnonzero(valid & (vals > t_f)).astype(np.int64, copy=False)
                        if op == "$gte":
                            return np.flatnonzero(valid & (vals >= t_f)).astype(np.int64, copy=False)
                        if op == "$lt":
                            return np.flatnonzero(valid & (vals < t_f)).astype(np.int64, copy=False)
                        return np.flatnonzero(valid & (vals <= t_f)).astype(np.int64, copy=False)

        # GENEL YOL: Bileşik/mantıksal filtreler
        matched = self._eval_to_set(where)
        if not matched:
            return np.empty(0, dtype=np.int64)
        arr = np.fromiter(matched, dtype=np.int64, count=len(matched))
        arr.sort()
        return arr

    def stats(self) -> Dict[str, Any]:
        return {
            "indexed_rows": self._size,
            "indexed_fields": len(self._eq_index),
            "cached_postings": len(self._posting_cache),
            "cached_numeric_columns": len(self._numeric_col_cache),
        }
