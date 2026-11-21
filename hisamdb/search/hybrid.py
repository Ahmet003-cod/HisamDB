"""
hisamdb.search.hybrid
---------------------------
Hibrit Arama (Hybrid Search) ve Füzyon Motoru.

Dense vektör skorları (ör. kosinüs [-1, 1]) ile Sparse/BM25 skorları (ör. [0, 30+])
farklı ölçeklerde olduğundan doğrudan toplanmamalıdır. Bu modül üç bilimsel yöntem sunar:
1. Reciprocal Rank Fusion (RRF): Ölçekten tamamen bağımsız sıra tabanlı füzyon (varsayılan).
2. Min-Max Ağırlıklı Füzyon: Her sonuç listesini [0, 1] aralığına ölçekleyip alpha/beta ile toplar.
3. Z-Score (Sigmoid) Ağırlıklı Füzyon: Aykırı skorlara dayanıklı istatistiksel normalizasyon.
"""

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np


def normalize_scores(
    scores: Sequence[float],
    method: str = "min_max",
) -> List[float]:
    """Bir skor listesini [0, 1] aralığına normalize eder."""
    n = len(scores)
    if n == 0:
        return []
    arr = np.asarray(scores, dtype=np.float64)

    if method == "min_max":
        mn = float(arr.min())
        mx = float(arr.max())
        span = mx - mn
        if span <= 1e-12:
            return [1.0 if float(x) > 0.0 else 0.0 for x in arr]
        return [float((x - mn) / span) for x in arr]

    if method == "z_score":
        mean = float(arr.mean())
        std = float(arr.std())
        if std <= 1e-12:
            return [0.5] * n
        # Sigmoid((x - mean) / std) -> (0, 1)
        return [1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, (float(x) - mean) / std)))) for x in arr]

    raise ValueError(f"Desteklenmeyen normalizasyon yöntemi: '{method}'. Desteklenenler: 'min_max', 'z_score'")


def fuse_rrf(
    ranked_lists: Sequence[Sequence[int]],
    weights: Optional[Sequence[float]] = None,
    rrf_k: int = 60,
    limit: int = 10,
) -> List[Tuple[int, float]]:
    """
    Reciprocal Rank Fusion (Cormack et al., 2009):
        RRF(d) = sum_i  weight_i / (rrf_k + rank_i(d))
    """
    if not ranked_lists:
        return []
    n_lists = len(ranked_lists)
    w_list = [1.0] * n_lists if weights is None else [float(w) for w in weights]
    if len(w_list) != n_lists:
        raise ValueError("Ağırlık sayısı sıralı liste sayısına eşit olmalıdır.")

    k_const = max(1, int(rrf_k))
    fused: Dict[int, float] = defaultdict(float)

    for r_list, w in zip(ranked_lists, w_list):
        if w <= 0.0:
            continue
        for rank_idx, doc_slot in enumerate(r_list, start=1):
            fused[int(doc_slot)] += w / (k_const + rank_idx)

    ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return ordered[:limit] if limit > 0 else []


def fuse_weighted(
    scored_lists: Sequence[Sequence[Tuple[int, float]]],
    weights: Optional[Sequence[float]] = None,
    normalization: str = "min_max",
    limit: int = 10,
) -> List[Tuple[int, float]]:
    """
    Normalize edilmiş ağırlıklı skor füzyonu:
        Score(d) = sum_i weight_i * norm(score_i(d)) / sum(weights)
    """
    if not scored_lists:
        return []
    n_lists = len(scored_lists)
    w_list = [1.0] * n_lists if weights is None else [float(w) for w in weights]
    if len(w_list) != n_lists:
        raise ValueError("Ağırlık sayısı skor listesi sayısına eşit olmalıdır.")

    w_sum = sum(w for w in w_list if w > 0.0)
    if w_sum <= 0.0:
        return []

    fused: Dict[int, float] = defaultdict(float)
    for s_list, w in zip(scored_lists, w_list):
        if w <= 0.0 or not s_list:
            continue
        normed = normalize_scores([s for _, s in s_list], method=normalization)
        w_eff = w / w_sum
        for (doc_slot, _), ns in zip(s_list, normed):
            fused[int(doc_slot)] += w_eff * ns

    ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return ordered[:limit] if limit > 0 else []
