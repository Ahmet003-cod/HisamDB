"""
microvectordb.indexes.optimizer
-------------------------------
Maliyet Tabanlı Sorgu Planlayıcısı (Cost-Based Query Optimizer).

Deneysel ölçümlere dayanan eşiklerle her sorgu için en hızlı yürütme planını seçer:
1. Seçicilik s = M / N <= 0.05 veya M <= 2048 -> `pre_filter_exact` (100% recall, ~0.2 ms)
2. HNSW aktif + s >= 0.08 -> `hnsw_post_filter` (C++ içinde sıfır GIL callback,
   hnsw_filter_cb'den 2.4x–3.4x daha hızlı! Yetersiz aday dönerse otomatik fallback).
3. HNSW aktif + 0.05 < s < 0.08 -> `hnsw_filter_cb` (graf içi filtreli geçiş).
4. IVF / IVF-PQ / DiskIndex aktif + filtre -> seçiciliğe göre uyarlamalı `nprobe` artışı
   (nprobe_eff = min(C, ceil(nprobe / sqrt(s)))).
5. `use_mmap=True` ve büyük Flat koleksiyon -> sabit RAM bütçeli `disk_stream` bloklu tarama.
6. Uzman kullanıcılar `strategy` veya `filter_strategy` ile manuel plan zorlayabilir.
"""

import math
from typing import Optional
from microvectordb.types import QueryPlan


class QueryOptimizer:
    """Maliyet tabanlı otomatik ve şeffaf sorgu planlayıcısı."""

    SMALL_EXACT_LIMIT = 2048
    LOW_SELECTIVITY_EXACT = 0.05
    HNSW_POST_FILTER_MIN_SEL = 0.08
    MMAP_STREAM_THRESHOLD = 8192

    @classmethod
    def plan(
        cls,
        total_vectors: int,
        dimension: int,
        k: int,
        effective_index: str,
        quantization: str,
        is_mmap: bool,
        has_filter: bool,
        candidate_count: int,
        has_raw_vectors: bool = True,
        hnsw_available: bool = False,
        ivf_ready: bool = False,
        ivf_n_clusters: int = 0,
        ivf_default_nprobe: int = 8,
        ivf_pq_ready: bool = False,
        disk_index_ready: bool = False,
        default_ef_search: int = 64,
        requested_ef_search: Optional[int] = None,
        requested_nprobe: Optional[int] = None,
        strategy: str = "auto",
        filter_strategy: str = "auto",
    ) -> QueryPlan:
        """
        Verilen koleksiyon durumu ve sorgu parametreleri için optimal `QueryPlan` üretir.
        """
        dim = max(1, dimension)

        if total_vectors == 0 or k <= 0 or (has_filter and candidate_count == 0):
            return QueryPlan(
                execution_path="empty",
                filter_strategy="short_circuit" if has_filter else "none",
                total_vectors=total_vectors,
                estimated_candidates=0,
                selectivity=0.0,
                k_requested=k,
                estimated_scan_ops=0,
                reason="Boş koleksiyon, k<=0 veya filtreyle eşleşen aday yok (0 işlem).",
            )

        cand = candidate_count if has_filter else total_vectors
        sel = cand / max(1, total_vectors)

        # Uyarlamalı ef_search ve nprobe
        base_ef = requested_ef_search if requested_ef_search is not None else default_ef_search
        eff_ef = max(base_ef, min(total_vectors, k * 2))

        base_np = requested_nprobe if requested_nprobe is not None else ivf_default_nprobe
        if has_filter and requested_nprobe is None and ivf_n_clusters > 0 and sel < 0.5:
            # Filtreli IVF / IVF-PQ / DiskIndex aramalarında Recall'u korumak için nprobe artır
            boost = int(math.ceil(base_np / math.sqrt(max(sel, 0.05))))
            eff_np = min(ivf_n_clusters, max(base_np, boost))
        else:
            eff_np = min(ivf_n_clusters, base_np) if ivf_n_clusters > 0 else base_np

        # Uzman kullanıcı zorlaması: exact
        if strategy == "exact" and has_raw_vectors:
            path = "disk_stream" if (is_mmap and cand >= cls.MMAP_STREAM_THRESHOLD) else "flat_exact"
            return QueryPlan(
                execution_path=path,
                filter_strategy="pre_filter" if has_filter else "none",
                total_vectors=total_vectors,
                estimated_candidates=cand,
                selectivity=sel,
                k_requested=k,
                estimated_scan_ops=cand * dim,
                reason="Kullanıcı tarafından manuel 'exact' stratejisi seçildi.",
            )

        if strategy == "disk_stream" and has_raw_vectors:
            return QueryPlan(
                execution_path="disk_stream",
                filter_strategy="pre_filter" if has_filter else "none",
                total_vectors=total_vectors,
                estimated_candidates=cand,
                selectivity=sel,
                k_requested=k,
                estimated_scan_ops=cand * dim,
                reason="Kullanıcı tarafından sabit-RAM 'disk_stream' taraması seçildi.",
            )

        # 1. Çok küçük filtreli alt-küme -> Pre-filter + Exact BLAS (hem en hızlı hem Recall=1.0)
        if (
            has_filter
            and filter_strategy in ("auto", "pre")
            and has_raw_vectors
            and (cand <= max(cls.SMALL_EXACT_LIMIT, k * 4) or sel < cls.LOW_SELECTIVITY_EXACT)
            and effective_index in ("hnsw", "ivf", "ivf_pq", "disk", "flat")
        ):
            path = "disk_stream" if (is_mmap and cand >= cls.MMAP_STREAM_THRESHOLD) else "flat_exact"
            return QueryPlan(
                execution_path=path,
                filter_strategy="pre_filter_exact",
                total_vectors=total_vectors,
                estimated_candidates=cand,
                selectivity=sel,
                k_requested=k,
                estimated_scan_ops=cand * dim,
                reason=(
                    f"Düşük filtre seçiciliği (M={cand:,}, s={sel:.1%}) -> "
                    f"pre-filter + exact BLAS, ANN callback yükünden çok daha hızlı ve %100 recall."
                ),
            )

        # 2. HNSW yolu
        if effective_index == "hnsw" and hnsw_available:
            if not has_filter:
                est_ops = int(eff_ef * max(1, math.log2(max(2, total_vectors))) * dim)
                return QueryPlan(
                    execution_path="hnsw",
                    filter_strategy="none",
                    total_vectors=total_vectors,
                    estimated_candidates=total_vectors,
                    selectivity=1.0,
                    k_requested=k,
                    adaptive_ef_search=eff_ef,
                    estimated_scan_ops=est_ops,
                    reason=f"Filtresiz HNSW grafik araması (ef_search={eff_ef}).",
                )

            # Filtreli HNSW: deneysel olarak sel >= 8% için post-filter oversample 2.4x–3.4x daha hızlı!
            use_post = (
                filter_strategy == "post"
                or (filter_strategy == "auto" and sel >= cls.HNSW_POST_FILTER_MIN_SEL)
            )
            if use_post:
                k_over = min(total_vectors, max(k * 3, int(math.ceil((k / max(sel, 0.05)) * 2.5))))
                ef_post = max(eff_ef, k_over)
                return QueryPlan(
                    execution_path="hnsw_post_filter",
                    filter_strategy="post_filter_oversample",
                    total_vectors=total_vectors,
                    estimated_candidates=cand,
                    selectivity=sel,
                    k_requested=k,
                    adaptive_ef_search=ef_post,
                    oversample_k=k_over,
                    estimated_scan_ops=ef_post * dim,
                    reason=(
                        f"Orta/yüksek seçicilik (s={sel:.1%}) -> C++ içinde sıfır-GIL HNSW "
                        f"post-filter oversampling (k_over={k_over}, yetersizse otomatik fallback)."
                    ),
                )

            return QueryPlan(
                execution_path="hnsw_filter_cb",
                filter_strategy="index_integrated_callback",
                total_vectors=total_vectors,
                estimated_candidates=cand,
                selectivity=sel,
                k_requested=k,
                adaptive_ef_search=eff_ef,
                estimated_scan_ops=int(eff_ef / max(sel, 0.05) * dim),
                reason=f"HNSW graf-içi filtre callback araması (s={sel:.1%}, ef_search={eff_ef}).",
            )

        # 3. DiskIndex (Bölümlenmiş MMap Disk İndeksi)
        if effective_index == "disk" and disk_index_ready:
            est_scan = int((eff_np / max(1, ivf_n_clusters)) * total_vectors)
            return QueryPlan(
                execution_path="disk_partitioned",
                filter_strategy="cluster_bitmask" if has_filter else "none",
                total_vectors=total_vectors,
                estimated_candidates=cand,
                selectivity=sel,
                k_requested=k,
                adaptive_nprobe=eff_np,
                estimated_scan_ops=est_scan * dim,
                reason=f"Bölümlenmiş DiskIndex (mmap, nprobe={eff_np}/{ivf_n_clusters}, düşük RAM).",
            )

        # 4. IVF-PQ (Residual Product Quantization)
        if effective_index == "ivf_pq" and ivf_pq_ready:
            est_scan = int((eff_np / max(1, ivf_n_clusters)) * total_vectors)
            return QueryPlan(
                execution_path="ivf_pq",
                filter_strategy="cluster_bitmask" if has_filter else "none",
                total_vectors=total_vectors,
                estimated_candidates=cand,
                selectivity=sel,
                k_requested=k,
                adaptive_nprobe=eff_np,
                estimated_scan_ops=est_scan * max(1, dim // 8),
                reason=f"IVF-PQ artık ADC araması (nprobe={eff_np}/{ivf_n_clusters}, 16x–32x düşük RAM).",
            )

        # 5. IVF (Packed BLAS)
        if effective_index == "ivf" and ivf_ready:
            est_scan = int((eff_np / max(1, ivf_n_clusters)) * total_vectors)
            return QueryPlan(
                execution_path="ivf",
                filter_strategy="cluster_bitmask" if has_filter else "none",
                total_vectors=total_vectors,
                estimated_candidates=cand,
                selectivity=sel,
                k_requested=k,
                adaptive_nprobe=eff_np,
                estimated_scan_ops=est_scan * dim,
                reason=f"Bitişik bloklu IVF BLAS araması (nprobe={eff_np}/{ivf_n_clusters}).",
            )

        # 6. Kuantize Flat yolları (PQ, INT8, Binary)
        if quantization == "pq":
            return QueryPlan(
                execution_path="quantized_pq",
                filter_strategy="pre_filter" if has_filter else "none",
                total_vectors=total_vectors,
                estimated_candidates=cand,
                selectivity=sel,
                k_requested=k,
                estimated_scan_ops=cand * max(1, dim // 8),
                reason="Flat Product Quantization (PQ) sütun-bitişik ADC taraması.",
            )
        if quantization == "int8":
            return QueryPlan(
                execution_path="quantized_int8",
                filter_strategy="pre_filter" if has_filter else "none",
                total_vectors=total_vectors,
                estimated_candidates=cand,
                selectivity=sel,
                k_requested=k,
                estimated_scan_ops=cand * dim,
                reason="Ölçek-ağırlıklı L2-cache bloklu INT8 ScalarQuantizer araması.",
            )
        if quantization == "binary":
            return QueryPlan(
                execution_path="quantized_binary",
                filter_strategy="pre_filter" if has_filter else "none",
                total_vectors=total_vectors,
                estimated_candidates=cand,
                selectivity=sel,
                k_requested=k,
                estimated_scan_ops=cand * max(1, dim // 64),
                reason="1-bit Binary SIMD POPCNT Hamming araması" + (" + float32 rescore." if has_raw_vectors else "."),
            )

        # 7. Büyük mmap Flat -> Sabit RAM bütçeli akış taraması
        if is_mmap and cand >= cls.MMAP_STREAM_THRESHOLD:
            return QueryPlan(
                execution_path="disk_stream",
                filter_strategy="pre_filter" if has_filter else "none",
                total_vectors=total_vectors,
                estimated_candidates=cand,
                selectivity=sel,
                k_requested=k,
                estimated_scan_ops=cand * dim,
                reason=f"MMap koleksiyon (M={cand:,}) -> sabit bellek bütçeli bloklu disk akış taraması.",
            )

        # 8. Varsayılan: Optimize Exact Flat BLAS
        return QueryPlan(
            execution_path="flat_exact",
            filter_strategy="pre_filter" if has_filter else "none",
            total_vectors=total_vectors,
            estimated_candidates=cand,
            selectivity=sel,
            k_requested=k,
            estimated_scan_ops=cand * dim,
            reason=f"Optimize Exact Flat BLAS ({cand:,} vektör, %100 recall).",
        )
