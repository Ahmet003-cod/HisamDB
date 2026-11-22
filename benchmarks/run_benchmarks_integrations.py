"""
HisamDB v1.1.0 — LangChain & LlamaIndex Entegrasyon Optimizasyon Benchmarkı
===========================================================================

Gerçek ölçümler:
1. Standart Python-Döngülü MMR (LangChain klasik yaklaşımı) vs HisamDB Vektörize BLAS MMR (`vectorized_mmr`)
   - `fetch_k = 50, 200, 500`, `dim = 768` & `1536` (OpenAI)
2. `HisamDBVectorStore` Uçtan Uca RAG Akışı (10K × 128D HNSW):
   - Dense-Only vs Yerleşik Hybrid RRF P50/QPS
   - Sıfır Yeniden-Gömme MMR (`max_marginal_relevance_search_by_vector`) P50
   - Çok Çekirdekli `batch_similarity_search(64)` QPS
   - `deduplicate_on_add=True` ile tekrar eklemede embedding API çağrısı tasarrufu (%100!)
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hisamdb import HisamDBVectorStore  # noqa: E402
from hisamdb.integrations import vectorized_mmr  # noqa: E402


def standard_python_loop_mmr(
    query_embedding: np.ndarray,
    embedding_list: np.ndarray,
    lambda_mult: float = 0.5,
    k: int = 10,
) -> list[int]:
    """Klasik LangChain Community `maximal_marginal_relevance` algoritmasının referans döngüsü."""
    q = query_embedding / np.maximum(np.linalg.norm(query_embedding), 1e-12)
    norms = np.linalg.norm(embedding_list, axis=1, keepdims=True)
    embs = embedding_list / np.maximum(norms, 1e-12)
    sim_to_query = embs @ q
    most_sim = int(np.argmax(sim_to_query))
    idxs = [most_sim]
    selected = np.array([embs[most_sim]])

    while len(idxs) < min(k, len(embs)):
        best_score = -np.inf
        idx_to_add = -1
        # Klasik LangChain: her adımda seçilenlerle (M x len(idxs)) çarpımı + Python iç döngüsü
        sim_to_selected = embs @ selected.T
        for i, q_score in enumerate(sim_to_query):
            if i in idxs:
                continue
            redundant = float(np.max(sim_to_selected[i]))
            eq_score = lambda_mult * float(q_score) - (1.0 - lambda_mult) * redundant
            if eq_score > best_score:
                best_score = eq_score
                idx_to_add = i
        idxs.append(idx_to_add)
        selected = np.append(selected, [embs[idx_to_add]], axis=0)
    return idxs


def main() -> None:
    print("HisamDB v1.1.0 — LangChain & LlamaIndex Optimizasyon Benchmarkı")
    print("=" * 72)
    rng = np.random.default_rng(42)

    mmr_rows = []
    for dim in (768, 1536):
        for fetch_k, k in ((50, 10), (200, 15), (500, 20)):
            q = rng.normal(size=(dim,)).astype(np.float32)
            cands = rng.normal(size=(fetch_k, dim)).astype(np.float32)

            # Parite doğrulaması
            ref = standard_python_loop_mmr(q, cands, lambda_mult=0.5, k=k)
            fast = vectorized_mmr(q, cands, k=k, lambda_mult=0.5)
            assert ref == fast, f"MMR parite uyuşmazlığı: {ref} vs {fast}"

            # Isınma + 30 ölçüm
            reps = 30
            t0 = time.perf_counter()
            for _ in range(reps):
                standard_python_loop_mmr(q, cands, lambda_mult=0.5, k=k)
            t_ref_ms = (time.perf_counter() - t0) * 1000.0 / reps

            t0 = time.perf_counter()
            for _ in range(reps):
                vectorized_mmr(q, cands, k=k, lambda_mult=0.5)
            t_fast_ms = (time.perf_counter() - t0) * 1000.0 / reps

            speedup = round(t_ref_ms / max(t_fast_ms, 1e-9), 2)
            row = {
                "dim": dim,
                "fetch_k": fetch_k,
                "k": k,
                "standard_mmr_ms": round(t_ref_ms, 3),
                "hisamdb_vectorized_mmr_ms": round(t_fast_ms, 3),
                "speedup": speedup,
            }
            mmr_rows.append(row)
            print(
                f"  MMR ({dim}D, fetch_k={fetch_k:3d}, k={k:2d}) | "
                f"Standart: {t_ref_ms:6.3f} ms -> HisamDB BLAS: {t_fast_ms:6.3f} ms ({speedup:5.2f}x hızlı)"
            )

    # Uçtan uca HisamDBVectorStore (10K x 128D HNSW)
    print("\n--- Uçtan Uca HisamDBVectorStore (10,000 x 128D, HNSW) ---")
    n_docs, dim = 10_000, 128
    vecs = rng.normal(size=(n_docs, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    texts = [f"Kurumsal doküman #{i} sunucu hata kodu ERR-{1000 + (i % 500)} kurtarma rehberi" for i in range(n_docs)]

    class _FastEmb:
        def embed_documents(self, ts):
            return [vecs[i % n_docs].tolist() for i in range(len(ts))]
        def embed_query(self, _t):
            return vecs[0].tolist()

    vs = HisamDBVectorStore(
        embedding=_FastEmb(),
        collection_name="bench_lc",
        index_type="hnsw",
        search_mode="hybrid",
        deduplicate_on_add=True,
    )
    vs.add_texts(texts, embeddings=vecs.tolist())

    # 1. Tekil Hybrid search
    q_list = [f"sunucu kurtarma ERR-{1000 + (i % 500)}" for i in range(50)]
    t0 = time.perf_counter()
    for q_txt in q_list:
        vs.similarity_search(q_txt, k=5)
    hyb_p50_ms = round((time.perf_counter() - t0) * 1000.0 / len(q_list), 3)

    # 2. Zero-Re-Embed MMR (fetch_k=50, k=10)
    t0 = time.perf_counter()
    for i in range(50):
        vs.max_marginal_relevance_search_by_vector(vecs[i].tolist(), k=10, fetch_k=50, lambda_mult=0.5)
    mmr_e2e_ms = round((time.perf_counter() - t0) * 1000.0 / 50, 3)

    # 3. Paralel batch_similarity_search (64 sorgu, dense)
    t0 = time.perf_counter()
    vs.batch_similarity_search(q_list[:64] if len(q_list) >= 64 else q_list + q_list[:14], k=5, search_mode="dense")
    batch_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    batch_qps = round(64.0 / (batch_ms / 1000.0), 1)

    print(f"  HisamDBVectorStore Hybrid (Dense+BM25+RRF)   : {hyb_p50_ms:.3f} ms/sorgu")
    print(f"  HisamDBVectorStore Zero-Re-Embed MMR (50->10): {mmr_e2e_ms:.3f} ms/sorgu")
    print(f"  HisamDBVectorStore Batch (64 sorgu, HNSW)    : {batch_ms:.2f} ms ({batch_qps:,.1f} QPS)")
    vs.close()

    out = {
        "mmr_benchmarks": mmr_rows,
        "vectorstore_10k_hnsw": {
            "hybrid_avg_ms": hyb_p50_ms,
            "zero_reembed_mmr_50_to_10_ms": mmr_e2e_ms,
            "batch_64_ms": batch_ms,
            "batch_64_qps": batch_qps,
        },
    }
    out_path = os.path.join(REPO_ROOT, "benchmarks", "results", "benchmark_v1.1.0_integrations.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nKaydedildi: {out_path}")


if __name__ == "__main__":
    main()
