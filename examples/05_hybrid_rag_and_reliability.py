"""
05_hybrid_rag_and_reliability.py
--------------------------------
MicroVectorDB v0.5.0 uçtan uca üretim örneği:
1. Hibrit Arama: Dense Vektör + Türkçe/Unicode Okapi BM25 + Sparse Embedding (RRF Füzyon)
2. RAG İlkelleri: chunk_text, add_documents, retrieve_with_context (Parent-Child),
   doküman gruplama, sonuç tekilleştirme, harici Reranker kancası ve Multi-Tenant izolasyonu
3. Güvenilirlik: <1 ms Tombstone silme, compact() ve SHA-256 doğrulamalı Snapshot/Restore
"""

import os
import shutil
import tempfile
import microvectordb
from microvectordb import MicroVectorDB, chunk_text


class KeywordBoostReranker:
    """Çekirdekten tamamen bağımsız harici reranker örneği (CrossEncoder/Cohere yerine)."""

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        q_words = set(query.lower().split())
        scores = []
        for doc in documents:
            d_low = doc.lower()
            overlap = sum(1.0 for w in q_words if w in d_low)
            scores.append(overlap + (5.0 if "err-9042" in d_low else 0.0))
        return scores


def main() -> None:
    tmp_root = tempfile.mkdtemp(prefix="mvdb_ex05_")
    db_path = os.path.join(tmp_root, "live_db")
    snap_path = os.path.join(tmp_root, "snapshot_01")
    restored_path = os.path.join(tmp_root, "restored_db")

    try:
        print(f"=== MicroVectorDB v{microvectordb.__version__} Hibrit RAG & Güvenilirlik Demosu ===\n")

        with MicroVectorDB(db_path) as db:
            coll = db.create_collection("enterprise_kb", dimension=4, metric="cosine")

            # 1. Uzun dokümanı örtüşmeli parçalara böl ve Parent-Child bağlamıyla ekle
            manual_text = (
                "Bölüm 1: MicroVectorDB yüksek hızlı vektör veritabanı kurulumu ve İstanbul veri merkezi mimarisi. "
                "Bölüm 2: Sunucu bağlantı hatası ERR-9042 ortaya çıktığında WAL replay ve checkpoint prosedürü uygulanır. "
                "Bölüm 3: Yedekleme sonrası SHA-256 doğrulamalı snapshot geri yükleme testi yapılmalıdır."
            )
            chunks = chunk_text(manual_text, chunk_size=110, chunk_overlap=20)
            chunk_vecs = []
            for ch in chunks:
                low = ch.lower()
                chunk_vecs.append([
                    0.95 if "istanbul" in low else 0.15,
                    0.90 if "err-9042" in low or "wal" in low else 0.15,
                    0.95 if "snapshot" in low or "sha-256" in low else 0.10,
                    0.10,
                ])

            ids = coll.add_documents(
                documents=chunks,
                embeddings=chunk_vecs,
                doc_ids=["ops_manual_v5"] * len(chunks),
                parent_ids=["parent_ops_manual"] * len(chunks),
                metadatas=[{"category": "ops", "year": 2026} for _ in chunks],
            )
            print(f"[1] {len(ids)} RAG parçası (chunk) eklendi: {ids}")

            # 2. Hibrit Arama (Dense + Okapi BM25 + RRF + Harici Reranker)
            hits = coll.hybrid_search(
                query_vector=[0.85, 0.45, 0.05, 0.00],
                query_text="İstanbul ERR-9042 bağlantı hatası",
                limit=2,
                alpha=0.5,
                beta=0.5,
                fusion="rrf",
                deduplicate=True,
                reranker=KeywordBoostReranker(),
            )
            print("\n[2] Hibrit + Harici Reranker Sonuçları:")
            for h in hits:
                print(f"    - {h.id} | skor={h.score:.3f} | {h.document[:75]}...")

            # 3. Parent-Child Bağlam Penceresi (Komşu chunk'ları otomatik birleştirir)
            ctx = coll.retrieve_with_context(
                query_vector=[0.45, 0.88, 0.05, 0.00],
                query_text="ERR-9042",
                limit=1,
                window=1,
            )
            print(f"\n[3] Bağlam Penceresi ({len(ctx[0]['neighbors'])} komşu parça birleştirildi):")
            print(f"    {ctx[0]['context'][:140]}...")

            # 4. Multi-Tenant (Çok Kiracılı) İzolasyon
            db.create_collection("saas_docs", dimension=4, metric="cosine")
            tenant_a = db.get_tenant_collection("saas_docs", "acme_corp")
            tenant_b = db.get_tenant_collection("saas_docs", "beta_ltd")
            tenant_a.add(ids=["a1"], vectors=[[1, 0, 0, 0]], documents=["ACME gizli finans raporu"])
            tenant_b.add(ids=["b1"], vectors=[[1, 0, 0, 0]], documents=["Beta gizli ürün yol haritası"])
            print(f"\n[4] Multi-Tenant İzolasyonu: ACME={tenant_a.count()}, Beta={tenant_b.count()}, "
                  f"Beta'nın ACME verisini silme girişimi={tenant_b.delete(ids=['a1'])} (engellendi!)")

            # 5. <1 ms Tombstone Silme + Compaction + SHA-256 Snapshot
            coll.delete(ids=[ids[-1]])
            st_before = coll.stats()
            reclaimed = coll.compact()
            manifest = db.create_snapshot(snap_path)
            print(
                f"\n[5] Tombstone & Snapshot: silme sonrası tombstones={st_before['tombstones']}, "
                f"compact() kazanılan slot={reclaimed}, snapshot dosya sayısı={len(manifest['files'])}"
            )

        # 6. SHA-256 Doğrulamalı Geri Yükleme
        MicroVectorDB.restore_snapshot(snap_path, restored_path)
        with MicroVectorDB(restored_path) as db_restored:
            c_res = db_restored.get_collection("enterprise_kb")
            print(f"[6] Snapshot geri yüklendi ve SHA-256 doğrulandı! Kayıt sayısı: {c_res.count()}")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
