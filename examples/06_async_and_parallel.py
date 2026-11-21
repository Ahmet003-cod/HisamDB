"""
06_async_and_parallel.py
------------------------
MicroVectorDB v0.6.0 (Faz 4) yüksek eşzamanlılık ve asenkron API örneği:
1. AsyncMicroVectorDB + asyncio.gather() ile bloklamayan eşzamanlı sorgulama
2. batch_search() ve batch_hybrid_search() ile çok çekirdekli toplu sorgulama
3. rebuild_index_async() ile okuyucuları durdurmadan arka planda indeks inşası
4. auto_compact_async=True ile yoğun silmede <1 ms düşük kuyruk gecikmesi
"""

import asyncio
import os
import shutil
import tempfile
import numpy as np
import hisamdb
from hisamdb import AsyncMicroVectorDB, is_hnsw_available


async def main() -> None:
    tmp_dir = tempfile.mkdtemp(prefix="mvdb_ex06_")
    db_path = os.path.join(tmp_dir, "async_prod_db")

    try:
        print(f"=== MicroVectorDB v{hisamdb.__version__} Async & Paralel Eşzamanlılık Demosu ===\n")
        rng = np.random.default_rng(42)
        dim = 32
        n_docs = 500
        vecs = rng.standard_normal((n_docs, dim)).astype(np.float32)
        docs = [f"Doküman {i}: {'İstanbul' if i % 2 == 0 else 'Ankara'} servis kodu SRV-{100 + i}" for i in range(n_docs)]
        metas = [{"city": "istanbul" if i % 2 == 0 else "ankara", "tier": i % 3} for i in range(n_docs)]

        idx_type = "hnsw" if is_hnsw_available() else "ivf"
        async with AsyncMicroVectorDB(db_path, max_workers=4) as db:
            coll = await db.create_collection(
                "async_kb",
                dimension=dim,
                metric="cosine",
                index_type=idx_type,
                index_params={
                    "auto_compact_async": True,
                    "compaction_threshold": 0.25,
                    "auto_compact_min_deleted": 32,
                },
            )
            ids = await coll.add_documents(documents=docs, embeddings=vecs, metadatas=metas)
            print(f"[1] {len(ids)} doküman '{idx_type}' koleksiyonuna asenkron eklendi.")

            # 2. asyncio.gather() ile 16 eşzamanlı hibrit sorgu fan-out
            q_vecs = rng.standard_normal((16, dim)).astype(np.float32)
            gather_results = await asyncio.gather(
                *(
                    coll.hybrid_search(
                        query_vector=q_vecs[i],
                        query_text=f"İstanbul SRV-{100 + i * 2}",
                        limit=3,
                        fusion="rrf",
                    )
                    for i in range(16)
                )
            )
            print(f"[2] asyncio.gather() ile {len(gather_results)} eşzamanlı hibrit sorgu tamamlandı "
                  f"(örnek ilk eşleşme: {gather_results[0][0].id}).")

            # 3. Çok çekirdekli batch_search() (sorgu başına ayrı metadata filtresiyle!)
            per_filters = [{"city": "istanbul" if i % 2 == 0 else "ankara"} for i in range(16)]
            batch_res = await coll.batch_search(q_vecs, limit=3, filters=per_filters, max_workers=4)
            print(f"[3] batch_search() ile {len(batch_res)} filtreli sorgu paralel çalıştırıldı.")

            # 4. Bloklamayan arka plan indeks yeniden inşası
            swapped = await coll.rebuild_index_async()
            print(f"[4] rebuild_index_async() okuyucuları bloklamadan tamamlandı (swapped={swapped}).")

            # 5. auto_compact_async=True ile yoğun silme + arka plan sıkıştırma
            deleted = await coll.delete(ids=ids[:150])
            await coll.compact_async()
            st = await coll.stats()
            print(f"[5] {deleted} kayıt silindi; arka plan compaction sonrası count={st['count']}, tombstones={st['tombstones']}.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
