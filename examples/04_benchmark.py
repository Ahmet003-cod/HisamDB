"""
04_benchmark.py
---------------
microvectordb performans ve hız testi (Benchmark).
1. 10.000 adet 128 boyutlu vektör ekleme hızını ölçer.
2. 1.000 adet sorgu çalıştırarak saniyedeki sorgu sayısını (QPS - Queries Per Second) hesaplar.
"""

import time
import numpy as np
import microvectordb

print("=== microvectordb Performans Benchmark Testi ===")

NUM_VECTORS = 10000
DIMENSION = 128
NUM_QUERIES = 1000
TOP_K = 5

print(f"Parametreler:")
print(f"  - Toplam Vektör Sayısı: {NUM_VECTORS:,}")
print(f"  - Vektör Boyutu:        {DIMENSION}")
print(f"  - Test Sorgu Sayısı:    {NUM_QUERIES:,}")
print(f"  - Top-K:                {TOP_K}\n")

# Rastgele veri üret
print("1. Test verileri bellekte üretiliyor...")
np.random.seed(42)
vectors = np.random.randn(NUM_VECTORS, DIMENSION).astype(np.float32)
ids = [f"vec_{i}" for i in range(NUM_VECTORS)]
docs = [f"Örnek doküman içeriği {i}" for i in range(NUM_VECTORS)]

client = microvectordb.EphemeralClient()
coll = client.create_collection("benchmark_coll", dimension=DIMENSION, metric="cosine")

# Ekleme testi
print("2. 10.000 vektör ekleniyor...")
t0 = time.time()
coll.add(ids=ids, embeddings=vectors, documents=docs)
t_insert = time.time() - t0
insert_rate = NUM_VECTORS / t_insert
print(f"[Ekleme Tamamlandı]")
print(f"  - Geçen Süre:        {t_insert:.3f} saniye")
print(f"  - Ekleme Hızı:       {insert_rate:,.0f} vektör/saniye")

# Arama testi
print(f"\n3. {NUM_QUERIES} adet benzerlik araması yapılıyor...")
query_vectors = np.random.randn(NUM_QUERIES, DIMENSION).astype(np.float32)

t0 = time.time()
for i in range(NUM_QUERIES):
    res = coll.query(query_embeddings=query_vectors[i], n_results=TOP_K)
t_query = time.time() - t0

qps = NUM_QUERIES / t_query
avg_latency_ms = (t_query / NUM_QUERIES) * 1000

print(f"[Arama Tamamlandı]")
print(f"  - Toplam Süre:       {t_query:.3f} saniye")
print(f"  - Hız (QPS):         {qps:,.1f} sorgu/saniye")
print(f"  - Ortalama Gecikme:  {avg_latency_ms:.3f} ms / sorgu")
print("\n=== Benchmark Başarıyla Tamamlandı ===")
