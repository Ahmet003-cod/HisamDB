"""
01_quickstart.py
----------------
microvectordb için 5 dakikalık hızlı başlangıç rehberi.
Bu örnekte:
1. Kalıcı bir veritabanı (PersistentClient) oluşturulur.
2. 'makaleler' adında bir koleksiyon oluşturulur (kosinüs benzerliği metriğiyle).
3. Örnek dokümanlar ve vektörleri eklenir.
4. Bir sorgu vektörü ile en benzer kayıtlar bulunur.
"""

import os
import shutil
import microvectordb

DB_PATH = "./demo_db"

# Temiz bir başlangıç için eski demo dizini varsa temizleyelim
if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH)

print("=== microvectordb Hızlı Başlangıç Demo ===")

# 1. Kalıcı istemci oluştur (veriler DB_PATH klasörüne kaydedilir)
client = microvectordb.PersistentClient(path=DB_PATH)
print(f"[OK] İstemci başlatıldı: {client}")

# 2. Koleksiyon oluştur (Örn: 4 boyutlu temsili embedding'ler)
collection = client.get_or_create_collection(
    name="teknoloji_rehberi",
    dimension=4,
    metric="cosine"  # "cosine", "dot_product" veya "euclidean"
)
print(f"[OK] Koleksiyon hazır: {collection}")

# 3. Örnek dokümanlar, ID'ler ve vektörler ekle
print("\n-> Veriler ekleniyor...")
ids = ["doc_ai", "doc_db", "doc_web", "doc_cloud"]
documents = [
    "Yapay zeka ve derin öğrenme algoritmaları hızla gelişiyor.",
    "İlişkisel veritabanları ve vektör indeksleme teknikleri.",
    "Modern web geliştirme, React ve FastAPI mimarileri.",
    "Bulut bilişim, Docker ve Kubernetes konteyner dağıtımı."
]
metadatas = [
    {"kategori": "ai", "yazar": "Ahmet", "yil": 2024},
    {"kategori": "database", "yazar": "Mehmet", "yil": 2023},
    {"kategori": "web", "yazar": "Ayşe", "yil": 2024},
    {"kategori": "devops", "yazar": "Fatma", "yil": 2024}
]
# Temsili 4 boyutlu embedding vektörleri:
embeddings = [
    [0.95, 0.10, 0.05, 0.02],  # AI ağırlıklı
    [0.10, 0.92, 0.20, 0.05],  # DB ağırlıklı
    [0.05, 0.15, 0.94, 0.10],  # Web ağırlıklı
    [0.02, 0.10, 0.12, 0.96],  # Cloud ağırlıklı
]

collection.add(
    ids=ids,
    embeddings=embeddings,
    documents=documents,
    metadatas=metadatas
)
print(f"[OK] Toplam kayıt sayısı: {collection.count()}")

# 4. Arama yap (Query)
# Yapay zeka konusuna yakın bir sorgu vektörü simüle edelim:
query_vector = [0.88, 0.12, 0.04, 0.01]

print("\n-> Arama yapılıyor (Sorgu: 'Yapay zeka konuları')...")
results = collection.query(
    query_embeddings=[query_vector],
    n_results=2  # En yakın 2 sonucu getir
)

print("\n--- Arama Sonuçları ---")
for i in range(len(results.ids[0])):
    item_id = results.ids[0][i]
    similarity = results.similarities[0][i]
    distance = results.distances[0][i]
    doc = results.documents[0][i]
    meta = results.metadatas[0][i]

    print(f"\nSonuç #{i+1}:")
    print(f"  ID:          {item_id}")
    print(f"  Benzerlik:   %{similarity * 100:.2f} (Mesafe: {distance:.4f})")
    print(f"  Doküman:     {doc}")
    print(f"  Metadata:    {meta}")

print("\n=== Demo başarıyla tamamlandı! ===")
