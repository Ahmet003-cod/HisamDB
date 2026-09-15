"""
02_metadata_filter.py
---------------------
microvectordb ile gelişmiş metadata filtreleme (where) örnekleri.
Desteklenen filtreler:
- Basit eşitlik: {"kategori": "elektronik"}
- Karşılaştırma operatörleri:
    $eq  : Eşit
    $ne  : Eşit değil
    $gt  : Büyüktür (>)
    $gte : Büyük eşittir (>=)
    $lt  : Küçüktür (<)
    $lte : Küçük eşittir (<=)
    $in  : Liste içinde var mı
    $nin : Liste içinde yok mu
"""

import microvectordb

# In-memory hızlı istemci
client = microvectordb.EphemeralClient()
coll = client.create_collection("urunler", dimension=3, metric="cosine")

# E-ticaret ürünleri ekleyelim
coll.add(
    ids=["u1", "u2", "u3", "u4", "u5"],
    embeddings=[
        [1.0, 0.2, 0.1],
        [0.9, 0.3, 0.1],
        [0.1, 1.0, 0.2],
        [0.2, 0.9, 0.3],
        [0.1, 0.1, 1.0],
    ],
    documents=[
        "iPhone 15 Pro Max 256GB Akıllı Telefon",
        "Samsung Galaxy S24 Ultra 512GB",
        "MacBook Pro 16 M3 Max 32GB RAM",
        "Dell XPS 15 OLED Laptop",
        "Sony WH-1000XM5 Kablosuz Kulaklık"
    ],
    metadatas=[
        {"kategori": "telefon", "fiyat": 75000, "marka": "apple", "stok": True},
        {"kategori": "telefon", "fiyat": 68000, "marka": "samsung", "stok": False},
        {"kategori": "laptop",  "fiyat": 110000, "marka": "apple", "stok": True},
        {"kategori": "laptop",  "fiyat": 85000,  "marka": "dell", "stok": True},
        {"kategori": "ses",     "fiyat": 14000,  "marka": "sony", "stok": True},
    ]
)

query_vector = [0.8, 0.2, 0.1]

print("=== Örnek 1: Sadece stokta olan telefonlar ===")
r1 = coll.query(
    query_embeddings=[query_vector],
    n_results=5,
    where={"kategori": "telefon", "stok": True}
)
for doc, meta in zip(r1.documents[0], r1.metadatas[0]):
    print(f"-> {doc} | Fiyat: {meta['fiyat']} TL | Stok: {meta['stok']}")

print("\n=== Örnek 2: Fiyatı 80.000 TL ve üzeri olanlar ($gte) ===")
r2 = coll.query(
    query_embeddings=[query_vector],
    n_results=5,
    where={"fiyat": {"$gte": 80000}}
)
for doc, meta in zip(r2.documents[0], r2.metadatas[0]):
    print(f"-> {doc} | Fiyat: {meta['fiyat']} TL")

print("\n=== Örnek 3: Markası Apple veya Sony olanlar ($in) ===")
r3 = coll.query(
    query_embeddings=[query_vector],
    n_results=5,
    where={"marka": {"$in": ["apple", "sony"]}}
)
for doc, meta in zip(r3.documents[0], r3.metadatas[0]):
    print(f"-> {doc} | Marka: {meta['marka']}")
