"""
03_rag_integration.py
---------------------
hisamdb'nin gerçek bir RAG (Retrieval-Augmented Generation)
projesinde ChromaDB veya FAISS yerine nasıl kullanılacağını gösteren uçtan uca örnek.

Bu senaryoda:
1. Bir SSS (FAQ) veya doküman listesi alınır.
2. Bir embedding modeli (veya simüle edilmiş embedding fonksiyonu) ile vektörleştirilir.
3. hisamdb içine kalıcı olarak kaydedilir.
4. Kullanıcı soru sorduğunda en alakalı bağlam (context) çekilir ve LLM prompt'u hazırlanır.
"""

import os
import shutil
import numpy as np
import hisamdb

# 1. Örnek RAG Bilgi Tabanı
KNOWLEDGE_BASE = [
    {
        "id": "faq_01",
        "question": "İade süresi kaç gündür?",
        "answer": "Satın aldığınız ürünleri teslim aldığınız tarihten itibaren 14 gün içinde koşulsuz şartsız iade edebilirsiniz.",
        "category": "iade"
    },
    {
        "id": "faq_02",
        "question": "Kargo ücreti ne kadar?",
        "answer": "500 TL ve üzeri tüm siparişlerde kargo ücretsizdir. Altındaki siparişlerde sabit 49 TL kargo ücreti uygulanır.",
        "category": "kargo"
    },
    {
        "id": "faq_03",
        "question": "Hangi ödeme yöntemlerini kabul ediyorsunuz?",
        "answer": "Kredi kartı, banka kartı, havale/EFT ve kapıda nakit ödeme seçeneklerimiz mevcuttur.",
        "category": "odeme"
    },
    {
        "id": "faq_04",
        "question": "Yurt dışına gönderim yapıyor musunuz?",
        "answer": "Şu an için yalnızca Türkiye sınırları içerisindeki adreslere teslimat yapılmaktadır.",
        "category": "kargo"
    }
]


# 2. Basit Vektörleştirici (Gerçek projede: SentenceTransformer("intfloat/multilingual-e5-large"))
# Burada harici model indirmeden hemen çalışması için deterministik bir semantik vektör simülasyonu yapıyoruz:
def simple_embed(text: str, dim: int = 64) -> list[float]:
    np.random.seed(abs(hash(text.lower().strip())) % (2**32))
    vec = np.random.randn(dim).astype(np.float32)
    # L2 normalizasyonu
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


def main():
    db_path = "./rag_vectordb_data"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    print("=== 1. hisamdb Başlatılıyor ===")
    client = hisamdb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name="faq_knowledge_base",
        metric="cosine"
    )

    print("=== 2. Dokümanlar İndeksleniyor ===")
    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for item in KNOWLEDGE_BASE:
        ids.append(item["id"])
        # Soru + Cevap birlikte indekslenir
        full_text = f"Soru: {item['question']}\nCevap: {item['answer']}"
        documents.append(full_text)
        embeddings.append(simple_embed(item["question"]))  # Soru vektörü
        metadatas.append({"category": item["category"], "question": item["question"]})

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    print(f"[OK] {collection.count()} adet bilgi başarıyla indekslendi ve diske yazıldı.")

    # 3. Kullanıcı Sorusu
    user_query = "Siparişi geri göndermek istiyorum, kaç gün vaktim var?"
    print(f"\n=== 3. Kullanıcı Sorusu Geldi ===")
    print(f"Kullanıcı: \"{user_query}\"")

    # Kullanıcı sorusunun embedding'i
    query_embedding = simple_embed("İade süresi kaç gündür?")  # Semantik olarak eşleştiğini simüle ediyoruz

    # 4. hisamdb ile En Alakalı Bağlamı Bul
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1
    )

    retrieved_doc = results.documents[0][0]
    similarity = results.similarities[0][0]

    print(f"\n[Retrieval] hisamdb en alakalı kaydı buldu (Benzerlik: %{similarity * 100:.2f}):")
    print(f"--- Çekilen Bilgi ---\n{retrieved_doc}\n---------------------")

    # 5. LLM Prompt Oluşturma
    system_prompt = (
        "Sen bir müşteri hizmetleri asistanısın. Yalnızca sağlanan bağlamı kullanarak cevap ver.\n\n"
        f"BAĞLAM:\n{retrieved_doc}\n\n"
        f"KULLANICI SORUSU: {user_query}\n\n"
        "BOT CEVABI:"
    )

    print("\n=== 4. LLM'e Gönderilecek Prompt Hazırlandı ===")
    print(system_prompt)
    print("\n(Bu prompt OpenAI GPT-4o-mini'ye verilerek kullanıcıya doğal cevap üretilir.)")


if __name__ == "__main__":
    main()
