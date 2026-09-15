"""
test_microvectordb.py
---------------------
microvectordb kütüphanesi için kapsamlı birim testleri (Unit Tests).
"""

import os
import shutil
import tempfile
import unittest
import numpy as np

import microvectordb
from microvectordb import (
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    DimensionMismatchError,
    EphemeralClient,
    PersistentClient,
    VectorDBError,
)


class TestMicroVectorDB(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_ephemeral_client_basic(self):
        """In-memory istemci temel işlemleri."""
        client = EphemeralClient()
        self.assertEqual(len(client.list_collections()), 0)

        coll = client.create_collection("test_coll", dimension=4, metric="cosine")
        self.assertEqual(coll.name, "test_coll")
        self.assertEqual(client.list_collections(), ["test_coll"])

        # Koleksiyon zaten varsa hata vermeli
        with self.assertRaises(CollectionAlreadyExistsError):
            client.create_collection("test_coll")

    def test_add_and_count(self):
        """Vektör ekleme ve sayma."""
        client = EphemeralClient()
        coll = client.create_collection("docs", dimension=3)

        ids = ["id1", "id2", "id3"]
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        documents = ["X ekseni", "Y ekseni", "Z ekseni"]
        metadatas = [{"eksen": "x"}, {"eksen": "y"}, {"eksen": "z"}]

        coll.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        self.assertEqual(coll.count(), 3)

        # Aynı ID tekrar eklenememeli
        with self.assertRaises(VectorDBError):
            coll.add(ids=["id1"], embeddings=[[1.0, 0.0, 0.0]])

        # Boyut uyuşmazlığı kontrolü
        with self.assertRaises(DimensionMismatchError):
            coll.add(ids=["id4"], embeddings=[[1.0, 0.0]])

    def test_query_cosine(self):
        """Kosinüs benzerliği ile en yakın komşu araması."""
        client = EphemeralClient()
        coll = client.create_collection("cosine_test", dimension=3, metric="cosine")

        coll.add(
            ids=["doc_x", "doc_y", "doc_z"],
            embeddings=[
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            documents=["Tamamen X", "Tamamen Y", "Tamamen Z"],
        )

        # Sorgu: [0.9, 0.1, 0.0] -> 'doc_x' en yakın olmalı
        res = coll.query(query_embeddings=[0.9, 0.1, 0.0], n_results=2)

        self.assertEqual(len(res.ids[0]), 2)
        self.assertEqual(res.ids[0][0], "doc_x")
        self.assertEqual(res.documents[0][0], "Tamamen X")
        # doc_x ile kosinüs benzerliği 0.99'dan büyük olmalı
        self.assertGreater(res.similarities[0][0], 0.95)

    def test_query_euclidean(self):
        """Öklid (L2) mesafesi testi."""
        client = EphemeralClient()
        coll = client.create_collection("l2_test", dimension=2, metric="euclidean")

        coll.add(
            ids=["point_a", "point_b"],
            embeddings=[[0.0, 0.0], [10.0, 10.0]],
            documents=["Merkez", "Uzak nokta"],
        )

        # [0.1, 0.1] noktasına point_a daha yakındır
        res = coll.query(query_embeddings=[0.1, 0.1], n_results=1)
        self.assertEqual(res.ids[0][0], "point_a")
        self.assertLess(res.distances[0][0], 1.0)

    def test_metadata_filter(self):
        """Metadata filtreleme ($eq, $gte, $in)."""
        client = EphemeralClient()
        coll = client.create_collection("filter_test", dimension=2)

        coll.add(
            ids=["p1", "p2", "p3", "p4"],
            embeddings=[[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.1, 0.9]],
            documents=["Telefon A", "Telefon B", "Bilgisayar A", "Bilgisayar B"],
            metadatas=[
                {"category": "phone", "price": 500, "brand": "apple"},
                {"category": "phone", "price": 300, "brand": "samsung"},
                {"category": "laptop", "price": 1200, "brand": "apple"},
                {"category": "laptop", "price": 800, "brand": "lenovo"},
            ],
        )

        # 1. Basit filtre: Sadece laptop'lar
        res1 = coll.query(
            query_embeddings=[1.0, 0.0],
            n_results=10,
            where={"category": "laptop"},
        )
        self.assertEqual(len(res1.ids[0]), 2)
        self.assertTrue(all("laptop" in doc.lower() or "bilgisayar" in doc.lower() for doc in res1.documents[0]))

        # 2. Operatör filtresi: Fiyatı 500 veya üstü olanlar ($gte)
        res2 = coll.query(
            query_embeddings=[1.0, 0.0],
            n_results=10,
            where={"price": {"$gte": 500}},
        )
        returned_ids = set(res2.ids[0])
        self.assertEqual(returned_ids, {"p1", "p3", "p4"})

        # 3. $in filtresi: Markası apple veya lenovo olanlar
        res3 = coll.query(
            query_embeddings=[1.0, 0.0],
            n_results=10,
            where={"brand": {"$in": ["apple", "lenovo"]}},
        )
        returned_ids = set(res3.ids[0])
        self.assertEqual(returned_ids, {"p1", "p3", "p4"})

    def test_upsert(self):
        """Upsert (güncelle veya ekle) testi."""
        client = EphemeralClient()
        coll = client.create_collection("upsert_test", dimension=2)

        coll.add(ids=["item1"], embeddings=[[1.0, 0.0]], documents=["İlk Hali"])
        self.assertEqual(coll.get(ids=["item1"])["documents"][0], "İlk Hali")

        # item1'i güncelle, item2'yi yeni ekle
        coll.upsert(
            ids=["item1", "item2"],
            embeddings=[[0.0, 1.0], [2.0, 2.0]],
            documents=["Güncellenmiş Hali", "Yeni Öğe"],
        )

        self.assertEqual(coll.count(), 2)
        self.assertEqual(coll.get(ids=["item1"])["documents"][0], "Güncellenmiş Hali")
        self.assertEqual(coll.get(ids=["item2"])["documents"][0], "Yeni Öğe")

    def test_delete(self):
        """Silme işlemi (ID ve filtre bazlı)."""
        client = EphemeralClient()
        coll = client.create_collection("delete_test", dimension=2)

        coll.add(
            ids=["a", "b", "c"],
            embeddings=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            metadatas=[{"status": "old"}, {"status": "new"}, {"status": "old"}],
        )
        self.assertEqual(coll.count(), 3)

        # ID ile sil
        coll.delete(ids=["b"])
        self.assertEqual(coll.count(), 2)
        self.assertEqual(coll.get()["ids"], ["a", "c"])

        # Filtre ile sil
        coll.delete(where={"status": "old"})
        self.assertEqual(coll.count(), 0)

    def test_persistence(self):
        """Kalıcılık testi: Diske yazma ve baştan açıldığında veriyi koruma."""
        # 1. İstemci aç ve veri ekle
        client1 = PersistentClient(path=self.temp_dir)
        coll1 = client1.create_collection("persistent_docs", dimension=3, metric="cosine")

        coll1.add(
            ids=["doc1", "doc2"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            documents=["Kalıcı Belge 1", "Kalıcı Belge 2"],
            metadatas=[{"tag": "v1"}, {"tag": "v2"}],
        )
        self.assertEqual(coll1.count(), 2)

        # 2. Yeni bağımsız istemci ile aynı klasörü aç (sanki program kapanıp yeniden açıldı)
        client2 = PersistentClient(path=self.temp_dir)
        self.assertIn("persistent_docs", client2.list_collections())

        coll2 = client2.get_collection("persistent_docs")
        self.assertEqual(coll2.count(), 2)
        self.assertEqual(coll2.dimension, 3)
        self.assertEqual(coll2.metric, "cosine")

        # Aramanın diskten yüklenen veriyle çalıştığını doğrula
        res = coll2.query(query_embeddings=[1.0, 0.0, 0.0], n_results=1)
        self.assertEqual(res.ids[0][0], "doc1")
        self.assertEqual(res.documents[0][0], "Kalıcı Belge 1")
        self.assertEqual(res.metadatas[0][0]["tag"], "v1")


if __name__ == "__main__":
    unittest.main()
