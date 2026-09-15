"""
microvectordb.collection
------------------------
Koleksiyon sınıfı: Vektörlerin saklandığı, benzerlik aramalarının (query),
metadata filtrelemelerinin ve CRUD işlemlerinin yönetildiği ana merkez.
ChromaDB ve FAISS benzeri kullanıcı dostu API sunar.
"""

import time
import numpy as np
from typing import Any, Dict, List, Optional, Union
from microvectordb.distance import compute_distances
from microvectordb.storage import StorageEngine
from microvectordb.types import (
    BatchVectors,
    DimensionMismatchError,
    Metadata,
    MetricType,
    QueryResult,
    VectorDBError,
    VectorLike,
)


class Collection:
    """
    Vektör koleksiyonu.
    Her koleksiyon bağımsız bir vektör matrisine ve metadata tablosuna sahiptir.
    """

    def __init__(
        self,
        name: str,
        storage: StorageEngine,
        dimension: Optional[int] = None,
        metric: MetricType = "cosine",
        created_at: Optional[float] = None,
    ):
        self.name = name
        self._storage = storage
        self.dimension = dimension
        self.metric: MetricType = metric
        self.created_at = created_at or time.time()

        # Depolama tablosunu oluştur
        self._storage.ensure_collection_storage(self.name)

        # Bellek içi veri yapıları (hızlı arama için)
        self._vectors: np.ndarray = np.empty((0, self.dimension or 0), dtype=np.float32)
        self._ids: List[str] = []
        self._id_to_idx: Dict[str, int] = {}
        self._documents: List[Optional[str]] = []
        self._metadatas: List[Optional[Metadata]] = []

        # Ön-normalizasyon cache'i (cosine aramaları için %30-40 hızlanma)
        self._vectors_normalized: Optional[np.ndarray] = None

        # Diskten mevcut verileri yükle
        self._load_from_storage()

    # ------------------------------------------------------------------ #
    # Normalizasyon Cache Yönetimi
    # ------------------------------------------------------------------ #
    def _invalidate_norm_cache(self) -> None:
        """Vektörler değiştiğinde normalizasyon cache'ini sıfırlar."""
        self._vectors_normalized = None

    def _get_normalized_vectors(self) -> np.ndarray:
        """
        L2-normalize edilmiş vektörleri döndürür (lazy cache).
        İlk çağrıda hesaplanır, sonraki çağrılarda cache'ten döner.
        Vektörler eklendiğinde/silindiğinde cache otomatik sıfırlanır.
        """
        if self._vectors_normalized is None and self._vectors.size > 0:
            from microvectordb.distance import normalize_vectors
            self._vectors_normalized = normalize_vectors(self._vectors)
        return self._vectors_normalized

    # ------------------------------------------------------------------ #
    # Bellek ve Disk Senkronizasyonu
    # ------------------------------------------------------------------ #
    def _load_from_storage(self) -> None:
        """Koleksiyon başlatıldığında diskteki verileri RAM'e aktarır."""
        items = self._storage.load_all_items(self.name)
        if not items:
            self._vectors = np.empty((0, self.dimension or 0), dtype=np.float32)
            return

        # Vektörleri yükle
        vecs = self._storage.load_vectors(self.name, self.dimension)
        if vecs.shape[0] != len(items):
            # Tutarsızlık varsa veya ilk yükleme ise
            pass

        self._vectors = vecs
        self._ids = [item["id"] for item in items]
        self._id_to_idx = {item["id"]: idx for idx, item in enumerate(items)}
        self._documents = [item["document"] for item in items]
        self._metadatas = [item["metadata"] for item in items]

        if self.dimension is None and self._vectors.shape[1] > 0:
            self.dimension = self._vectors.shape[1]

    def _persist(self) -> None:
        """Bellekteki son durumu diske kaydeder."""
        # 1. Koleksiyon üst bilgisini kaydet
        self._storage.save_collection_meta(
            self.name, self.dimension, self.metric, self.created_at
        )
        # 2. Vektör matrisini kaydet
        self._storage.save_vectors(self.name, self._vectors)
        # 3. Öğeleri kaydet
        batch_items = [
            (s_id, idx, self._documents[idx], self._metadatas[idx])
            for idx, s_id in enumerate(self._ids)
        ]
        self._storage.save_items_batch(self.name, batch_items)

    # ------------------------------------------------------------------ #
    # CRUD: Ekleme & Güncelleme
    # ------------------------------------------------------------------ #
    def add(
        self,
        ids: List[str],
        embeddings: BatchVectors,
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Metadata]] = None,
    ) -> None:
        """
        Koleksiyona yeni vektörler ekler.
        Mevcut bir ID tekrar eklenmek istenirse hata fırlatır (güncellemek için upsert kullanın).
        """
        arr = np.array(embeddings, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        n_items = len(ids)
        if arr.shape[0] != n_items:
            raise ValueError(f"ID sayısı ({n_items}) ile embedding sayısı ({arr.shape[0]}) eşit olmalıdır.")

        if documents is not None and len(documents) != n_items:
            raise ValueError("Document sayısı ID sayısıyla eşit olmalıdır.")
        if metadatas is not None and len(metadatas) != n_items:
            raise ValueError("Metadata sayısı ID sayısıyla eşit olmalıdır.")

        # Vektör boyutu doğrulaması
        if self.dimension is None:
            self.dimension = arr.shape[1]
        elif arr.shape[1] != self.dimension:
            raise DimensionMismatchError(
                f"Vektör boyutu uyuşmazlığı: Beklenen {self.dimension}, gelen {arr.shape[1]}"
            )

        # ID benzersizliği kontrolü
        for item_id in ids:
            if item_id in self._id_to_idx:
                raise VectorDBError(f"ID '{item_id}' zaten bu koleksiyonda mevcut. Güncellemek için 'upsert' kullanın.")

        # Ekle
        start_idx = len(self._ids)
        for i, item_id in enumerate(ids):
            self._ids.append(item_id)
            self._id_to_idx[item_id] = start_idx + i
            self._documents.append(documents[i] if documents else None)
            self._metadatas.append(metadatas[i] if metadatas else None)

        if self._vectors.size == 0:
            self._vectors = arr
        else:
            self._vectors = np.vstack([self._vectors, arr])

        # Diske yaz
        self._persist()
        self._invalidate_norm_cache()

    def upsert(
        self,
        ids: List[str],
        embeddings: BatchVectors,
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Metadata]] = None,
    ) -> None:
        """
        Vektörleri ekler veya varsa günceller (Add or Update).
        """
        arr = np.array(embeddings, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        n_items = len(ids)
        if arr.shape[0] != n_items:
            raise ValueError(f"ID sayısı ({n_items}) ile embedding sayısı ({arr.shape[0]}) eşit olmalıdır.")

        # Boyut kontrolü
        if self.dimension is None:
            self.dimension = arr.shape[1]
        elif arr.shape[1] != self.dimension:
            raise DimensionMismatchError(
                f"Vektör boyutu uyuşmazlığı: Beklenen {self.dimension}, gelen {arr.shape[1]}"
            )

        to_add_ids = []
        to_add_vecs = []
        to_add_docs = []
        to_add_metas = []

        for i, item_id in enumerate(ids):
            doc = documents[i] if documents else None
            meta = metadatas[i] if metadatas else None
            vec = arr[i]

            if item_id in self._id_to_idx:
                # Güncelle
                idx = self._id_to_idx[item_id]
                self._vectors[idx] = vec
                if documents:
                    self._documents[idx] = doc
                if metadatas:
                    self._metadatas[idx] = meta
            else:
                # Yeni eklenecekler listesi
                to_add_ids.append(item_id)
                to_add_vecs.append(vec)
                to_add_docs.append(doc)
                to_add_metas.append(meta)

        if to_add_ids:
            start_idx = len(self._ids)
            for i, item_id in enumerate(to_add_ids):
                self._ids.append(item_id)
                self._id_to_idx[item_id] = start_idx + i
                self._documents.append(to_add_docs[i])
                self._metadatas.append(to_add_metas[i])

            new_block = np.array(to_add_vecs, dtype=np.float32)
            if self._vectors.size == 0:
                self._vectors = new_block
            else:
                self._vectors = np.vstack([self._vectors, new_block])

        # Diske yaz
        self._persist()
        self._invalidate_norm_cache()

    # ------------------------------------------------------------------ #
    # Metadata Filtreleme Mantığı
    # ------------------------------------------------------------------ #
    @staticmethod
    def _matches_filter(item_meta: Optional[Metadata], where: Optional[Dict[str, Any]]) -> bool:
        """
        Metadata kaydının 'where' filtresine uyup uymadığını denetler.
        Örnek where filtreleri:
            where={"kategori": "ai"}
            where={"yas": {"$gte": 18}}
            where={"sehir": {"$in": ["Ankara", "İstanbul"]}}
        """
        if not where:
            return True
        if item_meta is None:
            return False

        for key, condition in where.items():
            val = item_meta.get(key)
            if isinstance(condition, dict):
                # Operatör kontrolleri ($eq, $ne, $gt, $gte, $lt, $lte, $in, $nin)
                for op, target in condition.items():
                    if op == "$eq" and val != target:
                        return False
                    elif op == "$ne" and val == target:
                        return False
                    elif op == "$gt" and (val is None or val <= target):
                        return False
                    elif op == "$gte" and (val is None or val < target):
                        return False
                    elif op == "$lt" and (val is None or val >= target):
                        return False
                    elif op == "$lte" and (val is None or val > target):
                        return False
                    elif op == "$in" and val not in target:
                        return False
                    elif op == "$nin" and val in target:
                        return False
            else:
                # Basit eşitlik kontrolü: {"category": "tech"}
                if val != condition:
                    return False

        return True

    # ------------------------------------------------------------------ #
    # Query / Arama İşlemi
    # ------------------------------------------------------------------ #
    def query(
        self,
        query_embeddings: Optional[Union[VectorLike, BatchVectors]] = None,
        query_embedding: Optional[Union[VectorLike, BatchVectors]] = None,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:
        """
        Benzerlik araması yapar (Nearest Neighbor Search).

        Args:
            query_embeddings: Tek bir vektör listesi veya çoklu sorgu matrisi.
            query_embedding: 'query_embeddings' için alternatif takma ad (alias).
            n_results: Getirilecek en benzer sonuç sayısı (Top-K).
            where: Metadata filtreleme sözlüğü (örn. {"kategori": "ai"}).

        Returns:
            QueryResult: ids, distances, similarities, documents, metadatas listeleri.
        """
        raw = query_embeddings if query_embeddings is not None else query_embedding
        if raw is None:
            raise ValueError("Arama için 'query_embeddings' veya 'query_embedding' parametresi gereklidir.")

        q_arr = np.array(raw, dtype=np.float32)
        if q_arr.ndim == 1:
            q_arr = q_arr.reshape(1, -1)

        n_queries = q_arr.shape[0]

        if self.count() == 0:
            return QueryResult(
                ids=[[] for _ in range(n_queries)],
                distances=[[] for _ in range(n_queries)],
                similarities=[[] for _ in range(n_queries)],
                documents=[[] for _ in range(n_queries)],
                metadatas=[[] for _ in range(n_queries)],
            )

        # Filtreleme varsa aday indeksleri belirle
        if where:
            candidate_indices = [
                idx for idx, meta in enumerate(self._metadatas)
                if self._matches_filter(meta, where)
            ]
            if not candidate_indices:
                return QueryResult(
                    ids=[[] for _ in range(n_queries)],
                    distances=[[] for _ in range(n_queries)],
                    similarities=[[] for _ in range(n_queries)],
                    documents=[[] for _ in range(n_queries)],
                    metadatas=[[] for _ in range(n_queries)],
                )
            sub_vectors = self._vectors[candidate_indices]
            sub_ids = [self._ids[i] for i in candidate_indices]
            sub_docs = [self._documents[i] for i in candidate_indices]
            sub_metas = [self._metadatas[i] for i in candidate_indices]
        else:
            sub_vectors = self._vectors
            sub_ids = self._ids
            sub_docs = self._documents
            sub_metas = self._metadatas

        # Mesafeleri ve benzerlikleri hesapla (NumPy / BLAS ile aşırı hızlı)
        # Cosine metriğinde, filtre yoksa önceden normalize edilmiş cache'i kullan
        pre_norm = None
        if self.metric == "cosine" and not where:
            pre_norm = self._get_normalized_vectors()
        elif self.metric == "cosine" and where and candidate_indices:
            # Filtre varsa cache'ten sadece aday indekslerini al
            full_norm = self._get_normalized_vectors()
            if full_norm is not None:
                pre_norm = full_norm[candidate_indices]
        dists, sims = compute_distances(q_arr, sub_vectors, metric=self.metric, pre_normalized_db=pre_norm)

        out_ids: List[List[str]] = []
        out_dists: List[List[float]] = []
        out_sims: List[List[float]] = []
        out_docs: List[List[Optional[str]]] = []
        out_metas: List[List[Optional[Metadata]]] = []

        k = min(n_results, len(sub_ids))

        for q_idx in range(n_queries):
            row_dists = dists[q_idx]
            row_sims = sims[q_idx]

            # En küçük K mesafeyi bul: O(N) partial sort (np.argsort O(N log N)'den çok daha hızlı)
            if k < len(row_dists):
                top_k_raw = np.argpartition(row_dists, k)[:k]
                top_k_indices = top_k_raw[np.argsort(row_dists[top_k_raw])]
            else:
                top_k_indices = np.argsort(row_dists)[:k]

            out_ids.append([sub_ids[i] for i in top_k_indices])
            out_dists.append([float(row_dists[i]) for i in top_k_indices])
            out_sims.append([float(row_sims[i]) for i in top_k_indices])
            out_docs.append([sub_docs[i] for i in top_k_indices])
            out_metas.append([sub_metas[i] for i in top_k_indices])

        return QueryResult(
            ids=out_ids,
            distances=out_dists,
            similarities=out_sims,
            documents=out_docs,
            metadatas=out_metas,
        )

    # ------------------------------------------------------------------ #
    # Diğer CRUD Metotları (get, delete, count, peek)
    # ------------------------------------------------------------------ #
    def get(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Kayıtları ID veya filtreye göre sorgular."""
        matched_indices = []
        for idx, (s_id, meta) in enumerate(zip(self._ids, self._metadatas)):
            if ids is not None and s_id not in ids:
                continue
            if where is not None and not self._matches_filter(meta, where):
                continue
            matched_indices.append(idx)
            if limit and len(matched_indices) >= limit:
                break

        return {
            "ids": [self._ids[i] for i in matched_indices],
            "documents": [self._documents[i] for i in matched_indices],
            "metadatas": [self._metadatas[i] for i in matched_indices],
        }

    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Kayıtları siler ve bellek matrisini yeniden düzenler."""
        if ids is None and where is None:
            raise ValueError("Silme için ya 'ids' ya da 'where' filtresi belirtilmelidir.")

        to_delete_set = set()
        if ids is not None:
            to_delete_set.update(ids)

        if where is not None:
            for s_id, meta in zip(self._ids, self._metadatas):
                if self._matches_filter(meta, where):
                    to_delete_set.add(s_id)

        if not to_delete_set:
            return 0

        keep_indices = [
            idx for idx, s_id in enumerate(self._ids)
            if s_id not in to_delete_set
        ]

        deleted_count = len(self._ids) - len(keep_indices)

        # Tablodan sil
        self._storage.delete_items_by_ids(self.name, list(to_delete_set))

        # Bellek yapılarını güncelle
        if keep_indices:
            self._vectors = self._vectors[keep_indices]
            self._ids = [self._ids[i] for i in keep_indices]
            self._documents = [self._documents[i] for i in keep_indices]
            self._metadatas = [self._metadatas[i] for i in keep_indices]
            self._id_to_idx = {s_id: idx for idx, s_id in enumerate(self._ids)}
        else:
            self._vectors = np.empty((0, self.dimension or 0), dtype=np.float32)
            self._ids.clear()
            self._documents.clear()
            self._metadatas.clear()
            self._id_to_idx.clear()

        # Diske yaz
        self._persist()
        self._invalidate_norm_cache()
        return deleted_count

    def get_all_ids(self) -> List[str]:
        """Koleksiyondaki tüm kayıt ID'lerinin kopyasını döndürür."""
        return list(self._ids)

    def count(self) -> int:
        """Koleksiyondaki toplam kayıt sayısını döndürür."""
        return len(self._ids)

    def peek(self, limit: int = 5) -> Dict[str, Any]:
        """Koleksiyondaki ilk birkaç kaydı gösterir."""
        return self.get(limit=limit)

    def __repr__(self) -> str:
        return (
            f"<Collection(name='{self.name}', count={self.count()}, "
            f"dimension={self.dimension}, metric='{self.metric}')>"
        )
