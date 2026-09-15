"""
microvectordb.storage
---------------------
Disk kalıcılığı (Persistence) ve Metadata depolama motoru.
Vektörleri yüksek hızlı binary formatta (.npy), metadata ve ID'leri ise
Python'un dahili ve güvenilir SQLite motorunda depolar.
"""

import os
import json
import sqlite3
import numpy as np
from typing import Any, Dict, List, Optional, Tuple


class StorageEngine:
    """
    Koleksiyonların vektörlerini ve metadata kayıtlarını disk üzerinde
    güvenli ve hızlı bir şekilde saklayan depolama motoru.
    """

    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir
        self.is_persistent = persist_dir is not None

        if self.is_persistent:
            os.makedirs(self.persist_dir, exist_ok=True)
            self._db_path = os.path.join(self.persist_dir, "vectordb.sqlite3")
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        else:
            self._db_path = ":memory:"
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)

        self._conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self) -> None:
        """Veritabanı tablolarını ve indekslerini hazırlar."""
        with self._conn:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS collections (
                    name TEXT PRIMARY KEY,
                    dimension INTEGER,
                    metric TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
            """)

    def _get_collection_table_name(self, coll_name: str) -> str:
        """Koleksiyon adını güvenli tablo adına dönüştürür."""
        clean = "".join(c if c.isalnum() else "_" for c in coll_name)
        return f"items_{clean}"

    def ensure_collection_storage(self, coll_name: str) -> None:
        """Belirtilen koleksiyon için metadata tablosunu oluşturur."""
        table = self._get_collection_table_name(coll_name)
        with self._conn:
            self._conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    id TEXT PRIMARY KEY,
                    int_id INTEGER UNIQUE,
                    document TEXT,
                    metadata TEXT
                );
            """)
            self._conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table}_int_id ON {table}(int_id);
            """)

    def save_collection_meta(self, name: str, dimension: Optional[int], metric: str, created_at: float) -> None:
        """Koleksiyon genel bilgilerini kaydeder veya günceller."""
        with self._conn:
            self._conn.execute("""
                INSERT INTO collections (name, dimension, metric, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    dimension=excluded.dimension,
                    metric=excluded.metric;
            """, (name, dimension, metric, created_at))

    def load_collections_meta(self) -> List[Dict[str, Any]]:
        """Mevcut tüm koleksiyonların üst bilgilerini döndürür."""
        cur = self._conn.cursor()
        cur.execute("SELECT name, dimension, metric, created_at FROM collections;")
        rows = cur.fetchall()
        return [dict(row) for row in rows]

    def delete_collection_storage(self, name: str) -> None:
        """Koleksiyonun tablosunu ve diskteki vektör dosyasını siler."""
        table = self._get_collection_table_name(name)
        with self._conn:
            self._conn.execute("DELETE FROM collections WHERE name = ?;", (name,))
            self._conn.execute(f"DROP TABLE IF EXISTS {table};")

        if self.is_persistent:
            vec_file = os.path.join(self.persist_dir, f"{name}_vectors.npy")
            if os.path.exists(vec_file):
                try:
                    os.remove(vec_file)
                except OSError:
                    pass

    # ------------------------------------------------------------------ #
    # Vektör Matrisi Okuma / Yazma
    # ------------------------------------------------------------------ #
    def save_vectors(self, coll_name: str, vectors: np.ndarray) -> None:
        """Vektör matrisini diske binary .npy olarak yazar."""
        if not self.is_persistent:
            return
        vec_file = os.path.join(self.persist_dir, f"{coll_name}_vectors.npy")
        np.save(vec_file, vectors.astype(np.float32))

    def load_vectors(self, coll_name: str, dimension: Optional[int], use_mmap: bool = False) -> np.ndarray:
        """
        Vektör matrisini diskten yükler.

        Args:
            coll_name: Koleksiyon adı.
            dimension: Vektör boyutu.
            use_mmap: True ise memory-mapped olarak yükler (RAM tasarrufu).
                      Büyük veri setlerinde (>1GB) önerilir.
        """
        if not self.is_persistent:
            dim = dimension or 0
            return np.empty((0, dim), dtype=np.float32)

        vec_file = os.path.join(self.persist_dir, f"{coll_name}_vectors.npy")
        if os.path.exists(vec_file):
            mmap_mode = 'r' if use_mmap else None
            arr = np.load(vec_file, mmap_mode=mmap_mode)
            return arr.astype(np.float32) if not use_mmap else arr

        dim = dimension or 0
        return np.empty((0, dim), dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Metadata ve Kayıt İşlemleri
    # ------------------------------------------------------------------ #
    def save_items_batch(
        self,
        coll_name: str,
        items: List[Tuple[str, int, Optional[str], Optional[Dict[str, Any]]]]
    ) -> None:
        """
        Toplu öğe (ID, int_id, document, metadata) kaydeder veya günceller.
        items: list of (str_id, int_id, document, metadata_dict)
        """
        table = self._get_collection_table_name(coll_name)
        records = [
            (
                s_id,
                int_id,
                doc,
                json.dumps(meta, ensure_ascii=False) if meta is not None else None
            )
            for s_id, int_id, doc, meta in items
        ]

        with self._conn:
            self._conn.executemany(f"""
                INSERT INTO {table} (id, int_id, document, metadata)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    int_id=excluded.int_id,
                    document=excluded.document,
                    metadata=excluded.metadata;
            """, records)

    def load_all_items(self, coll_name: str) -> List[Dict[str, Any]]:
        """Koleksiyondaki tüm kayıtları sırasıyla (int_id'ye göre) getirir."""
        table = self._get_collection_table_name(coll_name)
        cur = self._conn.cursor()
        cur.execute(f"SELECT id, int_id, document, metadata FROM {table} ORDER BY int_id ASC;")
        rows = cur.fetchall()

        results = []
        for row in rows:
            meta = json.loads(row["metadata"]) if row["metadata"] else None
            results.append({
                "id": row["id"],
                "int_id": row["int_id"],
                "document": row["document"],
                "metadata": meta
            })
        return results

    def delete_items_by_ids(self, coll_name: str, ids: List[str]) -> None:
        """Belirtilen ID'lere sahip öğeleri tablodan siler."""
        if not ids:
            return
        table = self._get_collection_table_name(coll_name)
        placeholders = ",".join("?" for _ in ids)
        with self._conn:
            self._conn.execute(f"DELETE FROM {table} WHERE id IN ({placeholders});", ids)

    def count_items(self, coll_name: str) -> int:
        """Koleksiyondaki kayıt sayısını döndürür."""
        table = self._get_collection_table_name(coll_name)
        cur = self._conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {table};")
        return cur.fetchone()[0]

    def close(self) -> None:
        """Veritabanı bağlantısını kapatır."""
        try:
            self._conn.close()
        except Exception:
            pass
