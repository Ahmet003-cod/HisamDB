"""
microvectordb.storage.engine
----------------------------
Güvenli, yüksek hızlı, atomik disk kalıcılığı ve metadata depolama motoru.

v0.3.0 Güvenlik ve Performans İyileştirmeleri:
1. Path Traversal koruması + SHA-256 tabanlı çakışmasız tablo/dosya adlandırması
   ("my-docs" ve "my_docs" artık asla aynı tabloyu paylaşmaz!).
2. Sürümlü `.mvdbvec` formatı ile O(M) artımlı ekleme, atomik yeniden yazma,
   CRC32 doğrulama ve gerçek `use_mmap=True` desteği.
3. Eski v0.2.0 `.npy` + `items_<clean>` veritabanları için otomatik geriye dönük uyumluluk.
4. SQLite WAL modu + yalnızca değişen satırları yazan delta kalıcılık + 500'lü chunked silme.
5. EphemeralClient (persist_dir=None) için sıfır-SQLite hızlı bellek yolu.
6. Thread-safe dahili RLock koruması ve temiz `close()` kaynak serbest bırakma.
"""

import hashlib
import json
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from microvectordb.storage.binary_format import (
    append_vectors,
    load_vectors_file,
    truncate_vectors_file,
    update_vector_row_in_file,
    write_vectors_atomic,
)
from microvectordb.types import (
    DataCorruptionError,
    InvalidCollectionNameError,
)

_DELETE_CHUNK_SIZE = 500
_MAX_COLLECTION_NAME_LEN = 128


def validate_collection_name(name: str) -> str:
    """Koleksiyon adının güvenli ve geçerli olduğunu doğrular."""
    if not isinstance(name, str) or not name.strip():
        raise InvalidCollectionNameError("Koleksiyon adı boş olmayan bir metin (str) olmalıdır.")
    if len(name) > _MAX_COLLECTION_NAME_LEN:
        raise InvalidCollectionNameError(
            f"Koleksiyon adı en fazla {_MAX_COLLECTION_NAME_LEN} karakter olabilir."
        )
    if "\x00" in name or any(ord(ch) < 32 for ch in name):
        raise InvalidCollectionNameError("Koleksiyon adı kontrol karakteri içeremez.")
    return name


def _collection_slug(coll_name: str) -> str:
    """
    Her koleksiyon adı için deterministik, SQL-güvenli, dosya sistemi-güvenli ve
    çakışmasız bir tanımlayıcı üretir: '<temiz_onek>_<sha256[:12]>'.
    """
    prefix = "".join(c if c.isalnum() else "_" for c in coll_name)[:32].strip("_") or "coll"
    digest = hashlib.sha256(coll_name.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _legacy_table_name(coll_name: str) -> str:
    """v0.2.0 tablo adlandırma kuralı (yalnızca eski DB göçü için)."""
    clean = "".join(c if c.isalnum() else "_" for c in coll_name)
    return f"items_{clean}"


def _is_safe_single_filename(name: str) -> bool:
    """Eski v0.2.0 .npy dosyasını ararken path traversal'ı engeller."""
    if not name or "/" in name or "\\" in name or ".." in name:
        return False
    return os.path.basename(name) == name


class StorageEngine:
    """
    Koleksiyonların vektörlerini (.mvdbvec) ve metadata kayıtlarını (SQLite WAL)
    disk üzerinde güvenli ve artımlı olarak saklayan depolama motoru.
    """

    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = os.path.abspath(persist_dir) if persist_dir is not None else None
        self.is_persistent = self.persist_dir is not None
        self._lock = threading.RLock()
        self._closed = False

        # Koleksiyon adı -> gerçek SQLite tablo adı & vektör dosya adı eşlemesi
        self._table_map: Dict[str, str] = {}
        self._vec_file_map: Dict[str, str] = {}

        if self.is_persistent:
            assert self.persist_dir is not None
            os.makedirs(self.persist_dir, exist_ok=True)
            self._db_path = os.path.join(self.persist_dir, "vectordb.sqlite3")
            self._conn: Optional[sqlite3.Connection] = sqlite3.connect(
                self._db_path,
                check_same_thread=False,
                timeout=30.0,
            )
            self._conn.row_factory = sqlite3.Row
            self._configure_sqlite()
            self._init_tables()
        else:
            self._db_path = ":memory:"
            self._conn = None
            # EphemeralClient için sıfır-SQLite saf bellek depoları
            self._mem_collections: Dict[str, Dict[str, Any]] = {}
            self._mem_items: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def _configure_sqlite(self) -> None:
        assert self._conn is not None
        with self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute("PRAGMA temp_store=MEMORY;")

    def _init_tables(self) -> None:
        assert self._conn is not None
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS collections (
                    name TEXT PRIMARY KEY,
                    dimension INTEGER,
                    metric TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    index_type TEXT DEFAULT 'flat',
                    quantization TEXT DEFAULT 'none',
                    storage_dtype TEXT DEFAULT 'float32',
                    table_name TEXT,
                    vector_file TEXT,
                    format_version INTEGER DEFAULT 2
                );
                """
            )
            # v0.2.0 veritabanından yükseltme: eksik sütunları güvenle ekle
            cur = self._conn.execute("PRAGMA table_info(collections);")
            existing_cols = {row["name"] for row in cur.fetchall()}
            migrations = [
                ("index_type", "TEXT DEFAULT 'flat'"),
                ("quantization", "TEXT DEFAULT 'none'"),
                ("storage_dtype", "TEXT DEFAULT 'float32'"),
                ("table_name", "TEXT"),
                ("vector_file", "TEXT"),
                ("format_version", "INTEGER DEFAULT 1"),
            ]
            for col_name, col_def in migrations:
                if col_name not in existing_cols:
                    self._conn.execute(f"ALTER TABLE collections ADD COLUMN {col_name} {col_def};")

    def _safe_persist_path(self, filename: str) -> str:
        """Hedef yolun kesinlikle persist_dir içinde kaldığını garanti eder."""
        assert self.persist_dir is not None
        if not _is_safe_single_filename(filename):
            raise InvalidCollectionNameError(f"Güvensiz dosya yolu reddedildi: '{filename}'")
        target = os.path.abspath(os.path.join(self.persist_dir, filename))
        base = os.path.abspath(self.persist_dir)
        if os.path.commonpath([base, target]) != base:
            raise InvalidCollectionNameError(f"Path traversal girişimi engellendi: '{filename}'")
        return target

    def _get_collection_table_name(self, coll_name: str) -> str:
        """Koleksiyon adını çakışmasız ve SQL-güvenli tablo adına dönüştürür."""
        validate_collection_name(coll_name)
        if coll_name in self._table_map:
            return self._table_map[coll_name]

        # Kalıcı DB'de daha önce kaydedilmiş tablo adı varsa (örn. v0.2.0 göçü) onu kullan
        if self.is_persistent and self._conn is not None:
            cur = self._conn.execute(
                "SELECT table_name FROM collections WHERE name = ?;", (coll_name,)
            )
            row = cur.fetchone()
            if row is not None and row["table_name"]:
                self._table_map[coll_name] = row["table_name"]
                return row["table_name"]
            # Eski v0.2.0 veritabanı: table_name NULL ama legacy tablo mevcut olabilir
            if row is not None:
                legacy = _legacy_table_name(coll_name)
                t_check = self._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (legacy,)
                ).fetchone()
                if t_check is not None:
                    self._table_map[coll_name] = legacy
                    return legacy

        table = f"items_{_collection_slug(coll_name)}"
        self._table_map[coll_name] = table
        return table

    def _get_vector_file_path(self, coll_name: str) -> str:
        """Yeni sürümlü `.mvdbvec` dosya yolunu güvenli biçimde döndürür."""
        validate_collection_name(coll_name)
        if coll_name in self._vec_file_map:
            return self._safe_persist_path(self._vec_file_map[coll_name])

        if self.is_persistent and self._conn is not None:
            cur = self._conn.execute(
                "SELECT vector_file FROM collections WHERE name = ?;", (coll_name,)
            )
            row = cur.fetchone()
            if row is not None and row["vector_file"]:
                self._vec_file_map[coll_name] = row["vector_file"]
                return self._safe_persist_path(row["vector_file"])

        fname = f"{_collection_slug(coll_name)}.mvdbvec"
        self._vec_file_map[coll_name] = fname
        return self._safe_persist_path(fname)

    def _get_legacy_npy_path(self, coll_name: str) -> Optional[str]:
        """v0.2.0 `{coll_name}_vectors.npy` dosyasını güvenli biçimde kontrol eder."""
        if not self.is_persistent or not _is_safe_single_filename(coll_name):
            return None
        candidate = self._safe_persist_path(f"{coll_name}_vectors.npy")
        return candidate if os.path.exists(candidate) else None

    def ensure_collection_storage(self, coll_name: str) -> None:
        validate_collection_name(coll_name)
        with self._lock:
            if not self.is_persistent:
                self._mem_items.setdefault(coll_name, {})
                return

            assert self._conn is not None
            table = self._get_collection_table_name(coll_name)
            with self._conn:
                self._conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                        id TEXT PRIMARY KEY,
                        int_id INTEGER UNIQUE,
                        document TEXT,
                        metadata TEXT
                    );
                    """
                )
                self._conn.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{table}_int_id ON {table}(int_id);"
                )

    def save_collection_meta(
        self,
        name: str,
        dimension: Optional[int],
        metric: str,
        created_at: float,
        index_type: str = "flat",
        quantization: str = "none",
        storage_dtype: str = "float32",
    ) -> None:
        validate_collection_name(name)
        with self._lock:
            if not self.is_persistent:
                self._mem_collections[name] = {
                    "name": name,
                    "dimension": dimension,
                    "metric": metric,
                    "created_at": created_at,
                    "index_type": index_type,
                    "quantization": quantization,
                    "storage_dtype": storage_dtype,
                }
                return

            assert self._conn is not None
            table = self._get_collection_table_name(name)
            vec_fname = self._vec_file_map.get(name) or f"{_collection_slug(name)}.mvdbvec"
            self._vec_file_map[name] = vec_fname
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO collections (
                        name, dimension, metric, created_at,
                        index_type, quantization, storage_dtype,
                        table_name, vector_file, format_version
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 2)
                    ON CONFLICT(name) DO UPDATE SET
                        dimension=excluded.dimension,
                        metric=excluded.metric,
                        index_type=excluded.index_type,
                        quantization=excluded.quantization,
                        storage_dtype=excluded.storage_dtype,
                        table_name=COALESCE(collections.table_name, excluded.table_name),
                        vector_file=COALESCE(collections.vector_file, excluded.vector_file),
                        format_version=2;
                    """,
                    (
                        name,
                        dimension,
                        metric,
                        created_at,
                        index_type,
                        quantization,
                        storage_dtype,
                        table,
                        vec_fname,
                    ),
                )

    def load_collections_meta(self) -> List[Dict[str, Any]]:
        with self._lock:
            if not self.is_persistent:
                return list(self._mem_collections.values())

            assert self._conn is not None
            cur = self._conn.execute(
                """
                SELECT name, dimension, metric, created_at,
                       index_type, quantization, storage_dtype,
                       table_name, vector_file, format_version
                FROM collections;
                """
            )
            results = []
            for row in cur.fetchall():
                d = dict(row)
                d["index_type"] = d.get("index_type") or "flat"
                d["quantization"] = d.get("quantization") or "none"
                d["storage_dtype"] = d.get("storage_dtype") or "float32"
                if d.get("table_name"):
                    self._table_map[d["name"]] = d["table_name"]
                if d.get("vector_file"):
                    self._vec_file_map[d["name"]] = d["vector_file"]
                results.append(d)
            return results

    def delete_collection_storage(self, name: str) -> None:
        validate_collection_name(name)
        with self._lock:
            if not self.is_persistent:
                self._mem_collections.pop(name, None)
                self._mem_items.pop(name, None)
                return

            assert self._conn is not None
            table = self._get_collection_table_name(name)
            vec_path = self._get_vector_file_path(name)
            legacy_npy = self._get_legacy_npy_path(name)

            with self._conn:
                self._conn.execute("DELETE FROM collections WHERE name = ?;", (name,))
                self._conn.execute(f"DROP TABLE IF EXISTS {table};")

            self._table_map.pop(name, None)
            self._vec_file_map.pop(name, None)

            for fpath in (vec_path, legacy_npy):
                if fpath and os.path.exists(fpath):
                    try:
                        os.remove(fpath)
                    except OSError:
                        pass

    # ------------------------------------------------------------------ #
    # Vektör Matrisi Okuma / Yazma / Artımlı Ekleme
    # ------------------------------------------------------------------ #
    def save_vectors(
        self,
        coll_name: str,
        vectors: np.ndarray,
        compute_crc: bool = True,
    ) -> None:
        """Vektör matrisini atomik olarak `.mvdbvec` formatında diske yazar."""
        if not self.is_persistent:
            return
        with self._lock:
            vec_path = self._get_vector_file_path(coll_name)
            write_vectors_atomic(vec_path, vectors, compute_data_crc=compute_crc)

    def append_vectors_delta(
        self,
        coll_name: str,
        new_vectors: np.ndarray,
        expected_prev_count: int,
    ) -> None:
        """Yeni eklenen vektörleri O(M) sürede mevcut `.mvdbvec` dosyasına ekler."""
        if not self.is_persistent:
            return
        with self._lock:
            vec_path = self._get_vector_file_path(coll_name)
            append_vectors(vec_path, new_vectors, expected_prev_count=expected_prev_count)

    def update_vector_row(self, coll_name: str, row_idx: int, vec: np.ndarray) -> None:
        """Tek bir vektör satırını diskte yerinde O(D) sürede günceller."""
        if not self.is_persistent:
            return
        with self._lock:
            vec_path = self._get_vector_file_path(coll_name)
            update_vector_row_in_file(vec_path, row_idx, vec)

    def truncate_vectors(self, coll_name: str, target_count: int) -> None:
        """Çökme kurtarmasında fazla yazılmış satırları temizler."""
        if not self.is_persistent:
            return
        with self._lock:
            vec_path = self._get_vector_file_path(coll_name)
            if os.path.exists(vec_path):
                truncate_vectors_file(vec_path, target_count)

    def load_vectors(
        self,
        coll_name: str,
        dimension: Optional[int],
        use_mmap: bool = False,
        verify_crc: bool = True,
    ) -> np.ndarray:
        """
        Vektör matrisini diskten yükler.
        Önce yeni `.mvdbvec` dosyasını kontrol eder; yoksa eski v0.2.0 `.npy`
        dosyasını okuyup otomatik olarak `.mvdbvec` formatına göç ettirir.
        """
        if not self.is_persistent:
            dim = dimension or 0
            return np.empty((0, dim), dtype=np.float32)

        with self._lock:
            vec_path = self._get_vector_file_path(coll_name)
            if os.path.exists(vec_path):
                return load_vectors_file(vec_path, use_mmap=use_mmap, verify_crc=verify_crc)

            # v0.2.0 otomatik göç (legacy .npy -> .mvdbvec)
            legacy_npy = self._get_legacy_npy_path(coll_name)
            if legacy_npy is not None:
                arr = np.load(legacy_npy, allow_pickle=False)
                arr_f32 = np.ascontiguousarray(arr, dtype=np.float32)
                write_vectors_atomic(vec_path, arr_f32, compute_data_crc=True)
                if use_mmap:
                    return load_vectors_file(vec_path, use_mmap=True, verify_crc=False)
                return arr_f32

            dim = dimension or 0
            return np.empty((0, dim), dtype=np.float32)

    def get_vector_file_size(self, coll_name: str) -> int:
        """Koleksiyonun diskteki vektör dosya boyutunu (byte) döndürür."""
        if not self.is_persistent:
            return 0
        with self._lock:
            vec_path = self._get_vector_file_path(coll_name)
            if os.path.exists(vec_path):
                return os.path.getsize(vec_path)
            legacy_npy = self._get_legacy_npy_path(coll_name)
            return os.path.getsize(legacy_npy) if legacy_npy else 0

    # ------------------------------------------------------------------ #
    # Metadata ve Kayıt İşlemleri (Delta-Optimized)
    # ------------------------------------------------------------------ #
    def save_items_batch(
        self,
        coll_name: str,
        items: List[Tuple[str, int, Optional[str], Optional[Dict[str, Any]]]],
    ) -> None:
        """Yalnızca verilen delta öğelerini kaydeder veya günceller."""
        if not items:
            return
        with self._lock:
            if not self.is_persistent:
                store = self._mem_items.setdefault(coll_name, {})
                for s_id, int_id, doc, meta in items:
                    store[s_id] = {
                        "id": s_id,
                        "int_id": int_id,
                        "document": doc,
                        "metadata": meta,
                    }
                return

            assert self._conn is not None
            table = self._get_collection_table_name(coll_name)
            records = [
                (
                    s_id,
                    int_id,
                    doc,
                    json.dumps(meta, ensure_ascii=False) if meta is not None else None,
                )
                for s_id, int_id, doc, meta in items
            ]
            with self._conn:
                self._conn.executemany(
                    f"""
                    INSERT INTO {table} (id, int_id, document, metadata)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        int_id=excluded.int_id,
                        document=excluded.document,
                        metadata=excluded.metadata;
                    """,
                    records,
                )

    def replace_all_items_after_compaction(
        self,
        coll_name: str,
        items: List[Tuple[str, int, Optional[str], Optional[Dict[str, Any]]]],
    ) -> None:
        """
        Silme/sıkıştırma sonrasında int_id sıralamasını tek bir atomik SQLite
        işleminde yeniden yazar (UNIQUE(int_id) çakışması olmadan).
        """
        with self._lock:
            if not self.is_persistent:
                self._mem_items[coll_name] = {
                    s_id: {"id": s_id, "int_id": int_id, "document": doc, "metadata": meta}
                    for s_id, int_id, doc, meta in items
                }
                return

            assert self._conn is not None
            table = self._get_collection_table_name(coll_name)
            records = [
                (
                    s_id,
                    int_id,
                    doc,
                    json.dumps(meta, ensure_ascii=False) if meta is not None else None,
                )
                for s_id, int_id, doc, meta in items
            ]
            with self._conn:
                self._conn.execute(f"DELETE FROM {table};")
                if records:
                    self._conn.executemany(
                        f"INSERT INTO {table} (id, int_id, document, metadata) VALUES (?, ?, ?, ?);",
                        records,
                    )

    def load_all_items(self, coll_name: str) -> List[Dict[str, Any]]:
        with self._lock:
            if not self.is_persistent:
                items = list(self._mem_items.get(coll_name, {}).values())
                items.sort(key=lambda x: x["int_id"])
                return items

            assert self._conn is not None
            table = self._get_collection_table_name(coll_name)
            cur = self._conn.execute(
                f"SELECT id, int_id, document, metadata FROM {table} ORDER BY int_id ASC;"
            )
            results = []
            for row in cur.fetchall():
                try:
                    meta = json.loads(row["metadata"]) if row["metadata"] else None
                except json.JSONDecodeError as exc:
                    raise DataCorruptionError(
                        f"Bozuk metadata JSON kaydı (id='{row['id']}'): {exc}"
                    ) from exc
                results.append(
                    {
                        "id": row["id"],
                        "int_id": row["int_id"],
                        "document": row["document"],
                        "metadata": meta,
                    }
                )
            return results

    def delete_items_by_ids(self, coll_name: str, ids: List[str]) -> None:
        """Belirtilen ID'leri 500'lü gruplar halinde siler (SQLite değişken sınırına güvenli)."""
        if not ids:
            return
        with self._lock:
            if not self.is_persistent:
                store = self._mem_items.get(coll_name)
                if store:
                    for item_id in ids:
                        store.pop(item_id, None)
                return

            assert self._conn is not None
            table = self._get_collection_table_name(coll_name)
            with self._conn:
                for start in range(0, len(ids), _DELETE_CHUNK_SIZE):
                    chunk = ids[start : start + _DELETE_CHUNK_SIZE]
                    placeholders = ",".join("?" for _ in chunk)
                    self._conn.execute(
                        f"DELETE FROM {table} WHERE id IN ({placeholders});", chunk
                    )

    def count_items(self, coll_name: str) -> int:
        with self._lock:
            if not self.is_persistent:
                return len(self._mem_items.get(coll_name, {}))
            assert self._conn is not None
            table = self._get_collection_table_name(coll_name)
            cur = self._conn.execute(f"SELECT COUNT(*) FROM {table};")
            return int(cur.fetchone()[0])

    def get_db_file_size(self) -> int:
        if not self.is_persistent or not os.path.exists(self._db_path):
            return 0
        total = os.path.getsize(self._db_path)
        for suffix in ("-wal", "-shm"):
            p = self._db_path + suffix
            if os.path.exists(p):
                total += os.path.getsize(p)
        return total

    def close(self) -> None:
        """Veritabanı bağlantısını güvenle kapatır."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None
