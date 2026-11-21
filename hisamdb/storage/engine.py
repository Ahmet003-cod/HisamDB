"""
hisamdb.storage.engine
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
import shutil
import sqlite3
import struct
import threading
import time
import zlib
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import orjson as _orjson

    def _json_loads(s: str) -> Any:
        return _orjson.loads(s)

    def _json_dumps(obj: Any) -> str:
        return _orjson.dumps(obj).decode("utf-8")

    def _json_loads_bytes(b: bytes) -> Any:
        return _orjson.loads(b)

    def _json_dumps_bytes(obj: Any) -> bytes:
        return _orjson.dumps(obj)
except ImportError:
    def _json_loads(s: str) -> Any:
        return json.loads(s)

    def _json_dumps(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

    def _json_loads_bytes(b: bytes) -> Any:
        return json.loads(b.decode("utf-8"))

    def _json_dumps_bytes(obj: Any) -> bytes:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

_COL_CACHE_MAGIC = b"MCOL"
_COL_CACHE_HDR_FMT = "<4sHQIB3xI"  # magic(4), ver(2), epoch(8), n_items(4), flags(1), pad(3), body_crc(4) = 26B
_COL_CACHE_HDR_SIZE = struct.calcsize(_COL_CACHE_HDR_FMT)

from hisamdb.storage.binary_format import (
    append_vectors,
    load_vectors_file,
    truncate_vectors_file,
    update_vector_row_in_file,
    write_vectors_atomic,
)
from hisamdb.types import (
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
                    format_version INTEGER DEFAULT 2,
                    data_epoch INTEGER DEFAULT 0,
                    index_params_json TEXT
                );
                """
            )
            # v0.2.0 / v0.3.0 veritabanından yükseltme: eksik sütunları güvenle ekle
            cur = self._conn.execute("PRAGMA table_info(collections);")
            existing_cols = {row["name"] for row in cur.fetchall()}
            migrations = [
                ("index_type", "TEXT DEFAULT 'flat'"),
                ("quantization", "TEXT DEFAULT 'none'"),
                ("storage_dtype", "TEXT DEFAULT 'float32'"),
                ("table_name", "TEXT"),
                ("vector_file", "TEXT"),
                ("format_version", "INTEGER DEFAULT 1"),
                ("data_epoch", "INTEGER DEFAULT 0"),
                ("index_params_json", "TEXT"),
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

    def get_index_file_path(self, coll_name: str, suffix: str) -> Optional[str]:
        """
        Koleksiyonun disk indeks dosyası için güvenli (path-traversal korumalı) yol döndürür.
        Örn. suffix='.ivf.mvdbidx', '.ivfpq.mvdbidx', '.hnsw.bin', '.hnsw.meta.json', '.mvdbdisk', '.disk.meta'
        """
        if not self.is_persistent:
            return None
        validate_collection_name(coll_name)
        fname = f"{_collection_slug(coll_name)}{suffix}"
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
                        metadata TEXT,
                        sparse_vector TEXT
                    );
                    """
                )
                # Eski tablolardan yükseltme: sparse_vector sütunu yoksa ekle
                cur = self._conn.execute(f"PRAGMA table_info({table});")
                cols = {row["name"] for row in cur.fetchall()}
                if "sparse_vector" not in cols:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN sparse_vector TEXT;")
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
        data_epoch: int = 0,
        index_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        validate_collection_name(name)
        params_json = json.dumps(index_params) if index_params else None
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
                    "data_epoch": int(data_epoch),
                    "index_params": dict(index_params or {}),
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
                        table_name, vector_file, format_version,
                        data_epoch, index_params_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 2, ?, ?)
                    ON CONFLICT(name) DO UPDATE SET
                        dimension=excluded.dimension,
                        metric=excluded.metric,
                        index_type=excluded.index_type,
                        quantization=excluded.quantization,
                        storage_dtype=excluded.storage_dtype,
                        table_name=COALESCE(collections.table_name, excluded.table_name),
                        vector_file=COALESCE(collections.vector_file, excluded.vector_file),
                        format_version=2,
                        data_epoch=excluded.data_epoch,
                        index_params_json=excluded.index_params_json;
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
                        int(data_epoch),
                        params_json,
                    ),
                )

    def update_collection_epoch(self, name: str, data_epoch: int) -> None:
        """Koleksiyonun veri sürüm sayacını (epoch) günceller."""
        if not self.is_persistent:
            return
        with self._lock:
            assert self._conn is not None
            with self._conn:
                self._conn.execute(
                    "UPDATE collections SET data_epoch = ? WHERE name = ?;",
                    (int(data_epoch), name),
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
                       table_name, vector_file, format_version,
                       data_epoch, index_params_json
                FROM collections;
                """
            )
            results = []
            for row in cur.fetchall():
                d = dict(row)
                d["index_type"] = d.get("index_type") or "flat"
                d["quantization"] = d.get("quantization") or "none"
                d["storage_dtype"] = d.get("storage_dtype") or "float32"
                d["data_epoch"] = int(d.get("data_epoch") or 0)
                raw_p = d.get("index_params_json")
                try:
                    d["index_params"] = json.loads(raw_p) if raw_p else {}
                except Exception:
                    d["index_params"] = {}
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
            index_paths = [
                self.get_index_file_path(name, suffix)
                for suffix in (
                    ".ivf.mvdbidx",
                    ".ivfpq.mvdbidx",
                    ".hnsw.bin",
                    ".hnsw.meta.json",
                    ".mvdbdisk",
                    ".disk.meta",
                    ".wal",
                    ".items.mvdbcol",
                )
            ]

            with self._conn:
                self._conn.execute("DELETE FROM collections WHERE name = ?;", (name,))
                self._conn.execute(f"DROP TABLE IF EXISTS {table};")

            self._table_map.pop(name, None)
            self._vec_file_map.pop(name, None)

            for fpath in [vec_path, legacy_npy, *index_paths]:
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
    # Metadata ve Kayıt İşlemleri (Delta-Optimized + Hızlı Sütunsal Yükleme)
    # ------------------------------------------------------------------ #
    def save_items_batch(
        self,
        coll_name: str,
        items: List[Tuple[Any, ...]],
    ) -> None:
        """
        Yalnızca verilen delta öğelerini kaydeder veya günceller.
        4-tuple (id, int_id, doc, meta) veya 5-tuple (id, int_id, doc, meta, sparse) kabul eder.
        """
        if not items:
            return
        with self._lock:
            if not self.is_persistent:
                store = self._mem_items.setdefault(coll_name, {})
                for tup in items:
                    s_id, int_id, doc, meta = tup[0], tup[1], tup[2], tup[3]
                    sp = tup[4] if len(tup) > 4 else None
                    store[s_id] = {
                        "id": s_id,
                        "int_id": int_id,
                        "document": doc,
                        "metadata": meta,
                        "sparse_vector": sp,
                    }
                return

            assert self._conn is not None
            table = self._get_collection_table_name(coll_name)
            records = []
            for tup in items:
                s_id, int_id, doc, meta = tup[0], tup[1], tup[2], tup[3]
                sp = tup[4] if len(tup) > 4 else None
                records.append(
                    (
                        s_id,
                        int_id,
                        doc,
                        _json_dumps(meta) if meta is not None else None,
                        _json_dumps(sp) if sp is not None else None,
                    )
                )
            with self._conn:
                self._conn.executemany(
                    f"""
                    INSERT INTO {table} (id, int_id, document, metadata, sparse_vector)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        int_id=excluded.int_id,
                        document=excluded.document,
                        metadata=excluded.metadata,
                        sparse_vector=excluded.sparse_vector;
                    """,
                    records,
                )

    def replace_all_items_after_compaction(
        self,
        coll_name: str,
        items: List[Tuple[Any, ...]],
    ) -> None:
        """
        Sıkıştırma sonrasında int_id sıralamasını tek bir atomik SQLite
        işleminde yeniden yazar (UNIQUE(int_id) çakışması olmadan).
        """
        with self._lock:
            if not self.is_persistent:
                self._mem_items[coll_name] = {
                    tup[0]: {
                        "id": tup[0],
                        "int_id": tup[1],
                        "document": tup[2],
                        "metadata": tup[3],
                        "sparse_vector": tup[4] if len(tup) > 4 else None,
                    }
                    for tup in items
                }
                return

            assert self._conn is not None
            table = self._get_collection_table_name(coll_name)
            records = [
                (
                    tup[0],
                    tup[1],
                    tup[2],
                    _json_dumps(tup[3]) if tup[3] is not None else None,
                    _json_dumps(tup[4]) if (len(tup) > 4 and tup[4] is not None) else None,
                )
                for tup in items
            ]
            with self._conn:
                self._conn.execute(f"DELETE FROM {table};")
                if records:
                    self._conn.executemany(
                        f"INSERT INTO {table} (id, int_id, document, metadata, sparse_vector) VALUES (?, ?, ?, ?, ?);",
                        records,
                    )

    def load_all_items_columnar(
        self,
        coll_name: str,
    ) -> Tuple[List[str], List[int], List[Optional[str]], List[Optional[Dict[str, Any]]], List[Optional[Dict[str, float]]]]:
        """
        Yüksek hızlı sütunsal yükleyici (Sorun #1 çözümü):
        - Ham tuple imleci (row_factory=None) -> 100K sqlite3.Row + ara dict tahsisini sıfırlar!
        - Kurulu ise orjson.loads (3-5x hızlı), yoksa standart json.loads kullanır.
        - None metadata/sparse alanları sıfır JSON çağrısıyla atlar.
        """
        with self._lock:
            if not self.is_persistent:
                items = list(self._mem_items.get(coll_name, {}).values())
                items.sort(key=lambda x: x["int_id"])
                return (
                    [x["id"] for x in items],
                    [x["int_id"] for x in items],
                    [x["document"] for x in items],
                    [x["metadata"] for x in items],
                    [x.get("sparse_vector") for x in items],
                )

            assert self._conn is not None
            table = self._get_collection_table_name(coll_name)
            cur = self._conn.cursor()
            cur.row_factory = None  # Ham C tuple: 100K'da 3x+ daha hızlı!
            cur.execute(
                f"SELECT id, int_id, document, metadata, sparse_vector FROM {table} ORDER BY int_id ASC;"
            )
            rows = cur.fetchall()
            n = len(rows)
            if n == 0:
                return [], [], [], [], []

            ids_t, int_ids_t, docs_t, metas_t, sparses_t = zip(*rows)
            ids = list(ids_t)
            int_ids = list(int_ids_t)
            docs = list(docs_t)
            loads_fn = _json_loads

            if any(metas_t):
                try:
                    metas: List[Optional[Dict[str, Any]]] = [
                        loads_fn(m) if m is not None else None for m in metas_t
                    ]
                except Exception:
                    metas = [None] * n
                    for i, m_raw in enumerate(metas_t):
                        if m_raw is not None:
                            try:
                                metas[i] = loads_fn(m_raw)
                            except Exception as exc:
                                raise DataCorruptionError(
                                    f"Bozuk metadata JSON kaydı (id='{ids[i]}'): {exc}"
                                ) from exc
            else:
                metas = [None] * n

            if any(sparses_t):
                try:
                    sparses: List[Optional[Dict[str, float]]] = [
                        loads_fn(s) if s is not None else None for s in sparses_t
                    ]
                except Exception:
                    sparses = [None] * n
                    for i, s_raw in enumerate(sparses_t):
                        if s_raw is not None:
                            try:
                                sparses[i] = loads_fn(s_raw)
                            except Exception as exc:
                                raise DataCorruptionError(
                                    f"Bozuk sparse_vector JSON kaydı (id='{ids[i]}'): {exc}"
                                ) from exc
            else:
                sparses = [None] * n

            return ids, int_ids, docs, metas, sparses

    def load_all_items(self, coll_name: str) -> List[Dict[str, Any]]:
        """Geriye dönük uyumlu öğe yükleyici."""
        ids, int_ids, docs, metas, sparses = self.load_all_items_columnar(coll_name)
        return [
            {
                "id": ids[i],
                "int_id": int_ids[i],
                "document": docs[i],
                "metadata": metas[i],
                "sparse_vector": sparses[i],
            }
            for i in range(len(ids))
        ]

    # ------------------------------------------------------------------ #
    # Sütunsal İkili Önbellek (.items.mvdbcol — 100K'da 10x+ hızlı açılış)
    # ------------------------------------------------------------------ #
    def invalidate_items_columnar_cache(self, coll_name: str) -> None:
        """Herhangi bir yazma/silme işleminde sütunsal önbelleği anında geçersiz kılar."""
        if not self.is_persistent:
            return
        cache_path = self.get_index_file_path(coll_name, ".items.mvdbcol")
        if cache_path and os.path.exists(cache_path):
            try:
                os.remove(cache_path)
            except OSError:
                pass

    def save_items_columnar_cache(
        self,
        coll_name: str,
        epoch: int,
        ids: List[str],
        int_ids: List[int],
        docs: List[Optional[str]],
        metas: List[Optional[Dict[str, Any]]],
        sparses: List[Optional[Dict[str, float]]],
    ) -> None:
        """
        SQLite doğruluk kaynağının yanında epoch+CRC32 korumalı sütunsal ikili önbellek yazar.
        100K kaydı SQLite'tan tek tek çekmek yerine ~10-40 ms içinde yüklenebilir hale getirir.
        """
        if not self.is_persistent:
            return
        cache_path = self.get_index_file_path(coll_name, ".items.mvdbcol")
        if not cache_path:
            return
        n = len(ids)
        if n == 0:
            self.invalidate_items_columnar_cache(coll_name)
            return

        contiguous = (int_ids[-1] == n - 1 and int_ids[0] == 0)
        has_docs = any(d is not None for d in docs)
        has_metas = any(m is not None for m in metas)
        has_sparses = any(s is not None for s in sparses)
        ids_has_nul = any("\x00" in s for s in ids)

        flags = (
            (1 if contiguous else 0)
            | (2 if has_docs else 0)
            | (4 if has_metas else 0)
            | (8 if has_sparses else 0)
            | (16 if ids_has_nul else 0)
        )

        ids_blob = _json_dumps_bytes(ids) if ids_has_nul else "\x00".join(ids).encode("utf-8")
        parts: List[bytes] = [struct.pack("<I", len(ids_blob)), ids_blob]
        if not contiguous:
            parts.append(np.asarray(int_ids, dtype=np.int32).tobytes())
        if has_docs:
            b_docs = _json_dumps_bytes(docs)
            parts.append(struct.pack("<I", len(b_docs)))
            parts.append(b_docs)
        if has_metas:
            b_metas = _json_dumps_bytes(metas)
            parts.append(struct.pack("<I", len(b_metas)))
            parts.append(b_metas)
        if has_sparses:
            b_sparses = _json_dumps_bytes(sparses)
            parts.append(struct.pack("<I", len(b_sparses)))
            parts.append(b_sparses)

        body = b"".join(parts)
        body_crc = zlib.crc32(body) & 0xFFFFFFFF
        header = struct.pack(
            _COL_CACHE_HDR_FMT, _COL_CACHE_MAGIC, 1, int(epoch), n, flags, body_crc
        )

        tmp_path = f"{cache_path}.tmp.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}"
        try:
            with open(tmp_path, "wb") as f:
                f.write(header)
                f.write(body)
                f.flush()
            os.replace(tmp_path, cache_path)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def load_items_columnar_cache(
        self,
        coll_name: str,
        expected_epoch: int,
    ) -> Optional[
        Tuple[
            List[str],
            List[int],
            List[Optional[str]],
            List[Optional[Dict[str, Any]]],
            List[Optional[Dict[str, float]]],
        ]
    ]:
        """
        Epoch ve CRC32 eşleşiyorsa sütunsal önbellekten milisaniyeler içinde yükler;
        yoksa veya geçersizse None döndürerek SQLite'a sorunsuz düşer.
        """
        if not self.is_persistent:
            return None
        cache_path = self.get_index_file_path(coll_name, ".items.mvdbcol")
        if not cache_path or not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path, "rb") as f:
                raw = f.read()
            if len(raw) < _COL_CACHE_HDR_SIZE:
                return None
            magic, ver, epoch, n, flags, expected_crc = struct.unpack(
                _COL_CACHE_HDR_FMT, raw[:_COL_CACHE_HDR_SIZE]
            )
            if magic != _COL_CACHE_MAGIC or ver != 1 or int(epoch) != int(expected_epoch):
                return None
            body = memoryview(raw)[_COL_CACHE_HDR_SIZE:]
            if (zlib.crc32(body) & 0xFFFFFFFF) != expected_crc:
                return None

            pos = 0
            (ids_len,) = struct.unpack_from("<I", body, pos)
            pos += 4
            ids_raw = bytes(body[pos : pos + ids_len])
            pos += ids_len
            if flags & 16:
                ids = _json_loads_bytes(ids_raw)
            else:
                ids = ids_raw.decode("utf-8").split("\x00")
            if len(ids) != n:
                return None

            if flags & 1:
                int_ids = list(range(n))
            else:
                int_ids = np.frombuffer(body[pos : pos + 4 * n], dtype=np.int32).tolist()
                pos += 4 * n

            if flags & 2:
                (d_len,) = struct.unpack_from("<I", body, pos)
                pos += 4
                docs = _json_loads_bytes(bytes(body[pos : pos + d_len]))
                pos += d_len
            else:
                docs = [None] * n

            if flags & 4:
                (m_len,) = struct.unpack_from("<I", body, pos)
                pos += 4
                metas = _json_loads_bytes(bytes(body[pos : pos + m_len]))
                pos += m_len
            else:
                metas = [None] * n

            if flags & 8:
                (s_len,) = struct.unpack_from("<I", body, pos)
                pos += 4
                sparses = _json_loads_bytes(bytes(body[pos : pos + s_len]))
                pos += s_len
            else:
                sparses = [None] * n

            return ids, int_ids, docs, metas, sparses
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # Snapshot / Backup & Verified Restore
    # ------------------------------------------------------------------ #
    @staticmethod
    def _file_sha256(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(4 * 1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def create_snapshot(self, snapshot_dir: str) -> Dict[str, Any]:
        """
        Tüm veritabanının (SQLite online backup + .mvdbvec + indeks dosyaları)
        SHA-256 doğrulamalı tutarlı anlık görüntüsünü (snapshot) oluşturur.
        """
        if not self.is_persistent or self.persist_dir is None or self._conn is None:
            raise ValueError("Snapshot yalnızca kalıcı (PersistentClient / MicroVectorDB(path)) modda alınabilir.")

        dest = os.path.abspath(snapshot_dir)
        os.makedirs(dest, exist_ok=True)
        files_manifest: Dict[str, Dict[str, Any]] = {}

        with self._lock:
            # 1. SQLite tutarlı online yedekleme
            dst_db = os.path.join(dest, "vectordb.sqlite3")
            dst_conn = sqlite3.connect(dst_db)
            try:
                self._conn.backup(dst_conn)
            finally:
                dst_conn.close()
            files_manifest["vectordb.sqlite3"] = {
                "size": os.path.getsize(dst_db),
                "sha256": self._file_sha256(dst_db),
            }

            # 2. Vektör, indeks, sütunsal önbellek ve WAL dosyalarını kopyala
            allowed_exts = (
                ".mvdbvec",
                ".mvdbidx",
                ".mvdbdisk",
                ".disk.meta",
                ".hnsw.bin",
                ".hnsw.meta.json",
                ".wal",
                ".items.mvdbcol",
            )
            for fname in sorted(os.listdir(self.persist_dir)):
                if fname.endswith(allowed_exts):
                    src_p = os.path.join(self.persist_dir, fname)
                    if os.path.isfile(src_p):
                        dst_p = os.path.join(dest, fname)
                        shutil.copy2(src_p, dst_p)
                        files_manifest[fname] = {
                            "size": os.path.getsize(dst_p),
                            "sha256": self._file_sha256(dst_p),
                        }

            manifest = {
                "format_version": 1,
                "created_at": time.time(),
                "files": files_manifest,
            }
            m_path = os.path.join(dest, "snapshot_manifest.json")
            with open(m_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            return manifest

    @classmethod
    def restore_snapshot(cls, snapshot_dir: str, target_dir: str, overwrite: bool = False) -> Dict[str, Any]:
        """
        Snapshot manifestindeki tüm SHA-256 sağlama toplamlarını doğrular ve
        veritabanını `target_dir` konumuna güvenle geri yükler.
        """
        src = os.path.abspath(snapshot_dir)
        dst = os.path.abspath(target_dir)
        m_path = os.path.join(src, "snapshot_manifest.json")
        if not os.path.exists(m_path):
            raise DataCorruptionError(f"Snapshot manifesti bulunamadı: '{m_path}'")

        with open(m_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        files_info: Dict[str, Dict[str, Any]] = manifest.get("files", {})
        if "vectordb.sqlite3" not in files_info:
            raise DataCorruptionError("Snapshot manifesti geçersiz: vectordb.sqlite3 eksik.")

        # 1. Tüm dosyaların bütünlüğünü kopyalamadan ÖNCE doğrula
        for fname, info in files_info.items():
            if not _is_safe_single_filename(fname):
                raise DataCorruptionError(f"Snapshot içinde güvensiz dosya adı: '{fname}'")
            src_p = os.path.join(src, fname)
            if not os.path.isfile(src_p):
                raise DataCorruptionError(f"Snapshot dosyası eksik: '{fname}'")
            actual_sha = cls._file_sha256(src_p)
            if actual_sha != info.get("sha256"):
                raise DataCorruptionError(f"Snapshot SHA-256 doğrulaması başarısız: '{fname}'")

        # 2. Hedef dizine güvenle geri yükle
        if os.path.exists(dst) and os.listdir(dst) and not overwrite:
            raise FileExistsError(
                f"Hedef dizin boş değil: '{dst}'. Üzerine yazmak için overwrite=True verin."
            )
        os.makedirs(dst, exist_ok=True)
        for fname in files_info:
            shutil.copy2(os.path.join(src, fname), os.path.join(dst, fname))
        return manifest

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
