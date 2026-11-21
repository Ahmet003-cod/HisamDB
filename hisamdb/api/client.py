"""
hisamdb.api.client
------------------------
Sıfır ek bağımlılıklı (yalnızca standart kütüphane `http.client` + Keep-Alive bağlantı havuzu)
MicroVectorDB Python HTTP İstemci SDK'sı:
- `RemoteMicroVectorDB`
- `RemoteCollection`
"""

import http.client
import json
import threading
import urllib.parse
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np

from hisamdb.types import (
    BatchVectors,
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    SearchResultItem,
    VectorDBError,
    VectorLike,
)

try:
    import orjson as _orjson

    def _dumps(obj: Any) -> bytes:
        return _orjson.dumps(obj)

    def _loads(raw: bytes) -> Any:
        return _orjson.loads(raw)
except ImportError:
    def _dumps(obj: Any) -> bytes:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    def _loads(raw: bytes) -> Any:
        return json.loads(raw.decode("utf-8"))


class RemoteAPIError(VectorDBError):
    """Uzak MicroVectorDB REST sunucusundan dönen HTTP hatası."""

    def __init__(self, status_code: int, message: str, error_code: Optional[str] = None) -> None:
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = int(status_code)
        self.error_code = error_code


def _to_float_list(vec: VectorLike) -> List[float]:
    if isinstance(vec, np.ndarray):
        return vec.astype(np.float32, copy=False).ravel().tolist()
    return [float(x) for x in vec]


def _to_batch_list(vecs: BatchVectors) -> List[List[float]]:
    if isinstance(vecs, np.ndarray):
        arr = np.asarray(vecs, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr.tolist()
    return [[float(x) for x in row] for row in vecs]


def _parse_search_items(raw_list: List[Dict[str, Any]]) -> List[SearchResultItem]:
    return [
        SearchResultItem(
            id=r["id"],
            score=float(r["score"]),
            distance=float(r["distance"]),
            document=r.get("document"),
            metadata=r.get("metadata"),
        )
        for r in raw_list
    ]


class RemoteCollection:
    """Uzak HisamDB REST sunucusu üzerindeki bir koleksiyon için ergonomik SDK."""

    def __init__(self, client: "RemoteHisamDB", name: str) -> None:
        self._client = client
        self.name = name
        self._base = f"/v1/collections/{urllib.parse.quote(name, safe='')}"

    def add(
        self,
        ids: List[str],
        embeddings: Optional[BatchVectors] = None,
        documents: Optional[List[Optional[str]]] = None,
        metadatas: Optional[List[Optional[Dict[str, Any]]]] = None,
        *,
        vectors: Optional[BatchVectors] = None,
        metadata: Optional[List[Optional[Dict[str, Any]]]] = None,
        sparse_vectors: Optional[List[Optional[Dict[Union[int, str], float]]]] = None,
    ) -> Dict[str, Any]:
        raw_v = embeddings if embeddings is not None else vectors
        if raw_v is None:
            raise ValueError("'vectors' veya 'embeddings' gereklidir.")
        raw_m = metadatas if metadatas is not None else metadata
        payload: Dict[str, Any] = {"ids": ids, "vectors": _to_batch_list(raw_v)}
        if documents is not None:
            payload["documents"] = documents
        if raw_m is not None:
            payload["metadatas"] = raw_m
        if sparse_vectors is not None:
            payload["sparse_vectors"] = sparse_vectors
        return self._client._request_json("POST", f"{self._base}/add", payload)

    def add_documents(
        self,
        documents: List[str],
        embeddings: BatchVectors,
        metadatas: Optional[List[Optional[Dict[str, Any]]]] = None,
        ids: Optional[List[str]] = None,
        doc_ids: Optional[List[str]] = None,
        parent_ids: Optional[List[str]] = None,
        sparse_vectors: Optional[List[Optional[Dict[Union[int, str], float]]]] = None,
    ) -> List[str]:
        payload: Dict[str, Any] = {
            "documents": documents,
            "embeddings": _to_batch_list(embeddings),
        }
        if metadatas is not None:
            payload["metadatas"] = metadatas
        if ids is not None:
            payload["ids"] = ids
        if doc_ids is not None:
            payload["doc_ids"] = doc_ids
        if parent_ids is not None:
            payload["parent_ids"] = parent_ids
        if sparse_vectors is not None:
            payload["sparse_vectors"] = sparse_vectors
        resp = self._client._request_json("POST", f"{self._base}/documents", payload)
        return resp["ids"]

    def upsert(
        self,
        ids: List[str],
        embeddings: Optional[BatchVectors] = None,
        documents: Optional[List[Optional[str]]] = None,
        metadatas: Optional[List[Optional[Dict[str, Any]]]] = None,
        *,
        vectors: Optional[BatchVectors] = None,
        metadata: Optional[List[Optional[Dict[str, Any]]]] = None,
        sparse_vectors: Optional[List[Optional[Dict[Union[int, str], float]]]] = None,
    ) -> Dict[str, Any]:
        raw_v = embeddings if embeddings is not None else vectors
        if raw_v is None:
            raise ValueError("'vectors' veya 'embeddings' gereklidir.")
        raw_m = metadatas if metadatas is not None else metadata
        payload: Dict[str, Any] = {"ids": ids, "vectors": _to_batch_list(raw_v)}
        if documents is not None:
            payload["documents"] = documents
        if raw_m is not None:
            payload["metadatas"] = raw_m
        if sparse_vectors is not None:
            payload["sparse_vectors"] = sparse_vectors
        return self._client._request_json("POST", f"{self._base}/upsert", payload)

    def search(
        self,
        query_vector: VectorLike,
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
        ef_search: Optional[int] = None,
        nprobe: Optional[int] = None,
        strategy: str = "auto",
        filter_strategy: str = "auto",
    ) -> List[SearchResultItem]:
        payload: Dict[str, Any] = {
            "query_vector": _to_float_list(query_vector),
            "limit": limit,
            "strategy": strategy,
            "filter_strategy": filter_strategy,
        }
        if filter is not None:
            payload["filter"] = filter
        if score_threshold is not None:
            payload["score_threshold"] = score_threshold
        if ef_search is not None:
            payload["ef_search"] = ef_search
        if nprobe is not None:
            payload["nprobe"] = nprobe
        resp = self._client._request_json("POST", f"{self._base}/search", payload)
        return _parse_search_items(resp["results"])

    def batch_search(
        self,
        query_vectors: BatchVectors,
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        filters: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        score_threshold: Optional[float] = None,
        ef_search: Optional[int] = None,
        nprobe: Optional[int] = None,
        max_workers: Optional[int] = None,
    ) -> List[List[SearchResultItem]]:
        payload: Dict[str, Any] = {
            "query_vectors": _to_batch_list(query_vectors),
            "limit": limit,
        }
        if filter is not None:
            payload["filter"] = filter
        if filters is not None:
            payload["filters"] = list(filters)
        if score_threshold is not None:
            payload["score_threshold"] = score_threshold
        if ef_search is not None:
            payload["ef_search"] = ef_search
        if nprobe is not None:
            payload["nprobe"] = nprobe
        if max_workers is not None:
            payload["max_workers"] = max_workers
        resp = self._client._request_json("POST", f"{self._base}/batch_search", payload)
        return [_parse_search_items(row) for row in resp["results"]]

    def sparse_search(
        self,
        query_text: Optional[str] = None,
        query_sparse: Optional[Dict[Union[int, str], float]] = None,
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResultItem]:
        payload: Dict[str, Any] = {"limit": limit}
        if query_text is not None:
            payload["query_text"] = query_text
        if query_sparse is not None:
            payload["query_sparse"] = query_sparse
        if filter is not None:
            payload["filter"] = filter
        if score_threshold is not None:
            payload["score_threshold"] = score_threshold
        resp = self._client._request_json("POST", f"{self._base}/sparse_search", payload)
        return _parse_search_items(resp["results"])

    def hybrid_search(
        self,
        query_vector: Optional[VectorLike] = None,
        query_text: Optional[str] = None,
        query_sparse: Optional[Dict[Union[int, str], float]] = None,
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        alpha: float = 0.5,
        beta: float = 0.5,
        sparse_weight: Optional[float] = None,
        fusion: str = "rrf",
        rrf_k: int = 60,
        score_threshold: Optional[float] = None,
        deduplicate: bool = False,
        group_by: Optional[str] = None,
        max_per_group: int = 1,
    ) -> List[SearchResultItem]:
        payload: Dict[str, Any] = {
            "limit": limit,
            "alpha": alpha,
            "beta": beta,
            "fusion": fusion,
            "rrf_k": rrf_k,
            "deduplicate": deduplicate,
            "max_per_group": max_per_group,
        }
        if query_vector is not None:
            payload["query_vector"] = _to_float_list(query_vector)
        if query_text is not None:
            payload["query_text"] = query_text
        if query_sparse is not None:
            payload["query_sparse"] = query_sparse
        if filter is not None:
            payload["filter"] = filter
        if sparse_weight is not None:
            payload["sparse_weight"] = sparse_weight
        if score_threshold is not None:
            payload["score_threshold"] = score_threshold
        if group_by is not None:
            payload["group_by"] = group_by
        resp = self._client._request_json("POST", f"{self._base}/hybrid_search", payload)
        return _parse_search_items(resp["results"])

    def retrieve_with_context(
        self,
        query_vector: Optional[VectorLike] = None,
        query_text: Optional[str] = None,
        limit: int = 5,
        window: int = 1,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        payload: Dict[str, Any] = {"limit": limit, "window": window}
        if query_vector is not None:
            payload["query_vector"] = _to_float_list(query_vector)
        if query_text is not None:
            payload["query_text"] = query_text
        if filter is not None:
            payload["filter"] = filter
        resp = self._client._request_json("POST", f"{self._base}/retrieve_context", payload)
        return resp["results"]

    def get(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if ids is not None:
            payload["ids"] = ids
        if where is not None:
            payload["where"] = where
        if limit is not None:
            payload["limit"] = limit
        return self._client._request_json("POST", f"{self._base}/get", payload)

    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> int:
        payload: Dict[str, Any] = {}
        if ids is not None:
            payload["ids"] = ids
        if where is not None:
            payload["where"] = where
        resp = self._client._request_json("POST", f"{self._base}/delete", payload)
        return int(resp["deleted"])

    def compact(self, background: bool = False) -> Dict[str, Any]:
        qs = "?background=true" if background else ""
        return self._client._request_json("POST", f"{self._base}/compact{qs}")

    def rebuild_index(self, wait: bool = True) -> Dict[str, Any]:
        qs = "?wait=true" if wait else "?wait=false"
        return self._client._request_json("POST", f"{self._base}/rebuild{qs}")

    def checkpoint(self) -> Dict[str, Any]:
        return self._client._request_json("POST", f"{self._base}/checkpoint")

    def explain(
        self,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"n_results": n_results}
        if where is not None:
            payload["where"] = where
        return self._client._request_json("POST", f"{self._base}/explain", payload)

    def stats(self) -> Dict[str, Any]:
        return self._client._request_json("GET", self._base)

    def count(self) -> int:
        return int(self.stats()["count"])

    def __repr__(self) -> str:
        return f"<RemoteCollection(name='{self.name}', base_url='{self._client.base_url}')>"


class RemoteHisamDB:
    """
    Sıfır ek bağımlılıklı, HTTP/1.1 Keep-Alive destekli HisamDB REST İstemcisi.

    Kullanım:
        with RemoteHisamDB("http://127.0.0.1:8000", api_key="hisam_...") as db:
            coll = db.get_or_create_collection("docs", dimension=128, index_type="hnsw")
            coll.add(ids=["1"], vectors=[[...]], documents=["Merhaba"])
            hits = coll.hybrid_search(query_vector=[...], query_text="Merhaba", limit=5)
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        parsed = urllib.parse.urlparse(base_url.rstrip("/"))
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError(f"Geçersiz sunucu URL'si: '{base_url}'")
        self.base_url = base_url.rstrip("/")
        self._scheme = parsed.scheme
        self._host = parsed.hostname
        self._port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self._base_path = parsed.path.rstrip("/")
        self._api_key = api_key
        self._timeout = float(timeout)
        self._local = threading.local()
        self._conns: List[http.client.HTTPConnection] = []
        self._conns_lock = threading.Lock()

    def _get_conn(self) -> http.client.HTTPConnection:
        conn: Optional[http.client.HTTPConnection] = getattr(self._local, "conn", None)
        if conn is None:
            if self._scheme == "https":
                conn = http.client.HTTPSConnection(self._host, self._port, timeout=self._timeout)
            else:
                conn = http.client.HTTPConnection(self._host, self._port, timeout=self._timeout)
            self._local.conn = conn
            with self._conns_lock:
                self._conns.append(conn)
        return conn

    def _reset_conn(self) -> None:
        conn: Optional[http.client.HTTPConnection] = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None

    def _request_raw(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> tuple[int, bytes]:
        full_path = f"{self._base_path}{path}"
        body = _dumps(payload) if payload is not None else None
        headers: Dict[str, str] = {
            "Connection": "keep-alive",
            "Accept": "application/json, text/plain",
        }
        if body is not None:
            headers["Content-Type"] = "application/json"
            headers["Content-Length"] = str(len(body))
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        for attempt in range(2):
            conn = self._get_conn()
            try:
                conn.request(method, full_path, body=body, headers=headers)
                resp = conn.getresponse()
                data = resp.read()
                return resp.status, data
            except (http.client.RemoteDisconnected, http.client.CannotSendRequest, ConnectionResetError, BrokenPipeError):
                self._reset_conn()
                if attempt == 1:
                    raise
            except Exception:
                self._reset_conn()
                raise
        raise RuntimeError("HTTP isteği tamamlanamadı.")

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Any:
        status_code, raw = self._request_raw(method, path, payload)
        try:
            parsed = _loads(raw) if raw else {}
        except Exception:
            text_err = raw.decode("utf-8", errors="replace")
            if status_code >= 400:
                raise RemoteAPIError(status_code, text_err)
            raise
        if status_code >= 400:
            detail = parsed.get("detail", str(parsed)) if isinstance(parsed, dict) else str(parsed)
            err_code = parsed.get("error") if isinstance(parsed, dict) else None
            if status_code == 404 and err_code == "collection_not_found":
                raise CollectionNotFoundError(str(detail))
            if status_code == 409 and err_code == "collection_already_exists":
                raise CollectionAlreadyExistsError(str(detail))
            raise RemoteAPIError(status_code, str(detail), error_code=err_code)
        return parsed

    def health(self) -> Dict[str, Any]:
        return self._request_json("GET", "/health")

    def ready(self) -> Dict[str, Any]:
        return self._request_json("GET", "/ready")

    def metrics(self) -> str:
        status_code, raw = self._request_raw("GET", "/metrics")
        if status_code >= 400:
            raise RemoteAPIError(status_code, raw.decode("utf-8", errors="replace"))
        return raw.decode("utf-8")

    def list_collections(self) -> List[str]:
        return self._request_json("GET", "/v1/collections")["collections"]

    def create_collection(
        self,
        name: str,
        dimension: Optional[int] = None,
        metric: str = "cosine",
        index_type: str = "flat",
        quantization: str = "none",
        storage_dtype: str = "float32",
        use_mmap: Optional[bool] = None,
        keep_raw_vectors: bool = True,
        index_params: Optional[Dict[str, Any]] = None,
        get_or_create: bool = False,
    ) -> RemoteCollection:
        payload: Dict[str, Any] = {
            "name": name,
            "metric": metric,
            "index_type": index_type,
            "quantization": quantization,
            "storage_dtype": storage_dtype,
            "keep_raw_vectors": keep_raw_vectors,
            "get_or_create": get_or_create,
        }
        if dimension is not None:
            payload["dimension"] = dimension
        if use_mmap is not None:
            payload["use_mmap"] = use_mmap
        if index_params is not None:
            payload["index_params"] = index_params
        self._request_json("POST", "/v1/collections", payload)
        return RemoteCollection(self, name)

    def get_or_create_collection(self, name: str, **kwargs: Any) -> RemoteCollection:
        return self.create_collection(name=name, get_or_create=True, **kwargs)

    def get_collection(self, name: str) -> RemoteCollection:
        coll = RemoteCollection(self, name)
        coll.stats()  # Var olduğunu doğrula (yoksa CollectionNotFoundError)
        return coll

    def delete_collection(self, name: str) -> None:
        self._request_json("DELETE", f"/v1/collections/{urllib.parse.quote(name, safe='')}")

    def create_snapshot(self, snapshot_name: str) -> Dict[str, Any]:
        return self._request_json("POST", "/v1/admin/snapshot", {"snapshot_name": snapshot_name})

    def close(self) -> None:
        with self._conns_lock:
            for c in self._conns:
                try:
                    c.close()
                except Exception:
                    pass
            self._conns.clear()
        self._local.conn = None

    def __enter__(self) -> "RemoteHisamDB":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<RemoteHisamDB(base_url='{self.base_url}')>"


# Geriye dönük uyumluluk takma adı
RemoteMicroVectorDB = RemoteHisamDB
