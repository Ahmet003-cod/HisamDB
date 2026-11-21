"""
microvectordb.concurrency.executor
----------------------------------
Çok çekirdekli paralel toplu sorgu yürütücüsü (`BatchQueryExecutor`) ve
bloklamayan arka plan bakım yöneticisi (`BackgroundTaskManager`).

Python GIL notu:
- NumPy/OpenBLAS, hnswlib C++ graf araması ve mmap I/O çağrıları GIL'i serbest
  bıraktığından, iş parçacığı havuzu (ThreadPoolExecutor) bellek kopyalamadan
  (zero-copy, zero-pickle) gerçek çok çekirdekli QPS ölçeklenmesi sağlar.
"""

import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def default_query_workers() -> int:
    """OpenBLAS / OS aşırı iş parçacığı çakışmasını önleyen dengeli varsayılan işçi sayısı."""
    cores = os.cpu_count() or 4
    return max(2, min(cores, 8))


class BatchQueryExecutor:
    """
    Yeniden kullanılabilir, tembel (lazy) başlatılan paralel sorgu iş parçacığı havuzu.
    """

    def __init__(self, max_workers: Optional[int] = None) -> None:
        self._max_workers = max(1, int(max_workers or default_query_workers()))
        self._lock = threading.Lock()
        self._pool: Optional[ThreadPoolExecutor] = None
        self._closed = False

    @property
    def max_workers(self) -> int:
        return self._max_workers

    def _get_pool(self) -> ThreadPoolExecutor:
        with self._lock:
            if self._closed:
                self._closed = False
            if self._pool is None:
                self._pool = ThreadPoolExecutor(
                    max_workers=self._max_workers,
                    thread_name_prefix="mvdb-query",
                )
            return self._pool

    def submit(self, fn: Callable[..., R], *args: Any, **kwargs: Any) -> Future:
        return self._get_pool().submit(fn, *args, **kwargs)

    def map_chunks(
        self,
        chunk_fn: Callable[[Sequence[T]], List[R]],
        items: Sequence[T],
        max_workers: Optional[int] = None,
    ) -> List[R]:
        """
        Girdi listesini `workers` adet bitişik dilime bölüp paralel yürütür ve
        orijinal sırayı birebir koruyarak birleştirir.
        """
        n = len(items)
        if n == 0:
            return []
        workers = self._max_workers if max_workers is None else max(1, int(max_workers))
        workers = min(workers, n)
        if workers <= 1:
            return chunk_fn(items)

        # Eşit büyüklükte bitişik dilimler -> en düşük Python zamanlama yükü
        base, rem = divmod(n, workers)
        slices: List[Sequence[T]] = []
        pos = 0
        for i in range(workers):
            sz = base + (1 if i < rem else 0)
            if sz > 0:
                slices.append(items[pos : pos + sz])
                pos += sz

        pool = self._get_pool()
        futures = [pool.submit(chunk_fn, sl) for sl in slices]
        results: List[R] = []
        for fut in futures:
            results.extend(fut.result())
        return results

    def shutdown(self, wait: bool = True) -> None:
        with self._lock:
            self._closed = True
            if self._pool is not None:
                self._pool.shutdown(wait=wait)
                self._pool = None


class BackgroundTaskManager:
    """
    Bloklamayan arka plan compaction, checkpoint ve indeks yeniden inşası yöneticisi.
    Aynı anahtar (ör. 'compact') için eşzamanlı tekrarlı istekleri tek bir işte birleştirir (coalescing).
    """

    def __init__(self, max_workers: int = 2) -> None:
        self._max_workers = max(1, int(max_workers))
        self._lock = threading.Lock()
        self._pool: Optional[ThreadPoolExecutor] = None
        self._inflight: Dict[str, Future] = {}
        self._closed = False

    def _get_pool_locked(self) -> ThreadPoolExecutor:
        if self._closed:
            raise RuntimeError("BackgroundTaskManager kapatıldı.")
        if self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=self._max_workers,
                thread_name_prefix="mvdb-bg",
            )
        return self._pool

    def submit_coalesced(self, key: str, fn: Callable[[], R]) -> Future:
        """
        Aynı `key` altında çalışan veya bekleyen bir görev varsa mevcut Future'ı döndürür;
        yoksa yeni arka plan görevi başlatır.
        """
        with self._lock:
            existing = self._inflight.get(key)
            if existing is not None and not existing.done():
                return existing

            pool = self._get_pool_locked()

            def _wrapped() -> R:
                try:
                    return fn()
                finally:
                    with self._lock:
                        cur = self._inflight.get(key)
                        if cur is fut:
                            self._inflight.pop(key, None)

            fut: Future = pool.submit(_wrapped)
            self._inflight[key] = fut
            return fut

    def has_pending(self) -> bool:
        with self._lock:
            return any(not f.done() for f in self._inflight.values())

    def wait_all(self, timeout: Optional[float] = None) -> None:
        with self._lock:
            futs = list(self._inflight.values())
        for f in futs:
            try:
                f.result(timeout=timeout)
            except Exception:
                pass

    def shutdown(self, wait: bool = True) -> None:
        with self._lock:
            self._closed = True
            pool = self._pool
            self._pool = None
            futs = list(self._inflight.values())
            self._inflight.clear()
        if wait:
            for f in futs:
                try:
                    f.result()
                except Exception:
                    pass
        if pool is not None:
            pool.shutdown(wait=wait)
