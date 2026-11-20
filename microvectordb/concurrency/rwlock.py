"""
microvectordb.concurrency.rwlock
--------------------------------
Yazıcı-öncelikli (writer-preferring) Reader-Writer Lock.
Çoklu eşzamanlı okuyucuların (query/search/get) birbirini engellemeden çalışmasına
izin verirken yazma işlemlerine (add/upsert/delete) münhasır (exclusive) erişim sağlar.
"""

import threading
from contextlib import contextmanager
from typing import Iterator


class RWLock:
    """
    Hafif, standart kütüphane tabanlı Reader-Writer Lock.
    Bekleyen yazıcı varken yeni okuyucular sıraya alınır (writer starvation önlenir).
    """

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0
        self._writer_active = False
        self._writers_waiting = 0

    def acquire_read(self) -> None:
        with self._cond:
            while self._writer_active or self._writers_waiting > 0:
                self._cond.wait()
            self._readers += 1

    def release_read(self) -> None:
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self) -> None:
        with self._cond:
            self._writers_waiting += 1
            try:
                while self._writer_active or self._readers > 0:
                    self._cond.wait()
                self._writer_active = True
            finally:
                self._writers_waiting -= 1

    def release_write(self) -> None:
        with self._cond:
            self._writer_active = False
            self._cond.notify_all()

    @contextmanager
    def read_lock(self) -> Iterator[None]:
        self.acquire_read()
        try:
            yield
        finally:
            self.release_read()

    @contextmanager
    def write_lock(self) -> Iterator[None]:
        self.acquire_write()
        try:
            yield
        finally:
            self.release_write()
