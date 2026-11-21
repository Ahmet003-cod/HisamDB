"""
microvectordb.storage.wal
-------------------------
İkili Write-Ahead Log (WAL) ve Çökme Kurtarma (Crash Recovery) Motoru.

Çerçeve (Frame) Biçimi (21-byte başlık + UTF-8 JSON yükü):
  [0..4)   Magic:      b"MWAL"
  [4..12)  LSN:        uint64 (monoton artan günlük sıra numarası)
  [12..13) OpType:     uint8  (1=ADD, 2=UPSERT, 3=DELETE, 4=CHECKPOINT, 5=COMPACT)
  [13..17) PayloadLen: uint32
  [17..21) CRC32:      uint32 (bytes [0..17) + payload üzerinden sağlama toplamı)
  [21..)   Payload:    UTF-8 JSON baytları

Kurtarma garantisi:
- Yazma ortasında elektrik/süreç kesintisi olursa sondaki yarım/bozuk çerçeve
  CRC32 doğrulamasıyla tespit edilir ve son sağlam çerçeveden itibaren temizce budanır.
- Tüm geçerli ADD/UPSERT/DELETE/CHECKPOINT kayıtları sırayla geri oynatılır (replay).
"""

import json
import os
import struct
import threading
import zlib
from dataclasses import dataclass
from typing import Any, BinaryIO, Dict, List, Optional, Tuple

_WAL_MAGIC = b"MWAL"
_WAL_HDR_FMT = "<4sQBII"  # magic(4), lsn(8), op(1), payload_len(4), crc32(4) = 21 bytes
_WAL_HDR_SIZE = struct.calcsize(_WAL_HDR_FMT)

OP_ADD = 1
OP_UPSERT = 2
OP_DELETE = 3
OP_CHECKPOINT = 4
OP_COMPACT = 5


@dataclass
class WALRecord:
    lsn: int
    op: int
    payload: Dict[str, Any]


class WriteAheadLog:
    """Thread-safe, CRC32 korumalı ikili Write-Ahead Log (kalıcı dosya tutamaçlı)."""

    def __init__(self, path: str, sync_on_write: bool = True) -> None:
        self.path = path
        self.sync_on_write = sync_on_write
        self._lock = threading.RLock()
        self._lsn: int = 0
        self._fh: Optional[BinaryIO] = None
        self._sync_lsn_from_disk()

    def _sync_lsn_from_disk(self) -> None:
        if not os.path.exists(self.path):
            return
        records, _ = self.replay(truncate_corrupt_tail=False)
        if records:
            self._lsn = records[-1].lsn

    def _ensure_open_locked(self) -> BinaryIO:
        if self._fh is None or self._fh.closed:
            parent = os.path.dirname(os.path.abspath(self.path))
            os.makedirs(parent, exist_ok=True)
            self._fh = open(self.path, "ab")
        return self._fh

    def _close_handle_locked(self) -> None:
        if self._fh is not None:
            try:
                if not self._fh.closed:
                    self._fh.flush()
                    self._fh.close()
            except OSError:
                pass
            self._fh = None

    def close(self) -> None:
        """Açık WAL dosya tutamacını temiz biçimde kapatır."""
        with self._lock:
            self._close_handle_locked()

    def __del__(self) -> None:
        try:
            self._close_handle_locked()
        except Exception:
            pass

    @property
    def current_lsn(self) -> int:
        return self._lsn

    def append(self, op: int, payload: Dict[str, Any], sync: Optional[bool] = None) -> int:
        """Yeni bir WAL kaydı ekler ve atanan LSN değerini döndürür."""
        do_sync = self.sync_on_write if sync is None else sync
        payload_bytes = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

        with self._lock:
            self._lsn += 1
            lsn = self._lsn
            prefix = struct.pack("<4sQBI", _WAL_MAGIC, lsn, int(op), len(payload_bytes))
            crc = zlib.crc32(payload_bytes, zlib.crc32(prefix)) & 0xFFFFFFFF
            frame = prefix + struct.pack("<I", crc) + payload_bytes

            fh = self._ensure_open_locked()
            fh.write(frame)
            fh.flush()
            if do_sync:
                os.fsync(fh.fileno())
            return lsn

    def append_add(self, count: int, start_slot: int, epoch: int) -> int:
        return self.append(OP_ADD, {"count": count, "start_slot": start_slot, "epoch": epoch})

    def append_delete(self, ids: List[str], slots: List[int], epoch: int) -> int:
        return self.append(OP_DELETE, {"ids": ids, "slots": slots, "epoch": epoch})

    def append_checkpoint(self, epoch: int, alive_count: int) -> int:
        return self.append(OP_CHECKPOINT, {"epoch": epoch, "alive_count": alive_count})

    def replay(self, truncate_corrupt_tail: bool = True) -> Tuple[List[WALRecord], int]:
        """
        WAL dosyasını baştan sona okur ve CRC32 doğrular.
        Yarım kalmış/bozuk kuyruk çerçevesi varsa güvenle budar.

        Dönüş: (geçerli_kayıtlar, budanan_bayt_sayısı)
        """
        with self._lock:
            if self._fh is not None and not self._fh.closed:
                self._fh.flush()
            if not os.path.exists(self.path):
                return [], 0

            with open(self.path, "rb") as f:
                data = f.read()

            total_len = len(data)
            pos = 0
            records: List[WALRecord] = []

            while pos < total_len:
                if pos + _WAL_HDR_SIZE > total_len:
                    break  # Yarım kalmış başlık (torn write)
                magic, lsn, op, plen, crc = struct.unpack(_WAL_HDR_FMT, data[pos : pos + _WAL_HDR_SIZE])
                if magic != _WAL_MAGIC:
                    break
                frame_end = pos + _WAL_HDR_SIZE + plen
                if frame_end > total_len:
                    break  # Yarım kalmış yük (torn write)

                prefix = data[pos : pos + 17]
                payload_bytes = data[pos + _WAL_HDR_SIZE : frame_end]
                actual_crc = zlib.crc32(payload_bytes, zlib.crc32(prefix)) & 0xFFFFFFFF
                if actual_crc != crc:
                    break

                try:
                    payload = json.loads(payload_bytes.decode("utf-8"))
                except Exception:
                    break

                records.append(WALRecord(lsn=int(lsn), op=int(op), payload=payload))
                pos = frame_end

            truncated = total_len - pos
            if truncated > 0 and truncate_corrupt_tail:
                self._close_handle_locked()
                with open(self.path, "r+b") as f:
                    f.truncate(pos)
                    f.flush()
                    os.fsync(f.fileno())

            if records:
                self._lsn = max(self._lsn, records[-1].lsn)
            return records, truncated

    def truncate(self) -> None:
        """Checkpoint / compaction sonrası WAL'ı sıfırlar."""
        with self._lock:
            self._close_handle_locked()
            if os.path.exists(self.path):
                with open(self.path, "wb") as f:
                    f.flush()
                    os.fsync(f.fileno())

    def size_bytes(self) -> int:
        with self._lock:
            if self._fh is not None and not self._fh.closed:
                self._fh.flush()
            return os.path.getsize(self.path) if os.path.exists(self.path) else 0
