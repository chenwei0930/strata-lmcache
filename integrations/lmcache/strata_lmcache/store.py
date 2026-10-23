# SPDX-License-Identifier: MIT
"""Content-addressed blob store behind the Strata LMCache server module.

One entry is one opaque session image (what ``strata --serve`` writes for
``SAVE``).  Entries are addressed by the key the Strata server computes from a
token prefix, so four Strata services can share one LMCache daemon without
knowing each other's GPU memory or engine state.

Layout under ``root``::

    objects/<aa>/<key>.bin     the image, exactly the bytes Strata wrote
    objects/<aa>/<key>.json    {"key", "meta", "bytes", "sha256", "accessed"}

The store is intentionally independent of LMCache's own KV page format: the
hybrid GDN/QSA state and Strata's quantized K/V are opaque to LMCache, and the
LMCache module contract passes opaque bytes to plugins.  LMCache still owns the
process, transport, lifecycle and observability; this module owns the bytes.

The class is pure Python and is unit-tested without LMCache installed.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$")
_CHUNK_MAX = 64 << 20


class StoreError(ValueError):
    """A refused operation: bad key, unknown upload, short write, quota, ..."""


class StoreFull(StoreError):
    """The new entry does not fit even after evicting every other entry."""


def check_key(key: str) -> str:
    if not isinstance(key, str) or not _KEY_RE.match(key):
        raise StoreError(f"invalid key {key!r}")
    return key


def _atomic_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)


class _Upload:
    __slots__ = ("key", "meta", "path", "file", "expected", "written", "hash")

    def __init__(self, key: str, meta: dict, path: Path, expected: int):
        self.key = key
        self.meta = meta
        self.path = path
        self.expected = expected
        self.written = 0
        self.hash = hashlib.sha256()
        self.file = open(path, "wb")


class BlobStore:
    """Thread-safe, quota-bounded, least-recently-used blob store."""

    def __init__(self, root: str | os.PathLike, max_bytes: int = 0, fsync: bool = True):
        self.root = Path(root)
        self.max_bytes = max(0, int(max_bytes))
        self.fsync = bool(fsync)
        self._lock = threading.RLock()
        self._index: dict[str, dict] = {}
        self._uploads: dict[str, _Upload] = {}
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._objects = self.root / "objects"
        self._objects.mkdir(parents=True, exist_ok=True)
        self._load()

    # ------------------------------------------------------------------ load
    def _load(self) -> None:
        now = time.time()
        broken = 0
        for folder in sorted(self._objects.iterdir()):
            if not folder.is_dir():
                continue
            for meta_path in folder.glob("*.json"):
                try:
                    doc = json.loads(meta_path.read_text(encoding="utf-8"))
                    key = check_key(doc["key"])
                    size = int(doc["bytes"])
                    blob = self._blob(key)
                    if not blob.is_file() or blob.stat().st_size != size:
                        raise StoreError("blob missing or wrong size")
                    doc.setdefault("meta", {})
                    doc.setdefault("accessed", now)
                    self._index[key] = doc
                    self._bytes += size
                except (OSError, ValueError, KeyError, TypeError):
                    broken += 1
                    try:
                        meta_path.unlink()
                    except OSError:
                        pass
        if broken:
            print(f"[strata-lmcache] dropped {broken} incomplete cache entr"
                  f"{'y' if broken == 1 else 'ies'}", flush=True)

    # ---------------------------------------------------------------- paths
    @staticmethod
    def _file_key(key: str) -> str:
        # A key is content, not a path: it may contain ':' (which Windows
        # forbids in file names).  Hash it for the filesystem only; the
        # original key stays in the metadata document.
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _blob(self, key: str) -> Path:
        h = self._file_key(key)
        return self._objects / h[:2] / f"{h}.bin"

    def _meta_path(self, key: str) -> Path:
        h = self._file_key(key)
        return self._objects / h[:2] / f"{h}.json"

    def _touch(self, key: str) -> None:
        doc = self._index.get(key)
        if doc is not None:
            doc["accessed"] = time.time()

    # --------------------------------------------------------------- lookup
    def match(self, keys) -> tuple[str, dict] | None:
        """The first key in ``keys`` that exists, with its metadata.

        ``keys`` is ordered by the caller (longest prefix first).  A hit
        refreshes the entry's LRU stamp; a missing entry is forgotten.
        """
        with self._lock:
            for key in keys:
                if not isinstance(key, str) or key not in self._index:
                    continue
                if not self._blob(key).is_file():
                    self._drop(key)
                    continue
                self._touch(key)
                self._hits += 1
                return key, dict(self._index[key].get("meta") or {})
            self._misses += 1
            return None

    def meta(self, key: str) -> dict | None:
        with self._lock:
            doc = self._index.get(check_key(key))
            if doc is None or not self._blob(key).is_file():
                return None
            self._touch(key)
            return {"meta": dict(doc.get("meta") or {}), "bytes": int(doc["bytes"]),
                    "sha256": doc.get("sha256", "")}

    def read_chunk(self, key: str, offset: int, length: int) -> bytes:
        """``length`` bytes at ``offset``; a short read at the end is normal."""
        if offset < 0 or length < 0 or length > _CHUNK_MAX:
            raise StoreError("bad read range")
        with self._lock:
            doc = self._index.get(check_key(key))
            if doc is None:
                raise StoreError(f"no such key {key!r}")
            self._touch(key)
            path, size = self._blob(key), int(doc["bytes"])
            if offset > size:
                raise StoreError("read starts past the end")
            length = min(length, size - offset)
            with open(path, "rb") as f:
                f.seek(offset)
                return f.read(length)

    # ----------------------------------------------------------------- write
    def begin_put(self, key: str, meta: dict, bytes_expected: int) -> str:
        """Start one upload.  The caller sends chunks, then ``end_put``."""
        check_key(key)
        if not isinstance(meta, dict):
            raise StoreError("meta must be an object")
        try:
            expected = int(bytes_expected)
        except (TypeError, ValueError):
            raise StoreError("bytes must be an integer") from None
        if expected < 0:
            raise StoreError("bytes must be >= 0")
        if self.max_bytes and expected > self.max_bytes:
            raise StoreFull(f"{expected} bytes is larger than the {self.max_bytes}-byte quota")
        with self._lock:
            upload_id = uuid.uuid4().hex
            h = self._file_key(key)
            part = self._objects / h[:2] / f".{h}.{upload_id}.part"
            part.parent.mkdir(parents=True, exist_ok=True)
            self._uploads[upload_id] = _Upload(key, dict(meta), part, expected)
            return upload_id

    def put_chunk(self, upload_id: str, offset: int, data: bytes) -> None:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise StoreError("chunk must be bytes")
        raw = bytes(data)
        if len(raw) > _CHUNK_MAX:
            raise StoreError(f"chunk is {len(raw)} bytes; the limit is {_CHUNK_MAX}")
        with self._lock:
            up = self._uploads.get(upload_id)
            if up is None:
                raise StoreError("unknown or finished upload")
            if offset != up.written:
                raise StoreError(f"chunk at {offset}, expected {up.written}")
            if up.written + len(raw) > up.expected:
                raise StoreError("chunk would exceed the announced size")
            up.file.write(raw)
            up.hash.update(raw)
            up.written += len(raw)

    def end_put(self, upload_id: str) -> dict:
        with self._lock:
            up = self._uploads.pop(upload_id, None)
            if up is None:
                raise StoreError("unknown or finished upload")
            try:
                up.file.flush()
                if self.fsync:
                    os.fsync(up.file.fileno())
                up.file.close()
                if up.written != up.expected:
                    raise StoreError(f"upload has {up.written} of {up.expected} bytes")
                digest = up.hash.hexdigest()
                final = self._blob(up.key)
                final.parent.mkdir(parents=True, exist_ok=True)
                os.replace(up.path, final)
                doc = {"key": up.key, "meta": up.meta, "bytes": up.written,
                       "sha256": digest, "created": time.time(), "accessed": time.time(),
                       "version": 1}
                _atomic_json(self._meta_path(up.key), doc)
            except Exception:
                try:
                    up.file.close()
                except Exception:
                    pass
                try:
                    up.path.unlink()
                except OSError:
                    pass
                raise
            old = self._index.get(up.key)
            if old is not None:
                self._bytes -= int(old["bytes"])
            self._index[up.key] = doc
            self._bytes += up.written
            self._evict(keep=up.key)
            return {"bytes": up.written, "sha256": digest}

    def abort_put(self, upload_id: str) -> None:
        with self._lock:
            up = self._uploads.pop(upload_id, None)
            if up is None:
                return
            try:
                up.file.close()
            except Exception:
                pass
            try:
                up.path.unlink()
            except OSError:
                pass

    # --------------------------------------------------------------- delete
    def _drop(self, key: str) -> None:
        doc = self._index.pop(key, None)
        if doc is None:
            return
        self._bytes -= int(doc.get("bytes") or 0)
        for path in (self._blob(key), self._meta_path(key)):
            try:
                path.unlink()
            except OSError:
                pass

    def delete(self, keys) -> int:
        removed = 0
        with self._lock:
            for key in keys:
                if isinstance(key, str) and key in self._index:
                    self._drop(key)
                    removed += 1
        return removed

    def clear(self) -> int:
        with self._lock:
            names = list(self._index)
            for key in names:
                self._drop(key)
            return len(names)

    # -------------------------------------------------------------- eviction
    def _evict(self, keep: str | None = None) -> None:
        if not self.max_bytes or self._bytes <= self.max_bytes:
            return
        if keep is not None and self._bytes > self.max_bytes:
            doc = self._index.get(keep)
            if doc is not None and int(doc["bytes"]) > self.max_bytes:
                self._drop(keep)
                raise StoreFull(f"entry is larger than the {self.max_bytes}-byte quota")
        victims = sorted(((d.get("accessed", 0), k) for k, d in self._index.items() if k != keep))
        for _, key in victims:
            if self._bytes <= self.max_bytes:
                break
            self._drop(key)
            self._evictions += 1
        if self._bytes > self.max_bytes:
            raise StoreFull("the quota cannot hold the new entry")

    # ----------------------------------------------------------------- stats
    def stats(self) -> dict:
        with self._lock:
            return {"entries": len(self._index), "bytes": self._bytes,
                    "max_bytes": self.max_bytes, "evictions": self._evictions,
                    "hits": self._hits, "misses": self._misses,
                    "uploads": len(self._uploads), "root": str(self.root)}

    def close(self) -> None:
        with self._lock:
            for upload_id in list(self._uploads):
                self.abort_put(upload_id)

    # for tests and operator tools
    def prune_orphans(self) -> int:
        """Remove ``.part`` files left by a crash.  Returns how many went."""
        removed = 0
        with self._lock:
            for folder in self._objects.iterdir():
                if not folder.is_dir():
                    continue
                for part in folder.glob(".*.part"):
                    try:
                        part.unlink()
                        removed += 1
                    except OSError:
                        pass
        return removed


class MemoryBlobStore:
    """The same contract as :class:`BlobStore`, held entirely in RAM.

    This is the recommended store for a host whose RAM is large and whose disk
    is not: a 63K-token Strata session is about 1.2 GiB, and a slow SSD makes a
    disk round trip the dominant cost.  Entries are ``bytearray`` objects, so
    an upload does not copy the image a second time when it completes.  The
    quota is enforced on admission *and* on completion; the least recently
    matched entry is evicted first.

    ``max_bytes=0`` means unlimited, which is only sensible when the operator
    has sized the machine.  A positive quota is strongly recommended.
    """

    def __init__(self, max_bytes: int = 64 << 30):
        self.max_bytes = max(0, int(max_bytes))
        self._lock = threading.RLock()
        self._entries: "collections.OrderedDict[str, dict]" = collections.OrderedDict()
        self._uploads: dict[str, dict] = {}
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    # ------------------------------------------------------------------ load
    def _entry(self, key: str) -> dict | None:
        e = self._entries.get(key)
        if e is not None:
            self._entries.move_to_end(key)
        return e

    # --------------------------------------------------------------- lookup
    def match(self, keys) -> tuple[str, dict] | None:
        with self._lock:
            for key in keys:
                if not isinstance(key, str):
                    continue
                e = self._entry(key)
                if e is None:
                    continue
                self._hits += 1
                return key, dict(e["meta"])
            self._misses += 1
            return None

    def meta(self, key: str) -> dict | None:
        with self._lock:
            e = self._entry(check_key(key))
            if e is None:
                return None
            return {"meta": dict(e["meta"]), "bytes": len(e["data"]),
                    "sha256": e["sha256"], "created": e["created"]}

    def read_chunk(self, key: str, offset: int, length: int) -> bytes:
        if offset < 0 or length < 0 or length > _CHUNK_MAX:
            raise StoreError("bad read range")
        with self._lock:
            e = self._entry(check_key(key))
            if e is None:
                raise StoreError(f"no such key {key!r}")
            data = e["data"]
            if offset > len(data):
                raise StoreError("read starts past the end")
            return bytes(data[offset:offset + min(length, len(data) - offset)])

    # ----------------------------------------------------------------- write
    def begin_put(self, key: str, meta: dict, bytes_expected: int) -> str:
        check_key(key)
        if not isinstance(meta, dict):
            raise StoreError("meta must be an object")
        try:
            expected = int(bytes_expected)
        except (TypeError, ValueError):
            raise StoreError("bytes must be an integer") from None
        if expected < 0:
            raise StoreError("bytes must be >= 0")
        with self._lock:
            if self.max_bytes and expected > self.max_bytes:
                raise StoreFull(f"{expected} bytes is larger than the {self.max_bytes}-byte quota")
            if self.max_bytes and self._bytes + expected > self.max_bytes:
                self._make_room(expected)
            upload_id = uuid.uuid4().hex
            self._uploads[upload_id] = {"key": key, "meta": dict(meta), "expected": expected,
                                        "buf": bytearray(), "offset": 0,
                                        "hash": hashlib.sha256()}
            return upload_id

    def put_chunk(self, upload_id: str, offset: int, data: bytes) -> None:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise StoreError("chunk must be bytes")
        raw = bytes(data)
        if len(raw) > _CHUNK_MAX:
            raise StoreError(f"chunk is {len(raw)} bytes; the limit is {_CHUNK_MAX}")
        with self._lock:
            up = self._uploads.get(upload_id)
            if up is None:
                raise StoreError("unknown or finished upload")
            if offset != up["offset"]:
                raise StoreError(f"chunk at {offset}, expected {up['offset']}")
            if up["offset"] + len(raw) > up["expected"]:
                raise StoreError("chunk would exceed the announced size")
            up["buf"].extend(raw)
            up["hash"].update(raw)
            up["offset"] += len(raw)

    def end_put(self, upload_id: str) -> dict:
        with self._lock:
            up = self._uploads.pop(upload_id, None)
            if up is None:
                raise StoreError("unknown or finished upload")
            if up["offset"] != up["expected"]:
                raise StoreError(f"upload has {up['offset']} of {up['expected']} bytes")
            key = up["key"]
            if self.max_bytes and len(up["buf"]) > self.max_bytes:
                raise StoreFull(f"entry is larger than the {self.max_bytes}-byte quota")
            old = self._entries.pop(key, None)
            if old is not None:
                self._bytes -= len(old["data"])
            digest = up["hash"].hexdigest()
            self._entries[key] = {"meta": up["meta"], "data": up["buf"], "sha256": digest,
                                  "created": time.time()}
            self._bytes += len(up["buf"])
            self._evict(keep=key)
            return {"bytes": len(up["buf"]), "sha256": digest}

    def abort_put(self, upload_id: str) -> None:
        with self._lock:
            self._uploads.pop(upload_id, None)

    # --------------------------------------------------------------- delete
    def delete(self, keys) -> int:
        removed = 0
        with self._lock:
            for key in keys:
                if not isinstance(key, str):
                    continue
                old = self._entries.pop(key, None)
                if old is not None:
                    self._bytes -= len(old["data"])
                    removed += 1
        return removed

    def clear(self) -> int:
        with self._lock:
            n = len(self._entries)
            self._entries.clear()
            self._bytes = 0
            return n

    def _evict(self, keep: str | None = None) -> None:
        if not self.max_bytes or self._bytes <= self.max_bytes:
            return
        for key in list(self._entries):
            if self._bytes <= self.max_bytes:
                break
            if key == keep:
                continue
            self._entries.pop(key, None)
            self._evictions += 1
        # Recompute after removals (the kept entry may itself be over quota).
        self._bytes = sum(len(e["data"]) for e in self._entries.values())
        if self._bytes > self.max_bytes:
            self._entries.pop(keep, None)
            self._bytes = sum(len(e["data"]) for e in self._entries.values())
            raise StoreFull("the quota cannot hold the new entry")

    def _make_room(self, incoming: int) -> None:
        """Evict least-recently-used entries until ``incoming`` fits too."""
        while self._entries and self._bytes + incoming > self.max_bytes:
            key = next(iter(self._entries))
            e = self._entries.pop(key)
            self._bytes -= len(e["data"])
            self._evictions += 1
        if self._bytes + incoming > self.max_bytes:
            raise StoreFull(f"{incoming} bytes does not fit the {self.max_bytes}-byte quota")

    # ----------------------------------------------------------------- stats
    def stats(self) -> dict:
        with self._lock:
            return {"entries": len(self._entries), "bytes": self._bytes,
                    "max_bytes": self.max_bytes, "evictions": self._evictions,
                    "hits": self._hits, "misses": self._misses,
                    "uploads": len(self._uploads), "storage": "memory"}

    def close(self) -> None:
        with self._lock:
            self._uploads.clear()
            self._entries.clear()
            self._bytes = 0

    def prune_orphans(self) -> int:
        """Memory has no crash leftovers; kept for the common module contract."""
        return 0
