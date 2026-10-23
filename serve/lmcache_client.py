# SPDX-License-Identifier: MIT
"""Blob clients for Strata's shared prefix cache.

Two providers are available:

``lmcache``
    Talks to a running ``lmcache server`` that loaded the ``strata_blob``
    module (``integrations/lmcache/strata_lmcache``).  LMCache owns the daemon,
    transport and lifecycle; blobs move in bounded chunks.

``filesystem``
    A shared directory.  No Python package besides Strata is needed; it is the
    fallback for one host with several cards, and it is what the unit tests use.
    It uses the same key space and metadata, so switching provider does not
    change what a key means.

Both clients implement the same tiny interface::

    match(keys) -> (key, meta) | None
    meta(key)   -> dict | None
    download(key, path) -> dict
    upload(key, path, meta) -> dict
    stats() -> dict
    close()
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import sys
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

CHUNK_BYTES = 8 << 20
_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,239}$")


class BlobError(RuntimeError):
    """A provider refused or failed an operation; the bridge degrades, never breaks a request."""


# The frame is defined beside the LMCache module.  Keep one implementation:
# the repository ships both sides, and a test asserts they agree.
try:  # pragma: no cover - exercised through the integration tests
    _HERE = Path(__file__).resolve().parent
    _INTEGRATION = _HERE.parent / "integrations" / "lmcache"
    if str(_INTEGRATION) not in sys.path:
        sys.path.insert(0, str(_INTEGRATION))
    from strata_lmcache.wire import WireError, decode, encode  # type: ignore
except Exception:  # pragma: no cover - a copied serve/ directory
    import struct

    class WireError(ValueError):
        pass

    def encode(header: dict, data: bytes = b"") -> bytes:
        blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
        return struct.pack(">II", len(blob), len(data)) + blob + bytes(data)

    def decode(payload: bytes, max_data: int | None = None) -> tuple[dict, bytes]:
        if len(payload) < 8:
            raise WireError("truncated frame")
        hl, dl = struct.unpack(">II", payload[:8])
        if hl <= 0 or hl > (1 << 20) or (max_data is not None and dl > max_data):
            raise WireError("bad frame lengths")
        end = 8 + hl
        if end + dl != len(payload):
            raise WireError("frame length mismatch")
        return json.loads(payload[8:end].decode("utf-8")), payload[end:]


def _check_key(key: str) -> str:
    if not isinstance(key, str) or not _KEY_RE.match(key):
        raise BlobError(f"invalid cache key {key!r}")
    return key


def _sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, doc: dict) -> None:
    tmp = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)


class FilesystemClient:
    """A shared directory of ``<key>.bin`` + ``<key>.json`` entries."""

    def __init__(self, root: str | os.PathLike, max_bytes: int = 0):
        self.root = Path(root)
        self.max_bytes = max(0, int(max_bytes or 0))
        self._objects = self.root / "objects"
        self._objects.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._entries = 0
        self._bytes = 0
        self._scan()

    def _blob(self, key: str) -> Path:
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self._objects / h[:2] / f"{h}.bin"

    def _meta(self, key: str) -> Path:
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self._objects / h[:2] / f"{h}.json"

    def _scan(self) -> None:
        entries = total = 0
        for folder in self._objects.iterdir():
            if not folder.is_dir():
                continue
            for mp in folder.glob("*.json"):
                try:
                    d = json.loads(mp.read_text(encoding="utf-8"))
                    entries += 1
                    total += int(d.get("bytes", 0))
                except (OSError, ValueError, KeyError, TypeError):
                    continue
        self._entries, self._bytes = entries, total

    def match(self, keys):
        with self._lock:
            for key in keys:
                try:
                    _check_key(key)
                except BlobError:
                    continue
                meta_path, blob = self._meta(key), self._blob(key)
                try:
                    doc = json.loads(meta_path.read_text(encoding="utf-8"))
                    if not blob.is_file() or blob.stat().st_size != int(doc.get("bytes", -1)):
                        raise ValueError("blob missing or wrong size")
                except (OSError, ValueError, TypeError):
                    continue
                doc["accessed"] = time.time()
                try:
                    _write_json(meta_path, doc)
                except OSError:
                    pass
                self._hits += 1
                return key, dict(doc.get("meta") or {})
            self._misses += 1
            return None

    def meta(self, key: str):
        _check_key(key)
        try:
            doc = json.loads(self._meta(key).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return {"meta": dict(doc.get("meta") or {}), "bytes": int(doc.get("bytes", 0)),
                "sha256": doc.get("sha256", "")}

    def download(self, key: str, path: str | os.PathLike) -> dict:
        _check_key(key)
        with self._lock:
            got = self.meta(key)
            if got is None:
                raise BlobError(f"no cache entry {key}")
            blob = self._blob(key)
            tmp = f"{path}.{uuid.uuid4().hex[:8]}.part"
            try:
                with open(blob, "rb") as src, open(tmp, "wb") as dst:
                    for block in iter(lambda: src.read(1 << 20), b""):
                        dst.write(block)
                if os.path.getsize(tmp) != got["bytes"]:
                    raise BlobError("downloaded size does not match the entry")
                digest = _sha256_file(tmp)
                if got["sha256"] and digest != got["sha256"]:
                    raise BlobError("downloaded checksum does not match the entry")
                os.replace(tmp, path)
            finally:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        return got

    def upload(self, key: str, path: str | os.PathLike, meta: dict) -> dict:
        _check_key(key)
        size = os.path.getsize(path)
        digest = _sha256_file(path)
        with self._lock:
            old = self.meta(key)
            blob, meta_path = self._blob(key), self._meta(key)
            blob.parent.mkdir(parents=True, exist_ok=True)
            tmp = f"{blob}.{uuid.uuid4().hex[:8]}.part"
            try:
                with open(path, "rb") as src, open(tmp, "wb") as dst:
                    for block in iter(lambda: src.read(1 << 20), b""):
                        dst.write(block)
                    dst.flush()
                    os.fsync(dst.fileno())
                os.replace(tmp, blob)
                doc = {"key": key, "meta": dict(meta or {}), "bytes": size, "sha256": digest,
                       "created": time.time(), "accessed": time.time(), "version": 1}
                _write_json(meta_path, doc)
            finally:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
            if old is not None:
                self._bytes -= int(old.get("bytes") or 0)
            else:
                self._entries += 1
            self._bytes += size
            self._evict(keep=key)
        return {"bytes": size, "sha256": digest}

    def _evict(self, keep: str | None = None) -> None:
        if not self.max_bytes:
            return
        docs = []
        total = 0
        for folder in self._objects.iterdir():
            if not folder.is_dir():
                continue
            for mp in folder.glob("*.json"):
                try:
                    d = json.loads(mp.read_text(encoding="utf-8"))
                    docs.append((float(d.get("accessed", 0)), d["key"], int(d.get("bytes", 0))))
                    total += int(d.get("bytes", 0))
                except (OSError, ValueError, KeyError, TypeError):
                    continue
        if total <= self.max_bytes:
            return
        for _, key, size in sorted(docs):
            if total <= self.max_bytes:
                break
            if key == keep:
                continue
            for p in (self._blob(key), self._meta(key)):
                try:
                    p.unlink()
                except OSError:
                    pass
            total -= size
            self._bytes -= size
            self._entries = max(0, self._entries - 1)
            self._evictions += 1

    def stats(self) -> dict:
        return {"provider": "filesystem", "entries": self._entries, "bytes": self._bytes,
                "max_bytes": self.max_bytes, "evictions": self._evictions,
                "hits": self._hits, "misses": self._misses, "root": str(self.root)}

    def close(self) -> None:
        return None


class HttpBlobClient:
    """The same interface over the module's HTTP endpoint.

    This is the Docker/default path: the Strata image needs **no** LMCache or
    torch.  The LMCache daemon runs the ``strata_blob`` module, which starts a
    small HTTP front (``http_port``) over the same in-memory store.  The frames
    are the package's ``wire`` frames, one per request, so the HTTP layer only
    moves bytes and adds no second schema.
    """

    def __init__(self, url: str, timeout: float = 30.0, chunk_bytes: int = CHUNK_BYTES,
                 match_timeout: float | None = None):
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise BlobError(f"not an HTTP endpoint: {url!r}")
        self.url = url
        self.scheme = parts.scheme
        self.host = parts.hostname
        self.port = parts.port or (443 if parts.scheme == "https" else 80)
        self.base = (parts.path or "").rstrip("/")
        self.timeout = float(timeout)
        self.match_timeout = float(match_timeout) if match_timeout else min(20.0, self.timeout)
        self.chunk_bytes = max(4096, min(int(chunk_bytes), 64 << 20))
        self._lock = threading.RLock()
        self._conn: http.client.HTTPConnection | None = None
        self._closed = False
        self._hits = 0
        self._misses = 0

    def _connection(self) -> http.client.HTTPConnection:
        if self._conn is None:
            cls = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
            self._conn = cls(self.host, self.port, timeout=self.timeout)
        return self._conn

    def _drop(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    def _call(self, method: str, header: dict, data: bytes = b"", timeout: float | None = None):
        if self._closed:
            raise BlobError("the HTTP cache client is closed")
        body = encode({"v": 1, **header}, data)
        path = f"{self.base}/strata_blob/{method}"
        with self._lock:
            last = None
            for attempt in (0, 1):
                try:
                    conn = self._connection()
                    conn.timeout = self.timeout if timeout is None else float(timeout)
                    conn.request("POST", path, body=body,
                                 headers={"Content-Type": "application/octet-stream",
                                          "Content-Length": str(len(body))})
                    resp = conn.getresponse()
                    raw = resp.read()
                    if resp.status != 200:
                        raise BlobError(f"cache HTTP {resp.status}: {raw[:200]!r}")
                    head, payload = decode(raw, max_data=self.chunk_bytes)
                    if not head.get("ok", False):
                        raise BlobError(f"{head.get('kind', 'error')}: {head.get('error', 'unknown error')}")
                    return head, payload
                except BlobError:
                    raise
                except (http.client.HTTPException, OSError) as exc:
                    last = exc
                    self._drop()
                    if attempt:
                        break
            raise BlobError(f"cache HTTP request failed: {last}")

    def ping(self) -> dict:
        head, _ = self._call("ping", {}, timeout=self.match_timeout)
        return head

    def match(self, keys):
        head, _ = self._call("match", {"keys": list(keys)}, timeout=self.match_timeout)
        if not head.get("found"):
            self._misses += 1
            return None
        self._hits += 1
        return head.get("key"), dict(head.get("meta") or {})

    def meta(self, key: str):
        _check_key(key)
        head, _ = self._call("meta", {"key": key})
        if not head.get("found"):
            return None
        return {"meta": dict(head.get("meta") or {}), "bytes": int(head.get("bytes", 0)),
                "sha256": head.get("sha256", "")}

    def download(self, key: str, path: str | os.PathLike) -> dict:
        _check_key(key)
        got = self.meta(key)
        if got is None:
            raise BlobError(f"no cache entry {key}")
        size = int(got["bytes"])
        tmp = f"{path}.{uuid.uuid4().hex[:8]}.part"
        h = hashlib.sha256()
        try:
            with open(tmp, "wb") as out:
                offset = 0
                while offset < size:
                    want = min(self.chunk_bytes, size - offset)
                    head, data = self._call("get_chunk", {"key": key, "offset": offset, "length": want})
                    if not data or len(data) != int(head.get("n", -1)):
                        raise BlobError("cache returned a short chunk")
                    out.write(data)
                    h.update(data)
                    offset += len(data)
            if h.hexdigest() != got["sha256"]:
                raise BlobError("downloaded checksum does not match the entry")
            os.replace(tmp, path)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        return got

    def upload(self, key: str, path: str | os.PathLike, meta: dict) -> dict:
        _check_key(key)
        size = os.path.getsize(path)
        head, _ = self._call("begin_put", {"key": key, "meta": dict(meta or {}), "bytes": size})
        upload = str(head.get("upload", ""))
        if not upload:
            raise BlobError("cache did not return an upload id")
        try:
            with open(path, "rb") as src:
                offset = 0
                while offset < size:
                    data = src.read(min(self.chunk_bytes, size - offset))
                    if not data:
                        raise BlobError("the session image ended early")
                    self._call("put_chunk", {"upload": upload, "offset": offset}, data)
                    offset += len(data)
            head, _ = self._call("end_put", {"upload": upload})
        except Exception:
            try:
                self._call("abort_put", {"upload": upload}, timeout=5)
            except Exception:
                pass
            raise
        return {"bytes": int(head.get("bytes", size)), "sha256": head.get("sha256", "")}

    def delete(self, keys) -> int:
        head, _ = self._call("delete", {"keys": list(keys)})
        return int(head.get("deleted", 0))

    def stats(self) -> dict:
        head, _ = self._call("stats", {}, timeout=min(2.0, self.timeout))
        return {"provider": "lmcache-http", "url": self.url, **dict(head.get("store") or {}),
                "hits": self._hits, "misses": self._misses}

    def close(self) -> None:
        self._closed = True
        with self._lock:
            self._drop()


class LmcacheClient:
    """Calls the ``strata_blob`` module on an LMCache multiprocess server."""

    def __init__(self, url: str, timeout: float = 30.0, chunk_bytes: int = CHUNK_BYTES,
                 match_timeout: float | None = None):
        if not isinstance(url, str) or not url.strip():
            raise BlobError("an LMCache URL is required (tcp://host:port or grpc://host:port)")
        self.url = url.strip()
        self.timeout = float(timeout)
        self.match_timeout = float(match_timeout) if match_timeout else min(20.0, self.timeout)
        chunk = max(4096, min(int(chunk_bytes), 64 << 20))
        if self.url.lower().startswith(("grpc://", "grpc+unix://")):
            # gRPC's default receive limit is 4 MiB; keep a margin for the
            # frame header and protobuf envelope.
            chunk = min(chunk, 2 << 20)
        self.chunk_bytes = chunk
        self._lock = threading.RLock()
        self._client = None
        self._request_type = None
        self._closed = False
        self._hits = 0
        self._misses = 0

    def _connect(self):
        if self._client is not None:
            return
        # Importing LMCache may import torch and ask the CUDA runtime whether a
        # GPU exists; on a 24 GB card that can create a context and take VRAM
        # from the engine.  The transport itself is pure CPU.  Hide the devices
        # for the import only, then put the environment back exactly as it was.
        saved = {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")}
        try:
            for k in saved:
                os.environ[k] = ""
            try:
                from lmcache.v1.multiprocess.protocols.server_module import ServerModuleCallRequest
                from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
            except Exception as exc:  # pragma: no cover - depends on the environment
                raise BlobError(
                    "the lmcache package is not importable in this environment; install LMCache or "
                    "use the \"filesystem\" provider"
                ) from exc
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self._request_type = ServerModuleCallRequest
        self._client = RequestClientFactory.create(self.url)

    def _call(self, method: str, header: dict, data: bytes = b"", timeout: float | None = None):
        if self._closed:
            raise BlobError("the LMCache client is closed")
        with self._lock:
            self._connect()
            assert self._client is not None and self._request_type is not None
            payload = encode({"v": 1, **header}, data)
            try:
                future = self._client.server_module_call(
                    self._request_type(method="strata_blob." + method, payload=payload))
                response = future.result(timeout=self.timeout if timeout is None else timeout)
            except TimeoutError as exc:
                raise BlobError(f"LMCache did not answer {method} within {timeout or self.timeout:.0f} s") from exc
            except Exception as exc:
                # A dead daemon must not take the Strata server down: raise a
                # BlobError, let the bridge log once and enter a cooldown.
                raise BlobError(f"LMCache transport failed: {exc}") from exc
            if not getattr(response, "success", False):
                raise BlobError(f"LMCache module error: {getattr(response, 'error', 'unknown error')}")
            try:
                head, body = decode(response.payload, max_data=self.chunk_bytes)
            except WireError as exc:
                raise BlobError(f"LMCache sent a malformed reply: {exc}") from exc
            if not head.get("ok", False):
                raise BlobError(f"{head.get('kind', 'error')}: {head.get('error', 'unknown error')}")
            return head, body

    def ping(self) -> dict:
        head, _ = self._call("ping", {})
        return head

    def match(self, keys):
        head, _ = self._call("match", {"keys": list(keys)}, timeout=self.match_timeout)
        if not head.get("found"):
            self._misses += 1
            return None
        self._hits += 1
        return head.get("key"), dict(head.get("meta") or {})

    def meta(self, key: str):
        _check_key(key)
        head, _ = self._call("meta", {"key": key})
        if not head.get("found"):
            return None
        return {"meta": dict(head.get("meta") or {}), "bytes": int(head.get("bytes", 0)),
                "sha256": head.get("sha256", "")}

    def download(self, key: str, path: str | os.PathLike) -> dict:
        _check_key(key)
        got = self.meta(key)
        if got is None:
            raise BlobError(f"no cache entry {key}")
        size = int(got["bytes"])
        tmp = f"{path}.{uuid.uuid4().hex[:8]}.part"
        h = hashlib.sha256()
        try:
            with open(tmp, "wb") as out:
                offset = 0
                while offset < size:
                    want = min(self.chunk_bytes, size - offset)
                    head, data = self._call("get_chunk", {"key": key, "offset": offset, "length": want})
                    if len(data) != int(head.get("n", -1)) or not data:
                        raise BlobError("LMCache returned a short chunk")
                    out.write(data)
                    h.update(data)
                    offset += len(data)
            if h.hexdigest() != got["sha256"]:
                raise BlobError("downloaded checksum does not match the entry")
            os.replace(tmp, path)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        return got

    def upload(self, key: str, path: str | os.PathLike, meta: dict) -> dict:
        _check_key(key)
        size = os.path.getsize(path)
        head, _ = self._call("begin_put", {"key": key, "meta": dict(meta or {}), "bytes": size})
        upload = str(head.get("upload", ""))
        if not upload:
            raise BlobError("LMCache did not return an upload id")
        try:
            with open(path, "rb") as src:
                offset = 0
                while offset < size:
                    data = src.read(min(self.chunk_bytes, size - offset))
                    if not data:
                        raise BlobError("the session image ended early")
                    self._call("put_chunk", {"upload": upload, "offset": offset}, data)
                    offset += len(data)
            head, _ = self._call("end_put", {"upload": upload})
        except Exception:
            try:
                self._call("abort_put", {"upload": upload}, timeout=5)
            except Exception:
                pass
            raise
        return {"bytes": int(head.get("bytes", size)), "sha256": head.get("sha256", "")}

    def delete(self, keys) -> int:
        head, _ = self._call("delete", {"keys": list(keys)})
        return int(head.get("deleted", 0))

    def stats(self) -> dict:
        head, _ = self._call("stats", {}, timeout=min(2.0, self.timeout))
        return {"provider": "lmcache", "url": self.url, **dict(head.get("store") or {}),
                "hits": self._hits, "misses": self._misses}

    def close(self) -> None:
        self._closed = True
        client, self._client = self._client, None
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


def create_client(spec: dict):
    """Build the provider named by the ``lmcache`` config block."""
    provider = str(spec.get("provider") or "lmcache").strip().lower()
    if provider in ("filesystem", "fs", "dir"):
        root = spec.get("path")
        if not root:
            raise BlobError('the "filesystem" provider needs "path": a directory all Strata servers can read/write')
        return FilesystemClient(root, int(spec.get("max_bytes") or 0))
    if provider in ("lmcache", "lmcache_server", "mp", "http"):
        url = str(spec.get("url") or "").strip()
        if url.lower().startswith(("http://", "https://")):
            return HttpBlobClient(url, float(spec.get("timeout_s") or 30.0),
                                  int(spec.get("chunk_bytes") or CHUNK_BYTES),
                                  spec.get("match_timeout_s"))
        return LmcacheClient(url, float(spec.get("timeout_s") or 30.0),
                             int(spec.get("chunk_bytes") or CHUNK_BYTES),
                             spec.get("match_timeout_s"))
    raise BlobError(f'unknown lmcache provider {provider!r} (use "lmcache" or "filesystem")')
