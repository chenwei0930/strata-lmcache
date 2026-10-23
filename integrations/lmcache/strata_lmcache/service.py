# SPDX-License-Identifier: MIT
"""Transport-neutral handlers for the ``strata_blob`` LMCache server module.

Keeping the bodies here, rather than in the LMCache module class, lets the
chunked protocol be tested without the LMCache package installed.  Each
``call`` takes one frame and returns one frame; the LMCache module is a thin
``@server_module_handler`` wrapper around it.
"""
from __future__ import annotations

from .store import StoreError, StoreFull
from .wire import MAGIC_VERSION, WireError, decode, encode

DEFAULT_CHUNK = 8 << 20


class BlobService:
    def __init__(self, store, chunk_bytes: int = DEFAULT_CHUNK, max_request_bytes: int = 64 << 20):
        self.store = store
        self.chunk_bytes = max(1, min(int(chunk_bytes), 64 << 20))
        self.max_request_bytes = max(1, min(int(max_request_bytes), 64 << 20))

    # --------------------------------------------------------------- helpers
    def _reply(self, header: dict | None = None, data: bytes = b"") -> bytes:
        return encode({"v": MAGIC_VERSION, "ok": True, **(header or {})}, data)

    def _error(self, exc: Exception, kind: str = "error") -> bytes:
        return encode({"v": MAGIC_VERSION, "ok": False, "kind": kind, "error": str(exc)[:2000]})

    def _request(self, payload: bytes, max_data: int) -> tuple[dict, bytes]:
        header, data = decode(payload, max_data=max_data)
        if int(header.get("v", -1)) != MAGIC_VERSION:
            raise WireError(f"unsupported wire version {header.get('v')!r}")
        return header, data

    # ----------------------------------------------------------------- calls
    def call(self, method: str, payload: bytes) -> bytes:
        fn = getattr(self, "_" + method, None)
        if fn is None:
            return self._error(StoreError(f"unknown method {method!r}"))
        return fn(payload)

    def _ping(self, payload: bytes) -> bytes:
        try:
            header, _ = self._request(payload, 0)
            return self._reply({"pong": True, "chunk_bytes": self.chunk_bytes,
                                "store": self.store.stats(), "echo": header.get("echo")})
        except Exception as exc:
            return self._error(exc, "wire")

    def _match(self, payload: bytes) -> bytes:
        try:
            header, _ = self._request(payload, 0)
            keys = header.get("keys")
            if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
                raise StoreError("keys must be a list of strings")
            hit = self.store.match(keys)
            if hit is None:
                return self._reply({"found": False})
            key, meta = hit
            return self._reply({"found": True, "key": key, "meta": meta})
        except Exception as exc:
            return self._error(exc, "invalid")

    def _meta(self, payload: bytes) -> bytes:
        try:
            header, _ = self._request(payload, 0)
            got = self.store.meta(str(header.get("key", "")))
            if got is None:
                return self._reply({"found": False})
            return self._reply({"found": True, **got})
        except Exception as exc:
            return self._error(exc, "invalid")

    def _get_chunk(self, payload: bytes) -> bytes:
        try:
            header, _ = self._request(payload, 0)
            key = str(header.get("key", ""))
            offset = int(header.get("offset", 0))
            length = min(int(header.get("length", self.chunk_bytes)), self.max_request_bytes)
            data = self.store.read_chunk(key, offset, length)
            return self._reply({"key": key, "offset": offset, "n": len(data)}, data)
        except Exception as exc:
            return self._error(exc, "invalid")

    def _begin_put(self, payload: bytes) -> bytes:
        try:
            header, _ = self._request(payload, 0)
            meta = header.get("meta") or {}
            if not isinstance(meta, dict):
                raise StoreError("meta must be an object")
            upload = self.store.begin_put(str(header.get("key", "")), meta, int(header.get("bytes", -1)))
            return self._reply({"upload": upload})
        except StoreFull as exc:
            return self._error(exc, "full")
        except Exception as exc:
            return self._error(exc, "invalid")

    def _put_chunk(self, payload: bytes) -> bytes:
        try:
            header, data = self._request(payload, self.max_request_bytes)
            self.store.put_chunk(str(header.get("upload", "")), int(header.get("offset", -1)), data)
            return self._reply({"n": len(data)})
        except Exception as exc:
            return self._error(exc, "invalid")

    def _end_put(self, payload: bytes) -> bytes:
        try:
            header, _ = self._request(payload, 0)
            done = self.store.end_put(str(header.get("upload", "")))
            return self._reply(done)
        except StoreFull as exc:
            return self._error(exc, "full")
        except Exception as exc:
            return self._error(exc, "invalid")

    def _abort_put(self, payload: bytes) -> bytes:
        try:
            header, _ = self._request(payload, 0)
            self.store.abort_put(str(header.get("upload", "")))
            return self._reply()
        except Exception as exc:
            return self._error(exc, "invalid")

    def _delete(self, payload: bytes) -> bytes:
        try:
            header, _ = self._request(payload, 0)
            keys = header.get("keys")
            if not isinstance(keys, list):
                raise StoreError("keys must be a list")
            return self._reply({"deleted": self.store.delete(keys)})
        except Exception as exc:
            return self._error(exc, "invalid")

    def _clear(self, payload: bytes) -> bytes:
        try:
            self._request(payload, 0)
            return self._reply({"cleared": self.store.clear()})
        except Exception as exc:
            return self._error(exc, "invalid")

    def _stats(self, payload: bytes) -> bytes:
        try:
            self._request(payload, 0)
            return self._reply({"store": self.store.stats(), "chunk_bytes": self.chunk_bytes})
        except Exception as exc:
            return self._error(exc, "invalid")
