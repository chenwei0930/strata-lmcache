# SPDX-License-Identifier: MIT
"""Strata's LMCache server module.

Run an LMCache multiprocess server with this module loaded::

    PYTHONPATH=/path/to/Strata/integrations/lmcache \\
    lmcache server --port 6555 --http-port 8080 \\
      --server-module '{"module_path": "strata_lmcache.module", "config": {
          "storage": "memory", "max_bytes": 68719476736,
          "http_host": "0.0.0.0", "http_port": 8081}}'

The REST of the file is the same module used over LMCache's own
``SERVER_MODULE_CALL`` transport.  ``http_port`` adds the small HTTP front that
the Strata container uses, so the Strata image needs neither LMCache nor torch.

The module stores opaque Strata session images keyed by a token-prefix hash
computed by the Strata server (``serve/lmcache_bridge.py``).  It deliberately
does not touch LMCache's KV page store: Strata's hybrid GDN/QSA state and its
quantized K/V are not vLLM KV pages, and LMCache's plugin contract is the
supported way to extend the daemon with an engine-specific object.

The default store is **RAM**: a 63K-token Strata session is about 1.2 GiB, and
the operation this feature is for - four A10 services sharing prefixes - is
bounded by memory bandwidth, not disk.  ``"storage": "filesystem"`` with a
``path`` is available for persistence, but it is not the recommended path.
"""
from __future__ import annotations

import os

from lmcache.logging import init_logger
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import (
    DiscoverableModule,
    EngineModule,
    ModuleBuildContext,
)
from lmcache.v1.multiprocess.ext_server_module import server_module_handler

from .service import DEFAULT_CHUNK, BlobService
from .store import BlobStore, MemoryBlobStore

logger = init_logger(__name__)

METHOD_PREFIX = "strata_blob."


class StrataBlobModule(DiscoverableModule):
    """Store and share Strata engine session images through LMCache."""

    module_name = "strata_blob"

    @classmethod
    def create(cls, build_ctx: ModuleBuildContext) -> EngineModule | None:
        cfg = dict(build_ctx.config_for(cls) or {})
        storage = str(cfg.get("storage") or ("filesystem" if cfg.get("path") else "memory")).strip().lower()
        max_bytes = int(cfg.get("max_bytes", 64 << 30))
        if max_bytes < 0:
            raise ValueError("strata_blob: max_bytes must be >= 0 (0 = unlimited)")
        if storage in ("filesystem", "fs", "dir"):
            root = cfg.get("path") or os.environ.get("STRATA_LMCACHE_STORE")
            if not root:
                raise ValueError(
                    "the strata_blob filesystem store needs a path: "
                    "--server-module '{\"module_path\": \"strata_lmcache.module\", "
                    "\"config\": {\"storage\": \"filesystem\", \"path\": \"/var/lib/strata-lmcache\"}}'"
                )
            store = BlobStore(root, max_bytes=max_bytes, fsync=bool(cfg.get("fsync", True)))
        elif storage in ("memory", "ram"):
            store = MemoryBlobStore(max_bytes=max_bytes)
        else:
            raise ValueError(f"strata_blob: unknown storage {storage!r} (memory or filesystem)")
        return cls(build_ctx.engine_context, store,
                   chunk_bytes=int(cfg.get("chunk_bytes", DEFAULT_CHUNK)),
                   max_request_bytes=int(cfg.get("max_request_bytes", 64 << 20)),
                   http_host=str(cfg.get("http_host", "127.0.0.1")),
                   http_port=int(cfg.get("http_port", 0) or 0))

    def __init__(self, ctx: MPCacheServerContext, store,
                 chunk_bytes: int = DEFAULT_CHUNK, max_request_bytes: int = 64 << 20,
                 http_host: str = "127.0.0.1", http_port: int = 0):
        self._ctx = ctx
        self.store = store
        self.service = BlobService(store, chunk_bytes=chunk_bytes, max_request_bytes=max_request_bytes)
        self.chunk_bytes = self.service.chunk_bytes
        self.max_request_bytes = self.service.max_request_bytes
        self.http = None
        if http_port:
            from .http_server import BlobHTTPServer
            self.http = BlobHTTPServer(self.service, http_host, http_port)
        store.prune_orphans()
        logger.info("strata_blob: %s%s", store.stats(),
                    f", HTTP {http_host}:{self.http.port}" if self.http is not None else "")

    # EngineModule protocol -------------------------------------------------
    @property
    def context(self) -> MPCacheServerContext:
        return self._ctx

    def report_status(self) -> dict:
        return {"strata_blob": self.store.stats(), "chunk_bytes": self.chunk_bytes,
                "storage": "memory" if isinstance(self.store, MemoryBlobStore) else "filesystem",
                **({"http": f"{self.http.host}:{self.http.port}"} if self.http is not None else {})}

    def close(self) -> None:
        if self.http is not None:
            self.http.close()
            self.http = None
        self.store.close()

    # handlers --------------------------------------------------------------
    @server_module_handler(METHOD_PREFIX + "ping")
    def ping(self, payload: bytes) -> bytes:
        return self.service.call("ping", payload)

    @server_module_handler(METHOD_PREFIX + "match")
    def match(self, payload: bytes) -> bytes:
        return self.service.call("match", payload)

    @server_module_handler(METHOD_PREFIX + "meta")
    def meta(self, payload: bytes) -> bytes:
        return self.service.call("meta", payload)

    @server_module_handler(METHOD_PREFIX + "get_chunk")
    def get_chunk(self, payload: bytes) -> bytes:
        return self.service.call("get_chunk", payload)

    @server_module_handler(METHOD_PREFIX + "begin_put")
    def begin_put(self, payload: bytes) -> bytes:
        return self.service.call("begin_put", payload)

    @server_module_handler(METHOD_PREFIX + "put_chunk")
    def put_chunk(self, payload: bytes) -> bytes:
        return self.service.call("put_chunk", payload)

    @server_module_handler(METHOD_PREFIX + "end_put")
    def end_put(self, payload: bytes) -> bytes:
        return self.service.call("end_put", payload)

    @server_module_handler(METHOD_PREFIX + "abort_put")
    def abort_put(self, payload: bytes) -> bytes:
        return self.service.call("abort_put", payload)

    @server_module_handler(METHOD_PREFIX + "delete")
    def delete(self, payload: bytes) -> bytes:
        return self.service.call("delete", payload)

    @server_module_handler(METHOD_PREFIX + "clear")
    def clear(self, payload: bytes) -> bytes:
        return self.service.call("clear", payload)

    @server_module_handler(METHOD_PREFIX + "stats")
    def stats(self, payload: bytes) -> bytes:
        return self.service.call("stats", payload)


__all__ = ["StrataBlobModule", "METHOD_PREFIX"]
