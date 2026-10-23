# SPDX-License-Identifier: MIT
"""LMCache server-module plugin that gives Strata a shared prefix cache.

Load the plugin by its module path, so importing this package does not require
LMCache (the store and wire helpers stay usable in a plain test environment)::

    lmcache server ... --server-module \
      '{"module_path": "strata_lmcache.module", "config": {"path": "..."}}'

The public entry point is :class:`strata_lmcache.module.StrataBlobModule`
(``strata_blob``).  The Strata server connects with its own client
(``serve/lmcache_client.py``), not with this package.
"""
from __future__ import annotations

__all__ = [
    "METHOD_PREFIX",
    "StrataBlobModule",
    "BlobService",
    "BlobStore",
    "MemoryBlobStore",
    "StoreError",
    "StoreFull",
    "WireError",
    "encode",
    "decode",
]


def __getattr__(name):
    """Import LMCache-facing pieces only when they are actually requested."""
    if name in ("StrataBlobModule", "METHOD_PREFIX"):
        from . import module
        return getattr(module, name)
    if name == "BlobService":
        from . import service
        return service.BlobService
    if name in ("BlobStore", "MemoryBlobStore", "StoreError", "StoreFull"):
        from . import store
        return getattr(store, name)
    if name in ("WireError", "encode", "decode"):
        from . import wire
        return getattr(wire, name)
    raise AttributeError(name)
