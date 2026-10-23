# SPDX-License-Identifier: MIT
"""Exercise the LMCache server-module handlers without LMCache installed.

The real daemon supplies the module base classes and the ``SERVER_MODULE_CALL``
transport.  This test stubs only those imports, then drives the exact handler
bytes the Strata bridge sends, so the protocol and the in-memory store are
covered on a machine with no LMCache package and no GPU.
"""
from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "integrations" / "lmcache"))

from strata_lmcache import wire  # noqa: E402


def _install_stubs():
    class _Logger:
        def info(self, *a, **k):
            pass

        def warning(self, *a, **k):
            pass

    logging = types.ModuleType("lmcache.logging")
    logging.init_logger = lambda *_a, **_k: _Logger()

    class DiscoverableModule:
        module_name = ""
        module_dependencies = ()
        deferred = False

        @classmethod
        def create(cls, build_ctx):
            raise NotImplementedError

    class ModuleBuildContext:
        pass

    engine_module = types.ModuleType("lmcache.v1.multiprocess.engine_module")
    engine_module.DiscoverableModule = DiscoverableModule
    engine_module.EngineModule = object
    engine_module.ModuleBuildContext = ModuleBuildContext
    engine_module.InstanceLivenessTarget = object

    engine_context = types.ModuleType("lmcache.v1.multiprocess.engine_context")
    engine_context.MPCacheServerContext = object

    def server_module_handler(method, **kwargs):
        def deco(fn):
            fn._strata_method = method
            return fn
        return deco

    ext = types.ModuleType("lmcache.v1.multiprocess.ext_server_module")
    ext.server_module_handler = server_module_handler

    mods = {
        "lmcache": types.ModuleType("lmcache"),
        "lmcache.logging": logging,
        "lmcache.v1": types.ModuleType("lmcache.v1"),
        "lmcache.v1.multiprocess": types.ModuleType("lmcache.v1.multiprocess"),
        "lmcache.v1.multiprocess.engine_module": engine_module,
        "lmcache.v1.multiprocess.engine_context": engine_context,
        "lmcache.v1.multiprocess.ext_server_module": ext,
    }
    for name in ("lmcache", "lmcache.v1", "lmcache.v1.multiprocess"):
        mods[name].__path__ = []
    previous = {name: sys.modules.get(name) for name in mods}
    sys.modules.update(mods)
    return previous


class ModuleHandlerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._previous = _install_stubs()
        sys.modules.pop("strata_lmcache.module", None)
        cls.mod = importlib.import_module("strata_lmcache.module")

    @classmethod
    def tearDownClass(cls):
        for name, old in cls._previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
        sys.modules.pop("strata_lmcache.module", None)

    def setUp(self):
        from strata_lmcache.store import MemoryBlobStore
        self.store = MemoryBlobStore(max_bytes=1 << 20)
        self.m = self.mod.StrataBlobModule(object(), self.store, chunk_bytes=4, max_request_bytes=64)

    def call(self, method, header, data=b""):
        payload = wire.encode(header, data)
        fn = getattr(self.m, method)
        return wire.decode(fn(payload), max_data=1 << 20)

    def test_ping(self):
        head, _ = self.call("ping", {"v": 1, "echo": "hi"})
        self.assertTrue(head["ok"])
        self.assertEqual(head["echo"], "hi")

    def test_put_match_chunk_meta(self):
        head, _ = self.call("begin_put", {"v": 1, "key": "model:8:abcd", "meta": {"tokens": 8}, "bytes": 8})
        upload = head["upload"]
        self.call("put_chunk", {"v": 1, "upload": upload, "offset": 0}, b"abcd")
        head, _ = self.call("put_chunk", {"v": 1, "upload": upload, "offset": 4}, b"efgh")
        self.assertEqual(head["n"], 4)
        head, _ = self.call("end_put", {"v": 1, "upload": upload})
        self.assertEqual(head["bytes"], 8)

        head, _ = self.call("match", {"v": 1, "keys": ["nope", "model:8:abcd"]})
        self.assertTrue(head["found"])
        self.assertEqual(head["meta"]["tokens"], 8)

        head, data = self.call("get_chunk", {"v": 1, "key": "model:8:abcd", "offset": 2, "length": 4})
        self.assertEqual(data, b"cdef")
        head, _ = self.call("meta", {"v": 1, "key": "model:8:abcd"})
        self.assertTrue(head["found"])
        self.assertEqual(head["bytes"], 8)

        head, _ = self.call("stats", {"v": 1})
        self.assertEqual(head["store"]["entries"], 1)

        head, _ = self.call("delete", {"v": 1, "keys": ["model:8:abcd"]})
        self.assertEqual(head["deleted"], 1)
        head, _ = self.call("match", {"v": 1, "keys": ["model:8:abcd"]})
        self.assertFalse(head["found"])

    def test_create_defaults_to_memory(self):
        class Ctx:
            engine_context = object()

            def config_for(self, _cls):
                return {"max_bytes": 1 << 20}
        mod = self.mod.StrataBlobModule.create(Ctx())
        self.assertIsNotNone(mod)
        self.assertEqual(mod.report_status()["storage"], "memory")
        mod.close()

    def test_bad_version_and_short_frame_refused(self):
        head, _ = wire.decode(self.m.ping(wire.encode({"v": 99})), max_data=8)
        self.assertFalse(head["ok"])
        head, _ = wire.decode(self.m.ping(b"\x00"), max_data=8)
        self.assertFalse(head["ok"])


if __name__ == "__main__":
    unittest.main()
