# SPDX-License-Identifier: MIT
"""The server hook: prepare() pins, run() restores, the worker saves.

Uses the real ``Service``, the real chat template and the byte tokenizer, with
a fake engine that implements only ``session_file``.  No model, GPU, engine
process, LMCache daemon or network.
"""
from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "integrations" / "lmcache"))

from serve import lmcache_bridge  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service  # noqa: E402
from strata_lmcache.http_server import BlobHTTPServer  # noqa: E402
from strata_lmcache.service import BlobService  # noqa: E402
from strata_lmcache.store import MemoryBlobStore  # noqa: E402


class _Engine(MockEngine):
    batch = 0
    gen = 1

    def __init__(self, tok, restore_ok=True):
        super().__init__(tok, "</think>\n\nhello", max_context=8192)
        self.calls = []
        self.restore_ok = restore_ok

    def alive(self):
        return True

    def session_file(self, action, path):
        self.calls.append(action)
        if action == "restore":
            if not self.restore_ok:
                raise ValueError("another model or configuration")
            return {"tokens": 419, "bytes": 5, "ms": 1.0}
        Path(path).write_bytes(b"STATE")
        return {"tokens": 646, "bytes": 5, "ms": 1.0}


class ServerHookTests(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.tpl = ChatTemplate(ROOT / "serve/chat_template.jinja")
        self.tmp = tempfile.TemporaryDirectory()
        self.msgs = [{"role": "system", "content": "S" * 200},
                     {"role": "user", "content": "U" * 200}]

    def tearDown(self):
        self.tmp.cleanup()

    def bridge(self):
        return lmcache_bridge.Bridge(
            {"provider": "filesystem", "path": self.tmp.name, "chunk_size": 16, "min_tokens": 32,
             "save_lock_timeout_s": 5, "retry_s": 0, "save_when_busy": True},
            "test-model", "0.1.48", 8192, "args", log=lambda *_a: None)

    def bridge_http(self, url):
        return lmcache_bridge.Bridge(
            {"provider": "lmcache", "url": url, "chunk_size": 16, "min_tokens": 32,
             "save_lock_timeout_s": 5, "retry_s": 0, "save_when_busy": True},
            "test-model", "0.1.48", 8192, "args", log=lambda *_a: None)

    def turn(self, svc, bridge):
        svc.lmcache = bridge
        req = {"max_tokens": 4}
        ids, thinking, max_new = svc.prepare(self.msgs, None, {}, 4, req=req)
        self.assertIsNotNone(req.get("strata_prefix"))
        list(svc.run(ids, thinking, None, max_new, req, threading.Event()))
        for t in list(bridge._threads):
            t.join(timeout=5)
        return ids

    def test_second_service_restores_the_system_prefix(self):
        one = Service(_Engine(self.tok), self.tok, self.tpl)
        b1 = self.bridge()
        self.turn(one, b1)
        self.assertEqual([c for c in one.engine.calls if c == "save"], ["save"])
        b1.close()

        two = Service(_Engine(self.tok, restore_ok=True), self.tok, self.tpl)
        b2 = self.bridge()
        self.turn(two, b2)
        self.assertEqual(two.engine.calls[:2], ["restore", "save"])
        self.assertEqual(b2.view()["restores"], 1)
        b2.close()

    def test_http_provider_shares_between_services(self):
        store = MemoryBlobStore(max_bytes=1 << 20)
        service = BlobService(store, chunk_bytes=4096, max_request_bytes=8192)
        server = BlobHTTPServer(service, "127.0.0.1", 0)
        try:
            url = f"http://127.0.0.1:{server.port}"
            one = Service(_Engine(self.tok), self.tok, self.tpl)
            b1 = self.bridge_http(url)
            self.turn(one, b1)
            b1.close()

            two = Service(_Engine(self.tok, restore_ok=True), self.tok, self.tpl)
            b2 = self.bridge_http(url)
            self.turn(two, b2)
            self.assertEqual(two.engine.calls[:2], ["restore", "save"])
            self.assertEqual(b2.view()["restores"], 1)
            b2.close()
        finally:
            server.close()

    def test_refused_restore_still_answers(self):
        one = Service(_Engine(self.tok), self.tok, self.tpl)
        b1 = self.bridge()
        self.turn(one, b1)
        b1.close()

        two = Service(_Engine(self.tok, restore_ok=False), self.tok, self.tpl)
        b2 = self.bridge()
        self.turn(two, b2)
        self.assertEqual(two.engine.calls[0], "restore")
        self.assertEqual(b2.view()["restore_failures"], 1)
        b2.close()


if __name__ == "__main__":
    unittest.main()
