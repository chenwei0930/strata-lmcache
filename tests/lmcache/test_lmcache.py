# SPDX-License-Identifier: MIT
"""Unit tests for Strata's LMCache bridge.

Run with ``python -m unittest discover -s tests/lmcache -v`` (no GPU, no
LMCache package, no model).  The LMCache transport itself is exercised by
``test_lmcache_transport.py`` when the package is installed.
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "integrations" / "lmcache"))

from serve import lmcache_bridge, lmcache_client, lmcache_keys  # noqa: E402
from strata_lmcache import store as blob_store  # noqa: E402
from strata_lmcache import wire  # noqa: E402


class WireTests(unittest.TestCase):
    def test_round_trip(self):
        frame = wire.encode({"v": 1, "op": "x", "n": 3}, b"abc")
        head, data = wire.decode(frame, max_data=8)
        self.assertEqual(head["n"], 3)
        self.assertEqual(data, b"abc")

    def test_truncated_refused(self):
        frame = wire.encode({"v": 1}, b"abc")
        with self.assertRaises(wire.WireError):
            wire.decode(frame[:-1])

    def test_oversize_refused(self):
        frame = wire.encode({"v": 1}, b"abcdef")
        with self.assertRaises(wire.WireError):
            wire.decode(frame, max_data=2)


class KeyTests(unittest.TestCase):
    def test_chain_is_stable_and_incremental(self):
        ids = list(range(1, 101))
        a = lmcache_keys.PrefixChain("m", 16, ids)
        b = lmcache_keys.PrefixChain("m", 16, ids)
        self.assertEqual([k.key for k in a], [k.key for k in b])
        self.assertEqual([k.tokens for k in a], list(range(16, 101, 16)))

    def test_prefix_chain_differs_after_divergence(self):
        a = lmcache_keys.PrefixChain("m", 16, list(range(100)))
        b = lmcache_keys.PrefixChain("m", 16, list(range(50)) + [999] + list(range(51, 100)))
        self.assertEqual(a.all()[0].key, b.all()[0].key)
        self.assertNotEqual(a.all()[-1].key, b.all()[-1].key)

    def test_any_prefix_key(self):
        ids = list(range(1, 101))
        c = lmcache_keys.PrefixChain("m", 16, ids)
        self.assertEqual(c.at(40).tokens, 40)
        self.assertEqual(c.at(40).key, lmcache_keys.PrefixChain("m", 16, ids[:40]).at(40).key)
        self.assertIsNone(c.at(0))

    def test_lookup_skips_short_prefixes(self):
        c = lmcache_keys.PrefixChain("m", 16, list(range(1, 65)))
        keys = c.lookup_keys(32)
        self.assertEqual(len(keys), 3)                  # 32, 48, 64
        self.assertTrue(keys[0].endswith(":64:" + c.at(64).digest.hex()))

    def test_last_before(self):
        c = lmcache_keys.PrefixChain("m", 16, list(range(1, 65)))
        self.assertEqual(c.last_before(47).tokens, 32)
        self.assertEqual(c.last_before(48).tokens, 48)

    def test_common_prefix(self):
        self.assertEqual(lmcache_keys.common_prefix([1, 2, 3], [1, 2, 4]), 2)
        self.assertEqual(lmcache_keys.common_prefix([], [1]), 0)


class MemoryStoreTests(unittest.TestCase):
    def _store(self, **kw):
        return blob_store.MemoryBlobStore(**kw)

    def test_put_get_match(self):
        s = self._store(max_bytes=1 << 20)
        up = s.begin_put("k1", {"tokens": 40}, 6)
        s.put_chunk(up, 0, b"abc")
        s.put_chunk(up, 3, b"def")
        got = s.end_put(up)
        self.assertEqual(got["bytes"], 6)
        key, meta = s.match(["missing", "k1"])
        self.assertEqual((key, meta["tokens"]), ("k1", 40))
        self.assertEqual(s.read_chunk("k1", 2, 3), b"cde")
        self.assertEqual(s.stats()["entries"], 1)
        self.assertEqual(s.delete(["k1"]), 1)
        self.assertIsNone(s.match(["k1"]))

    def test_quota_evicts_least_recent(self):
        s = self._store(max_bytes=10)
        for key, n in (("a", 6), ("b", 6)):
            up = s.begin_put(key, {}, n)
            s.put_chunk(up, 0, b"x" * n)
            s.end_put(up)
        self.assertIsNone(s.match(["a"]))
        self.assertIsNotNone(s.match(["b"]))
        self.assertEqual(s.stats()["evictions"], 1)

    def test_short_upload_refused(self):
        s = self._store()
        up = s.begin_put("k", {}, 5)
        s.put_chunk(up, 0, b"abc")
        with self.assertRaises(blob_store.StoreError):
            s.end_put(up)

    def test_bad_offset_refused(self):
        s = self._store()
        up = s.begin_put("k", {}, 10)
        with self.assertRaises(blob_store.StoreError):
            s.put_chunk(up, 2, b"abc")


class DiskStoreTests(unittest.TestCase):
    def test_round_trip_and_reload(self):
        with tempfile.TemporaryDirectory() as d:
            s = blob_store.BlobStore(d, max_bytes=1 << 20)
            up = s.begin_put("abc", {"tokens": 12}, 4)
            s.put_chunk(up, 0, b"data")
            s.end_put(up)
            self.assertEqual(s.read_chunk("abc", 0, 4), b"data")
            del s
            s2 = blob_store.BlobStore(d, max_bytes=1 << 20)
            self.assertEqual(s2.match(["abc"])[1]["tokens"], 12)
            self.assertEqual(s2.read_chunk("abc", 1, 2), b"at")


class FilesystemClientTests(unittest.TestCase):
    def test_upload_download_match(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as w:
            c = lmcache_client.FilesystemClient(d, max_bytes=1 << 20)
            src = os.path.join(w, "in.bin")
            Path(src).write_bytes(b"hello world")
            got = c.upload("m:12:abcd", src, {"tokens": 12})
            self.assertEqual(got["bytes"], 11)
            self.assertEqual(c.match(["m:12:abcd"])[1]["tokens"], 12)
            out = os.path.join(w, "out.bin")
            c.download("m:12:abcd", out)
            self.assertEqual(Path(out).read_bytes(), b"hello world")
            c.close()


class _FakeEngine:
    def __init__(self):
        self.batch = 0
        self.calls = []
        self.blob = None
        self.alive = lambda: True

    def session_file(self, action, path):
        self.calls.append((action, path))
        if action == "restore":
            if self.blob is None:
                raise ValueError("no session")
            Path(path).write_bytes(self.blob)
            return {"tokens": 40, "bytes": len(self.blob), "ms": 1.0}
        Path(path).write_bytes(b"session-image")
        return {"tokens": 99, "bytes": 13, "ms": 1.0}


class _FakeService:
    def __init__(self):
        self.engine = _FakeEngine()
        self.fifo = threading.Lock()
        self._lmcache_state_epoch = 1
        self.status_lock = threading.Lock()
        self.status = {"queued": 0}


class BridgeTests(unittest.TestCase):
    def _bridge(self, d, **spec):
        base = {"provider": "filesystem", "path": d, "chunk_size": 16, "min_tokens": 32,
                "save_lock_timeout_s": 5, "retry_s": 0, "save_when_busy": True}
        base.update(spec)
        return lmcache_bridge.Bridge(base, "test-model", "0.1.48", 4096, "args", log=lambda *_a: None)

    def test_new_conversation_pins_system_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            b = self._bridge(d)
            svc = _FakeService()
            ids = list(range(1, 101))
            msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
            plan = b.plan(svc, ids, msgs, None, {}, False, resolver=lambda: 40)
            self.assertIsNotNone(plan)
            self.assertEqual(plan.pin, 40)
            self.assertTrue(plan.save_key.endswith(f":40:{lmcache_keys.PrefixChain('x', 16, ids).at(40).digest.hex()}"))
            b.close()

    def test_continuation_pins_latest_boundary(self):
        with tempfile.TemporaryDirectory() as d:
            b = self._bridge(d)
            svc = _FakeService()
            first = list(range(1, 101))
            b._last_ids = first
            ids = first + list(range(101, 121))
            plan = b.plan(svc, ids, [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
                          None, {}, False, resolver=lambda: 40)
            self.assertGreater(plan.pin, 40)
            self.assertEqual(plan.pin % 16, 0)
            b.close()

    def test_restore_and_save_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            b = self._bridge(d)
            svc = _FakeService()
            ids = list(range(1, 101))
            sys_ids = ids[:40]
            # Seed the store with a snapshot for the system prefix.
            chain = lmcache_keys.PrefixChain(b.model, 16, ids)
            key = chain.at(40).key
            src = os.path.join(b.work_dir, "seed.bin")
            Path(src).write_bytes(b"seed-image")
            b._provider.upload(key, src, {"tokens": 40})
            os.unlink(src)

            plan = b.plan(svc, ids, [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
                          None, {}, False, resolver=lambda: 40)
            self.assertIsNotNone(plan.fetch)
            self.assertEqual(plan.tokens, 40)
            svc.engine.blob = b"seed-image"
            self.assertTrue(b.restore(svc, plan))
            self.assertEqual(svc.engine.calls[0][0], "restore")

            b.after_request(svc, plan, ids, "stop", 1, svc._lmcache_state_epoch)
            for t in list(b._threads):
                t.join(timeout=5)
            self.assertTrue(any(c[0] == "save" for c in svc.engine.calls))
            got = b._provider.match([plan.save_key])
            self.assertIsNotNone(got)
            self.assertEqual(got[1]["tokens"], plan.save_tokens)
            b.close()

    def test_no_restore_when_engine_holds_more(self):
        with tempfile.TemporaryDirectory() as d:
            b = self._bridge(d)
            svc = _FakeService()
            ids = list(range(1, 101))
            chain = lmcache_keys.PrefixChain(b.model, 16, ids)
            src = os.path.join(b.work_dir, "seed.bin")
            Path(src).write_bytes(b"x")
            b._provider.upload(chain.at(40).key, src, {"tokens": 40})
            os.unlink(src)
            b._held_tokens = 90
            b._last_ids = ids
            plan = b.plan(svc, ids, [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
                          None, {}, False, resolver=lambda: 40)
            self.assertIsNone(plan.fetch)
            b.close()

    def test_failed_restore_keeps_request(self):
        with tempfile.TemporaryDirectory() as d:
            b = self._bridge(d)
            svc = _FakeService()
            ids = list(range(1, 101))
            chain = lmcache_keys.PrefixChain(b.model, 16, ids)
            src = os.path.join(b.work_dir, "seed.bin")
            Path(src).write_bytes(b"x")
            b._provider.upload(chain.at(40).key, src, {"tokens": 40})
            os.unlink(src)

            class Refusing(_FakeEngine):
                def session_file(self, action, path):
                    raise ValueError("another model or configuration")
            svc.engine = Refusing()
            plan = b.plan(svc, ids, [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
                          None, {}, False, resolver=lambda: 40)
            self.assertFalse(b.restore(svc, plan))
            self.assertFalse(plan.restore_ok)
            b.close()


if __name__ == "__main__":
    unittest.main()
