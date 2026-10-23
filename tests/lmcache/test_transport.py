# SPDX-License-Identifier: MIT
"""The chunked LMCache client against the real handler code, in one process.

No LMCache package, daemon, GPU or network is involved: a fake transport
forwards ``SERVER_MODULE_CALL`` envelopes to :class:`BlobService`, which is the
same body the LMCache daemon runs.  This catches framing, chunk offsets,
checksums and error mapping, the parts most likely to drift.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "integrations" / "lmcache"))

from serve.lmcache_client import BlobError, LmcacheClient  # noqa: E402
from strata_lmcache.service import BlobService  # noqa: E402
from strata_lmcache.store import MemoryBlobStore  # noqa: E402


class _Response:
    success = True
    error = ""

    def __init__(self, payload):
        self.payload = payload


class _Future:
    def __init__(self, payload):
        self._payload = payload

    def result(self, timeout=None):
        return _Response(self._payload)


class _Request:
    def __init__(self, method, payload):
        self.method = method
        self.payload = payload


class _Transport:
    def __init__(self, service):
        self.service = service
        self.closed = False

    def server_module_call(self, request):
        name = request.method.split(".", 1)[1]
        return _Future(self.service.call(name, request.payload))

    def close(self):
        self.closed = True


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.store = MemoryBlobStore(max_bytes=1 << 20)
        self.service = BlobService(self.store, chunk_bytes=4096, max_request_bytes=8192)
        self.client = LmcacheClient("tcp://fake:0", chunk_bytes=4096)
        self.transport = _Transport(self.service)
        self.client._client = self.transport
        self.client._request_type = _Request
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.client.close()
        self.tmp.cleanup()

    def test_chunked_round_trip(self):
        src = os.path.join(self.tmp.name, "src.bin")
        payload = bytes(range(256)) * 80             # 20 KiB, five 4 KiB chunks
        Path(src).write_bytes(payload)
        got = self.client.upload("m:1200:abc", src, {"tokens": 1200})
        self.assertEqual(got["bytes"], len(payload))

        hit = self.client.match(["missing", "m:1200:abc"])
        self.assertIsNotNone(hit)
        self.assertEqual(hit[1]["tokens"], 1200)

        out = os.path.join(self.tmp.name, "out.bin")
        meta = self.client.download("m:1200:abc", out)
        self.assertEqual(meta["meta"]["tokens"], 1200)
        self.assertEqual(Path(out).read_bytes(), payload)

        stats = self.client.stats()
        self.assertEqual(stats["entries"], 1)
        self.assertEqual(stats["bytes"], len(payload))
        self.assertEqual(self.client.delete(["m:1200:abc"]), 1)
        self.assertIsNone(self.client.match(["m:1200:abc"]))

    def test_short_upload_aborts(self):
        src = os.path.join(self.tmp.name, "src.bin")
        Path(src).write_bytes(b"x" * 10)
        # A transport that loses a chunk must not leave a half entry behind.
        class Lossy(_Transport):
            def server_module_call(inner, request):
                if request.method.endswith("put_chunk"):
                    return _Future(b"")
                return super().server_module_call(request)
        self.client._client = Lossy(self.service)
        with self.assertRaises(BlobError):
            self.client.upload("m:1:abc", src, {})
        self.assertIsNone(self.store.match(["m:1:abc"]))

    def test_missing_key_is_an_error(self):
        with self.assertRaises(BlobError):
            self.client.download("m:1:none", os.path.join(self.tmp.name, "out"))


if __name__ == "__main__":
    unittest.main()
