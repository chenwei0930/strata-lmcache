# SPDX-License-Identifier: MIT
"""HTTP front + client, the Docker/default path (no LMCache, no torch)."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "integrations" / "lmcache"))

from serve.lmcache_client import BlobError, HttpBlobClient  # noqa: E402
from strata_lmcache.http_server import BlobHTTPServer  # noqa: E402
from strata_lmcache.service import BlobService  # noqa: E402
from strata_lmcache.store import MemoryBlobStore  # noqa: E402


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.store = MemoryBlobStore(max_bytes=1 << 20)
        self.service = BlobService(self.store, chunk_bytes=4096, max_request_bytes=8192)
        self.server = BlobHTTPServer(self.service, "127.0.0.1", 0)
        self.client = HttpBlobClient(f"http://127.0.0.1:{self.server.port}", chunk_bytes=4096, timeout=5)
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.client.close()
        self.server.close()
        self.tmp.cleanup()

    def test_chunked_round_trip(self):
        src = os.path.join(self.tmp.name, "src.bin")
        payload = bytes(range(256)) * 80
        Path(src).write_bytes(payload)
        got = self.client.upload("m:1200:abc", src, {"tokens": 1200})
        self.assertEqual(got["bytes"], len(payload))
        hit = self.client.match(["missing", "m:1200:abc"])
        self.assertIsNotNone(hit)
        self.assertEqual(hit[1]["tokens"], 1200)
        out = os.path.join(self.tmp.name, "out.bin")
        self.client.download("m:1200:abc", out)
        self.assertEqual(Path(out).read_bytes(), payload)
        stats = self.client.stats()
        self.assertEqual(stats["entries"], 1)
        self.assertEqual(self.client.delete(["m:1200:abc"]), 1)

    def test_missing_and_bad_body(self):
        with self.assertRaises(BlobError):
            self.client.download("m:1:none", os.path.join(self.tmp.name, "out"))
        self.assertIsNone(self.client.match(["m:1:none"]))

    def test_health(self):
        self.assertTrue(self.client.ping()["pong"])


if __name__ == "__main__":
    unittest.main()
