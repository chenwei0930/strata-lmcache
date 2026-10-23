# SPDX-License-Identifier: MIT
"""A tiny HTTP front for the ``strata_blob`` store.

The LMCache daemon is the owner of the cache; this endpoint lets a Strata
server use it **without importing LMCache or torch**.  That matters for the
container image: the Strata runtime keeps its pinned Python packages, and a
1.2 GiB session still moves at localhost/Docker-bridge speed in 8 MiB frames.

Protocol::

    POST /strata_blob/<method>      body: one ``wire`` frame
                                    reply: one ``wire`` frame, HTTP 200
    GET  /health                    {"ok": true}

The endpoint has no authentication of its own.  Bind it to a private Docker
network or loopback; it can read cached prompts and K/V.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .service import BlobService


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "strata-blob/1"

    def log_message(self, fmt, *args):  # keep the daemon log usable
        return

    def _send(self, status: int, body: bytes = b"", content_type: str = "application/octet-stream"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            try:
                self.wfile.write(body)
            except OSError:
                pass

    def do_GET(self):
        if self.path in ("/health", "/healthz"):
            self._send(200, b'{"ok":true}', "application/json")
        else:
            self._send(404, b'{"error":"not found"}', "application/json")

    def do_POST(self):
        service: BlobService = self.server.service  # type: ignore[attr-defined]
        prefix = "/strata_blob/"
        if not self.path.startswith(prefix):
            self._send(404, b'{"error":"not found"}', "application/json")
            return
        method = self.path[len(prefix):].split("?", 1)[0].strip("/")
        if not method or "/" in method:
            self._send(404, b'{"error":"not found"}', "application/json")
            return
        try:
            length = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            length = -1
        limit = service.max_request_bytes + (1 << 20)
        if length < 0 or length > limit:
            self._send(413, b'{"error":"body too large"}', "application/json")
            return
        try:
            body = self.rfile.read(length)
        except OSError:
            return
        if len(body) != length:
            self._send(400, b'{"error":"short body"}', "application/json")
            return
        try:
            reply = service.call(method, body)
        except Exception as exc:  # the service itself catches; this is the last resort
            reply = json.dumps({"v": 1, "ok": False, "kind": "error",
                                "error": str(exc)[:1000]}).encode("utf-8")
        self._send(200, reply)


class _Server(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class BlobHTTPServer:
    def __init__(self, service: BlobService, host: str = "127.0.0.1", port: int = 8081):
        self.service = service
        self._httpd = _Server((host, int(port)), _Handler)
        self._httpd.service = service  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        daemon=True, name="strata-blob-http")
        self._thread.start()

    @property
    def port(self) -> int:
        return int(self._httpd.server_address[1])

    @property
    def host(self) -> str:
        return str(self._httpd.server_address[0])

    def close(self) -> None:
        try:
            self._httpd.shutdown()
        except Exception:
            pass
        try:
            self._httpd.server_close()
        except Exception:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)
