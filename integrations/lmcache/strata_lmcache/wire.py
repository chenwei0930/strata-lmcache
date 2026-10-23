# SPDX-License-Identifier: MIT
"""Wire format shared by the Strata LMCache module and the Strata server.

The LMCache multiprocess ``SERVER_MODULE_CALL`` envelope carries opaque bytes,
so this package defines its own tiny frame instead of depending on a particular
serializer version:

    +----------------+----------------+------------------+--------------+
    | header_len u32 | data_len u32   | header (UTF-8)   | data (bytes) |
    +----------------+----------------+------------------+--------------+

Headers are JSON objects.  ``data`` carries a blob chunk for ``put_chunk`` and
``get_chunk`` and is empty for the small control calls.  The format is
deliberately boring: it is stable, it needs only the standard library, and a
truncated or oversized frame is refused instead of guessed at.
"""
from __future__ import annotations

import json
import struct

MAGIC_VERSION = 1
_HEADER = struct.Struct(">II")
MAX_HEADER = 1 << 20          # a control header larger than 1 MiB is a bug


class WireError(ValueError):
    """The frame is truncated, malformed, or larger than the caller allows."""


def encode(header: dict, data: bytes = b"") -> bytes:
    """One frame: the JSON header and optional raw payload bytes."""
    if not isinstance(header, dict):
        raise TypeError("header must be a dict")
    blob = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(blob) > MAX_HEADER:
        raise WireError(f"header is {len(blob)} bytes; the limit is {MAX_HEADER}")
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes")
    raw = bytes(data)
    return _HEADER.pack(len(blob), len(raw)) + blob + raw


def decode(payload: bytes, max_data: int | None = None) -> tuple[dict, bytes]:
    """Split one frame into ``(header, data)``.  Fails closed on any doubt."""
    if len(payload) < _HEADER.size:
        raise WireError("truncated frame: no header")
    header_len, data_len = _HEADER.unpack_from(payload, 0)
    if header_len <= 0 or header_len > MAX_HEADER:
        raise WireError(f"bad header length {header_len}")
    if max_data is not None and data_len > max_data:
        raise WireError(f"payload is {data_len} bytes; the limit is {max_data}")
    end_head = _HEADER.size + header_len
    end = end_head + data_len
    if end != len(payload):
        raise WireError(f"frame length {len(payload)} does not match its header ({end})")
    try:
        header = json.loads(payload[_HEADER.size:end_head].decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise WireError(f"header is not JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise WireError("header must be a JSON object")
    return header, payload[end_head:end]


def call_header(version: int = MAGIC_VERSION, **fields) -> dict:
    """Every request carries the version so a mixed deployment fails clearly."""
    return {"v": int(version), **fields}
