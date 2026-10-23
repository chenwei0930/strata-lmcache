# SPDX-License-Identifier: MIT
"""Stable token-prefix keys for Strata's LMCache bridge.

A key identifies *content*, never a file path, a GPU or an engine process: that
is what lets four Strata services on four A10s share one entry.  The chain is
incremental (each chunk hashes the previous hash plus its own token ids), so a
long conversation produces the same keys no matter which card wrote it or how
the prompt was split into HTTP requests.

The chunk size defaults to 400 because that is the unified block size of
Qwen3.8-Flash-Next (the model Strata serves); it is configurable for other
models.  Keys are only computed at full-chunk boundaries; the system-prompt
boundary is added separately by the bridge when it is known.
"""
from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass

DEFAULT_CHUNK = 400
_MAX_CHUNK = 65536


def check_chunk_size(n: int) -> int:
    if isinstance(n, bool) or not isinstance(n, int) or not 16 <= n <= _MAX_CHUNK:
        raise ValueError(f"chunk_size must be a whole number in 16..{_MAX_CHUNK}, not {n!r}")
    return n


def model_key(model_name: str, engine_version: str | None, max_context: int,
              model_paths: str = "", extra: str = "") -> str:
    """A short, stable identity for the model *and* the settings that change K/V.

    It is not a security boundary and it is not the session-file fingerprint
    (the engine checks that on restore).  It keeps entries from different
    models or KV layouts out of one lookup path, so a wrong-model restore is
    not even attempted.
    """
    payload = json.dumps([model_name or "", engine_version or "", int(max_context or 0),
                          model_paths or "", extra or ""], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _hash(prev: bytes, tokens: list[int]) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    h.update(prev)
    h.update(struct.pack(f">{len(tokens)}I", *[int(t) & 0xFFFFFFFF for t in tokens]))
    return h.digest()


@dataclass(frozen=True)
class PrefixKey:
    key: str
    tokens: int
    digest: bytes


class PrefixChain:
    """The full-chunk keys of one prompt, oldest first."""

    __slots__ = ("model", "chunk", "ids", "_keys")

    def __init__(self, model: str, chunk: int, ids: list[int] | tuple[int, ...]):
        self.model = model
        self.chunk = check_chunk_size(chunk)
        self.ids = list(ids)
        self._keys = self._build()

    def _make(self, tokens: int, digest: bytes) -> PrefixKey:
        return PrefixKey(f"{self.model}:{tokens}:{digest.hex()}", tokens, digest)

    def _build(self) -> list[PrefixKey]:
        out: list[PrefixKey] = []
        prev = b"\0" * 16
        c = self.chunk
        for end in range(c, len(self.ids) + 1, c):
            prev = _hash(prev, self.ids[end - c:end])
            out.append(self._make(end, prev))
        return out

    def __len__(self) -> int:
        return len(self._keys)

    def __iter__(self):
        return iter(self._keys)

    def all(self) -> list[PrefixKey]:
        return list(self._keys)

    def at(self, tokens: int) -> PrefixKey | None:
        """The key for any prefix length <= len(ids), not only a chunk boundary.

        Used for the system-prompt boundary, which is chosen by the chat
        template and is rarely a multiple of the chunk size.
        """
        n = int(tokens)
        if n <= 0 or n > len(self.ids):
            return None
        full = n // self.chunk * self.chunk
        prev = b"\0" * 16
        for end in range(self.chunk, full + 1, self.chunk):
            prev = _hash(prev, self.ids[end - self.chunk:end])
        if full != n:
            prev = _hash(prev, self.ids[full:n])
        return self._make(n, prev)

    def lookup_keys(self, min_tokens: int, extra: list["PrefixKey"] | None = None) -> list[str]:
        """Keys to ask the store for, longest prefix first.

        Entries shorter than ``min_tokens`` are not worth a restore: reading
        them again is cheaper than a file transfer, and they would churn the
        cache.  ``extra`` (the system-prompt key) is tried in length order too.
        """
        floor = max(1, int(min_tokens))
        items = [k for k in self._keys if k.tokens >= floor]
        seen = {k.key for k in items}
        for item in extra or ():
            if item is not None and item.tokens >= floor and item.key not in seen:
                items.append(item)
                seen.add(item.key)
        items.sort(key=lambda k: k.tokens, reverse=True)
        return [k.key for k in items]

    def last_before(self, limit: int) -> PrefixKey | None:
        """The deepest full-chunk key with ``tokens <= limit``."""
        best = None
        for k in self._keys:
            if k.tokens <= limit and (best is None or k.tokens > best.tokens):
                best = k
        return best


def common_prefix(a, b) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def system_prompt_hint(messages, tools=None, kwargs=None) -> str:
    """A cheap cache key for the first message (the system prompt).

    The bridge uses it to avoid re-tokenizing a long system prompt on every
    turn.  A changed tool list or template option changes the rendered head,
    so both are in the key.
    """
    if not isinstance(messages, list) or not messages:
        return ""
    first = messages[0] if isinstance(messages[0], dict) else {}
    role = first.get("role")
    if role not in ("system", "developer"):
        return ""
    body = first.get("content")
    if not isinstance(body, str):
        try:
            body = json.dumps(body, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            body = repr(body)
    try:
        tool_text = json.dumps(tools, sort_keys=True, ensure_ascii=False) if tools else ""
        kw_text = json.dumps(kwargs or {}, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        tool_text, kw_text = repr(tools), repr(kwargs)
    h = hashlib.sha256()
    h.update(body.encode("utf-8", "replace"))
    h.update(b"\0")
    h.update(tool_text.encode("utf-8", "replace"))
    h.update(b"\0")
    h.update(kw_text.encode("utf-8", "replace"))
    return h.hexdigest()
