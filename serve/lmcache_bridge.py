# SPDX-License-Identifier: MIT
"""The Strata <-> LMCache bridge: plan, restore, save.

Why a session image and not LMCache KV pages
--------------------------------------------
Strata's engine is not vLLM.  Its 48 layers alternate QSA full attention (12
paged K/V layers) with GDN linear attention (36 recurrent states), the K/V may
be quantized, and the draft layer has its own cache.  LMCache's vLLM connector
and KV-page layout describe vLLM's paged tensors, so they cannot carry this
state.  What LMCache *can* do for Strata, and what the engine already knows how
to produce, is an opaque, self-validating snapshot of the state at a token
boundary: ``SAVE`` / ``RESTORE`` (``conversation_file.hpp``).  This bridge puts
those snapshots in LMCache's daemon and keys them by a token-prefix chain, so a
card that never read a prompt can restore another card's prefix.

The engine pins a checkpoint exactly at the boundary this bridge chooses
(``strata_prefix`` / ``pin=N``), which is what makes a snapshot reusable by a
request that only shares a prefix.  A pinned checkpoint wins over a deeper
periodic one when a session file is written (``session_deepest_checkpoint``).
If pinning is unavailable the cache still works, only less precisely.

Safety rules
------------
* opt-in, and a failure here never changes an answer: the request falls back to
  a normal prefill;
* a restore is attempted only when it can beat what the engine already holds;
* SAVE/RESTORE are illegal with parallel requests and layer splits; the bridge
  refuses to start there instead of failing per request;
* every entry carries the model/KV identity and a SHA-256 of its bytes; the
  engine checks its own session fingerprint again on restore.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from serve.lmcache_client import BlobError, HttpBlobClient, LmcacheClient, create_client
from serve.lmcache_keys import (
    DEFAULT_CHUNK,
    PrefixChain,
    check_chunk_size,
    common_prefix,
    model_key,
    system_prompt_hint,
)


class ConfigError(ValueError):
    """The ``lmcache`` block is wrong; the server refuses to start."""


@dataclass
class Fetch:
    path: str
    thread: threading.Thread
    meta: dict = field(default_factory=dict)
    error: str = ""
    done: threading.Event = field(default_factory=threading.Event)


@dataclass
class Plan:
    """What this request should restore, pin and save (all optional)."""
    pin: int = 0
    save_key: str = ""
    save_tokens: int = 0
    key: str = ""
    tokens: int = 0
    meta: dict = field(default_factory=dict)
    fetch: Fetch | None = None
    restore_ok: bool = False
    restore_tried: bool = False
    save_queued: bool = False
    reason: str = ""

    @property
    def restores(self) -> bool:
        return self.fetch is not None


class Bridge:
    def __init__(self, spec: dict, model_name: str, engine_version: str | None,
                 max_context: int, model_paths: str = "", log=print):
        if not isinstance(spec, dict):
            raise ConfigError('"lmcache" must be true, false, or an object')
        self.spec = dict(spec)
        self.chunk = check_chunk_size(int(self.spec.get("chunk_size", DEFAULT_CHUNK)))
        self.min_tokens = int(self.spec.get("min_tokens", max(2 * self.chunk, 512)))
        if self.min_tokens < 1:
            raise ConfigError("lmcache.min_tokens must be >= 1")
        self.max_tokens = int(self.spec.get("max_tokens", 0) or 0)
        self.do_restore = bool(self.spec.get("restore", True))
        sv = self.spec.get("save", "auto")
        self.do_save = sv is not False and str(sv).strip().lower() not in ("off", "no", "false", "0")
        self.do_pin = bool(self.spec.get("pin", True))
        self.timeout_s = float(self.spec.get("timeout_s", 30.0))
        self.match_timeout_s = float(self.spec.get("match_timeout_s", 20.0))
        self.restore_timeout_s = float(self.spec.get("restore_timeout_s", 60.0))
        self.save_timeout_s = float(self.spec.get("save_timeout_s", 600.0))
        self.save_wait_s = float(self.spec.get("save_lock_timeout_s", 300.0))
        self.save_when_busy = bool(self.spec.get("save_when_busy", False))
        self.max_tokens = self.max_tokens if self.max_tokens > 0 else 0
        self.model = str(self.spec.get("model_key") or
                         model_key(model_name, engine_version, max_context, model_paths,
                                   extra=str(self.spec.get("model_extra") or "")))
        self.log = log or (lambda *_a, **_k: None)
        try:
            self._provider = create_client(self.spec)
        except BlobError as exc:
            raise ConfigError(str(exc)) from exc
        # An upload of a 1.2 GiB image is hundreds of chunk calls; a second
        # client keeps the next request's lookup from waiting behind it.
        if isinstance(self._provider, (LmcacheClient, HttpBlobClient)):
            try:
                self._uploader = create_client(self.spec)
            except BlobError:
                self._uploader = self._provider
        else:
            self._uploader = self._provider
        self._closed = threading.Event()
        self._lock = threading.RLock()
        self._cooldown_until = 0.0
        self._error_logged = 0.0
        self._threads: list[threading.Thread] = []
        self._sys_cache: dict[str, int] = {}
        self._last_ids: list[int] = []
        self._held_tokens = 0
        self._stats = {"plans": 0, "hits": 0, "restores": 0, "restore_failures": 0,
                       "saves": 0, "save_failures": 0, "skip_busy": 0, "skipped": 0}
        work = self.spec.get("work_dir")
        if work:
            Path(work).mkdir(parents=True, exist_ok=True)
        else:
            work = self._ram_work_dir()
        self.work_dir = tempfile.mkdtemp(prefix="strata-lmcache-", dir=work or None)
        try:
            free = shutil.disk_usage(self.work_dir).free
            where = "RAM (/dev/shm)" if os.path.abspath(self.work_dir).startswith("/dev/shm") else self.work_dir
            self.log(f"shared prefix cache: {self._describe()} (chunk {self.chunk}, "
                     f"model {self.model}, staging {where}, {free / (1 << 30):.0f} GiB free)")
            if free < (1 << 30):
                self.log(f"shared prefix cache: WARNING: the staging directory has only "
                         f"{free / (1 << 20):.0f} MiB free; a long conversation needs more (set "
                         f"lmcache.work_dir to a large tmpfs)")
        except Exception:
            pass

    @staticmethod
    def _ram_work_dir() -> str | None:
        """A tmpfs for the engine's SAVE/RESTORE when the disk is slow.

        The engine can only write a session to a path, so the bridge stages it
        in RAM and hands it to the LMCache module.  ``/dev/shm`` is the usual
        tmpfs on Linux; if it is too small (containers often mount 64 MiB) the
        operator can set ``lmcache.work_dir`` explicitly.
        """
        if os.name == "nt" or not os.path.isdir("/dev/shm"):
            return None
        try:
            free = shutil.disk_usage("/dev/shm").free
            if free >= (6 << 30) and os.access("/dev/shm", os.W_OK):
                return "/dev/shm"
        except OSError:
            pass
        return None

    # --------------------------------------------------------------- helpers
    def _describe(self) -> str:
        p = self.spec.get("provider") or "lmcache"
        if p == "filesystem":
            return f"filesystem {self.spec.get('path')}"
        url = str(self.spec.get("url") or "")
        if url.lower().startswith(("http://", "https://")):
            return f"LMCache HTTP {url}"
        return f"LMCache {url}"

    def _warn(self, text: str, throttle_s: float = 60.0) -> None:
        now = time.monotonic()
        if now - self._error_logged >= throttle_s:
            self._error_logged = now
            try:
                self.log(text)
            except Exception:
                pass

    def _available(self) -> bool:
        return not self._closed.is_set() and time.monotonic() >= self._cooldown_until

    def _failed(self, exc: Exception) -> None:
        self._cooldown_until = time.monotonic() + float(self.spec.get("retry_s", 15.0))
        self._warn(f"shared prefix cache unavailable for {self.spec.get('retry_s', 15.0)}s: {exc}")

    def _system_tokens(self, messages, tools, kwargs, resolver) -> int:
        """The last token of the shared system prompt, cached by its content."""
        hint = system_prompt_hint(messages, tools, kwargs)
        if not hint:
            return 0
        with self._lock:
            if hint in self._sys_cache:
                return self._sys_cache[hint]
        try:
            n = int(resolver() or 0)
        except Exception as exc:                      # never break the request for a hint
            self._warn(f"system-prefix hint unavailable: {exc}", throttle_s=300)
            n = 0
        with self._lock:
            if len(self._sys_cache) > 16:
                self._sys_cache.clear()
            self._sys_cache[hint] = n
        return n

    # ------------------------------------------------------------------ plan
    def plan(self, svc, ids, messages, tools, kwargs, has_images: bool, resolver=None) -> Plan | None:
        """Choose what to restore, where to pin and what to save.  Never raises."""
        try:
            return self._plan(svc, ids, messages, tools, kwargs, has_images, resolver)
        except Exception as exc:                       # a cache must not fail a request
            self._warn(f"plan skipped: {exc}")
            return None

    def _plan(self, svc, ids, messages, tools, kwargs, has_images, resolver) -> Plan | None:
        if self._closed.is_set() or has_images:
            return None
        if getattr(getattr(svc, "engine", None), "batch", 0):
            self._stats["skip_busy"] += 1
            return None
        if len(ids) < self.min_tokens or (self.max_tokens and len(ids) > self.max_tokens):
            return None
        chain = PrefixChain(self.model, self.chunk, ids)
        if not len(chain):
            return None
        # The last token of a prompt starts the next verify window, so a pin
        # must leave at least one token after it.
        boundary = chain.last_before(len(ids) - 2)
        if boundary is None:
            return None
        sys_n = 0
        if resolver is not None and len(messages or []) > 1:
            sys_n = self._system_tokens(messages, tools, kwargs, resolver)
            if sys_n and (sys_n >= len(ids) - 1 or sys_n < self.min_tokens):
                sys_n = 0
        sys_key = chain.at(sys_n) if sys_n else None

        local = common_prefix(self._last_ids, ids) if self._last_ids else 0
        keys = chain.lookup_keys(self.min_tokens, [sys_key] if sys_key else None)
        hit_key, hit_meta = None, {}
        if self._available():
            try:
                found = self._provider.match(keys)
                if found:
                    hit_key, hit_meta = found
                    if hit_meta.get("model") not in (None, "", self.model) or \
                            hit_meta.get("chunk") not in (None, 0, self.chunk):
                        hit_key, hit_meta = None, {}
                    else:
                        self._stats["hits"] += 1
            except Exception as exc:
                self._failed(exc)
        hit_tokens = int(hit_meta.get("tokens") or 0) if hit_key else 0

        # A new conversation with no cached prefix pins the shared system
        # prefix.  Once anything is cached (even just that prefix), pin the
        # newest whole-chunk boundary so the conversation itself is stored; the
        # system entry stays in the cache for the next new conversation.
        continuation = local > max(sys_n, 0) + 1 or hit_tokens > 0
        pin = boundary.tokens
        if self.do_pin and not continuation and sys_n and sys_n < len(ids) - 1:
            pin = sys_n
        if hit_tokens and hit_tokens >= pin:
            # The restored checkpoint is already at or past this boundary: do
            # not pin backwards (the engine cannot create an older checkpoint
            # without re-reading).  Keep the hit's own boundary instead.
            pin = hit_tokens if hit_tokens < len(ids) - 1 else 0
        if pin >= len(ids) - 1:
            earlier = chain.last_before(len(ids) - 2)
            pin = earlier.tokens if earlier is not None else 0
        if pin and pin < self.min_tokens:
            pin = 0
        save_key = ""
        save_tokens = 0
        if pin:
            key = chain.at(pin)
            if key is not None:
                save_key, save_tokens = key.key, pin
            else:
                pin = 0
        elif hit_key and hit_tokens:
            # An exact (or longer) hit: the restored file already holds the
            # pinned checkpoint, so a SAVE under the hit's key preserves it.
            save_key, save_tokens = hit_key, hit_tokens

        plan = Plan(pin=pin if self.do_pin else 0, save_key=save_key, save_tokens=save_tokens)
        if hit_key and self.do_restore and hit_tokens >= self.min_tokens:
            # Restore only if it beats whatever this engine already holds.  A
            # held prefix from an unrelated conversation is worth nothing, so
            # compare against the common prefix with the last prompt too.
            worth = min(self._held_tokens, local) if local else 0
            if hit_tokens > worth:
                plan.key, plan.tokens, plan.meta = hit_key, hit_tokens, hit_meta
                plan.fetch = self._start_fetch(hit_key, plan.meta)
                plan.reason = f"hit {hit_tokens} > {worth}"
            else:
                plan.reason = f"hit {hit_tokens} <= held {worth}"
        self._stats["plans"] += 1
        return plan

    def _start_fetch(self, key: str, meta: dict) -> Fetch:
        path = os.path.join(self.work_dir, f"{uuid.uuid4().hex}.bin")

        def work():
            try:
                got = self._provider.download(key, path)
                fetch.meta = got.get("meta") or meta
            except Exception as exc:
                fetch.error = str(exc)
                try:
                    os.unlink(path)
                except OSError:
                    pass
            finally:
                fetch.done.set()
        fetch = Fetch(path=path, thread=threading.Thread(target=work, daemon=True,
                                                         name="strata-lmcache-fetch"))
        self._threads.append(fetch.thread)
        fetch.thread.start()
        return fetch

    # --------------------------------------------------------------- restore
    def restore(self, svc, plan: Plan | None, cancel=None) -> bool:
        """Load the planned snapshot into the idle engine.  Called under the FIFO."""
        if plan is None or plan.fetch is None or plan.restore_tried:
            return False
        plan.restore_tried = True
        fetch = plan.fetch
        if cancel is not None and cancel.is_set():
            self._discard(fetch.path)
            return False
        if not fetch.done.wait(max(1.0, self.restore_timeout_s)):
            self._stats["restore_failures"] += 1
            self._warn(f"restore skipped: download of {plan.tokens} tokens did not finish in "
                       f"{self.restore_timeout_s:.0f}s")
            self._discard(fetch.path)
            return False
        if fetch.error:
            self._stats["restore_failures"] += 1
            self._warn(f"restore skipped: {fetch.error}")
            self._discard(fetch.path)
            return False
        if cancel is not None and cancel.is_set():
            self._discard(fetch.path)
            return False
        try:
            r = svc.engine.session_file("restore", fetch.path)
            with self._lock:
                self._held_tokens = int(plan.tokens)
            plan.restore_ok = True
            self._stats["restores"] += 1
            try:
                self.log(f"restored {plan.tokens} prompt tokens from the shared cache "
                         f"({r.get('bytes', 0) / (1 << 20):.0f} MiB in {r.get('ms', 0):.0f} ms)")
            except Exception:
                pass
            return True
        except Exception as exc:
            # SessionRefused/ValueError: the file did not match this engine; the
            # state is unchanged.  EngineDied (a transfer failed after device
            # writes) is the caller's to handle, so it is re-raised.
            if exc.__class__.__name__ in ("EngineDied", "EngineSilent", "EngineStuck"):
                raise
            self._stats["restore_failures"] += 1
            self._warn(f"restore refused: {exc}")
            return False
        finally:
            self._discard(fetch.path)

    def _discard(self, path: str) -> None:
        try:
            os.unlink(path)
        except OSError:
            pass

    def forget(self, plan: Plan | None) -> None:
        """Release a plan that ended without a restore (cancelled/refused)."""
        if plan is None:
            return
        fetch, plan.fetch = plan.fetch, None
        if fetch is None:
            return
        if fetch.done.is_set():
            self._discard(fetch.path)
            return

        def wait():
            fetch.done.wait(600)
            self._discard(fetch.path)
        threading.Thread(target=wait, daemon=True, name="strata-lmcache-discard").start()

    # ------------------------------------------------------------------ save
    def after_request(self, svc, plan: Plan | None, ids, finish: str,
                      generated: int = 0, state_epoch: int | None = None) -> None:
        """Remember what this engine holds and queue an asynchronous SAVE."""
        try:
            with self._lock:
                self._last_ids = list(ids)
                if plan is not None and plan.save_tokens:
                    self._held_tokens = plan.save_tokens
            if not self.do_save or plan is None or not plan.save_key:
                return
            if not self._available():
                self._stats["skipped"] += 1
                return
            if finish not in ("stop", "length"):
                return
            if plan.save_queued:
                return
            plan.save_queued = True
            if len(ids) < self.min_tokens:
                return
            if getattr(getattr(svc, "engine", None), "batch", 0):
                self._stats["skip_busy"] += 1
                return
            thread = threading.Thread(target=self._save, daemon=True, name="strata-lmcache-save",
                                      args=(svc, plan, list(ids), state_epoch))
            self._threads.append(thread)
            thread.start()
        except Exception as exc:
            self._warn(f"save not queued: {exc}")

    def _save(self, svc, plan: Plan, ids, state_epoch) -> None:
        path = os.path.join(self.work_dir, f"{uuid.uuid4().hex}.bin")
        acquired = False
        try:
            # A SAVE shares the engine's one request stream.  If a request is
            # already waiting, serving it is worth more than refreshing the
            # shared cache; the next quiet turn saves again.
            if not self.save_when_busy:
                with svc.status_lock:
                    waiting = int(svc.status.get("queued") or 0)
                if waiting:
                    self._stats["skipped"] += 1
                    return
            # The engine's control lines are free only between requests.  Take
            # the same FIFO a request takes; if a new request got there first
            # the state is no longer the one this plan describes, so give up.
            acquired = svc.fifo.acquire(timeout=max(1.0, self.save_wait_s))
            if not acquired:
                self._stats["skipped"] += 1
                return
            if self._closed.is_set():
                return
            if state_epoch is not None and getattr(svc, "_lmcache_state_epoch", None) != state_epoch:
                self._stats["skipped"] += 1
                return
            if not getattr(svc.engine, "alive", lambda: True)():
                self._stats["skipped"] += 1
                return
            try:
                r = svc.engine.session_file("save", path)
            except Exception as exc:
                if exc.__class__.__name__ in ("EngineDied", "EngineSilent", "EngineStuck"):
                    raise
                self._stats["save_failures"] += 1
                self._warn(f"SAVE skipped: {exc}")
                return
        except Exception as exc:
            if exc.__class__.__name__ in ("EngineDied", "EngineSilent", "EngineStuck"):
                self._warn(f"SAVE skipped: the engine ended ({exc})")
                return
            self._stats["save_failures"] += 1
            self._warn(f"SAVE failed: {exc}")
            return
        finally:
            if acquired:
                try:
                    svc.fifo.release()
                except Exception:
                    pass
        # The engine's state is now in a private file; the upload does not need
        # the request lock and must not hold up the next request on a slow link.
        try:
            meta = {"tokens": plan.save_tokens, "saved": int(r.get("tokens", 0)),
                    "model": self.model, "chunk": self.chunk, "created": time.time(),
                    "engine": self.spec.get("engine", ""), "source": self.spec.get("source", "")}
            try:
                got = self._uploader.upload(plan.save_key, path, meta)
                self._stats["saves"] += 1
                try:
                    self.log(f"saved {plan.save_tokens} prompt tokens to the shared cache "
                             f"({got.get('bytes', 0) / (1 << 20):.0f} MiB)")
                except Exception:
                    pass
            except Exception as exc:
                self._stats["save_failures"] += 1
                self._failed(exc)
        finally:
            self._discard(path)

    # ---------------------------------------------------------------- status
    def view(self) -> dict:
        out = {"enabled": True, "model": self.model, "chunk": self.chunk,
               "min_tokens": self.min_tokens, "held_tokens": self._held_tokens, **self._stats}
        try:
            out["store"] = self._provider.stats()
        except Exception as exc:
            out["store_error"] = str(exc)
        return out

    def close(self) -> None:
        self._closed.set()
        try:
            self._provider.close()
        except Exception:
            pass
        if self._uploader is not self._provider:
            try:
                self._uploader.close()
            except Exception:
                pass
        for t in list(self._threads):
            if t.is_alive():
                t.join(timeout=2.0)
        try:
            shutil.rmtree(self.work_dir, ignore_errors=True)
        except Exception:
            pass


def create(cfg: dict, cli, svc, model_name: str, log=print) -> Bridge | None:
    """Build the bridge from the run config and CLI flags, or None when off."""
    spec = cfg.get("lmcache")
    url = getattr(cli, "lmcache_url", None) or os.environ.get("STRATA_LMCACHE_URL")
    path = getattr(cli, "lmcache_path", None) or os.environ.get("STRATA_LMCACHE_PATH")
    if url:
        spec = {**(spec if isinstance(spec, dict) else {}), "provider": "lmcache", "url": url}
    if path:
        spec = {**(spec if isinstance(spec, dict) else {}), "provider": "filesystem", "path": path}
    chunk = getattr(cli, "lmcache_chunk_size", None)
    if chunk:
        spec = {**(spec if isinstance(spec, dict) else {}), "chunk_size": chunk}
    if spec is None or spec is False:
        return None
    if spec is True:
        spec = {}
    if not isinstance(spec, dict):
        raise ConfigError('"lmcache" must be true, false, or an object')
    if spec.get("enabled") is False:
        return None
    if cfg.get("replicas"):
        raise ConfigError("lmcache needs one engine per server: it cannot share the control lines with data-parallel "
                          "\"replicas\" (run one server per card, each with its own LMCache connection)")
    args = cfg.get("args") or []
    if isinstance(args, list) and "--layer-split" in args:
        raise ConfigError("lmcache needs a single-card engine: session SAVE/RESTORE does not support --layer-split")
    if isinstance(args, list) and ("--peer-device" in args or "--batch" in args or "--slots" in args):
        raise ConfigError("lmcache needs one card and one request at a time: remove --peer-device / --batch / --slots")
    if isinstance(args, list) and "--prompt-cache" in args[:-1]:
        try:
            pc = int(args[args.index("--prompt-cache") + 1])
        except (ValueError, IndexError):
            pc = 6
        if pc < 3:
            raise ConfigError("lmcache needs --prompt-cache 3 or more (the default is 6): the bridge pins a "
                              "checkpoint at the shared prefix, and the engine needs room for it")
    provider = str(spec.get("provider") or "lmcache").strip().lower()
    if provider in ("filesystem", "fs", "dir") and not spec.get("path"):
        raise ConfigError('lmcache: the "filesystem" provider needs "path" (a directory all Strata servers share)')
    if provider not in ("filesystem", "fs", "dir") and not str(spec.get("url") or "").strip():
        raise ConfigError('lmcache: the "lmcache" provider needs "url" (tcp://host:port, the LMCache '
                          'multiprocess server) or use {"provider": "filesystem", "path": "..."}')
    engine = getattr(svc, "engine", None)
    if getattr(engine, "batch", 0):
        raise ConfigError("lmcache needs one request at a time: it cannot share the engine's control lines "
                          "with \"parallel\" / --batch")
    if not hasattr(engine, "session_file"):
        raise ConfigError("lmcache needs the Strata engine (it uses the engine's session SAVE/RESTORE); "
                          "the mock engine has none")
    info = getattr(engine, "info", {}) or {}
    model_paths = str(spec.get("model_paths") or "")
    if not model_paths:
        try:
            a = list(cfg.get("args") or [])
            parts = []
            for flag in ("--native", "--pack", "--mmproj", "--max-context", "--kv"):
                if flag in a[:-1]:
                    parts.append(f"{flag}={a[a.index(flag) + 1]}")
            model_paths = "|".join(parts) or str(cfg.get("exe") or "")
        except Exception:
            model_paths = ""
    return Bridge(spec, model_name, str(info.get("version") or info.get("engine") or ""),
                  int(getattr(engine, "max_context", 0) or 0), model_paths, log=log)
