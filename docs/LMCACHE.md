# Shared prompt cache with LMCache

This is the opt-in bridge between Strata and [LMCache](https://github.com/LMCache/LMCache). It lets several Strata
services on the same host (for example four A10s, one service per card) reuse one another's prompt prefixes, so a
long system prompt, tool list or document is read once instead of once per card and once per conversation.

It is **not** the vLLM connector and it does not put vLLM KV pages into LMCache. Strata's engine is a different
model runtime: 36 of its 48 layers are GDN linear attention with recurrent state, 12 are QSA full attention with a
paged K/V, and the K/V may be quantized. LMCache's KV-page layout describes vLLM tensors. What Strata already knows
how to produce is a self-validating **session image** at a token boundary (`SAVE` / `RESTORE`, see
[DETAILS.md](DETAILS.md#session-files-disk)). This bridge stores that image in LMCache and keys it by a token-prefix
chain. LMCache owns the daemon, the transport, the RAM budget and the lifecycle; Strata owns the bytes.

```text
  card 0  Strata :8081 鈹€鈹?  card 1  Strata :8082 鈹€鈹?  one LMCache daemon (CPU/RAM only)
  card 2  Strata :8083 鈹€鈹尖攢鈻?strata_blob server module 鈹€鈻?RAM store (LRU, max_bytes)
  card 3  Strata :8084 鈹€鈹?       鈻?                                 鈹? SERVER_MODULE_CALL, 8 MiB chunks
                       serve/lmcache_bridge.py
```

## What it does per request

1. Strata computes the rolling hash of every whole `chunk_size` tokens (default **400**, the Qwen3.8-Flash-Next
   unified block size) and asks the LMCache module for the longest stored prefix.
2. A hit is downloaded to a **tmpfs** file while the request waits for the engine turn. When the turn starts the
   server sends the engine `RESTORE`; the engine then reads only what follows the restored checkpoint.
3. Strata tells the engine to pin a checkpoint at the boundary (`strata_prefix` / `pin=N`). With no cache hit the
   boundary is the end of the shared system prompt; once anything is cached (even just that prefix) it is the
   newest whole chunk, so the conversation itself is stored too. A pinned checkpoint is the one a session file
   keeps, which is what makes the image reusable by a request that shares only a prefix.
4. After the turn, when the engine is idle, a worker sends `SAVE`, uploads the image under the boundary key and
   deletes the tmpfs file. The worker takes the same request lock as a request and skips the save if a new request
   is already waiting (`save_when_busy: true` disables that politeness).

A failure anywhere in this path is a miss, never a wrong answer: the request reads its prompt normally. A restore
the engine refuses (another model, another KV setting, a corrupt file) leaves the session as it was. Every stored
entry also carries a SHA-256 and the model identity.

## Requirements

- Strata v0.2.3 (this tree). No engine rebuild is needed: the feature uses the engine's existing `SAVE`/`RESTORE`
  and `pin=N` (`--prompt-cache` 3 or more; the default is 6).
- **One card, one engine, one request at a time.** Session files are refused with `--layer-split`, `--peer-device`
  and `--batch` / `"parallel"`. The bridge refuses to start with them instead of failing per request.
- The same engine build, model files, tokenizer and KV settings (`--max-context`, `--kv`, rope, ...) on every card.
  The engine checks its own session fingerprint on restore; a different card is refused, not trusted.
- An LMCache build with the multiprocess server and `--server-module`. The 0.3.x line and the current `dev` branch
  have it. LMCache runs on the CPU and uses no A10 memory in this setup.
- **The HTTP interface (recommended, and what the container uses):** Strata talks to the module's HTTP endpoint
  (`http://host:8081`). The Strata environment then needs neither LMCache nor torch; only the daemon's venv has
  them. The native LMCache transport (`tcp://`/`grpc://`) is still available, but that path imports LMCache's
  transport client and its `torch` dependency into the Strata server; the server hides the GPUs for the import.
  Use HTTP unless you specifically need the native transport.
- RAM. The default store is memory. A 63K-token Strata session is about 1.2 GiB (measured on an RTX 4070 Ti,
  Ryzen 9 5900X, 64 GB RAM, NVMe, IQ3_XXS; [DETAILS.md](DETAILS.md#session-files-disk)); the size scales with the
  context and the K/V format. Size `max_bytes` for the working set, and leave one session of tmpfs per card for the
  staging file.

## 1. Start the LMCache daemon

Install LMCache in its own environment (it is not a Strata dependency):

```sh
python -m venv /opt/lmcache-venv
/opt/lmcache-venv/bin/pip install -U "lmcache>=0.3.1"     # or a current dev build
```

Put this repository's integration directory on the daemon's `PYTHONPATH` and load the module:

```sh
PYTHONPATH=/opt/Strata/integrations/lmcache \
CUDA_VISIBLE_DEVICES= \
/opt/lmcache-venv/bin/lmcache server \
    --host 127.0.0.1 --port 6555 \
    --http-host 127.0.0.1 --http-port 8080 \
    --l1-size-gb 1 \
    --server-module '{
      "module_path": "strata_lmcache.module",
      "config": {
        "storage": "memory",
        "max_bytes": 137438953472,
        "chunk_bytes": 8388608,
        "http_host": "0.0.0.0",
        "http_port": 8081
      }
    }'
```

`http_host`/`http_port` start the small HTTP front the Strata server uses. Bind it to loopback or a private Docker
network; it has no authentication. The LMCache transport (`tcp://127.0.0.1:6555`) remains available for operators
and for the optional native path.

`CUDA_VISIBLE_DEVICES=` keeps the daemon off every A10: this module needs no GPU. Strata's own server also hides the
GPUs for the few milliseconds it takes to import LMCache's transport client **only if you choose the native
transport**; the HTTP path imports neither LMCache nor torch.

- `max_bytes` is the whole shared cache. `137438953472` = 128 GiB; use what the machine can spare. The module
  evicts least-recently-used entries when it is full. `0` means unlimited and is only for a dedicated machine.
- `chunk_bytes` is the transport chunk (8 MiB). It does not change the token boundary; that is Strata's
  `chunk_size`. A `grpc://` URL is clamped to 2 MiB, below gRPC's default 4 MiB receive limit.
- `storage: "filesystem"` with `"path": DIR` is available for persistence, but it is not the recommended path for
  this use case: the image is large and a slow disk is exactly what the RAM store avoids.
- `--l1-size-gb` belongs to LMCache's own KV page cache, which this bridge does not use. Keep it small (1 GiB); the
  module's `max_bytes` is the shared cache. LMCache's `--chunk-size`, `--separate-object-groups` and
  `--supported-transfer-mode` flags are for its vLLM KV path; this module does not use that path.
- Bind to loopback. The module has no authentication of its own; anyone who can reach the port can read cached
  prompts and KV. If the daemon must be remote, put it behind a private network and LMCache's API key / transport
  security.
- LMCache's `--http-port` is only its own status API. Pick a port that does not collide with a Strata service:
  `8080` is Strata's default, so the examples below put Strata on `8081`-`8084` and LMCache on `8080`.

Check that the module loaded:

```sh
curl -s http://127.0.0.1:8080/status | grep -o '"strata_blob[^}]*' | head
```

or start Strata and look for `lmcache: shared prefix cache: LMCache tcp://127.0.0.1:6555`.

## 2. Start each Strata service

Run the modified **source checkout** (the engine binary can be an installed component; `strata.py` resolves it):

```sh
cd /opt/Strata
python strata.py serve --config /path/to/strata-q2_0.json
# or: python strata.py launch --model q2_0 --gpu 0 --port 8081 --yes --no-tui
```

A packaged install carries its own `serve/` copy; copy this checkout's `serve/` and
`integrations/lmcache/` over it, or run the checkout. Only Python files changed - the engine binary is not rebuilt.

Add the block to the run config of every card (`strata-<model>.json`), changing only `--gpu` / `--port`:

```json
{
  "lmcache": {
    "provider": "lmcache",
    "url": "http://127.0.0.1:8081",
    "chunk_size": 400,
    "work_dir": "/dev/shm/strata-lmcache",
    "save_when_busy": false
  }
}
```

or, for a direct `serve/server.py` start (advanced), `--lmcache-url tcp://127.0.0.1:6555`. With `strata launch`, use the
config block above: `strata launch` builds the run config from its saved profile, and the `lmcache` block is kept
as it is.

Start the four services (each reads the same config shape, with its own card and port):

```sh
strata launch --model q2_0 --gpu 0 --port 8081 --yes --no-tui
strata launch --model q2_0 --gpu 1 --port 8082 --yes --no-tui
strata launch --model q2_0 --gpu 2 --port 8083 --yes --no-tui
strata launch --model q2_0 --gpu 3 --port 8084 --yes --no-tui
```

Point the client or a load balancer at the four ports (nginx, Envoy, HAProxy, or a small round-robin proxy).
Sticky sessions are **not** required: a request that lands on another card restores the prefix from LMCache. They
still help, because a local hit avoids the restore transfer.

**Warm the shared prefix once.** The first conversation that runs stores the system-prompt entry; a later
conversation on another card hits it. If all four cards receive their first request at the same instant, each reads
the prompt once (none has stored it yet). A single short request with the real system prompt and tool list, sent
before the traffic starts, populates the entry for all four.

### Keys

| key | default | meaning |
| --- | --- | --- |
| `provider` | `lmcache` | `lmcache` (the daemon) or `filesystem` (a shared directory, for tests and fallback) |
| `url` | 鈥?| `http://host:8081` (the module's HTTP front; no LMCache/torch in Strata) or `tcp://host:6555` / `grpc://host:port` (native transport; needs LMCache in the Strata environment) |
| `path` | 鈥?| the directory for `provider: "filesystem"` |
| `chunk_size` | `400` | token boundary; 16..65536 |
| `min_tokens` | `2 * chunk_size`, at least 512 | a prefix shorter than this is not restored or stored |
| `max_tokens` | `0` | do not cache prompts longer than this (0 = no limit) |
| `restore` | `true` | allow a hit to be restored |
| `save` | `auto` | `off` disables storing (reads-only cache) |
| `pin` | `true` | ask the engine to pin the boundary checkpoint |
| `work_dir` | `/dev/shm` when it has 6 GiB free, else the temp dir | the engine's staging file; **use tmpfs** |
| `timeout_s` | `30` | one LMCache control call |
| `match_timeout_s` | `20` | the lookup call before a request (the first call also imports LMCache's client) |
| `restore_timeout_s` | `60` | the download wait before a restore is abandoned |
| `save_lock_timeout_s` | `300` | how long a save waits for the engine lock |
| `save_when_busy` | `false` | save even when a request is waiting (adds its latency to that request) |
| `retry_s` | `15` | after an LMCache failure, how long before it is tried again |
| `model_key` | derived | override the model identity in the key space (normally leave it) |

`strata_prefix` set by the **client** wins over the bridge; such a request is not stored or restored by it.

`min_tokens` is the point where a restore becomes worth its transfer. The default (`2 * chunk_size`, at least 512,
so 800 with the default chunk) shares most system prompts; a very small system prompt can use 400, and a deployment
that only cares about long RAG documents can raise it to 4096 to keep small side requests out of the cache.
`chunk_size` 400 matches the model's unified block size; smaller values give finer reuse at the cost of more index
keys and more entries. Changing it invalidates the old key space (the entries are still in LMCache until evicted,
but no new request looks for them).

## Memory budget

Let `S` be one session image. It is roughly the live prompt + K/V. From the measured 63K-token example, about
1.2 GiB; a 16K-token prompt is about one quarter of that. Then:

- LMCache module: `max_bytes` 鈮?number of prefixes worth keeping 脳 `S`.
- tmpfs (`work_dir`): at least 2 脳 `S` for four cards if saves and restores can overlap; allocate `4 脳 S` to be
  comfortable. Check with `df -h /dev/shm`; containers often mount only 64 MiB there.
- Strata's own `--session-min-free-mib` (default 4096) is checked by the engine before a save. Raise it if the
  tmpfs is small, or lower it only if you have checked the free space.
- Subtract the four engines' expert arenas and the OS from the machine's RAM before choosing `max_bytes` and the
  tmpfs size. The LMCache store and the tmpfs are **the same physical RAM** as the engines' experts; overcommitting
  will make the expert cache shrink or the OOM killer choose a process. On a 512 GiB host with four engines using
  80 GiB each, about 150 GiB is a sensible ceiling for LMCache + staging, not 400 GiB.

The first turn of a new conversation stores the **system-prompt prefix** (a small image: the prompt up to that
point). Later turns store deeper conversation prefixes (larger). A different conversation that shares the system
prompt therefore restores a small image, not the long conversation.

## Docker (CUDA 13.0, A10)

This fork ships two targets in one `Dockerfile` and a workflow that pushes them to the GitHub Container Registry:

| image | contents |
| --- | --- |
| `ghcr.io/chenwei0930/strata-lmcache:latest` | Strata v0.2.3, engine compiled with **CUDA 13.0** for **sm_86 (A10)** on `ubuntu:24.04`, plus this bridge |
| `ghcr.io/chenwei0930/strata-lmcache:lmcache` | the LMCache daemon (CPU/RAM) on `ubuntu:24.04` with the `strata_blob` module and its HTTP front |

The runtime image needs the NVIDIA driver **580 or newer** and the NVIDIA Container Toolkit. LMCache needs no GPU.
The first push to `main` runs `.github/workflows/docker.yml`; both images are built (the runtime image compiles the
engine for `sm_86`; `CUDA_ARCHITECTURES` is a workflow input). The LMCache packages on GHCR are private until you
change their visibility under **Packages 鈫?Package settings 鈫?Change visibility**.

```sh
# Strata runtime (build ~20 min on a GitHub runner; the engine is compiled)
docker build -t strata-lmcache:latest \
  --build-arg CUDA_IMAGE=nvidia/cuda:13.0.0-devel-ubuntu24.04 \
  --build-arg RUNTIME_BASE=ubuntu:24.04 \
  --build-arg CUDA_ARCHITECTURES=86 \
  --build-arg BUILD_VISION=1 .

# LMCache daemon (CPU/RAM; no CUDA toolkit needed)
docker build -t strata-lmcache:lmcache --target lmcache .
```

`deploy/docker-compose.yml` starts the daemon and four Strata services, one per A10. The Strata container reads
`STRATA_LMCACHE_URL` from its environment, so with compose it needs no edit to the generated run config:

```yaml
environment:
  STRATA_LMCACHE_URL: http://lmcache:8081
```

The container's `/dev/shm` is the staging area; give it `shm_size: 8g` or more (the compose file does). The model is
not in the image: mount your GGUF folder at `/models` and set `MODEL_PATH=/models`, or let the container download
one. `docs/DOCKER.md` explains the rest of the container's settings, which are unchanged from Strata.



- **Passed (this environment):** 33 unit tests in `tests/lmcache` (`python -m unittest discover -s tests/lmcache`)
  cover the token chain, the memory and disk stores, the chunked wire protocol, the module handlers, the client,
  the bridge, and the `Service` hook with a fake engine and the real chat template. The FS provider restored a
  prefix into a second `Service` and refused a wrong-engine restore without failing the request.
- **Not run here:** a real `lmcache server` process, a real LMCache transport, an A10, and a real Strata engine
  (this checkout has no GPU/CUDA build). Do not treat the A10 speed or the LMCache transfer rate as measured.
  Measure on the target with `/v1/status`'s `lmcache` block and the engine log's `restored N prompt tokens` /
  `saved N prompt tokens` lines.
- The engine's session files are documented as **not bit-exact across restarts** (expert residency and GDN rounding
  can differ). The same applies to a cache hit: validate with score-level checks, not a token diff.

## Troubleshooting

| symptom | cause and fix |
| --- | --- |
| `the lmcache package is not importable` | You chose the native `tcp://`/`grpc://` transport, which imports LMCache's client. Use the HTTP URL (`http://host:8081`) instead, or install LMCache into the Strata environment too. |
| `unknown server module method: strata_blob.match` | The daemon did not load the module, or an older LMCache has no `--server-module`. Check `PYTHONPATH`, the JSON's `module_path` (`strata_lmcache.module`), and the daemon log. |
| `restore refused: session ... another model or configuration` | The four cards are not identical (engine version, model files, `--max-context`, `--kv`, rope). Make them identical, or use a different `model_key`/`chunk_size` per group. |
| `pin=... ignored` in the engine log | `--prompt-cache` is 0 or below 3, or the boundary is too close to the end. Keep the default 6. |
| `SERR storage ...` / `507` | The tmpfs is full. Raise `df -h /dev/shm`, set `work_dir` to a large tmpfs, or lower `max_bytes`/context. |
| Every turn is slow | `save_when_busy` is true, or `work_dir` is a slow disk. Set `save_when_busy: false`; use `/dev/shm`; keep the cache warm. |
| No hits | The prompts differ before `min_tokens`, the cards use different tokenizers/settings, or `save` is off. Lower `min_tokens`, check the model identity, and look for `saved N prompt tokens` in the log. |

## Files

| file | role |
| --- | --- |
| `integrations/lmcache/strata_lmcache/` | the LMCache server module: `wire.py`, `service.py`, `store.py` (memory + disk), `module.py` |
| `serve/lmcache_keys.py` | token chunking and rolling hashes |
| `serve/lmcache_client.py` | the LMCache transport client and the filesystem provider |
| `serve/lmcache_bridge.py` | plan / restore / save, the engine's `pin`, the async worker |
| `serve/server.py` | the hook in `Service.prepare` and `Service._run`; the `lmcache` block; `/v1/status` |
| `tests/lmcache/` | unit and in-process integration tests |

