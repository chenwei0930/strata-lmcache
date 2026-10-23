# Strata + LMCache (community build)

This repository is based on **[Niko1221/Strata](https://github.com/Niko1221/Strata) v0.2.3**
(MIT).  It adds an opt-in shared prompt/KV cache for several Strata services on one host, backed by
[LMCache](https://github.com/LMCache/LMCache), and a CUDA 13.0 container build for the NVIDIA A10 (sm_86).

## What changed

- `serve/lmcache_bridge.py`, `serve/lmcache_keys.py`, `serve/lmcache_client.py`: the Strata side. It uses the
  engine's existing `SAVE` / `RESTORE` and `strata_prefix` / `pin=N`, so **the C++/CUDA engine is not rebuilt**.
- `integrations/lmcache/strata_lmcache/`: an LMCache server module that keeps opaque Strata session images in
  **RAM**, with an LRU quota and a small HTTP interface. No vLLM KV-page format is involved: Strata's GDN/QSA
  hybrid state is not a vLLM KV page.
- `serve/server.py`: opt-in `lmcache` config block (also `STRATA_LMCACHE_URL`), the `Service.prepare` / `_run`
  hooks, `/v1/status` reporting.
- `Dockerfile`: two targets. `runtime` compiles the engine with **CUDA 13.0** for `sm_86` (A10) on
  `ubuntu:24.04`; `lmcache` is the CPU/RAM daemon on the same base.
- `.github/workflows/docker.yml`: builds and pushes both images to GHCR.
- `deploy/docker-compose.yml`: four A10 services plus the LMCache daemon.
- `tests/lmcache/`: 33 tests that run without a GPU, LMCache, or a model.
- The upstream `.github/workflows` were removed from this fork so only the image workflow runs.

Read **[docs/LMCACHE.md](docs/LMCACHE.md)** for the design, configuration and limits.

## Images

```text
ghcr.io/chenwei0930/strata-lmcache:latest            Strata, CUDA 13.0, sm_86 (A10), with the bridge
ghcr.io/chenwei0930/strata-lmcache:lmcache           LMCache daemon + strata_blob module (CPU/RAM)
ghcr.io/chenwei0930/strata-lmcache:0.2.3-lmcache
ghcr.io/chenwei0930/strata-lmcache:lmcache-0.2.3
```

The first push to `main` triggers the workflow. The GHCR packages are **private by default**; after the first
run open **Packages**, the package, **Package settings**, and change visibility to Public if you want anonymous
pulls. The workflow logs in with the job's `GITHUB_TOKEN` (`packages: write`).

## Build locally

```sh
# Strata runtime, CUDA 13.0, A10
docker build -t strata-lmcache:latest \
  --build-arg CUDA_IMAGE=nvidia/cuda:13.0.0-devel-ubuntu24.04 \
  --build-arg RUNTIME_BASE=ubuntu:24.04 \
  --build-arg CUDA_ARCHITECTURES=86 \
  --build-arg BUILD_VISION=1 \
  .

# LMCache daemon, same ubuntu:24.04 base
docker build -t strata-lmcache:lmcache \
  --build-arg RUNTIME_BASE=ubuntu:24.04 \
  --build-arg LMCACHE_PIP_SPEC=lmcache==0.3.1 \
  --target lmcache .
```

The runtime image expects an NVIDIA driver **580 or newer** (CUDA 13.0) and the NVIDIA Container Toolkit.
The model is not baked into the image; mount your GGUF files or let the container download one.

## Run four A10 services

```sh
cd deploy
cp .env.example .env          # MODEL, API_KEY, model folder, RAM budget
docker compose up -d
```

`docker-compose.yml` starts one LMCache daemon and four Strata services, each pinned to one card, and points all
four at the daemon over HTTP.  See `docs/LMCACHE.md` for the per-card `lmcache` settings, sizing and limits.

## Upstream

The engine, model handling, CLI and documentation are Strata's.  Upstream fixes should be taken from
[Niko1221/Strata](https://github.com/Niko1221/Strata); this fork only carries the LMCache bridge and the container
build.  Strata is MIT-licensed; see [LICENSE](LICENSE).

