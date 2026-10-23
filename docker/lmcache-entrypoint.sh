#!/bin/sh
# Entry point of the LMCache container that backs Strata's shared prefix cache.
#
#   docker run ... strata-lmcache:lmcache
#
# Environment:
#   LMCACHE_MAX_BYTES   RAM budget of the shared store (default 64 GiB)
#   LMCACHE_PORT        LMCache ZMQ transport (default 6555; optional for Strata)
#   LMCACHE_HTTP_PORT   LMCache's own status API (default 8080)
#   LMCACHE_BLOB_PORT   the strata_blob HTTP data interface used by Strata (default 8081)
#   LMCACHE_L1_GB       LMCache's own KV page L1, unused by this module (default 1)
#   LMCACHE_STORAGE     memory (default) or filesystem
#   LMCACHE_PATH        the directory when LMCACHE_STORAGE=filesystem
set -eu

MAX_BYTES="${LMCACHE_MAX_BYTES:-68719476736}"
PORT="${LMCACHE_PORT:-6555}"
HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"
BLOB_PORT="${LMCACHE_BLOB_PORT:-8081}"
L1_GB="${LMCACHE_L1_GB:-1}"
STORAGE="${LMCACHE_STORAGE:-memory}"
PATH_DIR="${LMCACHE_PATH:-/data/lmcache}"

case "$MAX_BYTES" in ''|*[!0-9]*) echo "LMCACHE_MAX_BYTES must be a whole number of bytes" >&2; exit 2;; esac
case "$PORT$HTTP_PORT$BLOB_PORT$L1_GB" in *[!0-9]*) echo "ports and L1_GB must be numbers" >&2; exit 2;; esac

case "$STORAGE" in
  memory|ram)
    MODULE_CONFIG="{\"storage\": \"memory\", \"max_bytes\": $MAX_BYTES, \"http_host\": \"0.0.0.0\", \"http_port\": $BLOB_PORT}"
    ;;
  filesystem|fs|dir)
    MODULE_CONFIG="{\"storage\": \"filesystem\", \"path\": \"$PATH_DIR\", \"max_bytes\": $MAX_BYTES, \"http_host\": \"0.0.0.0\", \"http_port\": $BLOB_PORT}"
    ;;
  *)
    echo "LMCACHE_STORAGE must be memory or filesystem" >&2
    exit 2
    ;;
esac

MODULE="{\"module_path\": \"strata_lmcache.module\", \"config\": $MODULE_CONFIG}"
echo "[strata-lmcache] store=$STORAGE max_bytes=$MAX_BYTES blob_http=0.0.0.0:$BLOB_PORT"
echo "[strata-lmcache] Strata config: \"lmcache\": {\"provider\": \"lmcache\", \"url\": \"http://<this-host>:$BLOB_PORT\"}"

set -- server \
  --host 0.0.0.0 --port "$PORT" \
  --http-host 0.0.0.0 --http-port "$HTTP_PORT" \
  --l1-size-gb "$L1_GB" \
  --server-module "$MODULE"

if [ -x /opt/lmcache-venv/bin/lmcache ]; then
  exec /opt/lmcache-venv/bin/lmcache "$@"
fi
exec /opt/lmcache-venv/bin/python3 -m lmcache.cli.main "$@"
