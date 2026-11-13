#!/usr/bin/env bash
# Set up and start the kev server (Terminal A).
# Leave it running, then in Terminal B:  python main.py --config config.yaml
#
# Usage:
#   ./setup_kev.sh                         # kev-4B on port 8009
#   KEV_MODEL=jaredpalmer/kev-9b ./setup_kev.sh
#
# Settings (env vars, all optional):
#   KEV_REPO    path to the kev repo          (default: ../kev next to this project)
#   KEV_MODEL   Hub id or local checkpoint dir (default: jaredpalmer/kev-4b@v1.0 with a GPU,
#               jaredpalmer/kev-0.8b@v1.0 on a CPU-only Linux machine)
#   KEV_PORT    server port; must match kev.base_url in config.yaml (default: 8009)
#   KEV_DEVICE  auto | cuda | mps | cpu        (default: auto)
#   HF_HOME     where model weights are cached (default: ~/.cache/huggingface)
#   KEV_API_KEY if set, the server requires it (put the same key in config.yaml)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KEV_REPO="${KEV_REPO:-$SCRIPT_DIR/../kev}"
KEV_MODEL_SET_BY_USER="${KEV_MODEL:+yes}"
KEV_MODEL="${KEV_MODEL:-jaredpalmer/kev-4b@v1.0}"
KEV_PORT="${KEV_PORT:-8009}"
KEV_DEVICE="${KEV_DEVICE:-auto}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

log()  { printf '\n\033[1m[setup_kev]\033[0m %s\n' "$*"; }
warn() { printf '\n\033[33m[setup_kev] WARNING:\033[0m %s\n' "$*" >&2; }
die()  { printf '\n\033[31m[setup_kev] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# --- 1. kev repo -------------------------------------------------------------
if [[ ! -d "$KEV_REPO" ]]; then
  log "kev repo not found at $KEV_REPO; cloning it"
  git clone https://github.com/jaredpalmer/kev.git "$KEV_REPO"
fi
KEV_REPO="$(cd "$KEV_REPO" && pwd)"
[[ -f "$KEV_REPO/pyproject.toml" ]] || die "$KEV_REPO doesn't look like the kev repo (no pyproject.toml)"
log "Using kev repo: $KEV_REPO"

# --- 2. uv -------------------------------------------------------------------
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh || pip install --user uv
  command -v uv >/dev/null 2>&1 || die "uv install failed; see https://docs.astral.sh/uv/"
fi
log "uv $(uv --version | awk '{print $2}')"

# --- 3. Disk space for weights (kev-4B + base is roughly 8+ GB) ----------------
mkdir -p "$HF_HOME"
free_gb=$(df -Pk "$HF_HOME" | awk 'NR==2 {printf "%d", $4/1024/1024}')
log "Weights cache: $HF_HOME (${free_gb} GB free)"
if (( free_gb < 20 )); then
  warn "Less than 20 GB free for model weights. On SageMaker, increase the space's storage or set HF_HOME to a bigger volume."
fi

# --- 4. Install kev's environment (uv fetches Python 3.13 itself if needed) ---
cd "$KEV_REPO"
log "Installing kev dependencies (uv sync --extra serve)"
uv sync --extra serve

# On NVIDIA GPUs kev's README recommends flash-linear-attention for Qwen3.5 models
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
  log "Installing flash-linear-attention"
  uv pip install flash-linear-attention || warn "flash-linear-attention failed to install; kev will still run, just slower"
else
  warn "No NVIDIA GPU detected. kev will use MPS on a Mac or CPU otherwise (slow for large datasets)."
  if [[ "$(uname)" != "Darwin" && -z "$KEV_MODEL_SET_BY_USER" ]]; then
    # On CPU, 4B loads in fp32 (~16 GB, ~30 GB peak while loading); 0.8B needs a few GB
    KEV_MODEL="jaredpalmer/kev-0.8b@v1.0"
    warn "CPU only: using kev-0.8B instead of 4B. Fine for testing the pipeline; use a GPU instance for real runs. Set KEV_MODEL to override."
  fi
fi

# RAM check (Linux): rough peak needed to load each model on CPU in fp32
if [[ -r /proc/meminfo ]]; then
  ram_gb=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
  log "RAM: ${ram_gb} GB"
  case "$KEV_MODEL" in
    *0.8b*) need_gb=6 ;;
    *4b*)   need_gb=32 ;;
    *9b*)   need_gb=72 ;;
    *)      need_gb=0 ;;
  esac
  if ! command -v nvidia-smi >/dev/null 2>&1 && (( need_gb > 0 && ram_gb < need_gb )); then
    warn "Loading $KEV_MODEL on CPU can peak around ${need_gb} GB of RAM; this machine has ${ram_gb} GB. Expect swapping or a crash. Use a smaller model or a GPU instance."
  fi
fi

# --- 5. Port check ------------------------------------------------------------
if curl -s -o /dev/null "http://127.0.0.1:${KEV_PORT}/v1/models"; then
  die "Something is already answering on port $KEV_PORT (an earlier kev server?). Stop it or set KEV_PORT."
fi

# --- 6. Start the server ------------------------------------------------------
# A background watcher prints a clear line once the server answers.
(
  for _ in $(seq 1 720); do  # up to ~1 hour (first run downloads weights)
    if curl -sf -o /dev/null "http://127.0.0.1:${KEV_PORT}/v1/models"; then
      printf '\n\033[32m[setup_kev] kev is ready at http://127.0.0.1:%s, run the suite in Terminal B:\033[0m\n' "$KEV_PORT"
      printf '    python main.py --config config.yaml\n\n'
      exit 0
    fi
    sleep 5
  done
) &

log "Starting kev: $KEV_MODEL on port $KEV_PORT (device: $KEV_DEVICE). The first run downloads weights and can take a while. Ctrl-C to stop."
exec uv run --extra serve python -m kev.serve \
  --run "$KEV_MODEL" --port "$KEV_PORT" --device "$KEV_DEVICE"
