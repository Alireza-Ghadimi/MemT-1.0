#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INFERENCE_DIR="${ROOT_DIR}/inference"
VENV_DIR="${ROOT_DIR}/.venv"
MODEL_DIR="${MEMT_MODEL_DIR:-${ROOT_DIR}/checkpoints/MemT-1.0-Nano}"
STATE_DIR="${MEMT_STATE_DIRECTORY:-${ROOT_DIR}/memt-state}"
MODEL_REPO="MemTensor/MemT-1.0-Nano"
MODEL_REVISION="30fe0b22422c4cc4c87475f98db5c8baa5c1a902"

usage() {
  echo "Usage: $0 setup|serve|health"
  echo "  setup  Install the server and download the pinned Nano checkpoint (~70 GB)."
  echo "  serve  Start the official MemT SGLang server on 127.0.0.1:30000."
  echo "  health Check the running server's MemT capabilities."
}

require_gpu_host() {
  if [[ "$(uname -s)" != "Linux" ]]; then
    echo "MemT's official SGLang server requires Linux with NVIDIA CUDA." >&2
    exit 1
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is missing; the NVIDIA driver is required." >&2
    exit 1
  fi
  local gpu_count
  gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
  if (( gpu_count < 2 )); then
    echo "MemT-1.0-Nano's published TP2 launch needs two NVIDIA GPUs; found ${gpu_count}." >&2
    exit 1
  fi
}

check_checkpoint() {
  python3 - "$MODEL_DIR" <<'PY'
import json
from pathlib import Path
import sys

model = Path(sys.argv[1])
index_path = model / "model.safetensors.index.json"
for name in ("config.json", "tokenizer.json", "memt_block.safetensors"):
    if not (model / name).is_file():
        raise SystemExit(f"Checkpoint is missing {name}: {model}")
if not index_path.is_file():
    raise SystemExit(f"Checkpoint is missing {index_path}")
index = json.loads(index_path.read_text())
shards = set(index["weight_map"].values())
missing = sorted(name for name in shards if not (model / name).is_file())
if missing:
    raise SystemExit(f"Checkpoint is missing {len(missing)} weight shard(s): {missing[:3]}")
print(f"Checkpoint ready: {len(shards)} weight shards in {model}")
PY
}

case "${1:-}" in
  setup)
    require_gpu_host
    if ! command -v uv >/dev/null 2>&1; then
      echo "Install uv before running setup: https://docs.astral.sh/uv/" >&2
      exit 1
    fi
    python3 -c 'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ required"'
    uv venv --python "$(command -v python3)" "$VENV_DIR"
    (cd "$INFERENCE_DIR" && uv pip install --python "$VENV_DIR/bin/python" '.[server]')
    mkdir -p "$MODEL_DIR"
    "$VENV_DIR/bin/python" - "$MODEL_DIR" "$MODEL_REPO" "$MODEL_REVISION" <<'PY'
from huggingface_hub import snapshot_download
import sys

model_dir, repo_id, revision = sys.argv[1:]
snapshot_download(repo_id=repo_id, revision=revision, local_dir=model_dir)
PY
    check_checkpoint
    echo "Setup complete. Run: $0 serve"
    ;;
  serve)
    require_gpu_host
    if [[ ! -x "$VENV_DIR/bin/python" ]]; then
      echo "Server environment is missing. Run '$0 setup' first." >&2
      exit 1
    fi
    check_checkpoint
    mkdir -p "$STATE_DIR"
    export MEMT_PYTHON="$VENV_DIR/bin/python"
    export MEMT_STATE_DIRECTORY="$STATE_DIR"
    cd "$INFERENCE_DIR"
    exec sh ./start_memt_nano.sh "$MODEL_DIR"
    ;;
  health)
    curl --fail --silent --show-error --max-time 10 \
      "http://127.0.0.1:${MEMT_PORT:-30000}/memt/capabilities"
    echo
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
