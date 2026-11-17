#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-mac"
MODEL_DIR="${MEMT_MODEL_DIR:-${ROOT_DIR}/checkpoints/MemT-1.0-Nano}"
STATE_DIR="${MEMT_STATE_DIRECTORY:-${ROOT_DIR}/memt-state}"
MODEL_REPO="MemTensor/MemT-1.0-Nano"
MODEL_REVISION="30fe0b22422c4cc4c87475f98db5c8baa5c1a902"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This launcher requires an Apple Silicon Mac." >&2
  exit 1
fi

check_checkpoint() {
  python3 - "$MODEL_DIR" <<'PY'
import json
from pathlib import Path
import sys

model = Path(sys.argv[1])
required = ("config.json", "tokenizer.json", "memt_block.safetensors")
missing = [name for name in required if not (model / name).is_file()]
index_path = model / "model.safetensors.index.json"
if not index_path.is_file():
    missing.append(index_path.name)
if missing:
    raise SystemExit(f"Checkpoint missing files: {', '.join(missing)}")
shards = set(json.loads(index_path.read_text())["weight_map"].values())
missing_shards = sorted(name for name in shards if not (model / name).is_file())
if missing_shards:
    raise SystemExit(f"Checkpoint missing {len(missing_shards)} weight shards")
print(f"Checkpoint ready: {len(shards)} weight shards")
PY
}

check_mps() {
  "$VENV_DIR/bin/python" - <<'PY'
import torch
import torch.nn.functional as F

if not torch.backends.mps.is_available():
    raise SystemExit("PyTorch MPS is unavailable. Run in a normal macOS Terminal.")
query = torch.ones((1, 1, 4, 16), dtype=torch.bfloat16, device="mps")
F.scaled_dot_product_attention(query, query, query)
print("PyTorch MPS BF16 attention available")
PY
}

checkpoint_status() {
  python3 - "$MODEL_DIR" <<'PY'
from pathlib import Path
import re
import sys

model = Path(sys.argv[1])
shards = list(model.glob("model-*-of-*.safetensors"))
total = max((int(match.group(1)) for path in shards
             if (match := re.search(r"-of-(\d+)\.safetensors$", path.name))), default=26)
size = sum(path.stat().st_size for path in shards)
partials = list((model / ".cache/huggingface/download").glob("*.incomplete"))
partial_size = sum(path.stat().st_size for path in partials)
print(f"Checkpoint: {len(shards)}/{total} weight shards complete ({size / 2**30:.1f} GiB saved)")
print(f"Transfers in progress or interrupted: {len(partials)} ({partial_size / 2**20:.0f} MiB)")
print(f"Location: {model}")
PY
}

case "${1:-}" in
  setup)
    free_kib="$(df -Pk "$ROOT_DIR" | awk 'NR==2 {print $4}')"
    if (( free_kib < 85 * 1024 * 1024 )); then
      echo "At least 85 GiB of free disk space is needed for the 70 GB checkpoint." >&2
      exit 1
    fi
    if [[ ! -x "$VENV_DIR/bin/python" ]] || \
       ! "$VENV_DIR/bin/python" -c 'import torch, transformers, accelerate, safetensors, fastapi, uvicorn, huggingface_hub' >/dev/null 2>&1; then
      if ! command -v uv >/dev/null 2>&1; then
        echo "Install uv first: https://docs.astral.sh/uv/" >&2
        exit 1
      fi
      if [[ ! -x "$VENV_DIR/bin/python" ]]; then
        uv venv --python 3.12 "$VENV_DIR"
      fi
      uv pip install --python "$VENV_DIR/bin/python" \
        'torch==2.11.0' 'transformers==5.12.1' 'accelerate==1.15.0' \
        'safetensors==0.8.0' 'fastapi>=0.115,<1' 'uvicorn>=0.30,<1' \
        'huggingface_hub>=0.30,<2'
    else
      echo "Existing Python environment is ready; skipping package installation."
    fi
    check_mps
    checkpoint_status
    export HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-4}"
    "$VENV_DIR/bin/python" - "$MODEL_DIR" "$MODEL_REPO" "$MODEL_REVISION" <<'PY'
from huggingface_hub import snapshot_download
import sys
import time

model_dir, repo_id, revision = sys.argv[1:]
for attempt in range(1, 5):
    try:
        snapshot_download(repo_id=repo_id, revision=revision,
                          local_dir=model_dir, max_workers=2)
        break
    except RuntimeError as exc:
        if "CAS Client Error" not in str(exc) or attempt == 4:
            raise
        delay = 15 * attempt
        print(f"Xet transfer failed (attempt {attempt}/4). "
              f"Completed files are saved; retrying in {delay}s.", flush=True)
        time.sleep(delay)
PY
    check_checkpoint
    echo "Setup complete. Run: $0 serve"
    ;;
  serve)
    if [[ ! -x "$VENV_DIR/bin/python" ]]; then
      echo "Run '$0 setup' first." >&2
      exit 1
    fi
    check_checkpoint
    check_mps
    mkdir -p "$STATE_DIR"
    export MEMT_MODEL_DIR="$MODEL_DIR"
    export MEMT_STATE_DIRECTORY="$STATE_DIR"
    cd "$ROOT_DIR"
    exec "$VENV_DIR/bin/python" -m uvicorn mac_server:app \
      --host 127.0.0.1 --port "${MEMT_PORT:-30000}" --workers 1
    ;;
  health)
    curl --fail --silent --show-error --max-time 10 \
      "http://127.0.0.1:${MEMT_PORT:-30000}/health"
    echo
    ;;
  smoke)
    echo "Testing a four-token MemT response on port ${MEMT_PORT:-30000}..."
    curl --silent --show-error --include --max-time 300 \
      "http://127.0.0.1:${MEMT_PORT:-30000}/v1/chat/completions" \
      --header 'Content-Type: application/json' \
      --data-binary '{"model":"MemTensor/MemT-1.0-Nano","messages":[{"role":"user","content":"Hi"}],"max_tokens":4}'
    echo
    ;;
  status)
    checkpoint_status
    ;;
  *)
    echo "Usage: $0 setup|serve|health|smoke|status" >&2
    exit 2
    ;;
esac
