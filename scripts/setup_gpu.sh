#!/usr/bin/env bash
# Set up mjbot on a fresh GPU machine (e.g. AWS g6.8xlarge with the Deep Learning AMI).
#   bash scripts/setup_gpu.sh
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python -m pytest -q
.venv/bin/python - <<'PY'
import os, torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-", "cpus", os.cpu_count())
PY
cat <<'MSG'
Next:
  .venv/bin/python -m mjbot.tenhou index
  .venv/bin/python -m mjbot.tenhou fetch          # all indexed games, 1 req/s, resumable
  .venv/bin/python -m mjbot.tenhou convert
  .venv/bin/python -m mjbot.replay --check 1000
  .venv/bin/python -m mjbot.train_sl --logs data/tenhou/mjai --steps 200000 --bs 1024 \
      --ch 192 --blocks 10 --belief-to-policy --workers 24 --out checkpoints/houou.pt
MSG
