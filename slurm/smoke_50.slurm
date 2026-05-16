#!/bin/bash
#SBATCH -J arc-smoke50
#SBATCH -p gpu
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# Always run from the directory where sbatch was submitted (this repo).
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Auto-source a secrets file if the user created one (recommended).
# Create ~/.arc_secrets with:
#   export HUGGINGFACE_TOKEN=hf_xxx
#   export ANTHROPIC_API_KEY=sk-ant-xxx
# then chmod 600 ~/.arc_secrets
if [ -f "$HOME/.arc_secrets" ]; then
  # shellcheck disable=SC1091
  source "$HOME/.arc_secrets"
fi

: "${HUGGINGFACE_TOKEN:?HUGGINGFACE_TOKEN must be set (export it or put it in ~/.arc_secrets)}"
: "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY must be set (export it or put it in ~/.arc_secrets)}"

mkdir -p logs offload_dir results/eval_smoke50

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_ENABLE_HF_TRANSFER=1

PYTHON=/sw/software/Anaconda3/2024.02/bin/python3

echo "Running on $(hostname)"
$PYTHON - <<'PY'
import importlib
import sys

try:
    torch = importlib.import_module("torch")
    print("torch module:", getattr(torch, "__file__", "<unknown>"))
    print("torch version:", getattr(torch, "__version__", "<missing>"))
    print("cuda available:", getattr(torch.cuda, "is_available", lambda: False)())
    print("cuda devices:", getattr(torch.cuda, "device_count", lambda: 0)())
except Exception as exc:
    print("WARNING: torch sanity check failed:", repr(exc), file=sys.stderr)
    print("Continuing to run_eval.py so the real import error is visible there.", file=sys.stderr)
PY

# Smoke test: first 50 ARC-AGI-1 eval tasks (alphabetical by task id).
$PYTHON run_eval.py \
  --split arc1_eval \
  --data-root data/arc-agi1/evaluation \
  --output-dir results/eval_smoke50 \
  --limit 50 \
  --max-attempts 2 \
  --exec-timeout 10 \
  --save-renders results/eval_smoke50/_renders \
  --skip-existing
