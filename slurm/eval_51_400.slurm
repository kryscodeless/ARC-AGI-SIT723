#!/bin/bash
#SBATCH -J arc-eval51-400
#SBATCH -p gpu
#SBATCH --qos=batch-long
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Second-batch secrets: put a *different* ANTHROPIC_API_KEY here.
# File: ~/.arc_secrets_batch2  (chmod 600), with:
#   export HUGGINGFACE_TOKEN=hf_xxx
#   export ANTHROPIC_API_KEY=sk-ant-xxx   # the NEW key
if [ -f "$HOME/.arc_secrets_batch2" ]; then
  # shellcheck disable=SC1091
  source "$HOME/.arc_secrets_batch2"
fi

: "${HUGGINGFACE_TOKEN:?HUGGINGFACE_TOKEN must be set in ~/.arc_secrets_batch2}"
: "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY must be set in ~/.arc_secrets_batch2}"

mkdir -p logs offload_dir_batch2 results/eval_51_400

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_ENABLE_HF_TRANSFER=1

PYTHON=/sw/software/Anaconda3/2024.02/bin/python3

echo "Running on $(hostname)"
echo "Using ANTHROPIC key ending in: ...${ANTHROPIC_API_KEY: -6}"

$PYTHON - <<'PY'
import importlib, sys
try:
    torch = importlib.import_module("torch")
    print("torch:", getattr(torch, "__version__", "<missing>"))
    print("cuda available:", torch.cuda.is_available())
    print("cuda devices:", torch.cuda.device_count())
except Exception as exc:
    print("WARNING: torch sanity check failed:", repr(exc), file=sys.stderr)
PY

# Tasks 51..400 (run_eval.py sorts tasks alphabetically, so --offset 50 skips the
# first 50 tasks smoke50 already covered). No --limit => run through task 400.
$PYTHON run_eval.py \
  --split arc1_eval \
  --data-root data/arc-agi1/evaluation \
  --output-dir results/eval_51_400 \
  --offset 50 \
  --max-attempts 2 \
  --exec-timeout 10 \
  --offload-dir offload_dir_batch2 \
  --save-renders results/eval_51_400/_renders \
  --skip-existing
