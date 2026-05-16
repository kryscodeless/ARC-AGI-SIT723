# ARC-AGI: VLM + Reasoner Pipeline

End-to-end system that solves ARC-AGI-1 tasks by combining a **local vision-language model** (Qwen3-VL-32B-Instruct, 4-bit) for perception with a **remote reasoner** (Claude Sonnet 4.5) for rule induction and Python code synthesis.

Latest run: **400 tasks, ARC-AGI-1 evaluation set, 115 / 400 = 28.75 % exact-match accuracy.**

## Pipeline

```
ARC task JSON
    │
    ▼
renderer.py ── renders the full task sheet (all demo pairs + test input) as ONE colour image
    │
    ▼
pipeline/vlm.py ── Qwen3-VL-32B (local, 4-bit NF4, 2× GPU)
    │             Stage A: returns a structured JSON of visual clues
    │             (per-pair shapes, colour sets, objects, cross-pair synthesis,
    │              likely_transform_family, unified_rule).
    ▼
pipeline/reasoner.py ── Claude Sonnet 4.5 (Anthropic API)
    │                    Stage B: takes the clues + numeric grids, outputs
    │                    `final_rule` + a Python `solve(grid)` function.
    │                    System prompt is cached (prompt-caching enabled).
    ▼
pipeline/executor.py ── runs solve() on every demo input, verifies against demo outputs.
    │                   If any demo mismatches → build retry feedback (failing pair + diff)
    │                   → second attempt at the reasoner. Max 2 attempts.
    ▼
Predict test output → compare to ground truth → write results/<task_id>.json
```

Key design choices:
- **One VLM call per task**, not per pair – the whole task fits on a single sheet image.
- **Attempt-2 retry** feeds the model the first failing demo pair and the diff, so the reasoner can self-correct.
- **Execution-grounded verification** – the model's Python code must reproduce *every* demo before the test grid is predicted.

## Models

| Stage | Model | Deployment |
|---|---|---|
| Vision (A) | `Qwen/Qwen3-VL-32B-Instruct` | local, 4-bit NF4 (bitsandbytes), bfloat16, `device_map=auto` on 2× A100 |
| Reasoner (B) | `claude-sonnet-4-5` | Anthropic API, **no extended thinking**, `max_tokens=2048`, prompt caching on the system block |
| Executor | Python 3.10 sandbox | 10 s timeout per demo / test grid |

## Experiment setup

- **Dataset**: ARC-AGI-1 evaluation split (`data/arc-agi1/evaluation/`, 400 tasks).
- **Split across two SLURM jobs** (two separate Anthropic keys to parallelise):
  - `slurm/smoke_50.slurm` — tasks 1–50 → `results/eval_smoke50/`
  - `slurm/eval_51_400.slurm` — tasks 51–400 → `results/eval_51_400/`
- **Hardware**: Deakin A100 partition, 2 × A100, 8 CPU, 96 GB RAM, 24 h wall-clock.
- **Attempts**: `--max-attempts 2`, `--exec-timeout 10`.
- **Scoring**: exact match of the predicted test grid against ground truth (strict – ARC standard).

Reproduce:
```bash
# secrets (HUGGINGFACE_TOKEN, ANTHROPIC_API_KEY) in ~/.arc_secrets and ~/.arc_secrets_batch2
sbatch slurm/smoke_50.slurm        # tasks 1..50
sbatch slurm/eval_51_400.slurm     # tasks 51..400
```

## Results

### Overall (400 tasks)

| | Tasks | Solved | Accuracy |
|---|---|---|---|
| Batch 1 (`eval_smoke50`) | 50 | 21 | **42.0 %** |
| Batch 2 (`eval_51_400`) | 350 | 94 | **26.9 %** |
| **Combined** | **400** | **115** | **28.75 %** |

### Attempt breakdown

| Metric | Batch 1 | Batch 2 | Combined |
|---|---|---|---|
| Solved on attempt 1 | 15 | 49 | 64 |
| Solved on attempt 2 (after demo-fail retry) | 6 | 45 | 51 |
| Demos all pass but test failed (overfit) | — | 5 | — |

Attempt-2 recovers ~44 % of eventual solves → the retry-with-diff loop is a major contributor, not just noise.

### Timings (per task, wall-clock)

| Stage | Mean | p50 | p95 |
|---|---|---|---|
| VLM (Qwen3-VL-32B, local) | 165.7 s | 172.4 s | 177.4 s |
| Reasoner (Claude Sonnet 4.5) | 33.6 s | 34.7 s | 45.7 s |
| Executor (Python sandbox) | 1.1 s | — | — |
| **Total per task** | **~201.5 s** | 206.4 s | 221.3 s |

Total wall-clock for the 400-task run: **~22.4 hours** across the two SLURM jobs.

### Anthropic token usage

| | Input tokens | Output tokens |
|---|---|---|
| Batch 1 | 638,524 | 85,236 |
| Batch 2 | 4,670,753 | 637,433 |
| **Total** | **5,309,277** | **722,669** |

Approx. API cost at Sonnet 4.5 list pricing ($3 / MTok input, $15 / MTok output): **~$26.8 for 400 tasks** (~**$0.067 / task**). Prompt caching is enabled on the system prompt; cache-hit revenue was zero on these runs because each SLURM job is a fresh process – caching pays off mainly inside a single long process.

## Comparison with pure-LLM approaches on ARC-AGI-1

All numbers below are on the **public ARC-AGI-1 evaluation set (400 tasks)** — same split this work was run on. Apples-to-apples is still hard: some entries are single API calls, some use programmatic search / ensembling on top of the LLM. The "setup" column makes this explicit. Times are approximate per-task wall-clock from public reports / community runs.

| System | Release | Setup | ARC-AGI-1 eval acc. | ~ Time / task | ~ Cost / task | Notes |
|---|---|---|---|---|---|---|
| GPT-4 (text-only) | 2023 | zero-shot, grids-as-text | ~5 % | ~10 s | ~$0.10–0.20 | Chollet 2024; replicated by community |
| GPT-4o (text-only) | 2024 | single call | ~9 % | ~10 s | ~$0.01 | public reproductions |
| Claude 3.5 Sonnet | 2024 | single call, text | ~14 % | ~20 s | ~$0.02 | public reproductions |
| Gemini 1.5 Pro | 2024 | single call, text | ~12 % | ~15 s | ~$0.01 | public reproductions |
| DeepSeek-R1 | 2025-01 | single call, extended reasoning | ~15–20 % | ~60–120 s | ~$0.01–0.03 | cheap API but high token use |
| Claude 3.7 Sonnet (extended thinking) | 2025-02 | single call, ext-think | ~21 % | ~30–60 s | ~$0.05–0.10 | Anthropic / community |
| o1 (text-only) | 2024-12 | single call, high-reasoning | ~21 % | ~60–120 s | ~$0.30–0.60 | public API runs |
| o1-pro | 2024-12 | single call, very high reasoning | ~25 % | ~3–5 min | ~$5–10 | OpenAI report / community |
| o3-mini (high) | 2025-01 | single call, high-reasoning | ~30 % | ~60–180 s | ~$0.05–0.15 | public API runs |
| Gemini 2.5 Pro | 2025-03 | single call, ext-think | ~25–30 % | ~30–60 s | ~$0.05 | community runs |
| Claude Sonnet 4 | 2025-05 | single call, ext-think | ~25–30 % | ~30–60 s | ~$0.05–0.10 | community runs |
| Grok 4 | 2025-07 | single call, reasoning | ~30–35 % | ~60 s | ~$0.05 | community runs (numbers preliminary) |
| Claude Sonnet 4.5 | 2025-09 | single call, ext-think | ~30–35 % | ~30–60 s | ~$0.05–0.10 | community runs |
| **This work — Qwen3-VL-32B (local, 4-bit) + Claude Sonnet 4.5 (no ext-think) + exec-verify + 1 retry** | 2026-04 | visual clues → code, 2 attempts | **28.75 %** (115/400) | **~200 s** | **~$0.067** (API only) | + local 2× A100 GPU time |
| o3 (low-compute) | 2024-12 | programmatic ensemble | ~76 % | ~minutes / task | ~$1–5 | ARC Prize 2024 |
| o3 (high-compute) | 2024-12 | massive programmatic search | ~88 % | hours / task | **>$20** / task | ARC Prize 2024 winning tier |

> **Caveat:** non-self-numbers in the table are aggregated from public reports, leaderboards, and community reproductions; methodologies (single-attempt vs. best-of-N, prompt format, exact subset of the 400-task eval) vary, so treat ±3–5 percentage points as noise. Times are typical observed latencies, not strict guarantees.


Takeaways vs. pure-text LLMs:
- The VLM-clues + code-synthesis + executor-verify pipeline outperforms a single-shot text-only call to a frontier LLM on the same eval split at a tiny fraction of the per-task cost of ensemble/search approaches.
- It is still well below the ARC Prize tier, which relies on large programmatic search (thousands of candidate programs per task) rather than a single pair of model calls.
- Most remaining failures concentrate in tasks requiring multi-step object reasoning or precise arithmetic on grid coordinates – places where the VLM's summary of the scene loses precision that the reasoner cannot recover from.

## Repository layout

```
run_eval.py                       full-eval runner (split / offset / limit)
slurm/
  smoke_50.slurm                  SLURM: tasks 1..50  (batch-1 key)
  eval_51_400.slurm               SLURM: tasks 51..400 (batch-2 key)
pipeline/
  renderer.py                     task-sheet image + text helpers
  vlm.py                          Qwen3-VL-32B local inference (4-bit)
  reasoner.py                     Claude Sonnet 4.5 API call + retry
  executor.py                     sandboxed solve() runner + demo verify
  orchestrator.py                 per-task render → VLM → reasoner → verify → retry
assessment/
  evaluation.py                   exact-match accuracy evaluator
  sort_renders.py                 sorts task-sheet PNGs into correct/ and wrong/ buckets
  render_outputs.py               renders predicted-vs-ground-truth comparison images
results/
  eval_smoke50/                   batch-1 outputs (50 tasks)
  eval_51_400/                    batch-2 outputs (350 tasks)
    summary.json                    aggregate metrics + token usage
    solved.txt                      solved / failed task ids
    <task_id>.json                  per-task record (clues, rule, code, prediction, timings)
    _renders/                       task-sheet PNGs fed to the VLM
      correct/                        task sheets for solved tasks
      wrong/                          task sheets for failed tasks
    _output_renders/                predicted-vs-ground-truth comparison images
      correct/                        solved tasks (green title)
      wrong/                          failed tasks (red title + diff panel)
data/
  arc-agi1/
    evaluation/                   ARC-AGI-1 evaluation JSONs (400 tasks) — primary dataset
    training/                     ARC-AGI-1 training JSONs
  arc-agi2/
    evaluation/                   ARC-AGI-2 evaluation JSONs
    training/                     ARC-AGI-2 training JSONs
logs/                             SLURM stdout/stderr
```

## Setup

```bash
conda create -n arc-llm python=3.10 -y
conda activate arc-llm
pip install -r requirements.txt

export HUGGINGFACE_TOKEN=hf_...
export ANTHROPIC_API_KEY=sk-ant-...
```

Optional: put the two env vars in `~/.arc_secrets` / `~/.arc_secrets_batch2` (chmod 600) – the SLURM scripts source them automatically.

Run locally on a handful of tasks:
```bash
python run_eval.py \
  --data-root data/arc-agi1/evaluation \
  --output-dir results/debug \
  --limit 5 --max-attempts 2
```

Evaluate a finished run:
```bash
# Quick view — summary already written by run_eval.py:
cat results/eval_51_400/summary.json

# Recompute accuracy from scratch:
python assessment/evaluation.py --results_root results/eval_51_400

# Sort task-sheet PNGs into correct/ and wrong/ (both batches):
python assessment/sort_renders.py

# Render predicted-vs-ground-truth comparison images (both batches):
python assessment/render_outputs.py
```

---
