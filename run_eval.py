#!/usr/bin/env python
"""Full-eval runner: ARC-AGI-1 evaluation set through the VLM + Claude pipeline."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from pipeline.orchestrator import solve_task, summarise, write_result
from pipeline.reasoner import DEFAULT_MODEL as CLAUDE_MODEL, ClaudeReasoner
from pipeline.vlm import DEFAULT_MODEL_ID as QWEN_MODEL, VisionModel


def _load_tasks(split_dir: Path) -> list[tuple[str, dict]]:
    tasks: list[tuple[str, dict]] = []
    for p in sorted(split_dir.glob("*.json")):
        with p.open("r", encoding="utf-8") as f:
            tasks.append((p.stem, json.load(f)))
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser(description="ARC VLM+Claude pipeline runner")
    parser.add_argument("--split", default="arc1_eval", choices=["arc1_eval"])
    parser.add_argument("--data-root", default="data/arc-agi1/evaluation")
    parser.add_argument("--output-dir", default="results/eval_run")
    parser.add_argument("--limit", type=int, default=0, help="only run first N tasks (0 = all)")
    parser.add_argument("--offset", type=int, default=0, help="skip first N tasks")
    parser.add_argument("--task-ids", default="", help="comma-separated task ids to run (overrides limit/offset)")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--exec-timeout", type=float, default=10.0)
    parser.add_argument("--vlm-model", default=QWEN_MODEL)
    parser.add_argument("--reasoner-model", default=CLAUDE_MODEL)
    parser.add_argument("--quant-bits", type=int, default=4, choices=[4, 8])
    parser.add_argument("--offload-dir", default="offload_dir", help="directory for VLM weight offloading (use a unique path per concurrent job)")
    parser.add_argument("--save-renders", default="", help="optional directory to dump task-sheet PNGs")
    parser.add_argument("--skip-existing", action="store_true", help="skip tasks that already have a result JSON")
    parser.add_argument("--dry-run-no-vlm", action="store_true", help="skip VLM load; used for a structural smoke test only")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return 2

    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"ERROR: data root not found: {data_root}", file=sys.stderr)
        return 2

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    renders_dir = Path(args.save_renders) if args.save_renders else None

    tasks = _load_tasks(data_root)
    if args.task_ids.strip():
        wanted = {s.strip() for s in args.task_ids.split(",") if s.strip()}
        tasks = [t for t in tasks if t[0] in wanted]
    else:
        if args.offset:
            tasks = tasks[args.offset:]
        if args.limit:
            tasks = tasks[: args.limit]

    if args.skip_existing:
        tasks = [(tid, td) for tid, td in tasks if not (out_dir / f"{tid}.json").exists()]

    print(f"Tasks to run: {len(tasks)}")
    print(f"VLM: {args.vlm_model} (quant={args.quant_bits}-bit)")
    print(f"Reasoner: anthropic/{args.reasoner_model}")
    print(f"Output: {out_dir}")

    vlm = VisionModel(model_id=args.vlm_model, quant_bits=args.quant_bits, offload_dir=args.offload_dir)
    if not args.dry_run_no_vlm:
        print("Loading VLM ...")
        vlm.load()
        print("VLM loaded.")

    reasoner = ClaudeReasoner(model=args.reasoner_model)

    results: list[dict] = []
    for i, (tid, task) in enumerate(tasks, start=1):
        print(f"[{i}/{len(tasks)}] {tid} ...", flush=True)
        try:
            res = solve_task(
                tid,
                task,
                vlm,
                reasoner,
                max_attempts=args.max_attempts,
                exec_timeout=args.exec_timeout,
                save_renders_dir=renders_dir,
            )
        except Exception as e:
            print(f"  FATAL: {e}")
            res = {
                "task_id": tid,
                "predicted": None,
                "ground_truth": (task.get("test", [{}])[0] or {}).get("output"),
                "match": False,
                "fatal_error": str(e),
                "timings_sec": {},
            }
        write_result(out_dir, res)
        results.append(res)
        t = res.get("timings_sec", {})
        print(
            f"  match={res.get('match')} attempts={res.get('attempts_used', '?')} "
            f"demos={res.get('demo_pass_count', 0)}/{res.get('demo_total', 0)} "
            f"vlm={t.get('vlm', 0):.1f}s reasoner={sum(t.get('reasoner_attempts', [])):.1f}s "
            f"total={t.get('total', 0):.1f}s"
        )

    summary = summarise(results)
    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    solved_path = out_dir / "solved.txt"
    with solved_path.open("w", encoding="utf-8") as f:
        f.write("# Solved task ids\n")
        for tid in summary["solved_task_ids"]:
            f.write(tid + "\n")
        f.write("\n# Failed task ids\n")
        for tid in summary["failed_task_ids"]:
            f.write(tid + "\n")

    print(f"\nSummary: {summary['solved']}/{summary['tasks']} correct "
          f"({summary['accuracy']*100:.1f}%)")
    print(f"  attempt-1 solves: {summary['solved_on_attempt_1']}")
    print(f"  attempt-2 solves: {summary['solved_on_attempt_2']}")
    print(f"  demos-passed-but-test-failed: {len(summary['demos_passed_but_test_failed_ids'])}")
    print(f"  mean per-task: vlm {summary['mean_vlm_sec']:.1f}s (p95 {summary['p95_vlm_sec']:.1f}s) | "
          f"reasoner {summary['mean_reasoner_sec']:.1f}s (p95 {summary['p95_reasoner_sec']:.1f}s) | "
          f"total {summary['mean_total_sec']:.1f}s (p95 {summary['p95_total_sec']:.1f}s)")
    print(f"  Anthropic usage: {summary['anthropic_usage']}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {solved_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
