"""Per-task orchestration: render -> VLM -> reasoner -> verify -> retry -> predict.

Writes one JSON per task to the output directory specified by the caller
(e.g. results/eval_smoke50/ or results/eval_51_400/) and aggregates a
summary.json with accuracy + timings.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any

from .executor import run_solver, verify_on_demos
from .reasoner import ClaudeReasoner
from .renderer import render_task_sheet, image_to_bytes
from .vlm import VisionModel, get_clues


def _now() -> float:
    return time.perf_counter()


def solve_task(
    task_id: str,
    task: dict[str, Any],
    vlm: VisionModel,
    reasoner: ClaudeReasoner,
    max_attempts: int = 2,
    exec_timeout: float = 10.0,
    save_renders_dir: Path | None = None,
) -> dict[str, Any]:
    timings: dict[str, Any] = {"reasoner_attempts": [], "exec_attempts": []}
    t_start = _now()

    # Stage A: VLM clues (also returns the rendered task sheet PNG bytes).
    t0 = _now()
    try:
        clues, sheet_png = get_clues(vlm, task)
        vlm_error = None
    except Exception as e:
        clues = {"error": f"vlm_failed: {e}", "trace": traceback.format_exc()[:1500]}
        sheet_png = b""
        vlm_error = str(e)
    timings["vlm"] = _now() - t0

    if save_renders_dir is not None and sheet_png:
        save_renders_dir.mkdir(parents=True, exist_ok=True)
        (save_renders_dir / f"{task_id}.png").write_bytes(sheet_png)

    # Stage B: reasoner with up to max_attempts, retrying on failing demos.
    demos = task.get("train", [])
    best: dict[str, Any] = {}
    best_verify: dict[str, Any] = {}
    retry_feedback: dict[str, Any] | None = None
    reasoner_usages: list[dict[str, Any]] = []
    reasoner_errors: list[str] = []

    for attempt in range(1, max_attempts + 1):
        t0 = _now()
        try:
            proposal = reasoner.propose(task, clues, retry_feedback=retry_feedback)
            err: str | None = None
        except Exception as e:
            proposal = {"final_rule": "", "code": "", "raw": "", "usage": {}, "error": str(e)}
            err = f"{type(e).__name__}: {e}"
        timings["reasoner_attempts"].append(_now() - t0)
        reasoner_usages.append(proposal.get("usage", {}))
        if err:
            reasoner_errors.append(err)

        code = proposal.get("code", "")
        t0 = _now()
        verify = verify_on_demos(code, demos, timeout=exec_timeout) if code else {
            "ok": False,
            "all_pass": False,
            "pass_count": 0,
            "total": len(demos),
            "pair_results": [],
            "first_failure": {"error": "no_code_produced"},
        }
        timings["exec_attempts"].append(_now() - t0)

        better = (
            not best
            or verify.get("pass_count", 0) > best_verify.get("pass_count", -1)
        )
        if better:
            best = proposal
            best_verify = verify

        if verify.get("all_pass"):
            break

        # Build retry feedback for the next attempt.
        retry_feedback = {
            "previous_rule": proposal.get("final_rule", "")[:1000],
            "previous_code_head": (code or "")[:1500],
            "demos_total": verify.get("total"),
            "demos_passed": verify.get("pass_count"),
            "first_failure": verify.get("first_failure"),
        }

    # Prediction on test input(s) using the best code we have.
    t0 = _now()
    test_inputs = [ex["input"] for ex in task.get("test", [])]
    test_run = run_solver(best.get("code", ""), test_inputs, timeout=exec_timeout) if best.get("code") else {
        "ok": False,
        "error": "no_code",
    }
    timings["test_run"] = _now() - t0

    predictions: list[Any] = []
    if test_run.get("ok"):
        for rr in test_run.get("results", []):
            predictions.append(rr.get("grid") if rr.get("ok") else None)
    else:
        predictions = [None] * len(test_inputs)

    ground_truth = [ex.get("output") for ex in task.get("test", [])]

    # Single-test backward compatibility with evaluation.py (predicted/ground_truth fields).
    predicted_single = predictions[0] if predictions else None
    gt_single = ground_truth[0] if ground_truth else None
    match_single = predicted_single == gt_single if predicted_single is not None and gt_single is not None else False

    timings["total"] = _now() - t_start

    return {
        "task_id": task_id,
        "predicted": predicted_single,
        "ground_truth": gt_single,
        "match": match_single,
        "predictions": predictions,
        "ground_truths": ground_truth,
        "final_rule": best.get("final_rule", ""),
        "code": best.get("code", ""),
        "attempts_used": len(timings["reasoner_attempts"]),
        "demo_pass_count": best_verify.get("pass_count", 0),
        "demo_total": best_verify.get("total", len(demos)),
        "demo_all_pass": best_verify.get("all_pass", False),
        "clues": clues,
        "vlm_error": vlm_error,
        "reasoner_errors": reasoner_errors,
        "reasoner_usages": reasoner_usages,
        "test_run_error": None if test_run.get("ok") else test_run.get("error"),
        "timings_sec": timings,
        "provider": f"anthropic/{reasoner.model}",
        "vlm_model": vlm.model_id,
    }


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    solved = sum(1 for r in results if r.get("match"))
    by_attempt = {1: 0, 2: 0}
    demos_all_pass = 0
    solved_ids: list[str] = []
    failed_ids: list[str] = []
    demo_only_ids: list[str] = []
    for r in results:
        tid = r.get("task_id", "")
        if r.get("match"):
            by_attempt[r.get("attempts_used", 1)] = by_attempt.get(r.get("attempts_used", 1), 0) + 1
            solved_ids.append(tid)
        else:
            failed_ids.append(tid)
        if r.get("demo_all_pass"):
            demos_all_pass += 1
            if not r.get("match"):
                demo_only_ids.append(tid)

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    t_vlm = [r["timings_sec"].get("vlm", 0.0) for r in results]
    t_total = [r["timings_sec"].get("total", 0.0) for r in results]
    t_reasoner = [sum(r["timings_sec"].get("reasoner_attempts", [])) for r in results]
    t_exec = [sum(r["timings_sec"].get("exec_attempts", [])) for r in results]

    # Claude usage totals.
    in_tok = 0
    out_tok = 0
    cache_write = 0
    cache_read = 0
    for r in results:
        for u in r.get("reasoner_usages", []) or []:
            in_tok += (u.get("input_tokens") or 0)
            out_tok += (u.get("output_tokens") or 0)
            cache_write += (u.get("cache_creation_input_tokens") or 0)
            cache_read += (u.get("cache_read_input_tokens") or 0)

    return {
        "tasks": total,
        "solved": solved,
        "accuracy": (solved / total) if total else 0.0,
        "solved_on_attempt_1": by_attempt.get(1, 0),
        "solved_on_attempt_2": by_attempt.get(2, 0),
        "demos_all_pass_count": demos_all_pass,
        "solved_task_ids": solved_ids,
        "failed_task_ids": failed_ids,
        "demos_passed_but_test_failed_ids": demo_only_ids,
        "mean_vlm_sec": _mean(t_vlm),
        "p50_vlm_sec": _percentile(t_vlm, 50),
        "p95_vlm_sec": _percentile(t_vlm, 95),
        "mean_reasoner_sec": _mean(t_reasoner),
        "p50_reasoner_sec": _percentile(t_reasoner, 50),
        "p95_reasoner_sec": _percentile(t_reasoner, 95),
        "mean_exec_sec": _mean(t_exec),
        "mean_total_sec": _mean(t_total),
        "p50_total_sec": _percentile(t_total, 50),
        "p95_total_sec": _percentile(t_total, 95),
        "anthropic_usage": {
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cache_creation_input_tokens": cache_write,
            "cache_read_input_tokens": cache_read,
        },
    }


def write_result(out_dir: Path, result: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result['task_id']}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
