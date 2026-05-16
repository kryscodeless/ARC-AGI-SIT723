"""Sandboxed executor for LLM-generated solve() code.

Each call runs `solve(grid)` inside a short-lived subprocess with a hard timeout.
This protects the orchestrator from infinite loops, crashes, and memory spikes
without relying on complex in-process isolation.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

DRIVER = r'''
import json, sys, importlib.util, traceback
spec = importlib.util.spec_from_file_location("user_solver", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except Exception as e:
    print(json.dumps({"ok": False, "error": f"import_error: {type(e).__name__}: {e}", "trace": traceback.format_exc()}))
    sys.exit(0)

if not hasattr(mod, "solve"):
    print(json.dumps({"ok": False, "error": "missing_solve_function"}))
    sys.exit(0)

payload = json.load(sys.stdin)
grids = payload["grids"]
outputs = []
for g in grids:
    try:
        out = mod.solve(g)
        if hasattr(out, "tolist"):
            out = out.tolist()
        if not isinstance(out, list) or (out and not all(isinstance(r, list) for r in out)):
            outputs.append({"ok": False, "error": "solve_returned_non_grid"})
            continue
        out = [[int(v) for v in row] for row in out]
        outputs.append({"ok": True, "grid": out})
    except Exception as e:
        outputs.append({"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[:1500]})

print(json.dumps({"ok": True, "results": outputs}))
'''


def _write_user_code(code: str, tmpdir: Path) -> Path:
    path = tmpdir / "user_solver.py"
    path.write_text(code)
    return path


def run_solver(code: str, grids: list[list[list[int]]], timeout: float = 10.0) -> dict[str, Any]:
    """Run solve() on a list of input grids. Returns {'ok', 'results' or 'error'}."""
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        user_path = _write_user_code(code, tdp)
        driver_path = tdp / "_driver.py"
        driver_path.write_text(textwrap.dedent(DRIVER))
        try:
            proc = subprocess.run(
                [sys.executable, str(driver_path), str(user_path)],
                input=json.dumps({"grids": grids}),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "timeout", "results": []}
        if proc.returncode != 0:
            return {"ok": False, "error": f"nonzero_exit: {proc.returncode}", "stderr": proc.stderr[:1500]}
        try:
            return json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception as e:
            return {"ok": False, "error": f"bad_driver_output: {e}", "stdout": proc.stdout[:1500]}


def verify_on_demos(code: str, demos: list[dict[str, Any]], timeout: float = 10.0) -> dict[str, Any]:
    """Run solve() on every demo input and compare to its output.

    Returns a dict with:
      ok: True if imports + runtime succeeded (not whether outputs matched).
      pass_count, total, all_pass
      pair_results: per-pair {pair_index, match, expected, got, error?}
      first_failure: feedback blob for the reasoner, or None.
    """
    inputs = [d["input"] for d in demos]
    expected = [d["output"] for d in demos]
    res = run_solver(code, inputs, timeout=timeout)
    if not res.get("ok"):
        return {
            "ok": False,
            "error": res.get("error", "unknown"),
            "pass_count": 0,
            "total": len(demos),
            "all_pass": False,
            "pair_results": [],
            "first_failure": {"error": res.get("error"), "details": res.get("stderr") or res.get("stdout") or ""},
        }
    results = res.get("results", [])
    pair_results: list[dict[str, Any]] = []
    pass_count = 0
    first_failure: dict[str, Any] | None = None
    for i, (exp, rr) in enumerate(zip(expected, results), start=1):
        if rr.get("ok") and rr.get("grid") == exp:
            pair_results.append({"pair": i, "match": True})
            pass_count += 1
        else:
            got = rr.get("grid")
            err = rr.get("error")
            entry = {"pair": i, "match": False, "error": err, "expected": exp, "got": got}
            pair_results.append(entry)
            if first_failure is None:
                first_failure = {
                    "failing_pair": i,
                    "error": err,
                    "expected": exp,
                    "got": got,
                    "expected_shape": [len(exp), len(exp[0]) if exp else 0],
                    "got_shape": (
                        [len(got), len(got[0]) if got and got[0] else 0] if isinstance(got, list) else None
                    ),
                }
    total = len(demos)
    return {
        "ok": True,
        "pass_count": pass_count,
        "total": total,
        "all_pass": pass_count == total,
        "pair_results": pair_results,
        "first_failure": first_failure,
    }
