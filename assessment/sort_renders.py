#!/usr/bin/env python3
"""Sort task-sheet renders into correct/ and wrong/ subdirectories.

Reads solved.txt from a results directory, then hard-links (or copies) each
task-sheet PNG from _renders/ into _renders/correct/ or _renders/wrong/.

Usage:
    python assessment/sort_renders.py --results-dir results/eval_smoke50
    python assessment/sort_renders.py --results-dir results/eval_51_400
    python assessment/sort_renders.py  # runs both by default
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _parse_solved_txt(solved_txt: Path) -> tuple[set[str], set[str]]:
    """Return (solved_ids, failed_ids) from a solved.txt file."""
    solved: set[str] = set()
    failed: set[str] = set()
    current_bucket = solved
    for line in solved_txt.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            if "failed" in line.lower():
                current_bucket = failed
            continue
        current_bucket.add(line)
    return solved, failed


def sort_renders(results_dir: Path, copy: bool = False) -> None:
    renders_dir = results_dir / "_renders"
    solved_txt = results_dir / "solved.txt"

    if not renders_dir.exists():
        print(f"[skip] No _renders/ dir in {results_dir}")
        return
    if not solved_txt.exists():
        print(f"[skip] No solved.txt in {results_dir}")
        return

    solved_ids, failed_ids = _parse_solved_txt(solved_txt)

    correct_dir = renders_dir / "correct"
    wrong_dir = renders_dir / "wrong"
    correct_dir.mkdir(exist_ok=True)
    wrong_dir.mkdir(exist_ok=True)

    pngs = sorted(renders_dir.glob("*.png"))
    n_correct = n_wrong = n_skip = 0

    for src in pngs:
        task_id = src.stem
        if task_id in solved_ids:
            dst = correct_dir / src.name
            bucket = "correct"
        elif task_id in failed_ids:
            dst = wrong_dir / src.name
            bucket = "wrong"
        else:
            n_skip += 1
            continue

        if dst.exists():
            n_skip += 1
            continue

        if copy:
            shutil.copy2(src, dst)
        else:
            dst.hardlink_to(src)

        if bucket == "correct":
            n_correct += 1
        else:
            n_wrong += 1

    print(
        f"{results_dir.name}: {n_correct} correct, {n_wrong} wrong"
        f"{f', {n_skip} skipped (already exist or not in solved.txt)' if n_skip else ''}"
    )
    print(f"  correct/ → {correct_dir}")
    print(f"  wrong/   → {wrong_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sort _renders/ PNGs into correct/ and wrong/ subdirs")
    parser.add_argument(
        "--results-dir",
        default="",
        help="Single results directory to process (default: runs both eval_smoke50 and eval_51_400)",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of hard-linking (use when src and dst are on different filesystems)",
    )
    args = parser.parse_args()

    if args.results_dir:
        sort_renders(Path(args.results_dir), copy=args.copy)
    else:
        for d in ["results/eval_smoke50", "results/eval_51_400"]:
            sort_renders(Path(d), copy=args.copy)


if __name__ == "__main__":
    main()
