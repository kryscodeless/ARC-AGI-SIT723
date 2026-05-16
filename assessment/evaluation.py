#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_result_file(fp: Path) -> Tuple[str, Any, Any]:
    with fp.open('r', encoding='utf-8') as f:
        data = json.load(f)
    task_id = str(data.get('task_id') or fp.stem)
    predicted = data.get('predicted')
    ground_truth = data.get('ground_truth')
    return task_id, predicted, ground_truth


def grids_equal(a: Any, b: Any) -> bool:
    return a == b


def evaluate_dir(level_dir: Path) -> Dict[str, Any]:
    files = sorted([p for p in level_dir.glob('*.json') if p.is_file() and p.name != 'summary.json'])
    total = 0
    correct = 0
    wrong_ids: List[str] = []
    for p in files:
        try:
            task_id, pred, gt = load_result_file(p)
            if pred is None or gt is None:
                continue
            total += 1
            if grids_equal(pred, gt):
                correct += 1
            else:
                wrong_ids.append(task_id)
        except Exception:
            # Skip malformed files
            continue
    acc = (correct / total) if total else 0.0
    return {
        'level': level_dir.name,
        'total': total,
        'correct': correct,
        'wrong': len(wrong_ids),
        'accuracy': acc,
        'wrong_task_ids': wrong_ids,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate ARC predictions exact-match accuracy')
    parser.add_argument('--results_root', default='results/eval_51_400', help='Flat folder of per-task JSON outputs')
    parser.add_argument('--save_summary', default='', help='Path to save summary JSON (empty = skip)')
    args = parser.parse_args()

    base = Path(args.results_root)
    summaries: List[Dict[str, Any]] = []
    overall_total = 0
    overall_correct = 0

    if not base.exists():
        print(f"[error] Missing results dir: {base}")
        return
    s = evaluate_dir(base)
    summaries.append(s)
    overall_total += s['total']
    overall_correct += s['correct']

    overall_acc = (overall_correct / overall_total) if overall_total else 0.0

    # Print live results
    for s in summaries:
        print(f"Level {s['level']}: {s['correct']}/{s['total']} correct | accuracy={s['accuracy']*100:.1f}%")
        if s['wrong_task_ids']:
            print(f"  Wrong task_ids ({len(s['wrong_task_ids'])}): {', '.join(s['wrong_task_ids'])}")

    print(f"Overall: {overall_correct}/{overall_total} correct | accuracy={overall_acc*100:.1f}%")

    if args.save_summary:
        out_path = Path(args.save_summary)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open('w', encoding='utf-8') as f:
            json.dump({
                'per_level': summaries,
                'overall': {
                    'total': overall_total,
                    'correct': overall_correct,
                    'accuracy': overall_acc,
                }
            }, f, ensure_ascii=False, indent=2)
        print(f"Saved summary to {out_path}")


if __name__ == '__main__':
    main()


