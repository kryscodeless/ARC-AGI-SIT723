#!/usr/bin/env python3
"""Render predicted-vs-ground-truth comparison images for every task result.

Layout per image:
  [Predicted]  [Ground Truth]  [Diff]   ← diff only shown for wrong tasks

Correct tasks get a green title; wrong tasks get a red title and a diff panel
where mismatched cells are circled/crossed in bright red.

Outputs land in <results_dir>/_output_renders/{correct,wrong}/<task_id>.png

Usage:
    python assessment/render_outputs.py --results-dir results/eval_smoke50
    python assessment/render_outputs.py --results-dir results/eval_51_400
    python assessment/render_outputs.py          # runs both by default
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.renderer import ARC_COLORS, render_grid, _task_cell_size, _grid_shape

CELL_SIZE = 32          # fixed cell size for output renders (smaller = more tasks fit)
GAP = 10                # horizontal gap between panels
PANEL_GAP = 20
LABEL_H = 20
TITLE_H = 28
FONT_SIZE = 13
TITLE_FONT_SIZE = 14

CORRECT_TITLE_COLOR = (0, 140, 0)
WRONG_TITLE_COLOR = (200, 0, 0)
DIFF_MATCH_DIM = 0.25   # dim matched cells in diff panel to 25% brightness
DIFF_MISMATCH_BORDER = 4
DIFF_MISMATCH_BORDER_COLOR = (255, 0, 0)


def _font(size: int = FONT_SIZE) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _render_grid_fixed(
    grid: list[list[int]],
    label: str,
    cell_size: int = CELL_SIZE,
    label_color: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    h, w = _grid_shape(grid)
    border = 1
    img_w = w * cell_size + border * (w + 1)
    img_h = LABEL_H + h * cell_size + border * (h + 1)
    img = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(img)

    font = _font(FONT_SIZE)
    draw.text((2, 2), label, fill=label_color, font=font)

    y0 = LABEL_H
    for i in range(h):
        for j in range(w):
            val = int(grid[i][j])
            color = ARC_COLORS.get(val, (128, 128, 128))
            x = border + j * (cell_size + border)
            y = y0 + border + i * (cell_size + border)
            draw.rectangle(
                [x, y, x + cell_size - 1, y + cell_size - 1],
                fill=color,
                outline=(60, 60, 60),
                width=border,
            )
    return img


def _render_diff(
    predicted: list[list[int]],
    ground_truth: list[list[int]],
    cell_size: int = CELL_SIZE,
) -> Image.Image:
    """Diff panel: dimmed ground-truth colour for matching cells, bright red
    cross + correct colour border for mismatched cells."""
    h = max(len(predicted), len(ground_truth))
    w = max(len(predicted[0]) if predicted else 0, len(ground_truth[0]) if ground_truth else 0)
    border = 1
    img_w = w * cell_size + border * (w + 1)
    img_h = LABEL_H + h * cell_size + border * (h + 1)
    img = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(img)

    font = _font(FONT_SIZE)
    draw.text((2, 2), "Diff (red = wrong cell)", fill=(180, 0, 0), font=font)

    y0 = LABEL_H

    # handle size mismatch: treat out-of-bounds as -1
    def _get(grid: list[list[int]], r: int, c: int) -> int:
        try:
            return int(grid[r][c])
        except (IndexError, TypeError):
            return -1

    for i in range(h):
        for j in range(w):
            pred_val = _get(predicted, i, j)
            gt_val = _get(ground_truth, i, j)
            match = pred_val == gt_val

            gt_color = ARC_COLORS.get(gt_val, (128, 128, 128)) if gt_val >= 0 else (200, 200, 200)

            if match:
                # dim the colour
                fill = tuple(int(c * DIFF_MATCH_DIM + 255 * (1 - DIFF_MATCH_DIM)) for c in gt_color)
            else:
                # show predicted colour
                pred_color = ARC_COLORS.get(pred_val, (200, 200, 200)) if pred_val >= 0 else (200, 200, 200)
                fill = pred_color

            x = border + j * (cell_size + border)
            y = y0 + border + i * (cell_size + border)

            draw.rectangle(
                [x, y, x + cell_size - 1, y + cell_size - 1],
                fill=fill,
                outline=(60, 60, 60) if match else DIFF_MISMATCH_BORDER_COLOR,
                width=border if match else DIFF_MISMATCH_BORDER,
            )

            if not match:
                # draw a diagonal cross to mark the error
                x1, y1 = x + 3, y + 3
                x2, y2 = x + cell_size - 4, y + cell_size - 4
                draw.line([(x1, y1), (x2, y2)], fill=(255, 0, 0), width=2)
                draw.line([(x2, y1), (x1, y2)], fill=(255, 0, 0), width=2)

    return img


def render_comparison(result: dict[str, Any], cell_size: int = CELL_SIZE) -> Image.Image:
    task_id = result.get("task_id", "unknown")
    predicted = result.get("predicted")
    ground_truth = result.get("ground_truth")
    match = result.get("match", False)

    title_color = CORRECT_TITLE_COLOR if match else WRONG_TITLE_COLOR
    status = "CORRECT" if match else "WRONG"

    panels: list[Image.Image] = []

    if predicted is not None:
        panels.append(_render_grid_fixed(predicted, "Predicted", cell_size))
    if ground_truth is not None:
        panels.append(_render_grid_fixed(ground_truth, "Ground Truth", cell_size))
    if not match and predicted is not None and ground_truth is not None:
        panels.append(_render_diff(predicted, ground_truth, cell_size))

    if not panels:
        img = Image.new("RGB", (200, 60), "white")
        ImageDraw.Draw(img).text((4, 4), "No prediction data", fill="red", font=_font())
        return img

    total_w = sum(p.width for p in panels) + GAP * (len(panels) - 1)
    max_h = max(p.height for p in panels)
    canvas_h = TITLE_H + max_h + 6

    canvas = Image.new("RGB", (total_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    title = f"{task_id}  —  {status}"
    draw.text((4, 4), title, fill=title_color, font=_font(TITLE_FONT_SIZE))

    x = 0
    for panel in panels:
        canvas.paste(panel, (x, TITLE_H))
        x += panel.width + GAP

    return canvas


def process_results_dir(results_dir: Path, cell_size: int = CELL_SIZE) -> None:
    out_root = results_dir / "_output_renders"
    correct_dir = out_root / "correct"
    wrong_dir = out_root / "wrong"
    correct_dir.mkdir(parents=True, exist_ok=True)
    wrong_dir.mkdir(parents=True, exist_ok=True)

    jsons = sorted(p for p in results_dir.glob("*.json") if p.stem not in ("summary", "solved"))
    if not jsons:
        print(f"[skip] No task JSONs found in {results_dir}")
        return

    n_correct = n_wrong = n_skip = 0
    for p in jsons:
        dst_correct = correct_dir / f"{p.stem}.png"
        dst_wrong = wrong_dir / f"{p.stem}.png"
        if dst_correct.exists() or dst_wrong.exists():
            n_skip += 1
            continue

        with p.open(encoding="utf-8") as f:
            result = json.load(f)

        img = render_comparison(result, cell_size=cell_size)
        match = result.get("match", False)
        dst = dst_correct if match else dst_wrong
        img.save(dst)

        if match:
            n_correct += 1
        else:
            n_wrong += 1

    print(
        f"{results_dir.name}: rendered {n_correct} correct, {n_wrong} wrong"
        + (f" ({n_skip} already existed)" if n_skip else "")
    )
    print(f"  → {out_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render predicted-vs-ground-truth images for ARC results")
    parser.add_argument(
        "--results-dir",
        default="",
        help="Results directory to process (default: runs eval_smoke50 and eval_51_400)",
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=CELL_SIZE,
        help=f"Pixel size of each grid cell (default: {CELL_SIZE})",
    )
    args = parser.parse_args()

    if args.results_dir:
        process_results_dir(Path(args.results_dir), cell_size=args.cell_size)
    else:
        for d in ["results/eval_smoke50", "results/eval_51_400"]:
            process_results_dir(Path(d), cell_size=args.cell_size)


if __name__ == "__main__":
    main()
