"""Compact renderer for ARC task sheets.

- Adaptive cell size (clamped) so every sub-grid lands in a predictable pixel band.
- Single task-sheet image per task: one row per demo pair ([input -> output]),
  then one row for each test input.
- No in-cell coordinate labels (they clutter the pattern for VLMs).
- Small header label per sub-grid; optional axis rulers outside the grid.
- Official ARC palette.
"""
from __future__ import annotations

from typing import Any
from PIL import Image, ImageDraw, ImageFont

# Official ARC palette.
ARC_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),         # black
    1: (0, 116, 217),     # blue
    2: (255, 65, 54),     # red
    3: (46, 204, 64),     # green
    4: (255, 220, 0),     # yellow
    5: (170, 170, 170),   # grey
    6: (240, 18, 190),    # fuchsia
    7: (255, 133, 27),    # orange
    8: (127, 219, 255),   # light blue
    9: (135, 12, 37),     # maroon
}

TARGET_PX = 512
MIN_CELL = 16
MAX_CELL = 48
BORDER = 1
HEADER_H = 22
ARROW_W = 40
GAP = 12
ROW_GAP = 18


def _font(size: int = 13) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _grid_shape(grid: list[list[int]]) -> tuple[int, int]:
    h = len(grid)
    w = len(grid[0]) if h else 0
    return h, w


def _cell_size_for(h: int, w: int) -> int:
    longest = max(h, w, 1)
    cs = max(MIN_CELL, min(MAX_CELL, round(TARGET_PX / longest)))
    return int(cs)


def _task_cell_size(task: dict[str, Any]) -> int:
    """Single cell size for the whole task sheet (all sub-grids aligned)."""
    longest = 1
    for ex in task.get("train", []):
        for g in (ex["input"], ex["output"]):
            h, w = _grid_shape(g)
            longest = max(longest, h, w)
    for ex in task.get("test", []):
        h, w = _grid_shape(ex["input"])
        longest = max(longest, h, w)
    return max(MIN_CELL, min(MAX_CELL, round(TARGET_PX / longest)))


def render_grid(
    grid: list[list[int]],
    cell_size: int,
    label: str | None = None,
    show_axes: bool = False,
) -> Image.Image:
    """Render a single ARC grid as a PIL image."""
    h, w = _grid_shape(grid)
    font = _font(13)

    axis_pad = 14 if show_axes else 0
    header_h = HEADER_H if label else 0

    img_w = axis_pad + w * cell_size + BORDER * (w + 1)
    img_h = header_h + axis_pad + h * cell_size + BORDER * (h + 1)
    img = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(img)

    if label:
        draw.text((2, 3), label, fill="black", font=font)

    x0 = axis_pad
    y0 = header_h + axis_pad

    if show_axes:
        small = _font(10)
        for j in range(w):
            cx = x0 + BORDER + j * (cell_size + BORDER) + cell_size // 2 - 4
            draw.text((cx, header_h), str(j), fill="black", font=small)
        for i in range(h):
            cy = y0 + BORDER + i * (cell_size + BORDER) + cell_size // 2 - 6
            draw.text((0, cy), str(i), fill="black", font=small)

    for i in range(h):
        for j in range(w):
            val = grid[i][j]
            color = ARC_COLORS.get(int(val), (128, 128, 128))
            x = x0 + BORDER + j * (cell_size + BORDER)
            y = y0 + BORDER + i * (cell_size + BORDER)
            draw.rectangle(
                [x, y, x + cell_size - 1, y + cell_size - 1],
                fill=color,
                outline=(40, 40, 40),
                width=BORDER,
            )

    return img


def _draw_arrow(canvas: Image.Image, x: int, y_center: int, width: int = ARROW_W) -> None:
    draw = ImageDraw.Draw(canvas)
    shaft_y = y_center
    tip = x + width - 4
    draw.line([(x + 4, shaft_y), (tip - 6, shaft_y)], fill="black", width=2)
    draw.polygon(
        [(tip - 6, shaft_y - 6), (tip, shaft_y), (tip - 6, shaft_y + 6)],
        fill="black",
    )


def render_task_sheet(task: dict[str, Any], show_axes: bool = False) -> Image.Image:
    """Render the full task (all demo pairs + test inputs) as one PIL image."""
    cell_size = _task_cell_size(task)
    train = task.get("train", [])
    test = task.get("test", [])

    pair_rows: list[tuple[Image.Image, Image.Image | None, str]] = []
    for idx, ex in enumerate(train, start=1):
        ih, iw = _grid_shape(ex["input"])
        oh, ow = _grid_shape(ex["output"])
        in_img = render_grid(ex["input"], cell_size, f"Pair {idx} input {ih}x{iw}", show_axes)
        out_img = render_grid(ex["output"], cell_size, f"Pair {idx} output {oh}x{ow}", show_axes)
        pair_rows.append((in_img, out_img, f"pair{idx}"))

    for idx, ex in enumerate(test, start=1):
        ih, iw = _grid_shape(ex["input"])
        in_img = render_grid(ex["input"], cell_size, f"Test {idx} input {ih}x{iw}", show_axes)
        pair_rows.append((in_img, None, f"test{idx}"))

    row_widths: list[int] = []
    row_heights: list[int] = []
    for in_img, out_img, _ in pair_rows:
        if out_img is not None:
            row_widths.append(in_img.width + GAP + ARROW_W + GAP + out_img.width)
            row_heights.append(max(in_img.height, out_img.height))
        else:
            row_widths.append(in_img.width)
            row_heights.append(in_img.height)

    sheet_w = max(row_widths) + 2 * GAP
    sheet_h = sum(row_heights) + ROW_GAP * max(0, len(pair_rows) - 1) + 2 * GAP
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")

    y = GAP
    for (in_img, out_img, _), rh in zip(pair_rows, row_heights):
        x = GAP
        sheet.paste(in_img, (x, y + (rh - in_img.height) // 2))
        if out_img is not None:
            x += in_img.width + GAP
            _draw_arrow(sheet, x, y + rh // 2)
            x += ARROW_W + GAP
            sheet.paste(out_img, (x, y + (rh - out_img.height) // 2))
        y += rh + ROW_GAP

    return sheet


def shape_header(task: dict[str, Any]) -> str:
    """Short human-readable 'shape header' to sidecar with the image in Stage A."""
    parts = []
    for i, ex in enumerate(task.get("train", []), start=1):
        ih, iw = _grid_shape(ex["input"])
        oh, ow = _grid_shape(ex["output"])
        parts.append(f"pair{i}: {ih}x{iw} -> {oh}x{ow}")
    for i, ex in enumerate(task.get("test", []), start=1):
        ih, iw = _grid_shape(ex["input"])
        parts.append(f"test{i}: {ih}x{iw} -> ?")
    return "; ".join(parts)


def _grid_to_text(grid: list[list[int]]) -> str:
    return "\n".join(" ".join(str(int(v)) for v in row) for row in grid)


def grids_as_text(task: dict[str, Any]) -> str:
    """Compact numeric dump of all pairs + test inputs for the reasoner LLM."""
    lines: list[str] = []
    for i, ex in enumerate(task.get("train", []), start=1):
        ih, iw = _grid_shape(ex["input"])
        oh, ow = _grid_shape(ex["output"])
        lines.append(f"=== Demo pair {i}: input {ih}x{iw} ===")
        lines.append(_grid_to_text(ex["input"]))
        lines.append(f"--- Demo pair {i}: output {oh}x{ow} ---")
        lines.append(_grid_to_text(ex["output"]))
    for i, ex in enumerate(task.get("test", []), start=1):
        ih, iw = _grid_shape(ex["input"])
        lines.append(f"=== Test {i}: input {ih}x{iw} ===")
        lines.append(_grid_to_text(ex["input"]))
    return "\n".join(lines)


def image_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    import io
    buf = io.BytesIO()
    img.save(buf, format=fmt, optimize=True)
    return buf.getvalue()
