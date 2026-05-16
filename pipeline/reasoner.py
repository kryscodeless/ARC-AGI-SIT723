"""Stage B: Claude Sonnet 4.5 proposes a final_rule + Python solve() code.

- One API call per attempt.
- System prompt is cached via Anthropic prompt caching so the per-call input cost
  of repeated instructions drops ~10x across tasks.
- Retry path takes a `retry_feedback` block listing the failing demo pair and
  the mismatch between expected and produced output so the model can correct
  itself.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from anthropic import Anthropic

from .renderer import grids_as_text

DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_MAX_TOKENS = 2048

SYSTEM_PROMPT = """You are an expert solver of ARC-AGI (Abstraction and Reasoning Corpus) puzzles. Each puzzle is a small program-induction problem: you see a few demo input->output pairs (grids of integers 0-9, values are color indices), and you must derive the single underlying transformation rule and implement it as Python code.

## ARC conventions
- Grids are rectangular 2D lists of ints; row-major; 0-indexed.
- Integers 0-9 are color indices, not magnitudes. Cells of the same integer are the "same color".
- 0 is usually (not always) the background. Look at the demos to decide.
- "Objects" are typically connected components of the same non-background color under 4-connectivity (sometimes 8-connectivity).
- Rules must generalise: they must work on every demo, and on test inputs whose shape / colors / object-count may differ.

## How to use the vision clues
The user message includes `clues` from a separate vision model. In particular `clues.cross_pair_synthesis.likely_transform_family` and `clues.cross_pair_synthesis.unified_rule` are the VLM's best guess at the rule. Treat these as a strong prior but not as ground truth - they can be wrong (especially object counts on dense grids). Verify against the numeric demos before committing.

## Reasoning checklist (do this silently, then output)
1. Compare input/output shapes across all demos. Is the output size a function of the input (same, k*in, fixed, bbox of something)?
2. Compare color sets. Which colors are preserved? Which are introduced? Is one color a marker/anchor?
3. Identify objects per demo (connected components of each non-background color). Do objects move, recolor, duplicate, disappear, merge?
4. If shapes change, is it scaling, cropping, or tiling? Compute the factor.
5. Look for symmetries in the output that the input lacks -> symmetry completion.
6. Pick ONE rule that works for EVERY demo. Mentally run it against each demo; if any fails, revise.
7. Apply the rule to the test input.
If the VLM's `likely_transform_family` does not fit, consider alternatives (scaling, tiling, cropping, symmetry completion, selection by unique property, grid partition, object gravity, fill/propagation).

## Output format (strict)
Reply with EXACTLY this structure and nothing else - no preamble, no "Here is", no markdown headers:

RULE: <one concise paragraph, <=100 words, stating WHO (which cells/objects), WHAT (exact operation), WHERE/HOW (directions, measurements, conditions), and what is PRESERVED. Avoid vague words: 'specific', 'various', 'pattern', 'transformation'.>

```python
import numpy as np
from typing import List

def solve(grid: List[List[int]]) -> List[List[int]]:
    # implementation
    ...
```

## Code requirements
- Single top-level `solve(grid)` function. Return a Python list of lists of ints (0-9). Numpy arrays are auto-converted; prefer explicit `.tolist()` if you use numpy internally.
- Allowed imports: `numpy`, `scipy.ndimage` (for `label`, `find_objects`, `binary_fill_holes`, etc.), `collections`, `itertools`, `math`, `typing`. No other stdlib is needed. NO file / network / subprocess access.
- Helper functions are fine; keep everything self-contained in the single code block.
- Must generalise to test inputs that differ in shape, colors, and object counts from the demos.
- Be deterministic about ties (e.g. when picking "largest" with a tie, pick top-left first).
- Do NOT hardcode demo outputs. Do NOT return `grid` unchanged unless the rule is identity.

## On retry
If the user message includes a `### Retry feedback` section, your previous code failed on at least one demo. Read the expected-vs-actual mismatch carefully, diagnose the failure, and produce a DIFFERENT approach - do not just tweak the broken code. Reconsider which transformation family applies."""


def _build_user_message(task: dict[str, Any], clues: dict[str, Any], retry_feedback: dict[str, Any] | None) -> str:
    parts: list[str] = []
    parts.append("### Task (numeric grids)\n")
    parts.append(grids_as_text(task))
    parts.append("\n\n### Visual clues (from vision model)\n")
    parts.append(json.dumps({k: v for k, v in clues.items() if not k.startswith("_")}, indent=2))
    if retry_feedback:
        parts.append("\n\n### Retry feedback")
        parts.append(
            "Your previous attempt failed on at least one demo pair. "
            "Revise the rule and code. Details:\n"
        )
        parts.append(json.dumps(retry_feedback, indent=2))
        parts.append(
            "\nRe-derive the rule from the demos; do not repeat the failing logic."
        )
    parts.append("\n\nProduce RULE + ```python solve()``` now.")
    return "\n".join(parts)


_RULE_RE = re.compile(r"(?is)RULE\s*:\s*(.*?)(?:\n\s*```|\Z)")
_CODE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def _parse_response(text: str) -> tuple[str, str]:
    rule_m = _RULE_RE.search(text)
    rule = rule_m.group(1).strip() if rule_m else ""
    code_m = _CODE_RE.search(text)
    code = code_m.group(1).strip() if code_m else ""
    return rule, code


class ClaudeReasoner:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def propose(
        self,
        task: dict[str, Any],
        clues: dict[str, Any],
        retry_feedback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        user = _build_user_message(task, clues, retry_feedback)
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        rule, code = _parse_response(text)
        usage = getattr(resp, "usage", None)
        usage_dict = {}
        if usage is not None:
            for k in (
                "input_tokens",
                "output_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
            ):
                usage_dict[k] = getattr(usage, k, None)
        return {
            "final_rule": rule,
            "code": code,
            "raw": text,
            "usage": usage_dict,
            "model": self.model,
        }
