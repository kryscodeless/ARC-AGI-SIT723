"""Stage A: local Qwen3-VL-32B produces structured visual clues (one call per task).

The VLM receives a single task-sheet image + a short text header stating per-pair
grid shapes. It returns a strict JSON object describing what it sees - no long
chain-of-thought; the heavy reasoning happens in Stage B.
"""
from __future__ import annotations

import json
import math
import os
import re
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from .renderer import render_task_sheet, shape_header

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-32B-Instruct"

VLM_SYSTEM = (
    "You are a meticulous visual analyst for ARC-AGI puzzles. "
    "You look at a single image showing every demo pair of a task (input -> output) "
    "and the test input(s), and you report your systematic observations strictly as a "
    "JSON object. Use the ARC palette: 0=black, 1=blue, 2=red, 3=green, 4=yellow, "
    "5=grey, 6=fuchsia, 7=orange, 8=light-blue, 9=maroon. "
    "Be concrete with counts, colors by index, shapes, positions. "
    "FORBIDDEN WORDS: 'specific', 'particular', 'certain', 'various', 'pattern'. "
    "Do NOT wrap your answer in markdown fences, do NOT add prose outside the JSON."
)

VLM_USER_TEMPLATE = """Task shapes: {shapes}

The image shows every demo pair as a [input -> output] row, followed by the test input(s) on their own row(s).

For EACH demo pair, perform this 5-step systematic analysis, then write one JSON object. Keep each field short (1-2 sentences) but CONCRETE.

STEP 1 - FIRST IMPRESSION: "output is [bigger/smaller/same size/mirrored/filled in/cleaned up]", "I see [more/fewer] colored areas", "the overall pattern is [preserved/changed/partially transformed]".
STEP 2 - SIZE-BASED FAMILY (evaluate all three):
  * Same-size: recoloring / object removal / translation / reflection / fill ?
  * Proportionally scaled: uniform scale factor, tiling, outward propagation ?
  * Non-proportional: cropping with offsets, object extraction, summarisation ?
STEP 3 - SYSTEMATIC SCANNING (input vs output):
  * Vertical (top / middle / bottom region)
  * Horizontal (left / center / right region)
  * State what changed and what stayed identical.
STEP 4 - MECHANISM: the exact operation with measurements and directions (no vague terms).
STEP 5 - CANDIDATE RULE for THIS pair: one concrete sentence with WHO/WHAT/WHERE/HOW and PRESERVATION (what stays unchanged).

Then do a CROSS-PAIR synthesis: what is the single underlying rule consistent across all demos?

Output EXACTLY this JSON (no markdown, no extra keys):
{{
  "per_pair": [
    {{
      "pair": <1-indexed int>,
      "first_impression": "...",
      "size_family": "<same_size | proportional_scale | non_proportional>",
      "size_family_detail": "...",
      "scan_vertical": {{"top": "...", "middle": "...", "bottom": "..."}},
      "scan_horizontal": {{"left": "...", "center": "...", "right": "..."}},
      "objects": "<kind/shape/count/color of distinct objects in the input>",
      "colors_input": [<list of ARC color ints seen in input>],
      "colors_output": [<list of ARC color ints seen in output>],
      "background_color": <ARC int>,
      "mechanism": "...",
      "candidate_rule": "...",
      "preservation": "<what is unchanged from input to output>"
    }}
  ],
  "cross_pair_synthesis": {{
    "consistent_operation": "...",
    "consistent_parameters": "...",
    "likely_transform_family": "<pick the BEST match: recolor | reflect | rotate | translate | scale_up | scale_down | crop | extract | tile | fill | symmetry_complete | object_move | object_count | gravity | overlay | count_to_shape | grid_partition | other>",
    "unified_rule": "<one concise sentence stating the single rule that maps every demo input to its output>"
  }},
  "test_input_observations": "<briefly note what objects/colors/shape the test input has, and what the rule predicts should happen to it>"
}}"""


class VisionModel:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        quant_bits: int = 4,
        offload_dir: str = "offload_dir",
        gpu_headroom_gib: int = 2,
        max_new_tokens: int = 1500,
        temperature: float = 0.2,
    ) -> None:
        self.model_id = model_id
        self.quant_bits = quant_bits
        self.offload_dir = offload_dir
        self.gpu_headroom_gib = gpu_headroom_gib
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.model = None
        self.processor = None
        self.loaded = False

    def _bnb(self) -> BitsAndBytesConfig:
        if self.quant_bits == 4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        if self.quant_bits == 8:
            return BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
        raise ValueError("quant_bits must be 4 or 8")

    def _max_memory(self) -> dict[int | str, str]:
        mm: dict[int | str, str] = {}
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            allow = max(2, math.floor(total - self.gpu_headroom_gib))
            mm[i] = f"{allow}GiB"
        mm["cpu"] = "128GiB"
        return mm

    def load(self) -> None:
        if self.loaded:
            return
        assert torch.cuda.is_available(), "CUDA is required for the local VLM"
        os.makedirs(self.offload_dir, exist_ok=True)
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            quantization_config=self._bnb(),
            device_map="auto",
            max_memory=self._max_memory(),
            torch_dtype=torch.bfloat16,
            offload_folder=self.offload_dir,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        torch.set_grad_enabled(False)
        self.loaded = True

    def _generate(self, image, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user},
                ],
            },
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-5),
            )
        in_len = inputs["input_ids"].shape[1]
        gen = out[0][in_len:]
        return self.processor.decode(gen, skip_special_tokens=True)


def _parse_json_object(text: str) -> dict[str, Any]:
    """Extract the first top-level JSON object from the VLM response."""
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        candidate = fence.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        candidate = text[start : end + 1] if start != -1 and end != -1 else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return {"raw": text, "parse_error": True}


def get_clues(vlm: VisionModel, task: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
    """Run Stage A on one task. Returns (clues_dict, task_sheet_png_bytes)."""
    from .renderer import image_to_bytes

    sheet = render_task_sheet(task)
    sheet_png = image_to_bytes(sheet)

    user = VLM_USER_TEMPLATE.format(shapes=shape_header(task))
    raw = vlm._generate(sheet, VLM_SYSTEM, user)
    clues = _parse_json_object(raw)
    clues.setdefault("_raw", raw[:2000])
    return clues, sheet_png
