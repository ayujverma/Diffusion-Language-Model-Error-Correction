"""Synthetic Example records in the tiny backbone's vocabulary (software tests only)."""

from __future__ import annotations

import random

from diffusion_correction.data.tasks import Example

SLOT_LEN = 2  # tiny tests use 2-token "slots"


def make_tiny_examples(n: int, seed: int = 0, n_slots_choices=(2, 3, 4)) -> list[Example]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        n_slots = rng.choice(n_slots_choices)
        prompt = [rng.randrange(10, 17) for _ in range(rng.randint(3, 6))]
        response = [rng.randrange(0, 10) for _ in range(n_slots * SLOT_LEN)]
        out.append(Example(
            problem_id=f"tiny{i:04d}", family="arithmetic" if i % 2 == 0 else "programs", split="train",
            setting={"length": n_slots, "operand_digits": 1}, signature=[], heldout_composition=False,
            normalized_key=f"tiny{i}", source="", inputs={}, value_names=[f"v{k}" for k in range(n_slots)],
            values=[0] * n_slots, final_answer=0, value_bound=9, width=1, prompt_text="", response_text="",
            prompt_ids=prompt, response_ids=response,
            slot_spans=[[k * SLOT_LEN, (k + 1) * SLOT_LEN] for k in range(n_slots)]))
    return out
