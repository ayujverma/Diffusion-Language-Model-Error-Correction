"""Inject known bugs into a temp copy of src/ and confirm the milestone-3 tests catch each one.

Usage: .venv/bin/python scripts/mutation_check_m3.py
The working tree is never modified; each mutant runs the tiny-backbone suites only.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ["tests/test_masking.py", "tests/test_sampler.py", "tests/test_trainer.py"]

MUTANTS = [
    ("resume does not restore the mask stream", "training/trainer.py",
     "            g.set_state(rng[\"streams\"][name])", "            pass"),
    ("sampler may revise committed tokens", "sampling/monotone.py",
     "            masked = layout.editable_mask & (ids == mask_id)", "            masked = layout.editable_mask.clone()"),
    ("candidate ties pick the largest token ID", "sampling/monotone.py",
     "torch.where(restricted == best, vocab_ids, tok.vocab_size).min(dim=-1).values",
     "torch.where(restricted == best, vocab_ids, -1).max(dim=-1).values"),
    ("positions ranked by restricted policy probability", "sampling/monotone.py",
     "            score = raw_logp.gather(-1, cand[:, None]).squeeze(-1)", "            score = pol_logp"),
    ("position ties pick the rightmost position", "sampling/monotone.py",
     "key=lambda r: (-score_c[r], where[r][1])", "key=lambda r: (-score_c[r], -where[r][1])"),
    ("reveal schedule uses floor", "sampling/monotone.py",
     "k = [0] + [-(-s * length // calls) for s in range(1, calls + 1)]",
     "k = [0] + [s * length // calls for s in range(1, calls + 1)]"),
    ("one mask rate per batch", "training/masking.py",
     "        u = float(torch.rand(1, generator=generator, dtype=torch.float64).item())",
     "        u = us[0] if us else float(torch.rand(1, generator=generator, dtype=torch.float64).item())"),
    ("prompt tokens can be masked", "training/masking.py",
     "        pos = editable[b].nonzero().squeeze(-1)",
     "        pos = torch.arange(int(layout.attention_mask[b].sum()))"),
    ("zero-mask batch still steps the scheduler", "training/trainer.py",
     "            self.skipped_zero_mask += 1", "            self.skipped_zero_mask += 1\n            self.scheduler.step()"),
    ("evaluation draws from the training mask stream", "training/trainer.py",
     "        g = make_generator(self.cfg[\"seed\"], \"eval\")", "        g = self.gens[\"train_mask\"]"),
    ("LoRA leaves the base trainable", "training/lora.py",
     "        p.requires_grad_(False)", "        p.requires_grad_(True)"),
    ("sampler sees reference length", "evaluation/generation.py",
     "    return len(ex.value_names) * slot_len", "    return len(ex.response_ids) - (ex.response_ids[0] % 2)"),
]


def main() -> int:
    results = []
    for desc, rel, old, new in MUTANTS:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            shutil.copytree(ROOT / "src", src)
            path = src / "diffusion_correction" / rel
            text = path.read_text()
            if text.count(old) != 1:
                print(f"SETUP ERROR ({desc}): pattern found {text.count(old)} times")
                return 2
            path.write_text(text.replace(old, new))
            env = {**os.environ, "PYTHONPATH": str(src)}
            check = subprocess.run([sys.executable, "-c", "import diffusion_correction as d; print(d.__file__)"],
                                   env=env, capture_output=True, text=True, cwd=ROOT)
            assert str(src) in check.stdout, f"mutant copy not imported: {check.stdout}"
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", *TESTS], env=env, cwd=ROOT,
                               capture_output=True, text=True)
            caught = r.returncode != 0
            failing = [ln for ln in r.stdout.splitlines() if ln.startswith("FAILED")][:1]
            results.append((desc, caught, failing))
            print(f"{'CAUGHT ' if caught else 'MISSED '} {desc} {failing}", flush=True)
    missed = [d for d, c, _ in results if not c]
    print(f"\n{len(results) - len(missed)}/{len(results)} mutants caught")
    return 1 if missed else 0


if __name__ == "__main__":
    raise SystemExit(main())
