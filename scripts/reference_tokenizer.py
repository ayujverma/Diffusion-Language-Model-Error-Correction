"""Compare our tokenizer wrapper with the pinned reference tokenizers on a dataset.

Ours: ``tokenizers.Tokenizer.from_file(tokenizer.json)`` (main environment, D-008).
References, from the same pinned model directory via transformers 4.54.1:
- ``AutoTokenizer`` (fast; Rust, also reads tokenizer.json);
- ``Qwen2Tokenizer`` (slow; a separate pure-Python byte-level BPE built from
  vocab.json + merges.txt, so it does not share tokenizer.json at all).

For every record, the prompt, the response and the joint prompt+response text are
encoded by all three without special tokens and must give identical IDs; the stored
record IDs must match too. Edge strings (signs, zero, bounds, special tokens) are
checked as well.

    PYTHONPATH=src .venv-ref/bin/python scripts/reference_tokenizer.py --data-dir runs/data/pilot-seed20261116

Writes ``<data-dir>/tokenizer_reference_check.json``; exits non-zero on any mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import transformers
from transformers import AutoTokenizer, Qwen2Tokenizer

from diffusion_correction.data.splits import SPLITS, load_split
from diffusion_correction.models.open_dcoder import DEFAULT_LOCAL_DIR, load_tokenizer, verify_files

EDGE = [" +0000", " -0001", " +9999", " -9999", " +0045 +0135 +0116", " -0033 -0099 +0000", "# Answer:",
        "# Answer: +0045", "<M><M> +0045", "<|endoftext|>", "a=-0045", "a=+0045,b=-0135", "x = 17, y = 28.\n"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    args = p.parse_args()
    local = DEFAULT_LOCAL_DIR
    pinned = verify_files(local, check_weights=False)
    ours = load_tokenizer(local)
    fast = AutoTokenizer.from_pretrained(str(local))
    slow = Qwen2Tokenizer(vocab_file=str(local / "vocab.json"), merges_file=str(local / "merges.txt"))
    slow.add_special_tokens({"additional_special_tokens": ["<M>"]}) if "<M>" not in slow.get_vocab() else None

    def ids_ours(t):
        return ours.encode(t, add_special_tokens=False).ids

    def ids_ref(tok, t):
        return tok.encode(t, add_special_tokens=False)

    mismatches, n_strings, n_records = [], 0, 0
    for text in EDGE:
        a, b = ids_ours(text), ids_ref(fast, text)
        n_strings += 1
        if a != b:
            mismatches.append({"edge": text, "ours": a, "fast": b})
        if "<" not in text:  # the bare slow tokenizer is not given the model's added tokens
            c = ids_ref(slow, text)
            if a != c:
                mismatches.append({"edge": text, "ours": a, "slow": c})
    for split in SPLITS:
        for r in load_split(Path(args.data_dir), split):
            n_records += 1
            for name, text, stored in (("prompt", r.prompt_text, r.prompt_ids),
                                       ("response", r.response_text, r.response_ids),
                                       ("joint", r.prompt_text + r.response_text, r.prompt_ids + r.response_ids)):
                n_strings += 1
                a, b, c = ids_ours(text), ids_ref(fast, text), ids_ref(slow, text)
                if not (a == b == c == stored):
                    mismatches.append({"split": split, "problem_id": r.problem_id, "part": name})
    other = {f: hashlib.sha256((local / f).read_bytes()).hexdigest() for f in ("vocab.json", "merges.txt")}
    report = {"data_dir": args.data_dir, "transformers": transformers.__version__,
              "fast_class": type(fast).__name__, "slow_class": type(slow).__name__,
              "pinned_sha256": {k: pinned[k] for k in ("tokenizer.json", "tokenizer_config.json")},
              "unpinned_files_sha256": other, "n_records": n_records, "n_strings_compared": n_strings,
              "n_mismatches": len(mismatches), "first_mismatches": mismatches[:10], "passed": not mismatches}
    out = Path(args.data_dir) / "tokenizer_reference_check.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
