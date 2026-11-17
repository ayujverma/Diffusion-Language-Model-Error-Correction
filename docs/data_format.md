# Tasks, response format, checker and splits

The milestone-2 data contract. The code is in `src/diffusion_correction/data/`.
Decisions D-018 to D-021 explain the choices. Pilot settings are **provisional**
(D-019) until the milestone-3 validation calibration.

## 1. Task families

### Arithmetic chains (`data/arithmetic.py`)

A chain of n operations defines n values named `a, b, c, …`:

    a = 17 + 28        first value: constant op constant
    b = a * 3          each later value: previous value op constant
    c = b - 19

- `+` and `−` take a constant with exactly `operand_digits` digits (1..9 when d = 1;
  10..99 when d = 2; 100..999 when d = 3). The first constant has the same digit count.
- `×` takes a small integer k ∈ [`mul_min`, `mul_max`] (default 2..9).
- Bounds: every value, intermediates included, satisfies |v| ≤ `value_bound`
  (default 9999). If `allow_negative` is false, every value is also ≥ 0. Arithmetic
  uses exact Python integers, with no division and no rounding.
- Sampling: operator and operand are drawn uniformly at each step; a step that
  violates the bounds is redrawn, and the chain restarts after 50 failures. This is
  not uniform over valid chains (D-019).
- The composition signature is the tuple of operators, for example `('+', '*', '-')`.

### Short programs (`data/programs.py`)

Grammar (one statement per line):

    program   := statement ("\n" statement)*
    statement := NAME "=" expr
    expr      := atom | atom OP atom
    atom      := NAME | INT
    OP        := "+" | "-" | "*"
    NAME      := one lowercase ASCII letter
    INT       := "0" | [1-9][0-9]*

Spaces between tokens are optional. Nothing else is accepted: no parentheses, unary
minus, division, comments or blank lines.

Semantics:
- Inputs are named integers given with the program (the generator uses `x`, `y`).
  They are read-only and lie in [0, literal_max].
- A name must be defined, as an input or an earlier target, before use.
- Targets may be reassigned; the new value overwrites the old.
- `×` needs at least one INT operand in [mul_min, mul_max]. `NAME * NAME` is rejected.
- INT literals lie in [0, literal_max].
- After every statement the value must satisfy the same bounds as arithmetic.
- The trace is the value assigned by each statement, in order. The final answer is
  the last statement's value.
- Violations raise `ProgramError` with `kind` ∈ {syntax, undefined, multiply,
  literal, input, bound, length}.
- The interpreter is a hand-written lexer, parser and evaluator. Python `eval`/`exec`
  is never used, and model output is never executed.
- Bounded conditions are a planned later extension and are not in this language.

Generator: 2 inputs; statement 1 reads an input, and statement i > 1 reads statement
i − 1's target. Statement shapes are `add_const`, `sub_const`, `mul_const`,
`add_var`, `sub_var`. For the `_var` shapes the other operand is another defined
variable, and the operand order is random. The composition signature is the tuple
of shapes.

Arithmetic chains are valid programs with no inputs. Every stored record therefore
carries its problem as `source` + `inputs`, and the interpreter re-derives all labels
(`verify_dataset`).

## 2. Prompt

The prompt is built only from value-free public fields (`tasks.render_prompt`): the
statements, the inputs, the ordered names and the format description. It ends with
`"# Answer:"`.

    # Task: arithmetic chain. Compute each variable in order.
    a = 17 + 28
    b = a * 3
    c = b - 19
    # Output the values of a, b, c in order, a sign and 4 digits each.
    # Answer:

    # Task: run the program. Inputs: x = 17, y = 28.
    a = x + y
    b = a * 3
    c = b - x
    # Output the value assigned by each statement in order (a, b, c), a sign and 4 digits each.
    # Answer:

## 3. Response representation (`data/format.py`, D-018)

- One slot per value, with no names or separators: `" +0045 +0135 +0116"`.
- A slot is a sign token followed by `width` digit tokens:

  | Text | Token ID |
  | --- | --- |
  | `" +"` | 488 |
  | `" -"` | 481 |
  | `"0"`..`"9"` | 15..24 |

  These IDs are derived from the pinned tokenizer at run time, and each is asserted
  to be a single non-special token.
- `width = len(str(value_bound))`, which is 4 for the default 9999. A slot is
  therefore 5 tokens, and the response is 5 × (number of statements) tokens. That
  length is public.
- Zero is `" +0000"`. `" -0000"` is non-canonical and malformed.
- There is no EOS. The sequence ends after the last slot, and any padding is masked
  by length.
- Token-to-value spans: slot k occupies response tokens `[5k, 5k+5)`, stored as
  `slot_spans`. Token role: `sign`, then `digit0` (most significant) to `digit3`.
- Per record, `encode_and_verify` checks three things. The tokenizer's encoding of
  the response text must equal the slot template. Encoding prompt and response
  separately must equal encoding the joint text. The length must be the public length.

## 4. Record schema (`data/tasks.py::Example`; one JSON object per line)

| Field | Meaning |
| --- | --- |
| `problem_id` | first 16 hex digits of sha256(`normalized_key`); drafts and augmentations must carry it |
| `family`, `split` | `arithmetic` or `programs`; one of the five splits |
| `setting` | `{length, operand_digits[, n_inputs]}`, the public difficulty setting |
| `signature`, `heldout_composition` | composition signature; whether it contains a held-out pair |
| `normalized_key` | canonical program up to consistent variable renaming, plus inputs |
| `source`, `inputs`, `value_names` | the problem in the restricted language |
| `values`, `final_answer` | exact labels (**training/evaluation-only**) |
| `value_bound`, `width` | declared format parameters |
| `prompt_text`, `prompt_ids` | prompt and its token IDs |
| `response_text`, `response_ids` | canonical target response (**training/evaluation-only**) |
| `slot_spans` | `[start, end)` of each value within the response |

## 5. Checker (`data/checker.py`, D-020)

Predictions are scored by token ID at fixed positions; text is never decoded and
re-parsed. Four metrics:

| Level | Correct when | Denominator |
| --- | --- | --- |
| token | predicted ID = target ID at that response position | public response length, summed over examples |
| value | slot parses (status `ok`) and equals the target value | number of slots, summed |
| final answer | last slot is value-correct | examples |
| whole trace | every slot is value-correct | examples |

Malformed output:
- A slot with a bad sign token, a bad digit token, a special token (`<M>`, EOS/pad
  and the other added special tokens), an ID ≥ the tokenizer size, or `-0000` gets
  that reason as its status. It counts as an incorrect value; its tokens are still
  scored at token level.
- A prediction whose length differs from the public length is **not** trimmed,
  padded or zipped. All target tokens and slots count as incorrect, and the status is
  `length_mismatch`.
- `aggregate` reports counts and rates with explicit denominators, plus malformed
  example and slot rates with reasons. No example is dropped.

With the canonical format, a value is correct exactly when all of its slot's tokens
are correct (tested on random corruptions). The two levels are still reported
separately, because token accuracy credits partially correct values.

## 6. Splits and manifests (`data/splits.py`)

- Splits: `train`, `val`, `test_id` (same settings as train), `test_long` (longer
  lengths only), `test_unseen` (held-out compositions only).
- Held-out compositions are declared adjacent pairs in the signature: `-` then `*`
  for arithmetic, and `mul_const` then `sub_var` for programs. They never appear in
  train/val/test_id/test_long, and every test_unseen problem contains at least one.
  Seeds are not used to separate them.
- De-duplication is global across splits by `normalized_key`. Generation order is
  test_unseen, test_long, test_id, val, train, so a problem used for evaluation can
  never reappear in training. A setting whose problem space is too small stops with
  an explicit error.
- Each (split, family, length, digits) stream has its own seed derived from
  sha256(config seed, split, family, length, digits).
- Files written to `runs/data/<name>-seed<seed>/`:

  | File | Contents |
  | --- | --- |
  | `<split>.jsonl` | records; sorted keys, compact JSON |
  | `manifest.json` | deterministic: config, tokenizer revision and `tokenizer.json` sha256, format IDs, per-split file sha256 / counts / sha256 of sorted problem IDs, held-out definitions |
  | `provenance.json` | time, git state, config-file hash, manifest sha256, verification report |

- `dlmc verify-data` re-checks everything from the files alone: hashes, counts,
  interpreter labels, lengths, prompt rebuild from value-free fields,
  re-tokenization and spans. Structure and identity are **rebuilt from source**
  (D-024; `data/structure.py::derive_problem`), never taken from stored metadata:
  - The source must be an instance of the generated structure for its family and
    setting. Arithmetic: targets `a, b, …`; `K0 op K1`, then `previous op K`.
    Programs: inputs exactly `x, y[, z]`; targets in order, no reassignment; each
    statement reads its predecessor (statement 1: an input), as `P ± K`, `P * K`, or
    `P ± V` / `V ± P` with V another defined variable. Canonical spacing is required.
  - The held-out predicate is applied to the derived signature and must agree with
    the split and with `heldout_composition`. The stored signature must equal the
    derived one exactly.
  - `normalized_key` and `problem_id` must equal the ones rebuilt from source plus
    inputs, and cross-split uniqueness is checked on those rebuilt identities.
- `assert_grouped_by_problem(records, data_dir)` is the check that later draft banks
  must pass: every derived record sits in its problem's split.
- Difficulty settings are chosen on train/validation only (milestone 3). Test-set
  performance is never used to select them.

Current pilot dataset: `configs/data/pilot.yaml` → `runs/data/pilot-seed20261116`,
manifest sha256 `990ecd1dd2341b81bc742fa91189b129877cd9b48b196fbdbc208e8586c81536`.
It contains 18,000 / 3,600 / 3,600 / 900 / 1,800 problems. It is provisional and
will be regenerated, under a new hash, if calibration changes the settings.
