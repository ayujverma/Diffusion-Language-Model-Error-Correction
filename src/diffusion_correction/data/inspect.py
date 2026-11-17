"""One table exposing the tensors used for one example: tokens, layout masks, targets.

Rows come from the real :func:`build_layout` masks and the real
``align_logits`` convention (aligned row i = raw output row i-1). Generation
signals (c, q, -log q, stage bin) and correction-training quantities (current target
probability, weight, CE, weighted CE) do not exist until draft collection
(milestone 4) and correction training (milestone 5); their columns print ``n/a``.
An optional draft is HYPOTHETICAL (supplied on the command line) and is labelled as such.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models.alignment import FIRST_PREDICTABLE_POSITION, build_layout
from .checker import ExampleCheck, check_response
from .format import TokenizedFormat
from .tasks import Example

SIGNAL_COLUMNS = ("c", "q", "-log_q", "bin", "p_now", "w", "CE", "wCE")
UNAVAILABLE = "n/a"


@dataclass
class InspectTable:
    header: list[str]
    columns: list[str]
    rows: list[list[str]]
    footer: list[str]
    check: ExampleCheck | None

    def render(self) -> str:
        widths = [max(len(c), *(len(r[k]) for r in self.rows)) if self.rows else len(c)
                  for k, c in enumerate(self.columns)]
        line = "  ".join(c.ljust(w) for c, w in zip(self.columns, widths))
        out = list(self.header) + ["", line, "-" * len(line)]
        out += ["  ".join(v.ljust(w) for v, w in zip(r, widths)) for r in self.rows]
        return "\n".join(out + [""] + self.footer)

    def to_dict(self) -> dict:
        return {"header": self.header, "columns": self.columns,
                "rows": [dict(zip(self.columns, r)) for r in self.rows], "footer": self.footer,
                "check": self.check.to_dict() if self.check else None}


def token_text(tokenizer, tid: int) -> str:
    return repr(tokenizer.decode([tid], skip_special_tokens=False))


def inspect_example(ex: Example, tokenizer, tf: TokenizedFormat, pad_token_id: int, mask_token_id: int,
                    draft_ids: list[int] | None = None, n_pad: int = 2, prompt_rows: int = 4) -> InspectTable:
    n_p, n_r = len(ex.prompt_ids), len(ex.response_ids)
    lay = build_layout([ex.prompt_ids], [ex.response_ids], pad_token_id, total_length=n_p + n_r + n_pad)
    ids = lay.input_ids[0].tolist()
    attn, prm, rsp, edt = (m[0].tolist() for m in (lay.attention_mask, lay.prompt_mask, lay.response_mask,
                                                    lay.editable_mask))
    slot_of = {}
    for k, (s, e) in enumerate(ex.slot_spans):
        for j in range(s, e):
            slot_of[j] = k

    check = None
    draft_ok = draft_ids is not None and len(draft_ids) == n_r
    if draft_ids is not None:
        check = check_response(draft_ids, ex, tf)

    columns = ["pos", "seg", "attn", "prompt", "resp", "edit", "token_id", "token", "aligned_from_raw_row",
               "slot", "role", "slot_value"]
    if draft_ok:
        columns += ["draft_id", "draft_token", "label"]
    columns += list(SIGNAL_COLUMNS)

    rows = []
    first_shown = max(0, n_p - prompt_rows) if prompt_rows >= 0 else 0
    for i, tid in enumerate(ids):
        if i < first_shown:
            continue
        seg = "prompt" if prm[i] else "response" if rsp[i] else "pad"
        k = i - n_p
        in_resp = 0 <= k < n_r
        row = [str(i), seg, *("1" if m[i] else "0" for m in (attn, prm, rsp, edt)), str(tid),
               token_text(tokenizer, tid),
               str(i - 1) if in_resp and i >= FIRST_PREDICTABLE_POSITION else "-"]
        if in_resp:
            slot = slot_of[k]
            row += [f"{ex.value_names[slot]}[{slot}]", tf.token_role(k), str(ex.values[slot])]
        else:
            row += ["-", "-", "-"]
        if draft_ok:
            if in_resp:
                d = draft_ids[k]
                label = "E (wrong)" if d != tid else "K (keep)"
                row += [str(d), token_text(tokenizer, d), label]
            else:
                row += ["-", "-", "-"]
        row += [UNAVAILABLE if in_resp else "-"] * len(SIGNAL_COLUMNS)
        rows.append(row)

    header = [
        f"problem_id {ex.problem_id}   family {ex.family}   split {ex.split}   setting {ex.setting}",
        f"composition signature {ex.signature}   held-out composition: {ex.heldout_composition}",
        "prompt text:",
        *("    " + line for line in ex.prompt_text.split("\n")),
        f"target response text: {ex.response_text!r}",
        f"values ({', '.join(ex.value_names)}) = {ex.values}   final answer = {ex.final_answer}",
        f"format: bound |v| <= {ex.value_bound}, slot = sign + {ex.width} digits = {tf.fmt.slot_len} tokens, "
        f"response length {n_r} tokens (public), no EOS; sign IDs {tf.sign_ids}, digit IDs 0-9 = "
        f"{list(tf.digit_ids)}",
        f"slot spans (token offsets within the response, [start, end)): "
        + ", ".join(f"{n}={s}" for n, s in zip(ex.value_names, ex.slot_spans)),
        f"sequence: prompt positions 0..{n_p - 1}, response {n_p}..{n_p + n_r - 1}, "
        f"{n_pad} pad position(s) appended to show the pad mask (pad = EOS = {pad_token_id}; mask <M> = "
        f"{mask_token_id})",
    ]
    if first_shown:
        header.append(f"(prompt positions 0..{first_shown - 1} elided; use --prompt-rows -1 to show all)")

    footer = [
        "aligned_from_raw_row: the model output row that predicts this position (aligned[i] = raw[i-1]); "
        "position 0 is never a target.",
        "attn/prompt/resp/edit: the boolean masks built by models.alignment.build_layout from lengths, "
        "not from token IDs.",
        f"{', '.join(SIGNAL_COLUMNS)}: {UNAVAILABLE}. c, q, -log q and the stage bin are produced by draft "
        "collection (milestone 4); p_now, w, CE and wCE by correction training (milestone 5). Nothing is "
        "computed or estimated here.",
    ]
    if draft_ids is not None:
        footer.insert(0, "DRAFT: HYPOTHETICAL, supplied on the command line for inspection; not model output.")
        if not draft_ok:
            footer.insert(1, f"draft has {len(draft_ids)} tokens but the public response length is {n_r}: "
                             "rows are not aligned (no trimming or zipping); every target token and slot "
                             "counts as incorrect.")
        c = check
        footer.insert(1 if draft_ok else 2, (
            f"checker: tokens correct {sum(c.token_correct)}/{c.n_tokens}; values correct "
            f"{sum(c.value_correct)}/{len(c.value_correct)}; slot status {c.slot_status}; parsed "
            f"{c.parsed_values}; final answer correct {c.final_correct}; whole trace correct "
            f"{c.trace_correct}; well-formed {c.well_formed}; special-token positions "
            f"{c.special_token_positions}"))
    return InspectTable(header=header, columns=columns, rows=rows, footer=footer, check=check)
