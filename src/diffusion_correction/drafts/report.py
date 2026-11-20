"""Examples report from a finalized bank: a deterministic diagnostic sample, verified against the records.

Selection (fixed, documented in the report): every draft gets the key
``sha256("report-v1|<split>|<problem_id>|<budget>")``; the smallest key wins within a
category. Categories:

- coverage: for each train (setting, budget), an incorrect draft (any wrong token) if
  one exists, otherwise a correct one;
- per split and family: one fully correct draft, one with an early error (a wrong
  token committed while >= 60% of the response was masked), one with a late error
  (wrong token committed at < 40% masked), and up to three malformed drafts.

Each shown example stores its shard file, line number and the line's sha256; the
verification pass re-reads those lines and recomputes every displayed value.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
from pathlib import Path
from typing import Callable

from .bank import read_complete_bank, write_bytes_atomic, write_json_atomic

REPORT_SCHEMA = "dlmc.bank_report.v1"
MAX_TOKEN_ROWS = 8
EARLY_FRAC, LATE_FRAC = 0.6, 0.4


def selection_key(split: str, problem_id: str, budget: str) -> str:
    return hashlib.sha256(f"report-v1|{split}|{problem_id}|{budget}".encode()).hexdigest()


def _setting(r: dict) -> str:
    return f"{r['family']}:{r['length']}:{r['operand_digits']}"


def select_examples(shards: list[tuple[dict, list[dict]]]) -> list[dict]:
    best: dict[tuple, tuple] = {}
    malformed: dict[tuple, list] = {}

    def offer(cat: tuple, key: str, loc: tuple):
        if cat not in best or key < best[cat][0]:
            best[cat] = (key, loc)

    for s, recs in shards:
        for i, r in enumerate(recs):
            a, g = r["analysis"], r["generation"]
            key = selection_key(r["split"], r["problem_id"], r["budget"])
            loc = (s["file"], i)
            wrong = [p for p, ok in enumerate(a["token_correct"]) if not ok]
            if r["split"] == "train":
                offer(("coverage", "train", _setting(r), r["budget"], "incorrect" if wrong else "correct"), key, loc)
            fam = (r["split"], r["family"])
            if a["trace_correct"]:
                offer(("category", *fam, "correct"), key, loc)
            if any(g["mask_fraction_before"][p] >= EARLY_FRAC for p in wrong):
                offer(("category", *fam, "early_error"), key, loc)
            if any(g["mask_fraction_before"][p] < LATE_FRAC for p in wrong):
                offer(("category", *fam, "late_error"), key, loc)
            if not a["well_formed"]:
                malformed.setdefault(fam, []).append((key, loc))
    chosen = []
    cov = {}
    for cat, (key, loc) in best.items():
        if cat[0] == "coverage":
            cell = cat[1:4]
            if cat[4] == "incorrect" or cell not in cov:
                cov[cell] = (key, loc, cat[4])
    for cell in sorted(cov):
        key, loc, kind = cov[cell]
        chosen.append({"category": f"coverage ({kind})", "key": key, "file": loc[0], "line": loc[1]})
    for cat in sorted(c for c in best if c[0] == "category"):
        key, loc = best[cat]
        chosen.append({"category": f"{cat[1]} {cat[2]} {cat[3]}", "key": key, "file": loc[0], "line": loc[1]})
    for fam in sorted(malformed):
        for key, loc in sorted(malformed[fam])[:3]:
            chosen.append({"category": f"{fam[0]} {fam[1]} malformed", "key": key, "file": loc[0], "line": loc[1]})
    return chosen


def token_rows(r: dict) -> list[int]:
    g, a = r["generation"], r["analysis"]
    wrong = [p for p, ok in enumerate(a["token_correct"]) if not ok]
    order = sorted(range(r["response_length"]), key=lambda p: g["commit_order"][p])
    keep = list(dict.fromkeys([order[0]] + wrong))[:MAX_TOKEN_ROWS - 1]
    if order[-1] not in keep:
        keep.append(order[-1])
    return sorted(keep)


def display(r: dict, decode: Callable[[int], str], texts: dict) -> dict:
    """Everything the report shows for one record (also what the verification pass recomputes)."""
    g, a = r["generation"], r["analysis"]
    L = r["response_length"]
    slot_len = L // len(a["values"])
    slots = []
    for k in range(len(a["values"])):
        sl = slice(k * slot_len, (k + 1) * slot_len)
        slots.append({"slot": k, "reference": "".join(decode(t) for t in a["reference_ids"][sl]),
                      "draft": "".join(decode(t) for t in g["draft_ids"][sl]),
                      "status": a["slot_status"][k], "value_correct": a["value_correct"][k],
                      "parsed": a["parsed_values"][k], "true_value": a["values"][k]})
    rows = []
    for p in token_rows(r):
        rows.append({"position": p, "token_id": g["draft_ids"][p], "token": decode(g["draft_ids"][p]),
                     "reference_id": a["reference_ids"][p], "reference": decode(a["reference_ids"][p]),
                     "correct": a["token_correct"][p], "commit_step": g["commit_call"][p],
                     "commit_order": g["commit_order"][p], "mask_fraction_before": g["mask_fraction_before"][p],
                     "log_c_gen": g["log_c_gen"][p], "log_q_gen": a["log_q_gen"][p],
                     "rank_score": g["rank_score"][p], "policy_log_prob": g["policy_log_prob"][p]})
    return {"problem_id": r["problem_id"], "split": r["split"], "setting": _setting(r), "budget": r["budget"],
            "calls": r["calls"], "generator_model_sha256": r["generator_model_sha256"],
            "input": texts.get("prompt_text"), "reference_text": texts.get("response_text"),
            "draft_text": "".join(decode(t) for t in g["draft_ids"]),
            "labels": {k: a[k] for k in ("final_correct", "trace_correct", "well_formed", "n_token_errors",
                                         "n_value_errors")},
            "slots": slots, "tokens": rows, "n_wrong_tokens_total": a["n_token_errors"]}


def bank_counts(manifest: dict, shards: list[tuple[dict, list[dict]]]) -> list[dict]:
    cells: dict[tuple, dict] = {}
    for _, recs in shards:
        for r in recs:
            c = cells.setdefault((r["split"], r["family"], r["budget"]),
                                 {"drafts": 0, "incorrect": 0, "final_wrong": 0, "malformed": 0,
                                  "tokens": 0, "wrong_tokens": 0})
            a = r["analysis"]
            c["drafts"] += 1
            c["incorrect"] += a["n_token_errors"] > 0
            c["final_wrong"] += not a["final_correct"]
            c["malformed"] += not a["well_formed"]
            c["tokens"] += r["response_length"]
            c["wrong_tokens"] += a["n_token_errors"]
    order = {b: i for i, b in enumerate(manifest["identity"]["config"]["sampler"]["budgets"])}
    return [{"split": k[0], "family": k[1], "budget": k[2], **v}
            for k, v in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1], order.get(kv[0][2], 99)))]


def _f(x) -> str:
    return f"{x:.6g}" if isinstance(x, float) else str(x)


def render_markdown(rep: dict) -> str:
    out = [f"# Draft bank examples: {rep['bank']}", "",
           f"**Diagnostic sample, not a summary statistic.** {len(rep['examples'])} examples chosen by the fixed rule "
           "below from the saved bank. Bank manifest sha256 "
           f"`{rep['manifest_sha256']}`; generator model sha256 `{rep['generator_model_sha256']}` "
           f"(update {rep['generator_update']}).", "", "Selection rule:", "", rep["selection_rule"], "",
           "Probabilities: " + rep["probability_definition"], "", "## Full-bank counts", "",
           "| split | family | budget | drafts | incorrect | final wrong | malformed | wrong / tokens |",
           "|---|---|---|---|---|---|---|---|"]
    for c in rep["bank_counts"]:
        out.append(f"| {c['split']} | {c['family']} | {c['budget']} | {c['drafts']} | {c['incorrect']} | "
                   f"{c['final_wrong']} | {c['malformed']} | {c['wrong_tokens']} / {c['tokens']} |")
    out += ["", f"Malformed drafts in the bank: {rep['n_malformed']}"
            + (" (none exist, so no malformed examples are shown)." if rep["n_malformed"] == 0 else "."), ""]
    for i, ex in enumerate(rep["examples"], 1):
        d = ex["display"]
        out += [f"## {i}. {ex['category']}: `{d['problem_id']}`", "",
                f"split {d['split']}, setting {d['setting']}, budget {d['budget']} ({d['calls']} calls); "
                f"source `{ex['file']}` line {ex['line'] + 1}", "",
                "```text", f"{d['input']}", "```", "",
                f"- reference: `{d['reference_text']}`", f"- draft: `{d['draft_text']}`",
                f"- labels: {json.dumps(d['labels'], sort_keys=True)}", "",
                "| slot | reference | draft | status | value ok |", "|---|---|---|---|---|"]
        for s in d["slots"]:
            mark = "" if s["value_correct"] else " **≠**"
            out.append(f"| {s['slot']} | `{s['reference']}` | `{s['draft']}`{mark} | {s['status']} | {s['value_correct']} |")
        out += ["", "| pos | token (id) | reference (id) | ok | step | order | mask frac | log c | log q | rank score "
                "| policy log p |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for t in d["tokens"]:
            out.append(f"| {t['position']} | `{t['token']}` ({t['token_id']}) | `{t['reference']}` ({t['reference_id']}) "
                       f"| {t['correct']} | {t['commit_step']} | {t['commit_order']} | {_f(t['mask_fraction_before'])} "
                       f"| {_f(t['log_c_gen'])} | {_f(t['log_q_gen'])} | {_f(t['rank_score'])} | {_f(t['policy_log_prob'])} |")
        if d["n_wrong_tokens_total"] > sum(not t["correct"] for t in d["tokens"]):
            out.append(f"\n(table truncated to {MAX_TOKEN_ROWS} rows; {d['n_wrong_tokens_total']} wrong tokens in total)")
        out.append("")
    return "\n".join(out) + "\n"


def render_html(rep: dict) -> str:
    e = html.escape
    parts = ["<!doctype html><meta charset='utf-8'><title>Draft bank examples</title>",
             "<style>body{font-family:sans-serif;max-width:1100px;margin:auto}table{border-collapse:collapse;margin:6px 0}"
             "td,th{border:1px solid #ccc;padding:2px 6px;font-size:13px}.bad{background:#fdd}code{background:#f4f4f4}</style>",
             f"<h1>Draft bank examples: {e(rep['bank'])}</h1>",
             f"<p><b>Diagnostic sample, not a summary statistic.</b> Manifest <code>{e(rep['manifest_sha256'])}</code>; "
             f"generator <code>{e(rep['generator_model_sha256'])}</code> (update {rep['generator_update']}).</p>",
             f"<pre>{e(rep['selection_rule'])}</pre><p>{e(rep['probability_definition'])}</p><h2>Full-bank counts</h2><table>",
             "<tr><th>split</th><th>family</th><th>budget</th><th>drafts</th><th>incorrect</th><th>final wrong</th>"
             "<th>malformed</th><th>wrong / tokens</th></tr>"]
    for c in rep["bank_counts"]:
        parts.append("<tr>" + "".join(f"<td>{e(str(c[k]))}</td>" for k in
                                      ("split", "family", "budget", "drafts", "incorrect", "final_wrong", "malformed"))
                     + f"<td>{c['wrong_tokens']} / {c['tokens']}</td></tr>")
    parts.append(f"</table><p>Malformed drafts in the bank: {rep['n_malformed']}</p>")
    for i, ex in enumerate(rep["examples"], 1):
        d = ex["display"]
        parts += [f"<h2>{i}. {e(ex['category'])}: <code>{e(d['problem_id'])}</code></h2>",
                  f"<p>split {e(d['split'])}, setting {e(d['setting'])}, budget {e(d['budget'])} ({d['calls']} calls); "
                  f"source <code>{e(ex['file'])}</code> line {ex['line'] + 1}</p>",
                  f"<pre>{e(str(d['input']))}</pre><p>reference <code>{e(str(d['reference_text']))}</code>"
                  f"<br>draft <code>{e(d['draft_text'])}</code><br>labels {e(json.dumps(d['labels']))}</p><table>",
                  "<tr><th>slot</th><th>reference</th><th>draft</th><th>status</th></tr>"]
        for s in d["slots"]:
            cls = "" if s["value_correct"] else " class='bad'"
            parts.append(f"<tr{cls}><td>{s['slot']}</td><td><code>{e(s['reference'])}</code></td>"
                         f"<td><code>{e(s['draft'])}</code></td><td>{e(s['status'])}</td></tr>")
        parts.append("</table><table><tr><th>pos</th><th>token (id)</th><th>reference (id)</th><th>step</th><th>order</th>"
                     "<th>mask frac</th><th>log c</th><th>log q</th><th>rank score</th><th>policy log p</th></tr>")
        for t in d["tokens"]:
            cls = "" if t["correct"] else " class='bad'"
            parts.append(f"<tr{cls}><td>{t['position']}</td><td><code>{e(t['token'])}</code> ({t['token_id']})</td>"
                         f"<td><code>{e(t['reference'])}</code> ({t['reference_id']})</td><td>{t['commit_step']}</td>"
                         f"<td>{t['commit_order']}</td><td>{_f(t['mask_fraction_before'])}</td><td>{_f(t['log_c_gen'])}</td>"
                         f"<td>{_f(t['log_q_gen'])}</td><td>{_f(t['rank_score'])}</td><td>{_f(t['policy_log_prob'])}</td></tr>")
        parts.append("</table>")
    return "\n".join(parts) + "\n"


def build_report(bank_root: Path, decode: Callable[[int], str], texts: dict[tuple, dict],
                 out_dir: Path | None = None) -> dict:
    """``texts[(split, problem_id)]`` holds the dataset's prompt_text and response_text."""
    bank_root = Path(bank_root)
    out_dir = Path(out_dir) if out_dir else bank_root / "derived" / "report"
    manifest, shards = read_complete_bank(bank_root)
    by_file = {s["file"]: recs for s, recs in shards}
    chosen = select_examples(shards)
    for ex in chosen:
        r = by_file[ex["file"]][ex["line"]]
        ex["display"] = display(r, decode, texts.get((r["split"], r["problem_id"]), {}))
        raw = (bank_root / ex["file"]).read_bytes().split(b"\n")[ex["line"]]
        ex["line_sha256"] = hashlib.sha256(raw).hexdigest()
    gen = manifest["identity"]["generator"]
    rep = {"schema": REPORT_SCHEMA, "bank": manifest["bank"], "manifest_sha256": manifest["manifest_sha256"],
           "generator_model_sha256": gen["model_sha256"], "generator_update": gen["update"],
           "selection_rule": __doc__.split("\n\n", 1)[1].split("\n\nEach shown")[0],
           "probability_definition": manifest["identity"]["probability_definition"],
           "bank_counts": bank_counts(manifest, shards),
           "n_malformed": sum(not r["analysis"]["well_formed"] for _, recs in shards for r in recs),
           "examples": chosen}
    write_json_atomic(out_dir / "examples.json", rep)
    rep = json.loads((out_dir / "examples.json").read_text())
    write_bytes_atomic(out_dir / "examples.md", render_markdown(rep).encode())
    write_bytes_atomic(out_dir / "examples.html", render_html(rep).encode())
    ver = verify_report(bank_root, out_dir, decode, texts)
    return {**rep, "verification": ver}


def verify_report(bank_root: Path, out_dir: Path, decode: Callable[[int], str], texts: dict[tuple, dict]) -> dict:
    """Re-read every shown record from its shard line and recompute what the report displays."""
    bank_root, out_dir = Path(bank_root), Path(out_dir)
    rep = json.loads((out_dir / "examples.json").read_text())
    manifest, shards = read_complete_bank(bank_root)
    problems = []
    if rep["manifest_sha256"] != manifest["manifest_sha256"]:
        problems.append("report built from a different bank manifest")
    for ex in rep["examples"]:
        raw = (bank_root / ex["file"]).read_bytes().split(b"\n")[ex["line"]]
        if hashlib.sha256(raw).hexdigest() != ex["line_sha256"]:
            problems.append(f"{ex['file']}:{ex['line']}: line sha256 differs")
            continue
        r = json.loads(raw)
        again = json.loads(json.dumps(display(r, decode, texts.get((r["split"], r["problem_id"]), {}))))
        if again != ex["display"]:
            problems.append(f"{ex['file']}:{ex['line']}: displayed values differ from the stored record")
        for t in ex["display"]["tokens"]:
            if not (math.isfinite(t["log_c_gen"]) and t["rank_score"] == t["log_c_gen"]):
                problems.append(f"{r['problem_id']}: token {t['position']} rank score differs from log c")
    md = (out_dir / "examples.md").read_text()
    if render_markdown(rep) != md:
        problems.append("examples.md differs from a re-render of examples.json")
    if render_html(rep) != (out_dir / "examples.html").read_text():
        problems.append("examples.html differs from a re-render of examples.json")
    ver = {"passed": not problems, "problems": problems, "n_examples": len(rep["examples"]),
           "manifest_sha256": manifest["manifest_sha256"]}
    write_json_atomic(out_dir / "report_verification.json", ver)
    return ver
