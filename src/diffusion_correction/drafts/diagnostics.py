"""Signal diagnostics on a finalized train/val bank (numpy only; no model).

Every number comes with its denominator. Correlations are descriptive; nothing here
supports a claim that a weighting helps.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .bank import read_complete_bank, write_bytes_atomic, write_json_atomic
from .step0 import read_step0

STAGE_EDGES = (0.2, 0.4, 0.6, 0.8)          # D-011 mask-fraction bins
QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
DIAG_SCHEMA = "dlmc.bank_diagnostics.v1"


def rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based) with ties sharing their mean rank."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    bounds = np.flatnonzero(np.r_[True, sx[1:] != sx[:-1], True])
    counts = np.diff(bounds)
    avg = bounds[:-1] + (counts - 1) / 2.0 + 1.0
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.repeat(avg, counts)
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 3:
        return None
    ra, rb = rankdata(a), rankdata(b)
    sa, sb = ra.std(), rb.std()
    if sa == 0 or sb == 0:
        return None
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


def eta_squared(y: np.ndarray, groups: np.ndarray) -> float | None:
    """Share of the variance of ``y`` explained by group means."""
    if y.size < 2 or y.var() == 0:
        return None
    _, inv = np.unique(groups, return_inverse=True)
    means = np.bincount(inv, weights=y) / np.bincount(inv)
    return float(((means[inv] - y.mean()) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def stage_bin(frac: np.ndarray) -> np.ndarray:
    return np.digitize(frac, STAGE_EDGES, right=False)


def _q(x: np.ndarray) -> dict:
    if x.size == 0:
        return {"n": 0}
    return {"n": int(x.size), "mean": float(x.mean()),
            **{f"q{int(round(q * 100)):02d}": float(v) for q, v in zip(QUANTILES, np.quantile(x, QUANTILES))}}


def token_table(bank_root: Path) -> tuple[dict, dict[str, np.ndarray]]:
    manifest, shards = read_complete_bank(bank_root)
    step0 = read_step0(bank_root, manifest["manifest_sha256"])
    cols = defaultdict(list)
    drafts = defaultdict(list)
    for _, recs in shards:
        for r in recs:
            g, a = r["generation"], r["analysis"]
            L = r["response_length"]
            setting = f"{r['family']}:{r['length']}:{r['operand_digits']}"
            drafts["split"].append(r["split"])
            drafts["setting"].append(setting)
            drafts["budget"].append(r["budget"])
            drafts["final_correct"].append(a["final_correct"])
            drafts["trace_correct"].append(a["trace_correct"])
            drafts["well_formed"].append(a["well_formed"])
            drafts["n_token_errors"].append(a["n_token_errors"])
            s0 = step0.get((r["split"], r["problem_id"], r["budget"])) if step0 else None
            slot_len = L // len(a["values"])
            for p in range(L):
                cols["split"].append(r["split"])
                cols["family"].append(r["family"])
                cols["setting"].append(setting)
                cols["budget"].append(r["budget"])
                cols["position"].append(p)
                cols["slot"].append(p // slot_len)
                cols["L"].append(L)
                cols["frac"].append(g["mask_fraction_before"][p])
                cols["order_frac"].append(g["commit_order"][p] / L)
                cols["wrong"].append(not a["token_correct"][p])
                cols["log_c"].append(g["log_c_gen"][p])
                cols["log_q"].append(a["log_q_gen"][p])
                cols["policy"].append(g["policy_log_prob"][p])
                cols["raw_max"].append(g["raw_max_log_prob"][p])
                cols["s0_ref"].append(s0["step0_log_p_ref"][p] if s0 else np.nan)
                cols["s0_draft"].append(s0["step0_log_p_draft"][p] if s0 else np.nan)
    t = {k: np.asarray(v) for k, v in cols.items()}
    t["stage"] = stage_bin(t["frac"])
    d = {k: np.asarray(v) for k, v in drafts.items()}
    meta = {"manifest_sha256": manifest["manifest_sha256"], "bank": manifest["bank"], "has_step0": step0 is not None,
            "n_tokens": int(t["wrong"].size), "n_drafts": int(d["split"].size)}
    return meta, {**t, **{f"draft_{k}": v for k, v in d.items()}}


def _rate_rows(keys: list[np.ndarray], names: list[str], wrong: np.ndarray) -> list[dict]:
    combo = keys[0].astype(str)
    for k in keys[1:]:
        combo = np.char.add(np.char.add(combo, "|"), k.astype(str))
    uniq, inv = np.unique(combo, return_inverse=True)
    n = np.bincount(inv, minlength=uniq.size)
    w = np.bincount(inv, weights=wrong.astype(np.float64), minlength=uniq.size)
    out = [{**dict(zip(names, c.split("|"))), "tokens": int(n[i]), "wrong": int(w[i]), "rate": float(w[i] / n[i])}
           for i, c in enumerate(uniq.tolist())]
    return sorted(out, key=lambda r: tuple(_sortable(r[k]) for k in names))


def _sortable(v: str) -> tuple:
    if v.lstrip("-").isdigit():
        return (0, int(v), "")
    return (1, 0, v) if v != "L" else (0, 10**9, "")


def compute(bank_root: Path) -> dict:
    meta, t = token_table(Path(bank_root))
    out: dict = {"schema": DIAG_SCHEMA, **meta, "stage_edges": STAGE_EDGES,
                 "notes": ["log probabilities are float32, full vocabulary, natural log",
                           "stage = D-011 bin of the global mask fraction before the commit (0: <0.2 ... 4: >=0.8)",
                           "correlations are Spearman (average ranks for ties), descriptive only"]}
    sp = t["split"]
    # 1. counts and rates
    out["drafts"] = []
    for s in sorted(set(t["draft_split"].tolist())):
        for b in sorted(set(t["draft_budget"].tolist()), key=lambda x: (x == "L", x.zfill(3))):
            for st in sorted(set(t["draft_setting"].tolist())):
                sel = (t["draft_split"] == s) & (t["draft_budget"] == b) & (t["draft_setting"] == st)
                n = int(sel.sum())
                if n:
                    out["drafts"].append({"split": s, "budget": b, "setting": st, "drafts": n,
                                          "final_wrong": int((~t["draft_final_correct"][sel]).sum()),
                                          "trace_wrong": int((~t["draft_trace_correct"][sel]).sum()),
                                          "malformed": int((~t["draft_well_formed"][sel]).sum()),
                                          "token_errors": int(t["draft_n_token_errors"][sel].sum())})
    W = t["wrong"]
    out["rates"] = {
        "by_split_family_budget": _rate_rows([sp, t["family"], t["budget"]], ["split", "family", "budget"], W),
        "by_split_setting_budget": _rate_rows([sp, t["setting"], t["budget"]], ["split", "setting", "budget"], W),
        "by_split_budget_stage": _rate_rows([sp, t["budget"], t["stage"]], ["split", "budget", "stage"], W),
        "by_split_L_position": _rate_rows([sp, t["L"], t["position"]], ["split", "L", "position"], W),
        "by_split_family_slot": _rate_rows([sp, t["family"], t["slot"]], ["split", "family", "slot"], W),
    }
    # 2. distributions
    nlq = -t["log_q"]
    s0ce = -t["s0_ref"]
    out["distributions"] = {}
    for s in sorted(set(sp.tolist())):
        m = sp == s
        out["distributions"][s] = {
            "c_gen_wrong": _q(np.exp(t["log_c"][m & W])), "c_gen_correct": _q(np.exp(t["log_c"][m & ~W])),
            "neg_log_q_wrong": _q(nlq[m & W]), "neg_log_q_correct": _q(nlq[m & ~W]),
            "step0_ce_wrong": _q(s0ce[m & W][~np.isnan(s0ce[m & W])]),
            "step0_ce_correct": _q(s0ce[m & ~W][~np.isnan(s0ce[m & ~W])]),
        }
    # 3. relationships among wrong tokens (where a confidence weight would act)
    rel = {}
    for s in sorted(set(sp.tolist())):
        m = (sp == s) & W
        if not m.any():
            continue
        lc, lq, ce, cd = t["log_c"][m], t["log_q"][m], s0ce[m], t["s0_draft"][m]
        fr, bud, st, setting = t["frac"][m], t["budget"][m], t["stage"][m], t["setting"][m]
        has0 = not np.isnan(ce).any()
        r = {"n_wrong_tokens": int(m.sum()),
             "spearman": {"log_c_vs_neg_log_q": spearman(lc, -lq), "log_c_vs_mask_fraction": spearman(lc, fr),
                          "log_c_vs_commit_order": spearman(lc, t["order_frac"][m])}}
        set_rate = {row["setting"]: row["rate"] for row in out["rates"]["by_split_setting_budget"] if row["split"] == s}
        r["spearman"]["log_c_vs_setting_error_rate"] = spearman(lc, np.array([set_rate[x] for x in setting]))
        if has0:
            r["spearman"].update(log_c_vs_step0_ce=spearman(lc, ce), log_c_vs_step0_log_p_draft=spearman(lc, cd),
                                 neg_log_q_vs_step0_ce=spearman(-lq, ce))
        r["eta_squared_log_c"] = {
            "stage": eta_squared(lc, st), "budget": eta_squared(lc, bud), "setting": eta_squared(lc, setting),
            "budget_x_stage": eta_squared(lc, np.char.add(bud.astype(str), np.char.add("|", st.astype(str)))),
            "setting_x_budget_x_stage": eta_squared(lc, np.array([f"{a}|{b}|{c}" for a, b, c in zip(setting, bud, st)]))}
        within, overlap = [], []
        for key in sorted(set(zip(bud.tolist(), st.tolist()))):
            g = (bud == key[0]) & (st == key[1])
            n = int(g.sum())
            row = {"budget": key[0], "stage": int(key[1]), "n": n,
                   "spearman_log_c_vs_neg_log_q": spearman(lc[g], -lq[g])}
            if has0:
                row["spearman_log_c_vs_step0_ce"] = spearman(lc[g], ce[g])
                if n >= 20:
                    hc = lc[g] >= np.quantile(lc[g], 0.75)
                    hd = ce[g] >= np.quantile(ce[g], 0.75)
                    overlap.append({"budget": key[0], "stage": int(key[1]), "n": n, "top_c": int(hc.sum()),
                                    "top_c_also_top_step0_ce": int((hc & hd).sum()),
                                    "chance_fraction": float(hd.mean())})
            within.append(row)
        r["within_budget_stage"] = within
        r["top_quartile_overlap"] = overlap
        r["ties_and_saturation"] = {
            "log_c_equals_row_max": int((lc == t["raw_max"][m]).sum()),
            "c_gt_0.99": int((lc > np.log(0.99)).sum()), "c_gt_0.999": int((lc > np.log(0.999)).sum()),
            "distinct_log_c_values": int(np.unique(lc).size)}
        rel[s] = r
    out["relationships_wrong_tokens"] = rel
    # 4. weighting-group populations (wrong tokens per budget x stage, per setting x budget x stage)
    pops = {}
    for s in sorted(set(sp.tolist())):
        m = (sp == s) & W
        cells = defaultdict(int)
        for a, b, c in zip(t["setting"][m], t["budget"][m], t["stage"][m]):
            cells[(a, b, int(c))] += 1
        sizes = np.array(sorted(cells.values())) if cells else np.zeros(0)
        pops[s] = {"setting_budget_stage_cells_with_errors": len(cells),
                   "cell_size_min": int(sizes.min()) if sizes.size else None,
                   "cell_size_median": float(np.median(sizes)) if sizes.size else None,
                   "cells_below_20": int((sizes < 20).sum())}
    out["weighting_group_populations"] = pops
    # 5. Q-7: step-0 loss on correct (keep) tokens
    if meta["has_step0"]:
        q7 = {}
        for s in sorted(set(sp.tolist())):
            m = (sp == s) & ~W
            ce = s0ce[m]
            q7[s] = {"n_correct_tokens": int(m.sum()), "mean_step0_ce": float(ce.mean()) if ce.size else None,
                     "fraction_below_1e-3": float((ce < 1e-3).mean()) if ce.size else None,
                     "fraction_below_1e-2": float((ce < 1e-2).mean()) if ce.size else None}
        out["q7_step0_keep_loss"] = q7
    return out


def _fmt(x) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


def render_markdown(d: dict) -> str:
    lines = [f"# Bank signal diagnostics: {d['bank']}", "",
             f"Bank manifest sha256 `{d['manifest_sha256']}`; {d['n_drafts']} drafts, {d['n_tokens']} tokens. "
             f"Step-0 signals: {'present' if d['has_step0'] else 'not computed'}. Train and validation only.", ""]
    lines += [f"- {n}" for n in d["notes"]] + ["", "## Error rates by split, family and budget", "",
                                               "| split | family | budget | wrong / tokens | rate |", "|---|---|---|---|---|"]
    for r in d["rates"]["by_split_family_budget"]:
        lines.append(f"| {r['split']} | {r['family']} | {r['budget']} | {r['wrong']} / {r['tokens']} | {_fmt(r['rate'])} |")
    lines += ["", "## Error rates by budget and stage", "", "| split | budget | stage | wrong / tokens | rate |",
              "|---|---|---|---|---|"]
    for r in d["rates"]["by_split_budget_stage"]:
        lines.append(f"| {r['split']} | {r['budget']} | {r['stage']} | {r['wrong']} / {r['tokens']} | {_fmt(r['rate'])} |")
    lines += ["", "## Distributions (quantiles)", ""]
    for s, dist in d["distributions"].items():
        lines += [f"### {s}", "", "| signal | n | q05 | q25 | q50 | q75 | q95 |", "|---|---|---|---|---|---|---|"]
        for k, v in dist.items():
            lines.append(f"| {k} | {v['n']} | " + " | ".join(_fmt(v.get(q)) for q in ("q05", "q25", "q50", "q75", "q95")) + " |")
        lines.append("")
    lines += ["## Relationships among wrong tokens", ""]
    for s, r in d["relationships_wrong_tokens"].items():
        lines += [f"### {s} (n = {r['n_wrong_tokens']} wrong tokens)", ""]
        lines += [f"- Spearman {k}: {_fmt(v)}" for k, v in r["spearman"].items()]
        lines += [f"- eta^2 of log c on {k}: {_fmt(v)}" for k, v in r["eta_squared_log_c"].items()]
        lines += [f"- {k}: {v}" for k, v in r["ties_and_saturation"].items()]
        lines += ["", "| budget | stage | n | rho(log c, -log q) | rho(log c, step-0 CE) |", "|---|---|---|---|---|"]
        for w in r["within_budget_stage"]:
            lines.append(f"| {w['budget']} | {w['stage']} | {w['n']} | {_fmt(w['spearman_log_c_vs_neg_log_q'])} | "
                         f"{_fmt(w.get('spearman_log_c_vs_step0_ce'))} |")
        lines.append("")
    lines += ["## Weighting-group populations", "", "```json", json.dumps(d["weighting_group_populations"], indent=2), "```"]
    if "q7_step0_keep_loss" in d:
        lines += ["", "## Q-7: step-0 loss on correct tokens", "", "```json", json.dumps(d["q7_step0_keep_loss"], indent=2), "```"]
    return "\n".join(lines) + "\n"


def write_diagnostics(bank_root: Path, out_dir: Path | None = None) -> dict:
    bank_root = Path(bank_root)
    out_dir = Path(out_dir) if out_dir else bank_root / "derived" / "diagnostics"
    d = compute(bank_root)
    write_json_atomic(out_dir / "diagnostics.json", d)
    write_bytes_atomic(out_dir / "diagnostics.md", render_markdown(d).encode())
    return d
