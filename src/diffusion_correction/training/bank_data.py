"""Load immutable bank drafts for shared-bank correction training (milestone 5).

Drafts are read from a finalized bank only. Saved ``log_c_gen`` / ``log_q_gen`` /
``mask_fraction_before`` are never recomputed from the corrector. Train and val
splits stay separate; test splits are refused.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch

from ..drafts.bank import BANK_SPLITS, read_complete_bank
from ..drafts.identity import sha256_file
from ..models.alignment import SequenceLayout, build_layout

# Integer codes for (budget, stage) ranking groups. "L" is a fixed sentinel so all
# one-token-per-call drafts share one budget id regardless of response length.
BUDGET_CODES: dict[str, int] = {"1": 1, "2": 2, "4": 4, "8": 8, "L": 0}


@dataclass(frozen=True)
class BankIdentity:
    bank_dir: str
    bank_name: str
    identity_sha256: str
    manifest_sha256: str
    plan_sha256: str
    split: str
    budgets: tuple[str, ...]
    n_drafts: int
    draft_ids_sha256: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BankDraft:
    """One immutable bank record, reduced to the fields correction training needs."""

    problem_id: str
    split: str
    budget: str
    budget_code: int
    family: str
    length: int
    operand_digits: int
    setting: str
    prompt_ids: list[int]
    draft_ids: list[int]
    reference_ids: list[int]
    log_c_gen: list[float]
    log_q_gen: list[float]
    mask_fraction_before: list[float]
    token_correct: list[bool]
    final_correct: bool
    well_formed: bool
    n_token_errors: int
    shard: str
    identity_sha256: str


def budget_code(budget: str | int) -> int:
    b = str(budget)
    if b not in BUDGET_CODES:
        raise ValueError(f"unknown budget {budget!r}; expected one of {sorted(BUDGET_CODES)}")
    return BUDGET_CODES[b]


def _draft_ids_digest(drafts: Sequence[BankDraft]) -> str:
    from ..data.splits import sha256_bytes

    lines = [f"{d.split}\t{d.problem_id}\t{d.budget}" for d in drafts]
    return sha256_bytes("\n".join(lines).encode())


def _record_to_draft(rec: dict) -> BankDraft:
    g, a = rec["generation"], rec["analysis"]
    R = int(rec["response_length"])
    for name, arr in (("draft_ids", g["draft_ids"]), ("reference_ids", a["reference_ids"]),
                      ("log_c_gen", g["log_c_gen"]), ("log_q_gen", a["log_q_gen"]),
                      ("mask_fraction_before", g["mask_fraction_before"]),
                      ("token_correct", a["token_correct"])):
        if len(arr) != R:
            raise ValueError(f"{rec['problem_id']}/{rec['budget']}: {name} length {len(arr)} != {R}")
    return BankDraft(
        problem_id=rec["problem_id"], split=rec["split"], budget=str(rec["budget"]),
        budget_code=budget_code(rec["budget"]), family=rec["family"], length=int(rec["length"]),
        operand_digits=int(rec["operand_digits"]), setting=str(rec["setting"]),
        prompt_ids=list(rec["prompt_ids"]), draft_ids=list(g["draft_ids"]),
        reference_ids=list(a["reference_ids"]), log_c_gen=[float(x) for x in g["log_c_gen"]],
        log_q_gen=[float(x) for x in a["log_q_gen"]],
        mask_fraction_before=[float(x) for x in g["mask_fraction_before"]],
        token_correct=[bool(x) for x in a["token_correct"]], final_correct=bool(a["final_correct"]),
        well_formed=bool(a["well_formed"]), n_token_errors=int(a["n_token_errors"]),
        shard=str(rec.get("shard", "")), identity_sha256=str(rec["identity_sha256"]),
    )


def verify_bank_pins(bank_dir: str | Path, identity_sha256: str, manifest_sha256: str) -> dict:
    """Refuse a mismatched or incomplete bank. Returns the parsed manifest."""
    root = Path(bank_dir)
    if not (root / "BANK_COMPLETE").exists():
        raise ValueError(f"{root}: BANK_COMPLETE missing; correction training needs a finalized bank")
    marker = (root / "BANK_COMPLETE").read_text().split()
    msha = marker[marker.index("manifest_sha256") + 1]
    if msha != manifest_sha256:
        raise ValueError(f"{root}: BANK_COMPLETE manifest {msha} != configured {manifest_sha256}")
    if sha256_file(root / "MANIFEST.json") != msha:
        raise ValueError(f"{root}: MANIFEST.json does not match BANK_COMPLETE")
    manifest = json.loads((root / "MANIFEST.json").read_text())
    if manifest["identity_sha256"] != identity_sha256:
        raise ValueError(f"{root}: identity {manifest['identity_sha256']} != configured {identity_sha256}")
    if manifest.get("manifest_sha256", msha) != manifest_sha256:
        raise ValueError(f"{root}: manifest sha256 mismatch")
    return {**manifest, "manifest_sha256": msha}


def load_bank_drafts(bank_dir: str | Path, *, identity_sha256: str, manifest_sha256: str,
                     split: str, budgets: Sequence[str], verify_shards: bool = True
                     ) -> tuple[list[BankDraft], BankIdentity]:
    """Load train or val drafts at the requested budgets from a finalized bank."""
    if split not in BANK_SPLITS:
        raise ValueError(f"split {split!r} is not allowed (train/val only; test splits are never loaded)")
    budgets = tuple(str(b) for b in budgets)
    for b in budgets:
        budget_code(b)
    root = Path(bank_dir)
    manifest = verify_bank_pins(root, identity_sha256, manifest_sha256)
    if verify_shards:
        # Re-hash every shard against the manifest (same gate as read_complete_bank).
        _, shards = read_complete_bank(root)
        records = [r for _, recs in shards for r in recs]
    else:
        records = []
        for s in manifest["shards"]:
            p = root / s["file"]
            if not p.exists():
                raise ValueError(f"{p}: missing shard listed in the manifest")
            for line in p.read_text().splitlines():
                if line:
                    records.append(json.loads(line))
    want = set(budgets)
    drafts = [_record_to_draft(r) for r in records
              if r["split"] == split and str(r["budget"]) in want]
    if not drafts:
        raise ValueError(f"{root}: no drafts for split={split} budgets={list(budgets)}")
    for d in drafts:
        if d.identity_sha256 != identity_sha256:
            raise ValueError(f"{d.problem_id}/{d.budget}: record identity differs from the bank pin")
    # Stable order: shard order was plan order; keep file order within the filter.
    drafts.sort(key=lambda d: (d.budget, d.family, d.length, d.operand_digits, d.problem_id))
    digest = _draft_ids_digest(drafts)
    ident = BankIdentity(bank_dir=str(root), bank_name=manifest["bank"], identity_sha256=identity_sha256,
                         manifest_sha256=manifest_sha256, plan_sha256=manifest["plan_sha256"],
                         split=split, budgets=budgets, n_drafts=len(drafts), draft_ids_sha256=digest)
    return drafts, ident


@dataclass
class CorrectionBatch:
    """Collated physical minibatch for one correction update (mask + draft passes)."""

    target_layout: SequenceLayout          # prompt | reference | pad
    draft_layout: SequenceLayout           # prompt | draft | pad (same lengths)
    budgets: torch.Tensor                  # [B] int64 budget codes
    log_c_response: list[torch.Tensor]     # per example, length R
    log_q_response: list[torch.Tensor]
    mask_fraction_response: list[torch.Tensor]
    problem_ids: list[str]
    budget_labels: list[str]


def collate_correction_batch(drafts: Sequence[BankDraft], pad_token_id: int) -> CorrectionBatch:
    prompts = [d.prompt_ids for d in drafts]
    targets = [d.reference_ids for d in drafts]
    draft_ids = [d.draft_ids for d in drafts]
    # Same total length so the two layouts share masks.
    total = max(len(p) + len(r) for p, r in zip(prompts, targets))
    target_layout = build_layout(prompts, targets, pad_token_id, total_length=total)
    draft_layout = build_layout(prompts, draft_ids, pad_token_id, total_length=total)
    if not torch.equal(target_layout.editable_mask, draft_layout.editable_mask):
        raise AssertionError("target and draft layouts must share editable positions")
    return CorrectionBatch(
        target_layout=target_layout, draft_layout=draft_layout,
        budgets=torch.tensor([d.budget_code for d in drafts], dtype=torch.long),
        log_c_response=[torch.tensor(d.log_c_gen, dtype=torch.float32) for d in drafts],
        log_q_response=[torch.tensor(d.log_q_gen, dtype=torch.float32) for d in drafts],
        mask_fraction_response=[torch.tensor(d.mask_fraction_before, dtype=torch.float32) for d in drafts],
        problem_ids=[d.problem_id for d in drafts],
        budget_labels=[d.budget for d in drafts],
    )


def gather_error_signals(error_mask: torch.Tensor, editable_mask: torch.Tensor,
                         budgets: torch.Tensor, log_c_response: Sequence[torch.Tensor],
                         log_q_response: Sequence[torch.Tensor],
                         mask_fraction_response: Sequence[torch.Tensor]
                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten saved signals for error tokens in the same row-major order as ``selected_token_ce``."""
    if error_mask.shape != editable_mask.shape:
        raise ValueError("error_mask and editable_mask shape mismatch")
    B = error_mask.shape[0]
    log_c, log_q, mfrac, bud = [], [], [], []
    for b in range(B):
        # Response columns are exactly the editable positions for this row (right-padded layout).
        edit_idx = editable_mask[b].nonzero(as_tuple=True)[0]
        err_idx = error_mask[b].nonzero(as_tuple=True)[0]
        if edit_idx.numel() != log_c_response[b].numel():
            raise ValueError(f"example {b}: editable count {edit_idx.numel()} != response signal length "
                             f"{log_c_response[b].numel()}")
        # Map sequence positions -> response offsets.
        pos_to_resp = {int(p): i for i, p in enumerate(edit_idx.tolist())}
        for p in err_idx.tolist():
            if p not in pos_to_resp:
                raise ValueError(f"error position {p} is not editable")
            r = pos_to_resp[p]
            log_c.append(float(log_c_response[b][r]))
            log_q.append(float(log_q_response[b][r]))
            mfrac.append(float(mask_fraction_response[b][r]))
            bud.append(int(budgets[b]))
    device = error_mask.device
    return (torch.tensor(log_c, dtype=torch.float32, device=device),
            torch.tensor(log_q, dtype=torch.float32, device=device),
            torch.tensor(mfrac, dtype=torch.float32, device=device),
            torch.tensor(bud, dtype=torch.long, device=device))
