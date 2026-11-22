"""Model-call (NFE) accounting for every sampler (D-049).

One NFE is one successful call to ``DiffusionAdapter.aligned_logits``. The meter
increments inside ``forward``, after the call returns, so a sampler cannot spend a
model call without recording it. The budget is checked before the call: if the next
call would exceed ``max_calls``, the model is not run and ``forward`` returns None.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..models.alignment import SequenceLayout


@dataclass(frozen=True)
class CallRecord:
    index: int  # 1-based
    kind: str   # predictor | corrector | d3im | revision
    step: int   # 1-based sampler step this call belongs to


@dataclass
class ForwardMeter:
    """Counts aligned forwards for one trajectory.

    ``max_calls`` is a per-trajectory cap. A batched forward charges every example
    in the batch (they all took part), matching D-029. ``max_calls is None`` records
    without refusing.
    """

    adapter: object
    max_calls: int | None = None
    records: list[CallRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.max_calls is not None and self.max_calls < 1:
            raise ValueError(f"max_calls must be >= 1 or None, got {self.max_calls}")

    @property
    def nfe(self) -> int:
        return len(self.records)

    @property
    def kinds(self) -> list[str]:
        return [r.kind for r in self.records]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for kind in self.kinds:
            out[kind] = out.get(kind, 0) + 1
        return out

    def forward(self, layout: SequenceLayout, *, kind: str, step: int) -> torch.Tensor | None:
        if not kind:
            raise ValueError("call kind must be a non-empty string")
        if step < 1:
            raise ValueError("step is 1-based")
        if self.max_calls is not None and self.nfe >= self.max_calls:
            return None
        logits = self.adapter.aligned_logits(layout)
        self.records.append(CallRecord(index=self.nfe + 1, kind=kind, step=step))
        return logits
