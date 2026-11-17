"""Canonical fixed-slot numeric responses, verified against the pinned tokenizer.

Response text = one slot per value, nothing else::

    " +0045 +0135 +0116"

Each slot is a sign token (" +" or " -", leading space included) followed by
``width`` single-digit tokens, where ``width = len(str(value_bound))`` comes from
the declared bound of the task configuration, never from the answer. Zero is
``" +0000"``; ``" -0000"`` is non-canonical. Slot count = number of statements,
which the prompt states. There is no EOS token and no separator: the response
length is ``n_slots * (1 + width)`` tokens and is public (decision D-018).

Why this shape (measured on the pinned Qwen2 tokenizer, tests/test_data_format.py):
digits always split one per token, but punctuation merges with neighbours
(``"=-"``, ``",+"``, ``",-"``, ``"(+"`` are single tokens while ``"=+"`` is two), so
formats like ``a=+0045,b=-0135`` do not have a fixed token width. A leading space
before the sign isolates it: ``" +"`` and ``" -"`` are single tokens after a
digit and after the prompt's final ``":"``.

Parsing is by token ID at fixed positions. Text is never decoded and re-parsed,
so token merges or special tokens cannot be hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

SIGN_TEXT = {"+": " +", "-": " -"}


@dataclass(frozen=True)
class ResponseFormat:
    value_bound: int

    @property
    def width(self) -> int:
        return len(str(self.value_bound))

    @property
    def slot_len(self) -> int:
        return 1 + self.width

    def response_len(self, n_slots: int) -> int:
        return n_slots * self.slot_len

    def render_value(self, v: int) -> str:
        if abs(v) > self.value_bound:
            raise ValueError(f"{v} exceeds declared bound {self.value_bound}")
        return f"{SIGN_TEXT['-' if v < 0 else '+']}{abs(v):0{self.width}d}"

    def render(self, values: Sequence[int]) -> str:
        return "".join(self.render_value(v) for v in values)

    def slot_spans(self, n_slots: int) -> list[tuple[int, int]]:
        """Half-open token spans ``[start, end)`` of each value within the response."""
        return [(k * self.slot_len, (k + 1) * self.slot_len) for k in range(n_slots)]

    def describe(self) -> str:
        return f"a sign and {self.width} digits each"


@dataclass(frozen=True)
class SlotParse:
    status: str            # "ok" or a malformed reason
    value: int | None      # parsed integer when status == "ok"


@dataclass
class TokenizedFormat:
    """A :class:`ResponseFormat` bound to concrete token IDs of one tokenizer."""

    fmt: ResponseFormat
    sign_ids: dict[str, int]          # "+" / "-" -> token ID of " +" / " -"
    digit_ids: tuple[int, ...]        # token ID of "0".."9"
    special_ids: frozenset[int]       # every special added token (mask, EOS/pad, ...)
    tokenizer_size: int               # IDs >= this are unused output rows
    _sign_of: dict[int, str] = field(init=False, repr=False)
    _digit_of: dict[int, int] = field(init=False, repr=False)

    def __post_init__(self):
        self._sign_of = {i: s for s, i in self.sign_ids.items()}
        self._digit_of = {i: d for d, i in enumerate(self.digit_ids)}

    @classmethod
    def from_tokenizer(cls, tokenizer, fmt: ResponseFormat) -> "TokenizedFormat":
        def single(text: str) -> int:
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            if len(ids) != 1:
                raise ValueError(f"{text!r} is {len(ids)} tokens in this tokenizer; the format needs 1")
            return ids[0]

        sign_ids = {s: single(t) for s, t in SIGN_TEXT.items()}
        digit_ids = tuple(single(str(d)) for d in range(10))
        special = frozenset(i for i, t in tokenizer.get_added_tokens_decoder().items() if t.special)
        if special & (set(sign_ids.values()) | set(digit_ids)):
            raise ValueError("a format token is a special token")
        return cls(fmt=fmt, sign_ids=sign_ids, digit_ids=digit_ids, special_ids=special,
                   tokenizer_size=tokenizer.get_vocab_size(with_added_tokens=True))

    def encode_values(self, values: Sequence[int]) -> list[int]:
        """Template construction (no tokenizer call); see :func:`encode_and_verify`."""
        out = []
        for v in values:
            text = self.fmt.render_value(v)
            out.append(self.sign_ids[text[1]])
            out.extend(self.digit_ids[int(c)] for c in text[2:])
        return out

    def parse_slot(self, ids: Sequence[int]) -> SlotParse:
        if len(ids) != self.fmt.slot_len:
            return SlotParse("wrong_slot_length", None)
        if any(i in self.special_ids for i in ids):
            return SlotParse("special_token", None)
        if any(not 0 <= i < self.tokenizer_size for i in ids):
            return SlotParse("unknown_token_id", None)
        sign = self._sign_of.get(ids[0])
        if sign is None:
            return SlotParse("bad_sign_token", None)
        digits = [self._digit_of.get(i) for i in ids[1:]]
        if any(d is None for d in digits):
            return SlotParse("bad_digit_token", None)
        mag = int("".join(map(str, digits)))
        if sign == "-" and mag == 0:
            return SlotParse("negative_zero", None)
        return SlotParse("ok", -mag if sign == "-" else mag)

    def token_role(self, k: int) -> str:
        """Role of response token ``k``: ``sign`` or ``digit<j>`` (j = 0 is most significant)."""
        j = k % self.fmt.slot_len
        return "sign" if j == 0 else f"digit{j - 1}"


def boundary_merges(tokenizer, prompt_text: str, response_text: str) -> bool:
    """True if tokenizing the joint text differs from tokenizing the two parts separately."""
    p = tokenizer.encode(prompt_text, add_special_tokens=False).ids
    r = tokenizer.encode(response_text, add_special_tokens=False).ids
    return tokenizer.encode(prompt_text + response_text, add_special_tokens=False).ids != p + r


def encode_and_verify(tokenizer, tf: TokenizedFormat, prompt_text: str,
                      values: Sequence[int]) -> tuple[list[int], list[int], str]:
    """Tokenize prompt and response and check every layout invariant.

    Checks: (1) the tokenizer's own encoding of the response text equals the slot
    template; (2) encoding prompt and response separately and concatenating equals
    encoding the joint text (no merge across the boundary); (3) the response has the
    public length. Raises ValueError on any violation.
    """
    response_text = tf.fmt.render(values)
    p = tokenizer.encode(prompt_text, add_special_tokens=False).ids
    r = tokenizer.encode(response_text, add_special_tokens=False).ids
    template = tf.encode_values(values)
    if r != template:
        raise ValueError(f"tokenizer encodes {response_text!r} as {r}, template is {template}")
    if boundary_merges(tokenizer, prompt_text, response_text):
        raise ValueError("prompt/response boundary merges tokens: joint encoding differs from concatenation")
    if len(r) != tf.fmt.response_len(len(values)):
        raise ValueError("response length differs from the public length")
    if not p:
        raise ValueError("empty prompt")
    return p, r, response_text
