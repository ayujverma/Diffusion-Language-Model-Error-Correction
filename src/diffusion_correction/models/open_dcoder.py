"""Open-dCoder-0.5B adapter: pinned files, verified token IDs, our bidirectional forward.

The pinned remote code is NOT executed here. Its forward computation is
re-implemented in :mod:`qwen2_bidir` and checked against the unmodified upstream
class by ``scripts/reference_logits.py`` (run in a transformers==4.54.1 env) and
``tests/test_open_dcoder.py``. See docs/source_audit.md for why.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .adapter import DiffusionAdapter, TokenSpec
from .qwen2_bidir import BidirectionalQwen2, Qwen2Config

REPO_ID = "fredzzp/open-dcoder-0.5B"
REVISION = "d0d86d5b99960c05258bb1f8265dd91564dbac67"
DEFAULT_LOCAL_DIR = Path("runs/hf/open-dcoder-0.5B")

# sha256 of every file we read, at REVISION. The weights hash also appears in the
# CDLM training README's pinned-revision table.
PINNED_SHA256 = {
    "config.json": "fcd627396dda21d8f93425a6c18f1be20cd6bfe219f5e939fd33ba8357d7a1d9",
    "tokenizer.json": "4f3ead5ffbb2abfb9e51757b104af5174f59c994ff3b2ca715d344dc66ee1f4e",
    "tokenizer_config.json": "fd348836e1a0a9444e4ceca15c26169f95576ca65b570154a0a5dcc9111c2633",
    "special_tokens_map.json": "6e57d1b59798c99b30b53fa9536e328f8af2aa5616d2c03bea5f0f83f676989b",
    "generation_config.json": "c269052c80cbb710726e30ad35c24172486bd1be17c64681a2230cafc5057119",
    "modeling_qwen2.py": "7194b1fbabecf0ebdadebf14b345cc64c4ca0d16166f563877d5d62f81d719b1",
    "generation_utils.py": "f9f092c97da1e54d3024f62d6c5c21542f994a1e26a7cfc2dd7fd726a44ba701",
    "model.safetensors": "59c1a005f4b672bdd3bbdab6258b283dff3b87bab5cc4bbc0d479e8388f77b6e",
}
SMALL_FILES = [f for f in PINNED_SHA256 if f != "model.safetensors"]


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download(local_dir: Path = DEFAULT_LOCAL_DIR, include_weights: bool = True) -> Path:
    from huggingface_hub import snapshot_download

    patterns = list(SMALL_FILES) + ["merges.txt", "vocab.json", "added_tokens.json", "README.md", "__init__.py"]
    if include_weights:
        patterns.append("model.safetensors")
    snapshot_download(REPO_ID, revision=REVISION, allow_patterns=patterns, local_dir=str(local_dir))
    return local_dir


def verify_files(local_dir: Path, check_weights: bool = True) -> dict[str, str]:
    """Return {file: sha256}; raise if any pinned file is missing or differs."""
    out = {}
    for name, expected in PINNED_SHA256.items():
        if name == "model.safetensors" and not check_weights:
            continue
        path = local_dir / name
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run `dlmc download-backbone`")
        got = sha256_file(path)
        if got != expected:
            raise ValueError(f"{name}: sha256 {got} != pinned {expected}")
        out[name] = got
    return out


def resolve_token_spec(local_dir: Path) -> TokenSpec:
    """Read special-token IDs from the pinned tokenizer files and cross-check them.

    ``generation_config.json`` sets ``mask_token_id=151643`` (``<|endoftext|>``),
    but the tokenizer declares ``<M>`` (151665) as the mask token and the upstream
    trainer/sampler use ``tokenizer.mask_token_id``. We use ``<M>``; see
    docs/decisions.md (D-003).
    """
    tok = json.loads((local_dir / "tokenizer.json").read_text())
    stm = json.loads((local_dir / "special_tokens_map.json").read_text())
    cfg = json.loads((local_dir / "config.json").read_text())
    added = {a["content"]: a for a in tok["added_tokens"]}

    def special_id(entry) -> int:
        content = entry["content"] if isinstance(entry, dict) else entry
        if content not in added or not added[content]["special"]:
            raise ValueError(f"{content!r} is not a special added token in tokenizer.json")
        return added[content]["id"]

    mask_id = special_id(stm["mask_token"])
    pad_id = special_id(stm["pad_token"])
    eos_id = special_id(stm["eos_token"])
    if eos_id != cfg["eos_token_id"]:
        raise ValueError(f"tokenizer EOS {eos_id} != config eos_token_id {cfg['eos_token_id']}")
    vocab = cfg["vocab_size"]
    for name, tid in (("mask", mask_id), ("pad", pad_id), ("eos", eos_id)):
        if not 0 <= tid < vocab:
            raise ValueError(f"{name} id {tid} outside output vocabulary {vocab}")
    if mask_id in (pad_id, eos_id):
        raise ValueError("mask token must differ from pad/EOS")
    return TokenSpec(mask_token_id=mask_id, pad_token_id=pad_id, eos_token_id=eos_id, vocab_size=vocab,
                     forbidden_output_ids=(mask_id,))


def load_open_dcoder(local_dir: Path | str = DEFAULT_LOCAL_DIR, device: str | torch.device = "cpu",
                     dtype: torch.dtype = torch.float32, verify_weights_hash: bool = True) -> DiffusionAdapter:
    from safetensors.torch import load_file

    local_dir = Path(local_dir)
    hashes = verify_files(local_dir, check_weights=verify_weights_hash)
    cfg_dict = json.loads((local_dir / "config.json").read_text())
    cfg = Qwen2Config.from_hf_dict(cfg_dict)
    tokens = resolve_token_spec(local_dir)

    state = load_file(str(local_dir / "model.safetensors"))
    if cfg.tie_word_embeddings:
        if "lm_head.weight" in state and not torch.equal(state["lm_head.weight"], state["model.embed_tokens.weight"]):
            raise ValueError("config ties embeddings but checkpoint lm_head differs from embed_tokens")
        state.pop("lm_head.weight", None)
    with torch.device("meta"):
        model = BidirectionalQwen2(cfg)
    expected = {k for k in model.state_dict() if not (cfg.tie_word_embeddings and k == "lm_head.weight")}
    missing, unexpected = expected - state.keys(), state.keys() - expected
    if missing or unexpected:
        raise ValueError(f"state_dict mismatch: missing={sorted(missing)[:5]} unexpected={sorted(unexpected)[:5]}")
    model.load_state_dict(state, strict=False, assign=True)
    if cfg.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    model.model.rotary_emb = type(model.model.rotary_emb)(cfg.head_dim, cfg.rope_theta)
    model = model.to(device=device, dtype=dtype)
    provenance = {
        "backbone": REPO_ID, "revision": REVISION, "file_sha256": hashes,
        "checkpoint_dtype": cfg_dict.get("torch_dtype"), "runtime_dtype": str(dtype),
        "implementation": "diffusion_correction.models.qwen2_bidir (sdpa, bidirectional, no KV cache)",
    }
    return DiffusionAdapter(model=model, tokens=tokens, provenance=provenance)


def load_tokenizer(local_dir: Path | str = DEFAULT_LOCAL_DIR):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(Path(local_dir) / "tokenizer.json"))
