import pytest
import torch

from diffusion_correction.models.alignment import build_layout
from diffusion_correction.models.tiny import TINY_TOKENS, build_tiny_adapter

TOKENIZER_JSON = "runs/hf/open-dcoder-0.5B/tokenizer.json"


@pytest.fixture
def tiny():
    torch.manual_seed(0)
    return build_tiny_adapter(seed=0, dtype=torch.float64)


@pytest.fixture
def ragged_layout():
    """Two examples with different prompt and response lengths, right-padded."""
    prompts = [[1, 2, 3], [4, 5, 6, 7, 8]]
    responses = [[9, 0, 1, 2], [3]]
    return build_layout(prompts, responses, pad_token_id=TINY_TOKENS.pad_token_id)


@pytest.fixture(scope="session")
def task_tokenizer():
    """The pinned Open-dCoder tokenizer (sha256-checked small files only)."""
    from pathlib import Path

    from diffusion_correction.models.open_dcoder import DEFAULT_LOCAL_DIR, load_tokenizer, verify_files

    if not Path(TOKENIZER_JSON).exists():
        pytest.skip("pinned tokenizer files not downloaded (dlmc download-backbone --no-weights)")
    verify_files(DEFAULT_LOCAL_DIR, check_weights=False)
    return load_tokenizer(DEFAULT_LOCAL_DIR)
