import pytest
import torch

from diffusion_correction.models.alignment import build_layout
from diffusion_correction.models.tiny import TINY_TOKENS, build_tiny_adapter


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
