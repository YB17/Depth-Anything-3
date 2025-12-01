import importlib.util
from pathlib import Path
import sys

import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "depth_anything_3" / "model" / "lora" / "token_selective_lora.py"
spec = importlib.util.spec_from_file_location("token_selective_lora", MODULE_PATH)
token_selective_lora = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = token_selective_lora
spec.loader.exec_module(token_selective_lora)
TokenSelectiveLoRALinear = token_selective_lora.TokenSelectiveLoRALinear


def test_token_mask_blocks_lora_residual():
    base = torch.nn.Linear(4, 2, bias=False)
    torch.nn.init.eye_(base.weight)
    layer = TokenSelectiveLoRALinear(base, rank=1, alpha=1.0, enable=True)
    torch.nn.init.ones_(layer.lora_A)
    torch.nn.init.ones_(layer.lora_B)

    x = torch.arange(16, dtype=torch.float32).view(2, 2, 4)
    token_mask = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)

    out = layer(x, token_mask=token_mask)
    base_out = base(x)

    delta = out - base_out
    assert torch.allclose(delta[0, 1], torch.zeros_like(delta[0, 1]))
    assert torch.allclose(delta[1, 0], torch.zeros_like(delta[1, 0]))
    assert not torch.allclose(delta[0, 0], torch.zeros_like(delta[0, 0]))
    assert not torch.allclose(delta[1, 1], torch.zeros_like(delta[1, 1]))
