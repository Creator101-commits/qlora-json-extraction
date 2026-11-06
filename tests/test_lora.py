"""Tests for LoRA linear layer and parameter efficiency."""

from __future__ import annotations

import numpy as np

from qlora_extraction.models.lora import LoRALinear


def test_lora_linear_forward_shape() -> None:
    lora = LoRALinear(in_features=64, out_features=128, r=8, lora_alpha=16)
    x = np.random.randn(4, 64).astype(np.float32)
    out = lora.forward(x)
    assert out.shape == (4, 128)


def test_lora_parameter_counts() -> None:
    lora = LoRALinear(in_features=1024, out_features=1024, r=16)
    counts = lora.get_parameter_counts()
    base_params = counts["base_parameters"]
    trainable = counts["trainable_lora_parameters"]

    assert base_params == 1024 * 1024
    assert trainable == (1024 * 16) + (16 * 1024)
    assert counts["trainable_fraction"] < 0.05
