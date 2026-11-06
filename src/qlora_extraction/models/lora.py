"""Low-Rank Adaptation (LoRA) formulation and parameter efficiency diagnostics."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np


class LoRALinear:
    """Simulated Low-Rank Adapter layer: W = W_0 + (alpha / r) * (B @ A)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        lora_alpha: int = 32,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # Frozen base weight W_0
        self.weight = np.random.randn(out_features, in_features).astype(np.float32) / math.sqrt(
            in_features
        )
        self.weight_frozen = True

        # Trainable low-rank decomposition matrices
        # A is Gaussian initialized, B is zero-initialized
        self.lora_a = np.random.randn(r, in_features).astype(np.float32) / math.sqrt(in_features)
        self.lora_b = np.zeros((out_features, r), dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute y = x @ (W_0^T + (alpha/r) * A^T @ B^T)."""
        base_out = x @ self.weight.T
        lora_out = (x @ self.lora_a.T) @ self.lora_b.T * self.scaling
        return cast(np.ndarray, base_out + lora_out)

    def get_parameter_counts(self) -> dict[str, Any]:
        """Compute total, frozen, and trainable parameter counts."""
        base_params = self.out_features * self.in_features
        trainable_params = (self.out_features * self.r) + (self.r * self.in_features)
        return {
            "base_parameters": base_params,
            "trainable_lora_parameters": trainable_params,
            "trainable_fraction": trainable_params / (base_params + trainable_params),
        }
