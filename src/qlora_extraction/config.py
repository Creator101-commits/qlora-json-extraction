"""Configuration schemas and settings for QLoRA fine-tuning and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class QLoRAConfig:
    """Supervised QLoRA hyperparameter and model configuration."""

    base_model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    quantization_bits: int = 4
    compute_dtype: str = "bfloat16"

    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Training settings
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 1024
    warmup_ratio: float = 0.05
    max_steps: int = 250
    seed: int = 42

    # Validation
    repair_malformed_json: bool = True
    strict_schema_enforcement: bool = True
