"""Batched greedy text generation for the extraction benchmark (GPU-only).

Left padding keeps generated tokens aligned at the right edge; decoding is
greedy (``do_sample=False``) with the protocol ``max_new_tokens``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from qlora_extraction.data.cord import CordDocument
from qlora_extraction.prompting import (
    build_messages,
    minified_schema_json,
    render_prompt_text,
)

GENERATION_BATCH_SIZE = 8


@dataclass
class GenerationResult:
    """Generated raw texts plus token and timing records."""

    texts: list[str] = field(default_factory=list)
    input_token_counts: list[int] = field(default_factory=list)
    output_token_counts: list[int] = field(default_factory=list)
    gen_time_s: float = 0.0
    tokens_per_second: float = 0.0


def prompt_texts_for_docs(
    docs: list[CordDocument],
    condition: str,
    tokenizer: Any,
    demonstrations: list[tuple[str, str]] | None = None,
) -> list[str]:
    """Render chat-template prompt text for each document."""
    schema_json = minified_schema_json()
    texts: list[str] = []
    for doc in docs:
        messages = build_messages(condition, doc.ocr_text, schema_json, demonstrations)
        texts.append(render_prompt_text(tokenizer, messages))
    return texts


def generation_prompts_for_condition(
    condition: str,
    docs: list[CordDocument],
    tokenizer: Any,
    demonstrations: list[tuple[str, str]] | None = None,
) -> list[str]:
    """Render prompts; ``few_shot`` requires the frozen demonstration pairs."""
    if condition == "few_shot" and not demonstrations:
        raise ValueError("few_shot generation requires frozen demonstration pairs")
    return prompt_texts_for_docs(docs, condition, tokenizer, demonstrations)


def generate_texts(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    max_new_tokens: int,
    batch_size: int = GENERATION_BATCH_SIZE,
) -> GenerationResult:
    """Greedy generation over ``prompts`` in fixed-size left-padded batches."""
    import torch

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    result = GenerationResult()
    total_output_tokens = 0
    start = time.monotonic()
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False)
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )
        new_tokens = out[:, input_ids.shape[1] :]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        result.texts.extend(decoded)
        result.input_token_counts.extend(
            int(n) for n in attention_mask.sum(dim=1).cpu().tolist()
        )
        result.output_token_counts.extend(
            int(n) for n in (new_tokens != pad_id).sum(dim=1).cpu().tolist()
        )
        total_output_tokens += int(new_tokens.shape[0] * new_tokens.shape[1])
    elapsed = time.monotonic() - start
    result.gen_time_s = elapsed
    result.tokens_per_second = (
        total_output_tokens / elapsed if elapsed > 0 else 0.0
    )
    return result


def load_model_for_inference(
    adapter_dir: str | None,
    *,
    base_model: str,
    base_revision: str,
) -> tuple[Any, Any]:
    """Load the 4-bit base model plus an optional trained LoRA adapter."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(base_model, revision=base_revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        revision=base_revision,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer
