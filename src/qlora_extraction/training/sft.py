"""QLoRA supervised fine-tuning for CORD JSON extraction.

4-bit NF4 quantization with double quantization and fp16 compute, LoRA on
the seven projection modules, prompt-token loss masking, cosine schedule
with 3% warmup, effective batch 16, gradient clip 1.0, up to 5 epochs with
patience-1 early stopping on validation loss keeping the best epoch.

All torch/transformers/peft imports happen inside ``train_qlora_sft`` so the
module imports cleanly on CPU-only machines; the ``gpu`` extra provides the
real dependencies on Kaggle.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qlora_extraction.data.cord import CordDocument
from qlora_extraction.prompting import build_messages, minified_schema_json, render_prompt_text

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
BASE_MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass(frozen=True)
class SftConfig:
    """Frozen QLoRA recipe from the benchmark protocol."""

    base_model: str = BASE_MODEL
    base_revision: str = BASE_MODEL_REVISION
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = tuple(LORA_TARGET_MODULES)
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    effective_batch_size: int = 16
    per_device_batch_size: int = 8
    grad_clip: float = 1.0
    max_epochs: int = 5
    patience: int = 1
    seed: int = 0
    max_seq_len: int = 4096


@dataclass
class SftResult:
    """Training outcome plus resource and provenance records."""

    adapter_dir: str
    best_epoch: int | None
    history: dict[str, list[float]] = field(default_factory=dict)
    n_dropped_overlength: int = 0
    n_train: int = 0
    n_val: int = 0
    train_time_s: float = 0.0
    peak_gpu_mem_bytes: int = 0
    n_trainable_params: int = 0
    n_total_params: int = 0
    adapter_bytes: int = 0


def build_sft_example(
    doc: CordDocument,
    tokenizer: Any,
    schema_json: str,
    max_seq_len: int,
) -> dict[str, Any] | None:
    """Tokenize one doc into masked labels; ``None`` when over max_seq_len.

    The prompt is the zero-shot chat-template rendering; loss applies only
    to the assistant target tokens plus EOS.
    """
    messages = build_messages("zero_shot", doc.ocr_text, schema_json)
    prompt_text = render_prompt_text(tokenizer, messages)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    target_text = doc.target_json
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        target_ids = target_ids + [int(eos_id)]
    input_ids = prompt_ids + target_ids
    if len(input_ids) > max_seq_len:
        return None
    labels = [-100] * len(prompt_ids) + target_ids
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


def _collate(batch: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    import torch

    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, item in enumerate(batch):
        n = len(item["input_ids"])
        input_ids[i, :n] = torch.tensor(item["input_ids"], dtype=torch.long)
        labels[i, :n] = torch.tensor(item["labels"], dtype=torch.long)
        attention[i, :n] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention}


def train_qlora_sft(
    train_docs: list[CordDocument],
    val_docs: list[CordDocument],
    work_dir: Path,
    config: SftConfig | None = None,
) -> SftResult:
    """Train a QLoRA adapter on the zero-shot prompt; early stop on val loss."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    cfg = config or SftConfig()
    work_dir.mkdir(parents=True, exist_ok=True)
    schema_json = minified_schema_json()

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model, revision=cfg.base_revision
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    examples: list[dict[str, Any]] = []
    dropped = 0
    for doc in train_docs:
        ex = build_sft_example(doc, tokenizer, schema_json, cfg.max_seq_len)
        if ex is None:
            dropped += 1
        else:
            examples.append(ex)
    val_examples: list[dict[str, Any]] = []
    for doc in val_docs:
        ex = build_sft_example(doc, tokenizer, schema_json, cfg.max_seq_len)
        if ex is not None:
            val_examples.append(ex)

    train_ds = Dataset.from_list(examples)
    val_ds = Dataset.from_list(val_examples)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        revision=cfg.base_revision,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    grad_accum = max(1, cfg.effective_batch_size // cfg.per_device_batch_size)
    args = TrainingArguments(
        output_dir=str(work_dir / "checkpoints"),
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.warmup_ratio,
        max_grad_norm=cfg.grad_clip,
        num_train_epochs=cfg.max_epochs,
        seed=cfg.seed,
        data_seed=cfg.seed,
        fp16=True,
        bf16=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=cfg.max_epochs,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lambda batch: _collate(batch, pad_id),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=cfg.patience,
                early_stopping_threshold=0.0,
            )
        ],
    )

    start = time.monotonic()
    trainer.train()
    train_time = time.monotonic() - start

    adapter_dir = work_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    adapter_bytes = sum(f.stat().st_size for f in adapter_dir.rglob("*") if f.is_file())

    history: dict[str, list[float]] = {}
    for entry in trainer.state.log_history:
        for key, value in entry.items():
            if isinstance(value, int | float) and not isinstance(value, bool):
                history.setdefault(key, []).append(float(value))

    best_epoch: int | None = None
    eval_losses = [
        (e["epoch"], e["eval_loss"])
        for e in trainer.state.log_history
        if "eval_loss" in e and "epoch" in e
    ]
    if eval_losses:
        best_epoch = int(min(eval_losses, key=lambda t: t[1])[0])

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    peak_mem = 0
    if torch.cuda.is_available():
        peak_mem = int(torch.cuda.max_memory_allocated())

    return SftResult(
        adapter_dir=str(adapter_dir),
        best_epoch=best_epoch,
        history=history,
        n_dropped_overlength=dropped,
        n_train=len(examples),
        n_val=len(val_examples),
        train_time_s=train_time,
        peak_gpu_mem_bytes=peak_mem,
        n_trainable_params=n_trainable,
        n_total_params=n_total,
        adapter_bytes=adapter_bytes,
    )
