"""CUDA-only QLoRA pilot: CORD OCR words -> native CORD gt_parse JSON.

The native CORD target differs from the local synthetic ReceiptSchema.
No test partition is loaded, and this script does not publish test accuracy.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


def format_cord(example: dict[str, Any]) -> tuple[str, str]:
    """Extract only transcribed words as input, keeping annotation labels out."""
    annotation = json.loads(example["ground_truth"])
    lines = [
        " ".join(word["text"] for word in line.get("words", []))
        for line in annotation.get("valid_line", [])
    ]
    ocr = "\n".join(line for line in lines if line.strip())
    target = annotation.get("gt_parse")
    if not ocr or not isinstance(target, dict) or not target:
        raise ValueError("CORD row must contain OCR words and a nonempty gt_parse target.")
    return ocr, json.dumps(target, ensure_ascii=False, sort_keys=True)


def main() -> None:
    # Optional GPU libraries belong to the Kaggle environment, not the CPU package.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "peft", "bitsandbytes"])
    import torch
    from datasets import Dataset, load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("This 4-bit training job requires a CUDA GPU.")
    set_seed(42)
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        ),
        device_map={"": 0},
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    dataset_id = "naver-clova-ix/cord-v2"
    dataset_revision = "41b075798d0d9da590a950dac1e3d4a21826d4fd"
    train = load_dataset(dataset_id, revision=dataset_revision, split="train")
    validation = load_dataset(dataset_id, revision=dataset_revision, split="validation")
    fingerprints: set[str] = set()

    def encode_rows(rows: Any, training: bool) -> list[dict[str, list[int]]]:
        encoded = []
        for row in rows:
            ocr, target = format_cord(row)
            key = hashlib.sha256(re.sub(r"\s+", " ", ocr.casefold()).encode()).hexdigest()
            if key in fingerprints:
                continue
            if training:
                fingerprints.add(key)
            messages = [
                {"role": "system", "content": "Return only native CORD gt_parse JSON from OCR."},
                {"role": "user", "content": ocr},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
            ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
            if len(ids) > 1024:
                continue  # Never silently train on a truncated JSON target.
            encoded.append(
                {
                    "input_ids": ids,
                    "attention_mask": [1] * len(ids),
                    "labels": [-100] * len(prompt_ids) + target_ids + [tokenizer.eos_token_id],
                }
            )
        if not encoded:
            raise ValueError("No complete training sequences fit the token budget.")
        return encoded

    train_rows = encode_rows(train, True)
    validation_rows = encode_rows(validation, False)

    def collate(rows: list[dict[str, list[int]]]) -> dict[str, Any]:
        width = max(len(row["input_ids"]) for row in rows)
        return {
            key: torch.tensor([row[key] + [pad] * (width - len(row[key])) for row in rows])
            for key, pad in [
                ("input_ids", tokenizer.pad_token_id),
                ("attention_mask", 0),
                ("labels", -100),
            ]
        }

    trainer = Trainer(
        model=model,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(validation_rows),
        data_collator=collate,
        args=TrainingArguments(
            output_dir="./adapters/qlora-receipt-tinyllama",
            max_steps=50,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            warmup_steps=10,
            fp16=False,
            bf16=False,
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=25,
            save_strategy="steps",
            save_steps=25,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            seed=42,
        ),
    )
    trainer.train()
    output = Path("./artifacts/qlora_cord_adapter")
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    trainable, total = model.get_nb_trainable_parameters()
    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "model_revision": model.config._commit_hash,
                "dataset_id": dataset_id,
                "dataset_revision": dataset_revision,
                "train_examples": len(train_rows),
                "validation_examples": len(validation_rows),
                "train_ocr_sha256": sorted(fingerprints),
                "seed": 42,
                "max_steps": 50,
                "trainable_parameters": trainable,
                "total_parameters": total,
                "target_schema": "native CORD gt_parse (not ReceiptSchema)",
                "deduplication": "exact normalized OCR; near-duplicate audit remains required",
                "test_evaluated": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Saved validation-selected CORD adapter; no test accuracy was measured.")


if __name__ == "__main__":
    main()
