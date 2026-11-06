"""Kaggle GPU training script for 4-bit QLoRA JSON extraction on receipt OCR."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

# Ensure required libraries are installed in Kaggle environment
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "trl", "peft", "bitsandbytes"])

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing QLoRA fine-tuning on device: {device}")

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"

    # 4-bit NF4 Quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load 4-bit base model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA Adapter Configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load CORD OCR dataset
    dataset = load_dataset("naver-clova-ix/cord-v2")

    def format_cord(example: dict[str, Any]) -> dict[str, str]:
        return {"text": f"Receipt OCR:\n{example.get('id', '')}\nJSON:\n{example.get('ground_truth', '')}"}

    train_data = dataset["train"].map(format_cord)
    val_data = dataset["validation"].map(format_cord)

    sft_config = SFTConfig(
        output_dir="./adapters/qlora-receipt-qwen",
        dataset_text_field="text",
        max_seq_length=1024,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=50,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        logging_steps=5,
        save_strategy="steps",
        save_steps=25,
        report_to="none",
    )

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "train_dataset": train_data,
        "eval_dataset": val_data,
        "args": sft_config,
    }
    try:
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = SFTTrainer(tokenizer=tokenizer, **trainer_kwargs)

    print("Beginning 4-bit QLoRA Supervised Fine-Tuning...")
    trainer.train()

    os.makedirs("./artifacts", exist_ok=True)
    model.save_pretrained("./artifacts/qlora_cord_adapter")
    tokenizer.save_pretrained("./artifacts/qlora_cord_adapter")
    print("Fine-tuning complete. LoRA adapter artifacts successfully saved.")


if __name__ == "__main__":
    main()
