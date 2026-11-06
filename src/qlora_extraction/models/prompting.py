"""Prompt templates for zero-shot, few-shot, and instruction-tuned extraction."""

from __future__ import annotations

import json

from qlora_extraction.data.cord import ReceiptDocument

PROMPT_SCHEMA_SPEC = """{
  "merchant_name": "string",
  "transaction_date": "YYYY-MM-DD",
  "currency": "USD",
  "subtotal": 0.0,
  "tax": 0.0,
  "total": 0.0,
  "line_items": [
    {"description": "string", "quantity": 1, "price": 0.0}
  ]
}"""


def format_zero_shot_prompt(ocr_text: str) -> str:
    """Zero-shot extraction prompt with schema specification."""
    return (
        "Extract the receipt information from the OCR text into valid JSON matching this schema:\n"
        f"{PROMPT_SCHEMA_SPEC}\n\n"
        f"--- OCR TEXT ---\n{ocr_text}\n\n"
        "--- JSON OUTPUT ---"
    )


def format_few_shot_prompt(ocr_text: str, examples: list[ReceiptDocument]) -> str:
    """Few-shot extraction prompt with demonstrated OCR-to-JSON pairs."""
    blocks: list[str] = [
        "Extract receipt information from OCR text into valid JSON matching target schema.\n"
    ]

    for i, ex in enumerate(examples[:2], 1):
        ex_json = json.dumps(ex.ground_truth.model_dump(), indent=2)
        blocks.append(f"### Example {i}\nOCR:\n{ex.ocr_text}\nJSON:\n{ex_json}\n")

    blocks.append(f"### Target Document\nOCR:\n{ocr_text}\nJSON:")
    return "\n".join(blocks)


def format_qlora_instruction_prompt(ocr_text: str) -> str:
    """Supervised fine-tuning instruction prompt format for Llama-3 / Mistral chat templates."""
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are an expert OCR receipt information extraction model. "
        "Extract the structured receipt data and return ONLY a valid JSON object.<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"Receipt OCR Text:\n{ocr_text}<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
