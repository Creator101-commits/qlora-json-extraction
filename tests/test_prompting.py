"""Tests for prompt formatting functions."""

from __future__ import annotations

from qlora_extraction.data.cord import load_cord_dataset
from qlora_extraction.models.prompting import (
    format_few_shot_prompt,
    format_qlora_instruction_prompt,
    format_zero_shot_prompt,
)


def test_format_zero_shot_prompt() -> None:
    ocr = "STORE 123\nTOTAL $10.00"
    prompt = format_zero_shot_prompt(ocr)
    assert "--- OCR TEXT ---" in prompt
    assert ocr in prompt
    assert "merchant_name" in prompt


def test_format_few_shot_prompt() -> None:
    docs = load_cord_dataset()
    ocr = "STORE 456\nTOTAL $20.00"
    prompt = format_few_shot_prompt(ocr, docs[:2])
    assert "### Example 1" in prompt
    assert "### Target Document" in prompt
    assert ocr in prompt


def test_format_qlora_instruction_prompt() -> None:
    ocr = "STORE 789\nTOTAL $30.00"
    prompt = format_qlora_instruction_prompt(ocr)
    assert "<|start_header_id|>system<|end_header_id|>" in prompt
    assert ocr in prompt
