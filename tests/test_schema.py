"""Tests for Pydantic receipt schemas, JSON repair, and validation."""

from __future__ import annotations

import json

from qlora_extraction.config import QLoRAConfig
from qlora_extraction.schema import (
    ReceiptItem,
    ReceiptSchema,
    parse_and_validate_receipt_json,
    repair_json_string,
)


def test_qlora_config() -> None:
    cfg = QLoRAConfig()
    assert cfg.lora_r == 16
    assert cfg.lora_alpha == 32
    assert "q_proj" in cfg.target_modules
    assert cfg.quantization_bits == 4


def test_valid_receipt_schema() -> None:
    receipt = ReceiptSchema(
        merchant_name="Cafe Nero",
        transaction_date="2026-05-01",
        currency="USD",
        subtotal=10.0,
        tax=1.0,
        total=11.0,
        line_items=[
            ReceiptItem(description="Espresso", quantity=2, price=6.0),
            ReceiptItem(description="Croissant", quantity=1, price=4.0),
        ],
    )
    assert receipt.merchant_name == "Cafe Nero"
    assert len(receipt.line_items) == 2


def test_repair_json_markdown_and_trailing_commas() -> None:
    raw_markdown = """```json
    {
      "merchant_name": "Bakery",
      "transaction_date": "2026-05-01",
      "total": 5.50,
    }
    ```"""
    repaired = repair_json_string(raw_markdown)
    data = json.loads(repaired)
    assert data["merchant_name"] == "Bakery"
    assert data["total"] == 5.50


def test_repair_json_unclosed_braces() -> None:
    unclosed = (
        '{\n  "merchant_name": "Test Store",\n'
        '  "transaction_date": "2026-05-01",\n  "total": 10.0\n'
    )
    repaired = repair_json_string(unclosed)
    data = json.loads(repaired)
    assert data["merchant_name"] == "Test Store"


def test_parse_and_validate_receipt_json() -> None:
    valid_json = '{"merchant_name": "Target", "transaction_date": "2026-05-01", "total": 20.0}'
    schema, is_valid, err = parse_and_validate_receipt_json(valid_json)
    assert is_valid is True
    assert schema is not None
    assert schema.merchant_name == "Target"
    assert err is None

    invalid_json = '{"merchant_name": "Target"}'  # missing required transaction_date and total
    schema, is_valid, err = parse_and_validate_receipt_json(invalid_json)
    assert is_valid is False
    assert schema is None
    assert "validation error" in str(err).lower()
