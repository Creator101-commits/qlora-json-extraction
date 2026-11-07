"""Regression checks for truthful extraction, exact matching, and real OCR inputs."""

import importlib.util
import json
from pathlib import Path

import pytest

from qlora_extraction.evaluation.metrics import compute_extraction_metrics
from qlora_extraction.models.inference import ReceiptExtractor
from qlora_extraction.schema import ReceiptItem, ReceiptSchema, parse_and_validate_receipt_json


def test_total_is_not_tendered_cash_and_unknown_date_is_null():
    schema, valid, _ = ReceiptExtractor().extract(
        'SHOP "CENTRAL"\nITEM $7.50\nTOTAL $7.50\nCASH $20.00'
    )
    assert valid and schema.total == 7.5 and schema.transaction_date is None
    assert schema.merchant_name == 'Shop "Central"'
    assert len(schema.line_items) == 1


def test_missing_total_and_invalid_date_are_invalid_without_crash():
    for text in ["SHOP\nITEM $4.00", "SHOP\nDATE 2026-02-30\nTOTAL $4.00"]:
        schema, valid, _ = ReceiptExtractor().extract(text)
        assert schema is None and not valid


def test_schema_rejects_extra_keys_and_nonfinite_numbers():
    for payload in [
        dict(merchant_name="Store", transaction_date=None, total=5, unknown="x"),
        dict(merchant_name="Store", transaction_date=None, total=float("inf")),
    ]:
        assert not parse_and_validate_receipt_json(json.dumps(payload))[1]


def test_json_repair_preserves_brackets_and_commas_inside_quoted_text():
    raw = '{"merchant_name":"Shop {,}","transaction_date":null,"total":5,"line_items":[]'
    schema, valid, _ = parse_and_validate_receipt_json(raw)
    assert valid and schema.merchant_name == "Shop {,}"


def test_line_item_matching_counts_quantities_and_duplicate_rows():
    a = ReceiptItem(description="Coffee", quantity=1, price=2.0)
    truth = ReceiptSchema(
        merchant_name="Store", transaction_date=None, total=4.0, line_items=[a, a]
    )
    prediction = truth.model_copy(update={"merchant_name": "Sto", "line_items": [a]})
    metrics = compute_extraction_metrics([prediction], [truth])
    assert metrics.merchant_exact_match == 0.0
    assert metrics.line_item_f1 == pytest.approx(2 / 3)
    assert metrics.omission_rate == 1.0
    wrong_quantity = truth.model_copy(update={"line_items": [a.model_copy(update={"quantity": 2})]})
    assert compute_extraction_metrics([wrong_quantity], [truth]).line_item_f1 == 0.0


def test_json_parse_success_is_distinct_from_schema_success():
    truth = ReceiptSchema(merchant_name="Store", transaction_date=None, total=4.0)
    metric = compute_extraction_metrics([None], [truth], json_parsed=[True])
    assert metric.json_parse_rate == 1 and metric.schema_validity_rate == 0
    with pytest.raises(ValueError):
        compute_extraction_metrics([], [truth])


def test_kaggle_formatter_uses_ocr_words_and_only_native_target():
    path = Path(__file__).resolve().parents[1] / "kaggle/train_kaggle.py"
    spec = importlib.util.spec_from_file_location("cord_training", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ocr, target = module.format_cord(
        {
            "id": "not-ocr",
            "ground_truth": json.dumps(
                {
                    "valid_line": [
                        {"words": [{"text": "COFFEE"}, {"text": "10,000"}], "category": "menu"}
                    ],
                    "gt_parse": {"menu": {"nm": "COFFEE", "price": "10,000"}},
                    "meta": {"secret": "ignored"},
                }
            ),
        }
    )
    assert ocr == "COFFEE 10,000" and "not-ocr" not in ocr and "category" not in ocr
    assert json.loads(target) == {"menu": {"nm": "COFFEE", "price": "10,000"}}
