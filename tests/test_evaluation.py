"""Tests for evaluation metrics and benchmark harness."""

from __future__ import annotations

import pytest

from qlora_extraction.data.cord import load_cord_dataset
from qlora_extraction.evaluation.metrics import compute_extraction_metrics
from qlora_extraction.evaluation.runner import evaluate_extraction_condition
from qlora_extraction.schema import ReceiptItem, ReceiptSchema


def test_compute_extraction_metrics_perfect() -> None:
    schema = ReceiptSchema(
        merchant_name="Apple Store",
        transaction_date="2026-06-01",
        currency="USD",
        subtotal=100.0,
        tax=8.0,
        total=108.0,
        line_items=[ReceiptItem(description="Case", quantity=1, price=40.0)],
    )

    metrics = compute_extraction_metrics([schema], [schema])
    assert metrics.json_parse_rate == pytest.approx(1.0)
    assert metrics.schema_validity_rate == pytest.approx(1.0)
    assert metrics.merchant_exact_match == pytest.approx(1.0)
    assert metrics.date_exact_match == pytest.approx(1.0)
    assert metrics.total_exact_match == pytest.approx(1.0)
    assert metrics.line_item_f1 == pytest.approx(1.0)
    assert metrics.macro_field_f1 == pytest.approx(1.0)
    assert metrics.hallucination_rate == pytest.approx(0.0)
    assert metrics.omission_rate == pytest.approx(0.0)


def test_compute_extraction_metrics_none() -> None:
    schema = ReceiptSchema(
        merchant_name="Store",
        transaction_date="2026-06-01",
        total=10.0,
    )
    metrics = compute_extraction_metrics([None], [schema])
    assert metrics.json_parse_rate == pytest.approx(0.0)
    assert metrics.omission_rate == pytest.approx(1.0)


def test_evaluate_extraction_condition() -> None:
    docs = load_cord_dataset()
    metrics = evaluate_extraction_condition("qlora", docs)
    assert 0.0 <= metrics.json_parse_rate <= 1.0
    assert 0.0 <= metrics.macro_field_f1 <= 1.0
