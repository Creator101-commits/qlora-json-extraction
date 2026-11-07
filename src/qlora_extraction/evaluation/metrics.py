"""Receipt metrics on aligned predictions, including multiplicity and quantities."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal

import numpy as np

from qlora_extraction.schema import ReceiptSchema


@dataclass(frozen=True)
class ExtractionMetrics:
    """Rates are per receipt; line F1 matches description, quantity, and price."""

    json_parse_rate: float
    schema_validity_rate: float
    merchant_exact_match: float
    date_exact_match: float
    total_exact_match: float
    line_item_f1: float
    macro_field_f1: float
    hallucination_rate: float
    omission_rate: float


def compute_extraction_metrics(
    predictions: list[ReceiptSchema | None],
    ground_truths: list[ReceiptSchema],
    json_parsed: list[bool] | None = None,
) -> ExtractionMetrics:
    """Score canonical values; raw parse flags distinguish syntax from schema failures.

    Without raw flags, parse rate is a lower bound from validated predictions.
    Hallucination means an extra/mismatched line or incorrect supplied scalar;
    omission means an invalid receipt, missing scalar, or unmatched target line.
    """
    n = len(ground_truths)
    if len(predictions) != n or (json_parsed is not None and len(json_parsed) != n):
        raise ValueError("Predictions, truths, and parse flags must be aligned.")
    if n == 0:
        return ExtractionMetrics(*([0.0] * 9))
    valid = np.array([p is not None for p in predictions])
    parsed = np.asarray(json_parsed, dtype=bool) if json_parsed is not None else valid
    if np.any(valid & ~parsed):
        raise ValueError("A schema-valid prediction must also be parsed JSON.")

    merchant_matches = date_matches = total_matches = hallucinations = omissions = 0
    line_scores: list[float] = []

    def norm(value: str) -> str:
        return " ".join(value.casefold().split())

    def cents(value: float) -> Decimal:
        return Decimal(str(value)).quantize(Decimal("0.01"))

    for pred, gt in zip(predictions, ground_truths, strict=True):
        if pred is None:
            omissions += 1
            line_scores.append(0.0)
            continue
        merchant_ok = norm(pred.merchant_name) == norm(gt.merchant_name)
        date_ok = pred.transaction_date == gt.transaction_date
        total_ok = cents(pred.total) == cents(gt.total)
        merchant_matches += merchant_ok
        date_matches += date_ok
        total_matches += total_ok
        pred_items = Counter(
            (norm(it.description), it.quantity, cents(it.price)) for it in pred.line_items
        )
        true_items = Counter(
            (norm(it.description), it.quantity, cents(it.price)) for it in gt.line_items
        )
        tp = sum((pred_items & true_items).values())
        n_pred, n_true = sum(pred_items.values()), sum(true_items.values())
        line_scores.append(2 * tp / (n_pred + n_true) if n_pred + n_true else 1.0)
        hallucinations += int(
            n_pred > tp
            or not merchant_ok
            or not total_ok
            or (pred.transaction_date is not None and not date_ok)
        )
        omissions += int(
            n_true > tp or (gt.transaction_date is not None and pred.transaction_date is None)
        )

    line_f1 = float(np.mean(line_scores))
    return ExtractionMetrics(
        json_parse_rate=float(np.mean(parsed)),
        schema_validity_rate=float(np.mean(valid)),
        merchant_exact_match=merchant_matches / n,
        date_exact_match=date_matches / n,
        total_exact_match=total_matches / n,
        line_item_f1=line_f1,
        macro_field_f1=(merchant_matches / n + date_matches / n + total_matches / n + line_f1) / 4,
        hallucination_rate=hallucinations / n,
        omission_rate=omissions / n,
    )
