"""Structured extraction evaluation metrics: field precision, recall, and parse validity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qlora_extraction.schema import ReceiptSchema


@dataclass(frozen=True)
class ExtractionMetrics:
    """Consolidated evaluation metrics for structured receipt extraction."""

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
) -> ExtractionMetrics:
    """Compute structural and field-level extraction metrics over aligned datasets."""
    n = len(ground_truths)
    if n == 0:
        return ExtractionMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    parsed_count = sum(1 for p in predictions if p is not None)
    json_parse_rate = parsed_count / n
    schema_validity_rate = json_parse_rate

    merchant_matches = 0
    date_matches = 0
    total_matches = 0
    line_f1_scores: list[float] = []
    hallucination_counts = 0
    omission_counts = 0

    for pred, gt in zip(predictions, ground_truths, strict=True):
        if pred is None:
            omission_counts += 1
            line_f1_scores.append(0.0)
            continue

        # Merchant match (case-insensitive substring)
        if (
            pred.merchant_name.lower() in gt.merchant_name.lower()
            or gt.merchant_name.lower() in pred.merchant_name.lower()
        ):
            merchant_matches += 1

        # Date match
        if pred.transaction_date == gt.transaction_date:
            date_matches += 1

        # Total amount match within 1 cent tolerance
        if abs(pred.total - gt.total) < 0.02:
            total_matches += 1

        # Line item comparison
        pred_items = {it.description.lower(): it.price for it in pred.line_items}
        gt_items = {it.description.lower(): it.price for it in gt.line_items}

        if not gt_items:
            line_f1 = 1.0 if not pred_items else 0.0
        else:
            tp = sum(
                1 for desc, pr in pred_items.items()
                if desc in gt_items and abs(pr - gt_items[desc]) < 0.02
            )
            fp = len(pred_items) - tp
            fn = len(gt_items) - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            line_f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            if fp > 0:
                hallucination_counts += 1

        line_f1_scores.append(line_f1)

    merchant_em = merchant_matches / n
    date_em = date_matches / n
    total_em = total_matches / n
    mean_line_f1 = float(np.mean(line_f1_scores))

    macro_f1 = float((merchant_em + date_em + total_em + mean_line_f1) / 4.0)
    hallucination_rate = hallucination_counts / n
    omission_rate = omission_counts / n

    return ExtractionMetrics(
        json_parse_rate=float(json_parse_rate),
        schema_validity_rate=float(schema_validity_rate),
        merchant_exact_match=float(merchant_em),
        date_exact_match=float(date_em),
        total_exact_match=float(total_em),
        line_item_f1=mean_line_f1,
        macro_field_f1=macro_f1,
        hallucination_rate=float(hallucination_rate),
        omission_rate=float(omission_rate),
    )
