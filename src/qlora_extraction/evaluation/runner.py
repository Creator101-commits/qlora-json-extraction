"""Benchmark evaluation runner across extraction conditions."""

from __future__ import annotations

import json

from qlora_extraction.data.cord import ReceiptDocument
from qlora_extraction.evaluation.metrics import ExtractionMetrics, compute_extraction_metrics
from qlora_extraction.models.inference import ReceiptExtractor
from qlora_extraction.schema import ReceiptSchema


def evaluate_extraction_condition(
    condition: str,
    documents: list[ReceiptDocument],
) -> ExtractionMetrics:
    """Run extraction pipeline on a list of receipt documents and compute metrics."""
    extractor = ReceiptExtractor(condition=condition)
    predictions: list[ReceiptSchema | None] = []
    ground_truths: list[ReceiptSchema] = []
    json_parsed: list[bool] = []

    for doc in documents:
        schema, _, raw = extractor.extract(doc.ocr_text)
        try:
            json.loads(raw)
            json_parsed.append(True)
        except json.JSONDecodeError:
            json_parsed.append(False)
        predictions.append(schema)
        ground_truths.append(doc.ground_truth)

    return compute_extraction_metrics(predictions, ground_truths, json_parsed)
