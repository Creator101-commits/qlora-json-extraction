"""Benchmark evaluation runner across extraction conditions."""

from __future__ import annotations

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

    for doc in documents:
        schema, _, _ = extractor.extract(doc.ocr_text)
        predictions.append(schema)
        ground_truths.append(doc.ground_truth)

    return compute_extraction_metrics(predictions, ground_truths)
