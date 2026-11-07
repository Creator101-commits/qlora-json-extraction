"""Tests for CORD receipt data records and schema integrity."""

from __future__ import annotations

from qlora_extraction.data.cord import load_cord_dataset


def test_load_cord_dataset() -> None:
    docs = load_cord_dataset()
    assert len(docs) >= 5

    splits = {d.split for d in docs}
    assert "train" in splits
    assert "val" in splits
    assert "test" in splits

    for doc in docs:
        assert doc.doc_id.startswith("synthetic_")
        assert len(doc.ocr_text) > 20
        assert doc.ground_truth.total > 0.0
        assert len(doc.ground_truth.merchant_name) > 0
