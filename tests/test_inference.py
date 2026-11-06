"""Tests for receipt extraction inference under different conditions."""

from __future__ import annotations

from qlora_extraction.models.inference import ReceiptExtractor


def test_extractor_zero_shot() -> None:
    ocr = "COFFEE SHOP\n2026-05-10\nTOTAL $7.50"
    extractor = ReceiptExtractor(condition="zero_shot")
    schema, is_valid, raw = extractor.extract(ocr)
    assert is_valid is True
    assert schema is not None
    assert schema.total == 7.50


def test_extractor_few_shot() -> None:
    ocr = "RESTAURANT\n2026-05-11\nTAX $1.50\nTOTAL $18.50"
    extractor = ReceiptExtractor(condition="few_shot")
    schema, is_valid, raw = extractor.extract(ocr)
    assert is_valid is True
    assert schema is not None
    assert schema.total == 18.50
    assert schema.tax == 1.50


def test_extractor_qlora() -> None:
    ocr = "GROCERY\n2026-05-12\nITEM A $4.00\nITEM B $6.00\nTAX $1.00\nTOTAL $11.00"
    extractor = ReceiptExtractor(condition="qlora")
    schema, is_valid, raw = extractor.extract(ocr)
    assert is_valid is True
    assert schema is not None
    assert schema.total == 11.00
    assert len(schema.line_items) == 2
