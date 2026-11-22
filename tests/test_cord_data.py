"""Tests for CORD record loading, OCR reconstruction, and lazy targets."""

from __future__ import annotations

import json

from qlora_extraction.data.cord import (
    CordDocument,
    cord_documents_from_records,
    normalized_ocr_sha256,
    ocr_text_from_valid_line,
)


def _gt(valid_line: list[dict[str, object]], gt_parse: dict[str, object]) -> str:
    return json.dumps(
        {
            "gt_parse": gt_parse,
            "valid_line": valid_line,
            "meta": {"image_id": 7, "split": "train"},
        }
    )


def _line(y: int, x: int, *texts: str) -> dict[str, object]:
    return {
        "words": [
            {
                "text": t,
                "quad": {
                    "x1": x + i * 50,
                    "y1": y,
                    "x2": x + i * 50 + 40,
                    "y2": y,
                    "x3": x + i * 50 + 40,
                    "y3": y + 20,
                    "x4": x + i * 50,
                    "y4": y + 20,
                },
            }
            for i, t in enumerate(texts)
        ],
        "category": "menu.nm",
    }


def test_ocr_reconstruction_sorts_lines_and_words() -> None:
    valid_line = [
        _line(100, 300, "TOTAL", "10,000"),
        _line(50, 200, "TEH", "BOTOL"),
        _line(50, 20, "KOPI"),
    ]
    text = ocr_text_from_valid_line(valid_line)
    assert text.split("\n") == ["KOPI", "TEH BOTOL", "TOTAL 10,000"]


def test_ocr_skips_empty_lines() -> None:
    valid_line = [_line(10, 10, "A"), {"words": [], "category": "x"}]
    assert ocr_text_from_valid_line(valid_line) == "A"


def test_documents_from_records_lazy_target() -> None:
    gt_parse = {"menu": {"nm": "TEH", "cnt": "1"}, "total": {"total_price": "5,000"}}
    records = [{"ground_truth": _gt([_line(1, 1, "TEH")], gt_parse)}]
    docs = cord_documents_from_records(records, "train")
    assert len(docs) == 1
    doc = docs[0]
    assert doc.doc_id == "train:7"
    assert doc.split == "train"
    assert doc.ocr_text == "TEH"
    assert doc.target is None
    target = doc.get_target()
    assert target["menu"] == [{"nm": "TEH", "cnt": "1"}]
    assert doc.target_json == json.dumps(target, ensure_ascii=False, sort_keys=True)


def test_duplicate_doc_ids_rejected() -> None:
    gt_parse: dict[str, object] = {"menu": []}
    records = [
        {"ground_truth": _gt([], gt_parse)},
        {"ground_truth": _gt([], gt_parse)},
    ]
    try:
        cord_documents_from_records(records, "train")
    except ValueError as exc:
        assert "duplicate doc_id" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_normalized_ocr_sha256_ignores_whitespace() -> None:
    a = CordDocument("s:1", "train", "TEH  BOTOL\nKOPI", "{}")
    b = CordDocument("s:2", "train", "  TEH BOTOL KOPI  ", "{}")
    assert normalized_ocr_sha256(a.ocr_text) == normalized_ocr_sha256(b.ocr_text)
