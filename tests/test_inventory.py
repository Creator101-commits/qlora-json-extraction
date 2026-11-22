"""Tests for the schema inventory audit and gap reporting."""

from __future__ import annotations

import json

from qlora_extraction.data.cord import CordDocument
from qlora_extraction.data.inventory import audit_inventories, inventory_for_docs


def _doc(doc_id: str, target: dict[str, object]) -> CordDocument:
    return CordDocument(doc_id, "train", "OCR", json.dumps(target))


def test_inventory_counts_paths_and_types() -> None:
    docs = [
        _doc(
            "train:1",
            {
                "menu": [{"nm": "A", "price": "1"}, {"nm": "B"}],
                "sub_total": {"etc": ["x", "y"]},
                "total": {"total_price": "2"},
            },
        )
    ]
    inv = inventory_for_docs(docs)
    assert inv["n_docs"] == 1
    paths = inv["paths"]
    assert paths["menu[]"]["list"] == 1
    assert paths["menu[].nm"]["str"] == 2
    assert paths["menu[].price"]["str"] == 1
    assert paths["sub_total.etc[]"]["str"] == 2
    assert paths["total.total_price"]["str"] == 1


def test_schema_gaps_report_val_only_keys() -> None:
    train = [_doc("train:1", {"menu": [], "total": {"total_price": "1"}})]
    val = [
        _doc(
            "validation:1",
            {"menu": [], "total": {"total_price": "1", "new_key": "x"}},
        )
    ]
    report = audit_inventories(train, {"validation": val})
    gaps = report["splits"]["validation"]["gaps"]
    assert gaps["missing_paths"] == ["total.new_key"]
    assert gaps["type_mismatches"] == {}


def test_schema_gaps_empty_when_subset() -> None:
    train = [_doc("train:1", {"menu": [{"nm": "A"}], "total": {"total_price": "1"}})]
    test = [_doc("test:1", {"menu": [], "total": {}})]
    report = audit_inventories(train, {"test": test})
    gaps = report["splits"]["test"]["gaps"]
    assert gaps["missing_paths"] == []
