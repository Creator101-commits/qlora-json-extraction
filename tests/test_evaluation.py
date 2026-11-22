"""Tests for normalization, flattening, field-F1, nTED, and aggregation."""

from __future__ import annotations

import json

import pytest

from qlora_extraction.data.cord import CordDocument
from qlora_extraction.evaluation import (
    aggregate_scores,
    flatten_leaves,
    normalize_value,
    nted_score,
    score_doc,
)
from qlora_extraction.inference.parse import parse_model_output


def _doc(target: dict[str, object]) -> CordDocument:
    return CordDocument("test:1", "test", "OCR", json.dumps(target))


def test_normalize_value_collapses_whitespace_only() -> None:
    assert normalize_value("  TEH   BOTOL \n KOPI ") == "TEH BOTOL KOPI"
    assert normalize_value("Teh123") == "Teh123"  # case and digits untouched


def test_flatten_leaves_multiset_and_groups() -> None:
    obj = {
        "menu": [{"nm": "A", "price": "1"}, {"nm": "B"}],
        "sub_total": {"etc": ["x", "y"]},
    }
    pairs = flatten_leaves(obj)
    assert pairs[("menu.nm", "A")] == 1
    assert pairs[("menu.nm", "B")] == 1
    assert pairs[("sub_total.etc", "x")] == 1
    assert pairs[("sub_total.etc", "y")] == 1
    assert pairs[("menu.price", "1")] == 1


def test_flatten_normalizes_values() -> None:
    pairs = flatten_leaves({"total": {"total_price": "  10,000  "}})
    assert pairs[("total.total_price", "10,000")] == 1


def test_score_doc_perfect_match() -> None:
    target = {"menu": [{"nm": "TEH", "price": "4,000"}], "total": {"total_price": "4,000"}}
    doc = _doc(target)
    pred_text = json.dumps(target)
    parsed = parse_model_output(pred_text)
    score = score_doc(doc, parsed)
    assert score.parse_ok and score.schema_valid and score.nted == pytest.approx(1.0)
    assert score.tp == score.n_ref_pairs and score.fp == 0 and score.fn == 0
    assert score.exact_match


def test_score_doc_missing_field_counts_fn() -> None:
    target = {"menu": [{"nm": "A", "price": "1"}], "total": {"total_price": "1"}}
    doc = _doc(target)
    pred = {"menu": [{"nm": "A"}], "total": {"total_price": "1"}}
    score = score_doc(doc, parse_model_output(json.dumps(pred)))
    # ref has 4 leaves (menu.nm, menu.price, total.total_price, plus? menu[] no leaf)
    assert score.fn == 1  # menu.price omitted
    assert score.tp == score.n_ref_pairs - 1
    assert score.omitted == 1


def test_score_doc_hallucinated_field() -> None:
    target = {"menu": [], "total": {"total_price": "1"}}
    doc = _doc(target)
    pred = {"menu": [], "total": {"total_price": "1", "cashprice": "9"}}
    score = score_doc(doc, parse_model_output(json.dumps(pred)))
    assert score.fp == 1 and score.hallucinated == 1


def test_score_doc_forbidden_key_flagged() -> None:
    target = {"menu": []}
    doc = _doc(target)
    pred = {"menu": [], "bogus": "x"}
    score = score_doc(doc, parse_model_output(json.dumps(pred)))
    assert score.forbidden_keys is True
    assert score.schema_valid is False


def test_score_doc_parse_failure_zero_semantic() -> None:
    doc = _doc({"menu": [{"nm": "A"}], "total": {"total_price": "1"}})
    score = score_doc(doc, parse_model_output("no json here"))
    assert score.parse_ok is False
    assert score.tp == 0 and score.fn == score.n_ref_pairs
    assert score.nted == 0.0 and score.exact_match is False
    assert score.has_commentary is True


def test_aggregate_scores_hand_computed() -> None:
    target = {"menu": [{"nm": "A"}], "total": {"total_price": "1"}}
    good = _doc(target)
    bad_target = {"menu": [{"nm": "B"}], "total": {"total_price": "2"}}
    bad = _doc(bad_target)
    scores = [
        score_doc(good, parse_model_output(good.target_json)),
        score_doc(bad, parse_model_output(good.target_json)),
    ]
    agg = aggregate_scores(scores)
    assert agg["n_docs"] == 2
    assert agg["parse_rate"] == 1.0
    assert agg["exact_match_rate"] == 0.5
    assert agg["nted"] == pytest.approx((1.0 + nted_score(good.get_target(), bad.get_target())) / 2)
    g = agg["per_group_f1"]
    assert "menu" in g and "sub_total" in g and "total" in g


def test_nted_identical_and_changed_leaf() -> None:
    assert nted_score({"a": "1"}, {"a": "1"}) == 1.0
    # {"a":"1"} tree: {} -> a -> "1" = 3 nodes; one leaf relabel -> dist 1
    assert nted_score({"a": "2"}, {"a": "1"}) == pytest.approx(1 - 1 / 3)
    # empty vs non-empty
    assert nted_score({}, {"a": "1"}) < 1.0


def test_commentary_outside_object_flagged() -> None:
    parsed = parse_model_output('Here is the JSON: {"menu": []} hope this helps')
    assert parsed.parse_ok and parsed.obj == {"menu": []}
    assert parsed.has_commentary is True


def test_unbalanced_object_is_parse_failure() -> None:
    parsed = parse_model_output('{"menu": [{"nm": "A"}')
    assert parsed.parse_ok is False
    assert parsed.json_text is None
