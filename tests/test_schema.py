"""Tests for the CordParseV1 schema, canonicalization, and key rejection."""

from __future__ import annotations

import json

from qlora_extraction.schema import (
    CordParseV1,
    canonicalize_cord_parse,
    has_forbidden_keys,
    validate_cord_parse,
)


def _target(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "menu": [{"nm": "Nasi", "cnt": "1", "price": "75,000"}],
        "sub_total": {"subtotal_price": "75,000"},
        "total": {"total_price": "75,000", "cashprice": "80,000"},
    }
    base.update(overrides)
    return base


def test_canonicalize_wraps_dict_menu_and_sub() -> None:
    raw = {
        "menu": {"nm": "TEH", "cnt": "2", "sub": {"nm": "SUSU", "price": "5,000"}},
        "total": {"total_price": "10,000"},
    }
    out = canonicalize_cord_parse(raw)
    assert isinstance(out["menu"], list)
    assert out["menu"][0]["sub"] == [{"nm": "SUSU", "price": "5,000"}]


def test_canonicalize_leaves_lists_and_scalars() -> None:
    raw = {"sub_total": {"etc": ["1,000", "2,000"]}, "menu": []}
    out = canonicalize_cord_parse(raw)
    assert out["sub_total"]["etc"] == ["1,000", "2,000"]


def test_validate_accepts_train_shape_with_optional_groups() -> None:
    parsed = validate_cord_parse(_target())
    assert parsed is not None
    assert parsed.menu[0].nm == "Nasi"


def test_validate_accepts_sub_total_as_list() -> None:
    target = _target(sub_total=[{"subtotal_price": "1"}, {"tax_price": "2"}])
    assert validate_cord_parse(target) is not None


def test_forbidden_top_level_key_rejected() -> None:
    target = _target()
    target["merchant_name"] = "X"
    assert validate_cord_parse(target) is None
    assert has_forbidden_keys(target) is True


def test_forbidden_nested_key_rejected() -> None:
    target = _target()
    target["total"]["store_name"] = "X"  # type: ignore[index]
    assert validate_cord_parse(target) is None
    assert has_forbidden_keys(target) is True


def test_missing_menu_rejected() -> None:
    target = _target()
    del target["menu"]
    assert validate_cord_parse(target) is None


def test_void_menu_single_dict_canonicalizes_to_list() -> None:
    target = _target(void_menu={"nm": "TEH BOTOL", "price": "4,000"})
    canonical = canonicalize_cord_parse(target)
    assert canonical["void_menu"] == [{"nm": "TEH BOTOL", "price": "4,000"}]
    assert validate_cord_parse(target) is not None


def test_schema_json_is_compact_object() -> None:
    schema = CordParseV1.model_json_schema()
    assert schema["additionalProperties"] is False
    minified = json.dumps(schema, separators=(",", ":"), sort_keys=True)
    assert '"menu"' in minified
