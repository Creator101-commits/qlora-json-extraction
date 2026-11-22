"""Versioned Pydantic schema for canonicalized CORD-v2 ``gt_parse`` output.

The key inventory was audited on the TRAIN split only. Validation and test
splits are checked against this schema and any keys they contain that the
schema rejects are reported as schema gaps, never added to the schema.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

CordScalar = str | list[str]
"""Leaf values are strings; a few gt_parse leaves are lists of strings."""

_KEYS_WRAPPED_AS_LIST = frozenset({"menu", "sub", "void_menu"})
"""CORD stores these fields as either a single dict or a list of dicts."""


class CordSubItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nm: CordScalar | None = None
    cnt: CordScalar | None = None
    price: CordScalar | None = None
    unitprice: CordScalar | None = None


class CordMenuItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nm: CordScalar | None = None
    cnt: CordScalar | None = None
    price: CordScalar | None = None
    unitprice: CordScalar | None = None
    num: CordScalar | None = None
    discountprice: CordScalar | None = None
    itemsubtotal: CordScalar | None = None
    etc: CordScalar | None = None
    vatyn: CordScalar | None = None
    sub: list[CordSubItem] | None = None


class CordSubTotal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subtotal_price: CordScalar | None = None
    service_price: CordScalar | None = None
    tax_price: CordScalar | None = None
    etc: CordScalar | None = None
    discount_price: CordScalar | None = None
    othersvc_price: CordScalar | None = None


class CordTotal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_price: CordScalar | None = None
    cashprice: CordScalar | None = None
    changeprice: CordScalar | None = None
    menutype_cnt: CordScalar | None = None
    menuqty_cnt: CordScalar | None = None
    total_etc: CordScalar | None = None
    creditcardprice: CordScalar | None = None
    emoneyprice: CordScalar | None = None


class CordVoidMenuItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nm: CordScalar | None = None
    price: CordScalar | None = None


class CordParseV1(BaseModel):
    """Canonicalized CORD-v2 ``gt_parse`` document (schema version 1)."""

    model_config = ConfigDict(extra="forbid")

    menu: list[CordMenuItem]
    sub_total: CordSubTotal | list[CordSubTotal] | None = None
    total: CordTotal | None = None
    void_menu: list[CordVoidMenuItem] | None = None


def canonicalize_cord_parse(obj: Any) -> Any:
    """Return a copy with dict-valued ``menu``/``sub``/``void_menu`` as lists.

    CORD stores these fields inconsistently: a single entry may appear as a
    bare dict instead of a one-element list. The canonical form always uses
    lists so downstream parsing and scoring see one shape.
    """
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            if key in _KEYS_WRAPPED_AS_LIST and isinstance(value, dict):
                value = [value]
            out[key] = canonicalize_cord_parse(value)
        return out
    if isinstance(obj, list):
        return [canonicalize_cord_parse(item) for item in obj]
    return obj


def key_paths(obj: Any, path: str = "") -> set[str]:
    """All key paths in a parsed object, ``.`` for fields, ``[]`` for lists."""
    paths: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{path}.{key}" if path else str(key)
            paths.add(child)
            paths |= key_paths(value, child)
    elif isinstance(obj, list):
        marker = f"{path}[]" if path else "[]"
        paths.add(marker)
        for item in obj:
            paths |= key_paths(item, marker)
    return paths


def validate_cord_parse(obj: Any) -> CordParseV1 | None:
    """Validate a parsed JSON object against ``CordParseV1``.

    Returns the model on success, ``None`` on any validation failure.
    """
    try:
        return CordParseV1.model_validate(canonicalize_cord_parse(obj))
    except Exception:
        return None


def has_forbidden_keys(obj: Any) -> bool:
    """True when the object contains keys rejected by ``extra="forbid"``."""
    try:
        CordParseV1.model_validate(canonicalize_cord_parse(obj))
    except Exception as exc:
        errors = getattr(exc, "errors", None)
        if callable(errors):
            return any(err.get("type") == "extra_forbidden" for err in errors())
        return False
    return False
