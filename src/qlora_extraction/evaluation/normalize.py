"""Leaf-value normalization for extraction metrics.

Frozen rule: strip leading/trailing whitespace and collapse internal
whitespace runs to single spaces. Case and digits are untouched.
"""

from __future__ import annotations

import json
from typing import Any


def normalize_value(value: str) -> str:
    return " ".join(value.split())


def normalize_leaf(value: Any) -> str:
    """Normalize a scalar leaf; non-strings use compact JSON rendering."""
    if isinstance(value, str):
        return normalize_value(value)
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return json.dumps(value)
    return normalize_value(json.dumps(value, ensure_ascii=False, sort_keys=True))


def normalize_tree(obj: Any) -> Any:
    """Apply ``normalize_leaf`` to every scalar leaf, recursively."""
    if isinstance(obj, dict):
        return {key: normalize_tree(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [normalize_tree(item) for item in obj]
    return normalize_leaf(obj)
