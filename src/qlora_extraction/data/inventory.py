"""Schema inventory audit over CORD ``gt_parse`` targets.

The inventory walks canonicalized targets and records every key path with
the value types observed. The schema is defined from TRAIN only; paths seen
only in validation or test are reported as schema gaps.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from qlora_extraction.data.cord import CordDocument


def _walk(obj: Any, path: str, out: Counter[tuple[str, str]]) -> None:
    """Collect ``(key_path, json_type)`` counts for every node."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{path}.{key}" if path else str(key)
            _walk(value, child, out)
    elif isinstance(obj, list):
        out[(f"{path}[]", "list")] += 1
        for item in obj:
            _walk(item, f"{path}[]", out)
    else:
        out[(path, type(obj).__name__)] += 1


def inventory_for_docs(docs: list[CordDocument]) -> dict[str, Any]:
    """Per-split key-path inventory with counts and value types.

    Materializes targets, so callers must only pass splits whose ground
    truth may be read at that point in the protocol.
    """
    counts: Counter[tuple[str, str]] = Counter()
    n_docs = 0
    for doc in docs:
        target = doc.get_target()
        n_docs += 1
        _walk(target, "", counts)
    paths: dict[str, dict[str, int]] = {}
    for (path, json_type), count in sorted(counts.items()):
        paths.setdefault(path, {})[json_type] = count
    return {"n_docs": n_docs, "paths": paths}


def schema_gaps(
    reference_inventory: dict[str, Any], other_inventory: dict[str, Any]
) -> dict[str, Any]:
    """Key paths (and type mismatches) present outside the reference split.

    ``reference_inventory`` is the TRAIN-derived inventory that defines the
    schema; ``other_inventory`` is validation or test. Paths in the other
    split but absent from the reference are schema gaps.
    """
    ref_paths = set(reference_inventory["paths"])
    other_paths = set(other_inventory["paths"])
    gaps = sorted(other_paths - ref_paths)
    shared = other_paths & ref_paths
    type_mismatches = {
        path: {
            "reference": reference_inventory["paths"][path],
            "other": other_inventory["paths"][path],
        }
        for path in sorted(shared)
        if set(reference_inventory["paths"][path]) != set(other_inventory["paths"][path])
    }
    return {"missing_paths": gaps, "type_mismatches": type_mismatches}


def audit_inventories(
    train_docs: list[CordDocument],
    other_docs: dict[str, list[CordDocument]],
) -> dict[str, Any]:
    """Full audit: train inventory plus per-split gap reports."""
    train_inv = inventory_for_docs(train_docs)
    report: dict[str, Any] = {"train": train_inv, "splits": {}}
    for split, docs in other_docs.items():
        inv = inventory_for_docs(docs)
        report["splits"][split] = {"inventory": inv, "gaps": schema_gaps(train_inv, inv)}
    return report


def inventory_to_json(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2)
