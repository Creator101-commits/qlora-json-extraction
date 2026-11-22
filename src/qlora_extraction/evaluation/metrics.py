"""Per-document and aggregate extraction metrics.

Field-level scoring flattens each canonicalized parse to a multiset of
``(key_path, normalized_leaf_value)`` pairs; indices are dropped so menu
order does not affect field F1. Document exact match compares normalized
JSON serializations with sorted keys, so list order still matters there.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from qlora_extraction.data.cord import CordDocument
from qlora_extraction.evaluation.normalize import normalize_leaf, normalize_tree
from qlora_extraction.evaluation.nted import nted_score
from qlora_extraction.inference.parse import ParsedOutput
from qlora_extraction.schema import (
    canonicalize_cord_parse,
    has_forbidden_keys,
    validate_cord_parse,
)

TOP_LEVEL_GROUPS = ("menu", "sub_total", "total")


def flatten_leaves(obj: Any, path: str = "") -> Counter[tuple[str, str]]:
    """Multiset of ``(key_path, normalized_value)`` leaf pairs.

    Dict children extend the path with ``.key``; list elements recurse at
    the same path so list order and cardinality are preserved as repeated
    pairs while indices never appear.
    """
    out: Counter[tuple[str, str]] = Counter()
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{path}.{key}" if path else str(key)
            out.update(flatten_leaves(value, child))
    elif isinstance(obj, list):
        for item in obj:
            out.update(flatten_leaves(item, path))
    else:
        out[(path, normalize_leaf(obj))] += 1
    return out


def _group_of(path: str) -> str | None:
    head = path.split(".", 1)[0].split("[", 1)[0]
    return head if head in TOP_LEVEL_GROUPS else None


@dataclass
class DocScore:
    """All per-document metric inputs needed for aggregation and bootstrap."""

    doc_id: str
    parse_ok: bool
    schema_valid: bool
    forbidden_keys: bool
    has_commentary: bool
    n_pred_pairs: int
    n_ref_pairs: int
    tp: int
    fp: int
    fn: int
    group_counts: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    nted: float = 0.0
    exact_match: bool = False
    hallucinated: int = 0
    omitted: int = 0


def _normalized_json(obj: Any) -> str:
    return json.dumps(
        normalize_tree(obj), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def score_doc(doc: CordDocument, parsed: ParsedOutput) -> DocScore:
    """Score one parsed prediction against the document's canonical target."""
    ref = doc.get_target()
    ref_pairs = flatten_leaves(ref)
    base = DocScore(
        doc_id=doc.doc_id,
        parse_ok=parsed.parse_ok,
        schema_valid=False,
        forbidden_keys=False,
        has_commentary=parsed.has_commentary,
        n_pred_pairs=0,
        n_ref_pairs=sum(ref_pairs.values()),
        tp=0,
        fp=0,
        fn=sum(ref_pairs.values()),
        nted=0.0,
        exact_match=False,
        hallucinated=0,
        omitted=sum(ref_pairs.values()),
    )
    if not parsed.parse_ok or parsed.obj is None:
        return base

    canonical_pred = canonicalize_cord_parse(parsed.obj)
    schema_ok = validate_cord_parse(parsed.obj) is not None
    forbidden = has_forbidden_keys(parsed.obj)
    pred_pairs = flatten_leaves(canonical_pred)

    tp = sum((pred_pairs & ref_pairs).values())
    fp = sum((pred_pairs - ref_pairs).values())
    fn = sum((ref_pairs - pred_pairs).values())

    group_counts: dict[str, list[int]] = {g: [0, 0, 0] for g in TOP_LEVEL_GROUPS}
    for (path, _), count in (pred_pairs & ref_pairs).items():
        g = _group_of(path)
        if g:
            group_counts[g][0] += count
    for (path, _), count in (pred_pairs - ref_pairs).items():
        g = _group_of(path)
        if g:
            group_counts[g][1] += count
    for (path, _), count in (ref_pairs - pred_pairs).items():
        g = _group_of(path)
        if g:
            group_counts[g][2] += count

    base.schema_valid = schema_ok
    base.forbidden_keys = forbidden
    base.n_pred_pairs = sum(pred_pairs.values())
    base.tp = tp
    base.fp = fp
    base.fn = fn
    base.group_counts = {g: (v[0], v[1], v[2]) for g, v in group_counts.items()}
    base.nted = nted_score(canonical_pred, ref)
    base.exact_match = _normalized_json(canonical_pred) == _normalized_json(ref)
    base.hallucinated = fp
    base.omitted = fn
    return base


def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def aggregate_scores(scores: list[DocScore]) -> dict[str, Any]:
    """Aggregate per-document scores into the reported metric block."""
    n = len(scores)
    tp = sum(s.tp for s in scores)
    fp = sum(s.fp for s in scores)
    fn = sum(s.fn for s in scores)
    precision, recall, f1 = _f1(tp, fp, fn)

    groups: dict[str, Any] = {}
    for group in TOP_LEVEL_GROUPS:
        gtp = sum(s.group_counts.get(group, (0, 0, 0))[0] for s in scores)
        gfp = sum(s.group_counts.get(group, (0, 0, 0))[1] for s in scores)
        gfn = sum(s.group_counts.get(group, (0, 0, 0))[2] for s in scores)
        gp, gr, gf1 = _f1(gtp, gfp, gfn)
        if gtp + gfp + gfn == 0:
            gf1 = 1.0
        groups[group] = {"precision": gp, "recall": gr, "f1": gf1}

    halluc = [
        s.hallucinated / s.n_pred_pairs if s.n_pred_pairs else 0.0 for s in scores
    ]
    omit = [s.omitted / s.n_ref_pairs if s.n_ref_pairs else 0.0 for s in scores]

    return {
        "n_docs": n,
        "parse_rate": sum(s.parse_ok for s in scores) / n if n else 0.0,
        "schema_valid_rate": sum(s.schema_valid for s in scores) / n if n else 0.0,
        "forbidden_key_rate": sum(s.forbidden_keys for s in scores) / n if n else 0.0,
        "extra_commentary_rate": (
            sum(s.has_commentary for s in scores) / n if n else 0.0
        ),
        "field_precision": precision,
        "field_recall": recall,
        "field_f1": f1,
        "nted": sum(s.nted for s in scores) / n if n else 0.0,
        "exact_match_rate": sum(s.exact_match for s in scores) / n if n else 0.0,
        "per_group_f1": groups,
        "hallucination_rate": sum(halluc) / n if n else 0.0,
        "omission_rate": sum(omit) / n if n else 0.0,
    }
