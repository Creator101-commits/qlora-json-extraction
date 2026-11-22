"""Merge the two split-kernel outputs into one benchmark result.

Part A and part B each run ``prepare`` + their own jobs + ``collect``
under the same frozen plan. This module proves the parts belong together
(same git SHA, same ``plan_sha256``, disjoint job sets, ``ordering_ok``
in both manifests) and writes the combined ``metrics.json`` /
``manifest.json``. Headline comparisons and bootstraps come from part A;
QLoRA per-seed numbers aggregate seeds 0, 1 and 2 across both parts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

SEED_CONDITIONS = ("qlora_sft_seed0", "qlora_sft_seed1", "qlora_sft_seed2")
BUDGET_POINTS = ("200", "400", "800")


def _load_part(part_dir: Path, label: str) -> dict[str, Any]:
    part_dir = Path(part_dir)
    metrics = json.loads((part_dir / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((part_dir / "manifest.json").read_text(encoding="utf-8"))
    plan = json.loads((part_dir / "analysis_plan.json").read_text(encoding="utf-8"))
    return {"dir": part_dir, "label": label, "metrics": metrics,
            "manifest": manifest, "plan": plan}


def _assert_parts_compatible(a: dict[str, Any], b: dict[str, Any]) -> None:
    ma, mb = a["manifest"], b["manifest"]
    if not ma.get("git_sha") or ma.get("git_sha") != mb.get("git_sha"):
        raise ValueError(
            f"git_sha mismatch or missing: {ma.get('git_sha')!r} vs {mb.get('git_sha')!r}"
        )
    pa = a["plan"].get("plan_sha256")
    pb = b["plan"].get("plan_sha256")
    if not pa or pa != pb:
        raise ValueError(f"plan_sha256 mismatch or missing: {pa!r} vs {pb!r}")
    ja = set(ma.get("conditions_run", []))
    jb = set(mb.get("conditions_run", []))
    overlap = ja & jb
    if overlap:
        raise ValueError(f"job sets are not disjoint, shared: {sorted(overlap)}")
    for part, manifest in ((a["label"], ma), (b["label"], mb)):
        if not manifest.get("ordering_ok"):
            raise ValueError(f"{part} manifest ordering_ok is not true")


def _seed_stats(metrics_a: dict[str, Any], metrics_b: dict[str, Any]) -> dict[str, Any]:
    """Per-seed test aggregates plus mean/sd/range over seeds 0, 1, 2."""
    per_seed: dict[str, Any] = {}
    for cond in SEED_CONDITIONS:
        block = metrics_a["test"].get(cond) or metrics_b["test"].get(cond)
        if block:
            per_seed[cond] = block
    out: dict[str, Any] = {"per_seed": per_seed}
    for metric in ("field_f1", "nted"):
        values = [b[metric] for b in per_seed.values() if metric in b]
        if values:
            arr = np.asarray(values, dtype=np.float64)
            out[f"{metric}_mean"] = float(arr.mean())
            out[f"{metric}_sd"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            out[f"{metric}_min"] = float(arr.min())
            out[f"{metric}_max"] = float(arr.max())
            out[f"{metric}_range"] = float(arr.max() - arr.min())
    return out


def _merge_budget_curve(
    metrics_a: dict[str, Any], metrics_b: dict[str, Any]
) -> dict[str, Any]:
    curve: dict[str, Any] = {}
    for point in BUDGET_POINTS:
        block = (metrics_a.get("budget_curve_validation") or {}).get(point)
        if block is None:
            block = (metrics_b.get("budget_curve_validation") or {}).get(point)
        curve[point] = block
    return curve


def _merge_slices(metrics_a: dict[str, Any], metrics_b: dict[str, Any]) -> dict[str, Any]:
    slices: dict[str, Any] = {"ocr_tertiles": {}, "near_dup": {}}
    for src in (metrics_a, metrics_b):
        s = src.get("slices") or {}
        for tertile, conds in (s.get("ocr_tertiles") or {}).items():
            slices["ocr_tertiles"].setdefault(tertile, {}).update(conds)
        for cond, entry in (s.get("near_dup") or {}).items():
            slices["near_dup"].setdefault(cond, entry)
    return slices


def merge_parts(part_a_dir: Path, part_b_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Merge two part outputs; returns the merged summary dict."""
    a = _load_part(part_a_dir, "A")
    b = _load_part(part_b_dir, "B")
    _assert_parts_compatible(a, b)

    ma, mb = a["metrics"], b["metrics"]
    merged: dict[str, Any] = {"protocol": ma.get("protocol", mb.get("protocol"))}
    for split in ("test", "validation"):
        merged[split] = {}
        for src in (ma, mb):
            for cond, block in (src.get(split) or {}).items():
                merged[split].setdefault(cond, block)

    merged["qlora_seeds"] = _seed_stats(ma, mb)
    merged["budget_curve_validation"] = _merge_budget_curve(ma, mb)
    if ma.get("bootstrap") is not None:
        merged["bootstrap"] = ma["bootstrap"]
    merged["slices"] = _merge_slices(ma, mb)
    excl: dict[str, Any] = {}
    for src in (ma, mb):
        for cond, block in (src.get("test_excluding_near_dups") or {}).items():
            excl.setdefault(cond, block)
    merged["test_excluding_near_dups"] = excl

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "protocol": merged["protocol"],
        "git_sha": a["manifest"]["git_sha"],
        "plan_sha256": a["plan"]["plan_sha256"],
        "ordering_ok": True,
        "parts": {
            "A": {
                "conditions_run": a["manifest"].get("conditions_run"),
                "n_scored": a["manifest"].get("n_scored"),
                "first_test_read_epoch_ns": a["manifest"].get(
                    "first_test_read_epoch_ns"
                ),
            },
            "B": {
                "conditions_run": b["manifest"].get("conditions_run"),
                "n_scored": b["manifest"].get("n_scored"),
                "first_test_read_epoch_ns": b["manifest"].get(
                    "first_test_read_epoch_ns"
                ),
            },
        },
        "conditions_run": sorted(
            set(a["manifest"].get("conditions_run", []))
            | set(b["manifest"].get("conditions_run", []))
        ),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"metrics": merged, "manifest": manifest}
