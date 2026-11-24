"""Tests for the two-part merge and the nested budget subsets."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qlora_extraction.data.cord import CordDocument
from qlora_extraction.experiments.merge import merge_parts
from qlora_extraction.experiments.runner import (
    BenchmarkConfig,
    job_specs,
    subsample_docs,
)


def _write_part(
    root: Path,
    *,
    git_sha: str = "abc123",
    plan_sha: str = "planhash",
    conditions: list[str],
    ordering_ok: bool = True,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "analysis_plan.json").write_text(
        json.dumps({"plan_sha256": plan_sha}), encoding="utf-8"
    )
    metrics: dict = {
        "protocol": "cord-qlora-benchmark-v1",
        "test": {
            cond: {"field_f1": 0.5, "nted": 0.4, "n_docs": 3}
            for cond in conditions
        },
        "validation": {},
        "qlora_seeds": {},
        "budget_curve_validation": {},
        "slices": {"ocr_tertiles": {}, "near_dup": {}},
        "test_excluding_near_dups": {},
    }
    (root / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    manifest = {
        "git_sha": git_sha,
        "ordering_ok": ordering_ok,
        "conditions_run": conditions,
        "n_scored": {},
        "first_test_read_epoch_ns": 1,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_merge_happy_path(tmp_path: Path) -> None:
    a = _write_part(tmp_path / "a", conditions=["zero_shot", "qlora_sft_seed0"])
    b = _write_part(tmp_path / "b", conditions=["qlora_sft_seed1", "qlora_sft_seed2"])
    out = merge_parts(a, b, tmp_path / "merged")
    metrics = json.loads((tmp_path / "merged" / "metrics.json").read_text())
    assert set(metrics["test"]) == {
        "zero_shot", "qlora_sft_seed0", "qlora_sft_seed1", "qlora_sft_seed2"
    }
    seeds = metrics["qlora_seeds"]["per_seed"]
    assert set(seeds) == {"qlora_sft_seed0", "qlora_sft_seed1", "qlora_sft_seed2"}
    assert metrics["qlora_seeds"]["field_f1_mean"] == pytest.approx(0.5)
    assert metrics["qlora_seeds"]["field_f1_sd"] == pytest.approx(0.0)
    assert out["manifest"]["ordering_ok"] is True
    assert out["manifest"]["plan_sha256"] == "planhash"


def test_merge_rejects_git_sha_mismatch(tmp_path: Path) -> None:
    a = _write_part(tmp_path / "a", conditions=["zero_shot"])
    b = _write_part(tmp_path / "b", conditions=["few_shot"], git_sha="other")
    with pytest.raises(ValueError, match="git_sha"):
        merge_parts(a, b, tmp_path / "merged")


def test_merge_rejects_plan_sha_mismatch(tmp_path: Path) -> None:
    a = _write_part(tmp_path / "a", conditions=["zero_shot"])
    b = _write_part(tmp_path / "b", conditions=["few_shot"], plan_sha="different")
    with pytest.raises(ValueError, match="plan_sha256"):
        merge_parts(a, b, tmp_path / "merged")


def test_merge_rejects_overlapping_jobs(tmp_path: Path) -> None:
    a = _write_part(tmp_path / "a", conditions=["zero_shot"])
    b = _write_part(tmp_path / "b", conditions=["zero_shot", "few_shot"])
    with pytest.raises(ValueError, match="disjoint"):
        merge_parts(a, b, tmp_path / "merged")


def test_merge_rejects_bad_ordering(tmp_path: Path) -> None:
    a = _write_part(tmp_path / "a", conditions=["zero_shot"], ordering_ok=False)
    b = _write_part(tmp_path / "b", conditions=["few_shot"])
    with pytest.raises(ValueError, match="ordering_ok"):
        merge_parts(a, b, tmp_path / "merged")


def test_nested_budget_subsets() -> None:
    docs = [
        CordDocument(f"train:{i}", "train", f"ocr {i}", '{"total": {}}')
        for i in range(800)
    ]
    b200 = subsample_docs(docs, 200)
    b400 = subsample_docs(docs, 400)
    ids200 = {d.doc_id for d in b200}
    ids400 = {d.doc_id for d in b400}
    assert len(b200) == 200 and len(b400) == 400
    assert ids200 < ids400
    assert {d.doc_id for d in subsample_docs(docs, 200)} == ids200
    # the 800 point is the full train set, so 400 nests inside it trivially
    assert ids400 < {d.doc_id for d in docs}


def test_job_specs_parts_disjoint_and_b_test_only() -> None:
    cfg = BenchmarkConfig()
    plan: dict = {}
    a = {s["name"] for s in job_specs(cfg, plan, "A")}
    b = {s["name"] for s in job_specs(cfg, plan, "B")}
    assert a == {"zero_shot", "few_shot", "qlora_sft_seed0", "budget200"}
    assert b == {"qlora_sft_seed1", "qlora_sft_seed2", "budget400"}
    assert not (a & b)
    b_specs = {s["name"]: s for s in job_specs(cfg, plan, "B")}
    assert b_specs["qlora_sft_seed1"]["splits"] == ["test"]
    assert b_specs["qlora_sft_seed2"]["splits"] == ["test"]
    assert b_specs["budget400"]["splits"] == ["validation"]
    a_specs = {s["name"]: s for s in job_specs(cfg, plan, "A")}
    assert a_specs["qlora_sft_seed0"]["splits"] == ["validation", "test"]
