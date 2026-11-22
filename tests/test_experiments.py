"""End-to-end runner test with a fake job executor (no GPU, no real data)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from qlora_extraction.data.cord import CordDocument
from qlora_extraction.experiments import (
    BenchmarkConfig,
    job_specs,
    prepare,
    run_benchmark,
)


class FakeTokenizer:
    eos_token = "<eos>"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> Any:
        parts = [f"<{m['role']}>{m['content']}</" for m in messages]
        if add_generation_prompt:
            parts.append("<|assistant|>")
        return "\n".join(parts)

    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))


def _target(i: int) -> dict[str, object]:
    return {
        "menu": [{"nm": f"ITEM{i}", "price": f"{i}000"}],
        "total": {"total_price": f"{i}000"},
    }


def _doc(doc_id: str, split: str, i: int, target_words: int = 0) -> CordDocument:
    target = _target(i)
    if target_words:
        target["total"] = {"total_price": " ".join(["x"] * target_words)}
    return CordDocument(doc_id, split, f"OCR LINE {i} TEH BOTOL", json.dumps(target))


def _make_docs(split: str, n: int, prefix: str, **kw: Any) -> list[CordDocument]:
    return [_doc(f"{prefix}:{i}", split, i, **kw) for i in range(n)]


def test_prepare_writes_plan_before_test_targets(tmp_path: Path) -> None:
    train = _make_docs("train", 20, "train")
    val = _make_docs("validation", 10, "validation")
    test = _make_docs("test", 10, "test")
    plan = prepare(tmp_path, train, val, test, FakeTokenizer(), BenchmarkConfig())  # type: ignore[arg-type]
    assert (tmp_path / "analysis_plan.json").exists()
    assert (tmp_path / "schema_inventory.json").exists()
    assert (tmp_path / "dedup_report.json").exists()
    assert len(plan["demonstration_doc_ids"]) == 3
    assert plan["max_new_tokens"] > 0
    assert plan["max_seq_len"] > 0
    assert plan["split_counts"] == {"train": 20, "validation": 10, "test": 10}
    assert all(d.target is None for d in test)


def test_job_specs_cover_conditions_and_budgets() -> None:
    specs = job_specs(BenchmarkConfig(), {})
    names = [s["name"] for s in specs]
    assert names == [
        "zero_shot",
        "few_shot",
        "qlora_sft_seed0",
        "qlora_sft_seed1",
        "qlora_sft_seed2",
        "budget200",
        "budget400",
    ]
    smoke_specs = job_specs(BenchmarkConfig(smoke=True), {})
    assert [s["name"] for s in smoke_specs] == [
        "zero_shot",
        "few_shot",
        "qlora_sft_seed0",
        "qlora_sft_seed1",
        "qlora_sft_seed2",
        "budget200",
        "budget400",
    ]


def test_run_benchmark_end_to_end(tmp_path: Path) -> None:
    train = _make_docs("train", 12, "train")
    val = _make_docs("validation", 6, "validation")
    test = _make_docs("test", 6, "test")
    docs_by_split = {"train": train, "validation": val, "test": test}

    def executor(spec: dict[str, Any], out_dir: Path) -> None:
        job_dir = out_dir / "jobs" / spec["name"]
        job_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for split in spec.get("splits") or ["test"]:
            for doc in docs_by_split[split]:
                rows.append(
                    {
                        "doc_id": doc.doc_id,
                        "split": split,
                        "raw_text": doc.target_json,
                        "prompt_tokens": 10,
                        "output_tokens": 5,
                    }
                )
        with (job_dir / "predictions.jsonl").open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

    cfg = BenchmarkConfig(n_boot=50)
    result = run_benchmark(
        tmp_path, train, val, test, FakeTokenizer(), cfg, executor  # type: ignore[arg-type]
    )
    manifest = result["manifest"]
    assert manifest["ordering_ok"] is True
    assert manifest["card_write_epoch_ns"] < manifest["first_test_read_epoch_ns"]
    metrics = result["metrics"]
    assert metrics["test"]["zero_shot"]["field_f1"] == pytest.approx(1.0)
    assert metrics["test"]["qlora_sft_seed0"]["exact_match_rate"] == 1.0
    assert "bootstrap" in metrics
    assert metrics["budget_curve_validation"]["800"]["field_f1"] == pytest.approx(1.0)
    assert all(d.target is not None for d in test)


def test_prepare_rejects_schema_invalid_train_target(tmp_path: Path) -> None:
    train = _make_docs("train", 10, "train")
    bad = CordDocument("train:bad", "train", "OCR", '{"bogus_key": 1}')
    train.append(bad)
    val = _make_docs("validation", 5, "validation")
    test = _make_docs("test", 5, "test")
    with pytest.raises(ValueError, match="train:bad"):
        prepare(tmp_path, train, val, test, FakeTokenizer(), BenchmarkConfig())  # type: ignore[arg-type]


def test_prepare_records_train_schema_valid_count(tmp_path: Path) -> None:
    train = _make_docs("train", 12, "train")
    val = _make_docs("validation", 5, "validation")
    test = _make_docs("test", 5, "test")
    plan = prepare(tmp_path, train, val, test, FakeTokenizer(), BenchmarkConfig())  # type: ignore[arg-type]
    assert plan["train_schema_valid"] == 12
    assert plan["plan_sha256"]


def test_max_tokens_derived_from_train_only(tmp_path: Path) -> None:
    # Test targets are enormous; the plan must still reflect train p99.
    train = _make_docs("train", 15, "train")
    val = _make_docs("validation", 5, "validation")
    test = _make_docs("test", 5, "test", target_words=500)
    plan = prepare(tmp_path, train, val, test, FakeTokenizer(), BenchmarkConfig())  # type: ignore[arg-type]
    basis = plan["max_new_tokens_basis"]["p99_target_tokens"]
    assert plan["max_new_tokens"] == math.ceil(1.25 * basis)
    # Train targets are a handful of words; a 500-word test target would
    # produce a much larger p99 if test leaked into the derivation.
    assert basis < 100
    assert all(d.target is None for d in test)
