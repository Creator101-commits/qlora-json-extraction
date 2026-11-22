"""Tests for paired bootstrap determinism and difference intervals."""

from __future__ import annotations

import pytest

from qlora_extraction.evaluation.bootstrap import DocStats, paired_bootstrap


def _stats(f1_tp: int, fp: int, fn: int, nted: float, n: int) -> list[DocStats]:
    # Alternate per-doc counts so resamples actually vary.
    return [
        DocStats(
            tp=f1_tp + (i % 2),
            fp=fp + (i % 3),
            fn=fn,
            nted=nted + (i % 5) * 0.01,
        )
        for i in range(n)
    ]


def test_bootstrap_deterministic() -> None:
    stats = {
        "zero_shot": _stats(8, 2, 2, 0.5, 10),
        "few_shot": _stats(9, 1, 1, 0.7, 10),
    }
    a = paired_bootstrap(stats, [("few_shot", "zero_shot")], n_boot=200, seed=0)
    b = paired_bootstrap(stats, [("few_shot", "zero_shot")], n_boot=200, seed=0)
    assert a == b


def test_bootstrap_ci_contains_point_estimate_region() -> None:
    stats = {
        "zero_shot": _stats(10, 0, 0, 1.0, 20),
        "few_shot": _stats(10, 0, 0, 1.0, 20),
    }
    out = paired_bootstrap(stats, [("few_shot", "zero_shot")], n_boot=100, seed=0)
    cond = out["conditions"]["zero_shot"]  # type: ignore[index]
    assert cond["field_f1"]["ci_low"] <= cond["field_f1"]["mean"] <= cond["field_f1"]["ci_high"]  # type: ignore[index]
    diff = out["diffs"]["few_shot-zero_shot"]["field_f1"]  # type: ignore[index]
    assert diff["ci_low"] <= 0.0 <= diff["ci_high"] or diff["ci_low"] == diff["ci_high"] == 0.0  # type: ignore[index]


def test_bootstrap_rejects_mismatched_lengths() -> None:
    stats = {"a": _stats(1, 0, 0, 1.0, 5), "b": _stats(1, 0, 0, 1.0, 6)}
    with pytest.raises(ValueError, match="same document order"):
        paired_bootstrap(stats, [], n_boot=10)


def test_bootstrap_different_seeds_differ() -> None:
    stats = {"a": _stats(7, 3, 3, 0.4, 30)}
    a = paired_bootstrap(stats, [], n_boot=100, seed=0)
    b = paired_bootstrap(stats, [], n_boot=100, seed=1)
    assert a != b
