"""Paired document-level bootstrap for metric uncertainty.

Resamples test document indices with replacement (2000 replicates, seed 0)
and recomputes micro field-F1 and mean nTED per condition per replicate.
Percentile 95% intervals are reported for each condition and for paired
differences against the declared reference conditions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0
CI_ALPHA = 0.05


@dataclass(frozen=True)
class DocStats:
    """Per-document sufficient statistics for bootstrap recomputation."""

    tp: int
    fp: int
    fn: int
    nted: float


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def _percentile_interval(values: np.ndarray) -> dict[str, float]:
    lo, hi = np.percentile(values, [100 * CI_ALPHA / 2, 100 * (1 - CI_ALPHA / 2)])
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(lo),
        "ci_high": float(hi),
    }


def paired_bootstrap(
    stats_by_condition: dict[str, list[DocStats]],
    diff_pairs: list[tuple[str, str]],
    *,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Bootstrap field-F1 and nTED per condition plus paired differences.

    ``stats_by_condition`` maps condition name to per-document stats in a
    fixed document order shared across conditions. ``diff_pairs`` lists
    ``(condition, reference)`` pairs whose replicate differences are
    reported as ``"<condition>-<reference>"``.
    """
    rng = np.random.default_rng(seed)
    n_docs = len(next(iter(stats_by_condition.values())))
    if n_docs == 0:
        return {"n_boot": n_boot, "seed": seed, "ci_alpha": CI_ALPHA,
                "conditions": {}, "diffs": {}}
    for stats in stats_by_condition.values():
        if len(stats) != n_docs:
            raise ValueError("all conditions must share the same document order")

    cond_names = list(stats_by_condition)
    f1_reps: dict[str, list[float]] = {c: [] for c in cond_names}
    nted_reps: dict[str, list[float]] = {c: [] for c in cond_names}
    diff_reps: dict[str, dict[str, list[float]]] = {
        f"{a}-{b}": {"field_f1": [], "nted": []} for a, b in diff_pairs
    }

    for _ in range(n_boot):
        idx = rng.integers(0, n_docs, size=n_docs)
        rep_f1: dict[str, float] = {}
        rep_nted: dict[str, float] = {}
        for cond in cond_names:
            stats = [stats_by_condition[cond][i] for i in idx]
            tp = sum(s.tp for s in stats)
            fp = sum(s.fp for s in stats)
            fn = sum(s.fn for s in stats)
            rep_f1[cond] = _f1_from_counts(tp, fp, fn)
            rep_nted[cond] = float(np.mean([s.nted for s in stats])) if stats else 0.0
            f1_reps[cond].append(rep_f1[cond])
            nted_reps[cond].append(rep_nted[cond])
        for a, b in diff_pairs:
            key = f"{a}-{b}"
            diff_reps[key]["field_f1"].append(rep_f1[a] - rep_f1[b])
            diff_reps[key]["nted"].append(rep_nted[a] - rep_nted[b])

    conditions: dict[str, object] = {}
    for cond in cond_names:
        conditions[cond] = {
            "field_f1": _percentile_interval(np.array(f1_reps[cond])),
            "nted": _percentile_interval(np.array(nted_reps[cond])),
        }
    diffs: dict[str, object] = {}
    for key, metrics in diff_reps.items():
        diffs[key] = {
            "field_f1": _percentile_interval(np.array(metrics["field_f1"])),
            "nted": _percentile_interval(np.array(metrics["nted"])),
        }
    return {
        "n_boot": n_boot,
        "seed": seed,
        "ci_alpha": CI_ALPHA,
        "conditions": conditions,
        "diffs": diffs,
    }
