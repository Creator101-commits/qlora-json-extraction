"""End-to-end benchmark orchestration for the CORD QLoRA study.

Protocol ``cord-qlora-benchmark-v1``: prepare freezes every configuration
(demonstrations, token limits, dedup flags, subsets) into
``analysis_plan.json`` before any test ground truth is materialized; jobs
then produce predictions; collect parses, scores, bootstraps, and writes
``metrics.json`` and ``manifest.json`` exactly once.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from qlora_extraction.data.cord import CordDocument
from qlora_extraction.data.dedup import dedup_audit, report_to_dict
from qlora_extraction.data.inventory import audit_inventories
from qlora_extraction.evaluation import (
    DocScore,
    DocStats,
    aggregate_scores,
    paired_bootstrap,
    score_doc,
)
from qlora_extraction.evaluation.bootstrap import N_BOOTSTRAP
from qlora_extraction.inference.parse import parse_model_output
from qlora_extraction.prompting import (
    build_messages,
    demonstrations_for_prompt,
    minified_schema_json,
    render_prompt_text,
    schema_token_count,
    select_demonstrations,
)
from qlora_extraction.prompting.templates import ChatTemplateTokenizer

PROTOCOL = "cord-qlora-benchmark-v1"


@dataclass(frozen=True)
class BenchmarkConfig:
    """Frozen benchmark shape; smoke mode shrinks every stage."""

    seeds: tuple[int, ...] = (0, 1, 2)
    budget_sizes: tuple[int, ...] = (200, 400)
    main_budget: int = 800
    n_boot: int = N_BOOTSTRAP
    smoke: bool = False
    smoke_train: int = 16
    smoke_val: int = 8
    smoke_test: int = 8
    smoke_epochs: int = 1
    max_epochs: int = 5


class JobExecutor(Protocol):
    """Runs one job spec, writing ``jobs/<name>/`` outputs under out_dir."""

    def __call__(self, spec: dict[str, Any], out_dir: Path) -> None: ...


def _now_ns() -> int:
    return time.time_ns()


def _encode_ids(tokenizer: ChatTemplateTokenizer, text: str) -> list[int]:
    return tokenizer.encode(text)


def _target_token_len(tokenizer: ChatTemplateTokenizer, doc: CordDocument) -> int:
    return len(_encode_ids(tokenizer, doc.target_json))


def _prompt_token_len(
    tokenizer: ChatTemplateTokenizer, doc: CordDocument, schema_json: str
) -> int:
    messages = build_messages("zero_shot", doc.ocr_text, schema_json)
    return len(_encode_ids(tokenizer, render_prompt_text(tokenizer, messages)))


def _p99(values: list[int]) -> int:
    if not values:
        return 0
    return int(np.percentile(np.array(values, dtype=np.float64), 99))


def subsample_docs(
    docs: list[CordDocument], n: int, seed: int = 0
) -> list[CordDocument]:
    """Deterministic seeded subsample preserving a stable doc order.

    One permutation per ``(docs, seed)``; taking ``perm[:n]`` makes every
    smaller subset nested inside every larger one (200 inside 400 inside
    800), which the budget curve requires.
    """
    if n >= len(docs):
        return list(docs)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(docs))
    idx = np.sort(perm[:n])
    return [docs[int(i)] for i in idx]


def prepare(
    out_dir: Path,
    train_docs: list[CordDocument],
    validation_docs: list[CordDocument],
    test_docs: list[CordDocument],
    tokenizer: ChatTemplateTokenizer,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    """Freeze all configurations into ``analysis_plan.json``.

    Reads train and validation ground truth plus test OCR only; test
    ``gt_parse`` stays unmaterialized until collect time.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = config
    if cfg.smoke:
        train_docs = subsample_docs(train_docs, cfg.smoke_train)
        validation_docs = subsample_docs(validation_docs, cfg.smoke_val)
        test_docs = subsample_docs(test_docs, cfg.smoke_test)

    # Train-schema gate: every training target must validate CordParseV1.
    from qlora_extraction.schema import validate_cord_parse

    invalid_ids = [
        d.doc_id for d in train_docs if validate_cord_parse(d.get_target()) is None
    ]
    if invalid_ids:
        shown = ", ".join(invalid_ids[:20])
        raise ValueError(
            f"{len(invalid_ids)} train targets fail CordParseV1: {shown}"
            + (" ..." if len(invalid_ids) > 20 else "")
        )

    # Data audit: train inventory defines the schema; val/test gaps reported.
    inventory = audit_inventories(train_docs, {"validation": validation_docs})
    (out_dir / "schema_inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )

    dedup = dedup_audit(train_docs, validation_docs, test_docs)
    (out_dir / "dedup_report.json").write_text(
        json.dumps({k: report_to_dict(v) for k, v in dedup.items()}, indent=2),
        encoding="utf-8",
    )
    test_near_dup_ids = set(dedup["test_vs_train_validation"].flagged_query_ids)

    demos = select_demonstrations(train_docs)
    demo_ids = [d.doc_id for d in demos]

    schema_json = minified_schema_json()
    schema_hash = hashlib.sha256(schema_json.encode("utf-8")).hexdigest()

    target_lens = [_target_token_len(tokenizer, d) for d in train_docs]
    prompt_lens = [_prompt_token_len(tokenizer, d, schema_json) for d in train_docs]
    total_lens = [p + t + 1 for p, t in zip(prompt_lens, target_lens, strict=True)]
    p99_target = _p99(target_lens)
    max_new_tokens = int(np.ceil(1.25 * p99_target))
    max_seq_len = _p99(total_lens)

    ocr_lens = sorted(len(d.ocr_text) for d in train_docs)
    tertiles = [
        int(ocr_lens[int(0.33 * (len(ocr_lens) - 1))]),
        int(ocr_lens[int(0.67 * (len(ocr_lens) - 1))]),
    ] if ocr_lens else [0, 0]

    plan = {
        "protocol": PROTOCOL,
        "cord_repo": "naver-clova-ix/cord-v2",
        "cord_revision": "41b075798d0d9da590a950dac1e3d4a21826d4fd",
        "schema": {
            "name": "CordParseV1",
            "json_sha256": schema_hash,
            "token_count": schema_token_count(tokenizer),
        },
        "smoke": cfg.smoke,
        "split_counts": {
            "train": len(train_docs),
            "validation": len(validation_docs),
            "test": len(test_docs),
        },
        "train_doc_ids": [d.doc_id for d in train_docs],
        "validation_doc_ids": [d.doc_id for d in validation_docs],
        "test_doc_ids": [d.doc_id for d in test_docs],
        "demonstration_doc_ids": demo_ids,
        "demonstrations": demonstrations_for_prompt(demos),
        "budget_rule": {
            "seed": 0,
            "method": "numpy default_rng(0).permutation prefix over train order",
            "sizes": [cfg.main_budget, *cfg.budget_sizes],
        },
        "train_schema_valid": len(train_docs),
        "seeds": list(cfg.seeds),
        "max_epochs": cfg.smoke_epochs if cfg.smoke else cfg.max_epochs,
        "max_new_tokens": max_new_tokens,
        "max_new_tokens_basis": {"p99_target_tokens": p99_target, "factor": 1.25},
        "max_seq_len": max_seq_len,
        "max_seq_len_basis": {"p99_prompt_plus_target": int(np.percentile(total_lens, 99))},
        "n_overlength_train": sum(1 for n in total_lens if n > max_seq_len),
        "dedup": {
            "test_near_dup_ids": sorted(test_near_dup_ids),
            "validation_flagged_ids": list(
                dedup["validation_vs_train"].flagged_query_ids
            ),
        },
        "ocr_length_tertiles": tertiles,
        "conditions": ["zero_shot", "few_shot", "qlora_sft"],
        "n_boot": cfg.n_boot,
        "bootstrap_seed": 0,
        "normalization": "strip and collapse whitespace; case and digits untouched",
        "created_epoch_ns": _now_ns(),
        "card_write_epoch_ns": _now_ns(),
    }
    # plan_sha256 proves both split kernels froze the identical plan; the
    # volatile fields (epochs, timestamps) cannot differ between parts.
    volatile = {"created_epoch_ns", "card_write_epoch_ns", "max_epochs"}
    fp_obj = {k: v for k, v in plan.items() if k not in volatile}
    plan["plan_sha256"] = hashlib.sha256(
        json.dumps(fp_obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    (out_dir / "analysis_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return plan


# Job ownership for the two-kernel split: the weekly GPU quota cannot hold
# the whole protocol in one run, so part A carries the prompted conditions,
# the seed-0 adapter and the 200-doc budget point; part B carries the two
# remaining seeds and the 400-doc point. Both parts share one frozen plan.
PART_A_JOBS = frozenset({"zero_shot", "few_shot", "qlora_sft_seed0", "budget200"})
PART_B_JOBS = frozenset({"qlora_sft_seed1", "qlora_sft_seed2", "budget400"})


def job_specs(
    config: BenchmarkConfig, plan: dict[str, Any], part: str = "all"
) -> list[dict[str, Any]]:
    """Declare benchmark jobs for one part; Kaggle runs each in a subprocess."""
    specs: list[dict[str, Any]] = [
        {"name": "zero_shot", "kind": "prompted", "condition": "zero_shot"},
        {"name": "few_shot", "kind": "prompted", "condition": "few_shot"},
    ]
    for seed in config.seeds:
        specs.append(
            {
                "name": f"qlora_sft_seed{seed}",
                "kind": "qlora",
                "condition": "qlora_sft",
                "seed": seed,
                "budget": config.main_budget,
                "splits": ["validation", "test"],
            }
        )
    for budget in config.budget_sizes:
        specs.append(
            {
                "name": f"budget{budget}",
                "kind": "qlora",
                "condition": "qlora_sft",
                "seed": 0,
                "budget": budget,
                "splits": ["validation"],
            }
        )
    if part == "all":
        return specs
    allowed = PART_A_JOBS if part == "A" else PART_B_JOBS
    kept = [s for s in specs if s["name"] in allowed]
    if part == "B":
        # Part B runs after part A; seed-1/2 validation outputs drive
        # nothing, so only test generation is needed there.
        for spec in kept:
            if spec["kind"] == "qlora" and spec["name"].startswith("qlora_sft_seed"):
                spec["splits"] = ["test"]
    return kept


def _read_predictions(job_dir: Path) -> list[dict[str, Any]]:
    path = job_dir / "predictions.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def collect(
    out_dir: Path,
    test_docs: list[CordDocument],
    validation_docs: list[CordDocument],
    config: BenchmarkConfig,
) -> dict[str, Any]:
    """Score all job outputs and write ``metrics.json`` + ``manifest.json``.

    This is the first point where test ``gt_parse`` is materialized, so the
    epoch recorded here must exceed ``card_write_epoch_ns`` in the plan.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = json.loads((out_dir / "analysis_plan.json").read_text(encoding="utf-8"))
    first_test_read_epoch_ns = _now_ns()

    test_by_id = {d.doc_id: d for d in test_docs}
    val_by_id = {d.doc_id: d for d in validation_docs}

    # condition (= job name) -> split -> doc_id -> DocScore
    scores: dict[str, dict[str, dict[str, DocScore]]] = {}
    for job_dir in sorted((out_dir / "jobs").glob("*")):
        if not job_dir.is_dir():
            continue
        cond = job_dir.name
        for row in _read_predictions(job_dir):
            split = row["split"]
            docs_by_id = test_by_id if split == "test" else val_by_id
            doc = docs_by_id.get(row["doc_id"])
            if doc is None:
                continue
            parsed = parse_model_output(row["raw_text"])
            score = score_doc(doc, parsed)
            scores.setdefault(cond, {}).setdefault(split, {})[doc.doc_id] = score

    metrics: dict[str, Any] = {"protocol": PROTOCOL, "test": {}, "validation": {}}
    for cond, by_split in scores.items():
        for split, doc_scores in by_split.items():
            ordered = [doc_scores[k] for k in sorted(doc_scores)]
            metrics[split][cond] = aggregate_scores(ordered)

    # QLoRA per-seed aggregation on test.
    seed_scores = {
        seed: scores.get(f"qlora_sft_seed{seed}", {}).get("test", {})
        for seed in config.seeds
    }
    if any(seed_scores.values()):
        per_seed = {}
        for seed, doc_map in seed_scores.items():
            if doc_map:
                ordered = [doc_map[k] for k in sorted(doc_map)]
                per_seed[f"seed{seed}"] = aggregate_scores(ordered)
        f1s = [m["field_f1"] for m in per_seed.values()]
        nteds = [m["nted"] for m in per_seed.values()]
        metrics["qlora_seeds"] = {
            "per_seed": per_seed,
            "field_f1_mean": float(np.mean(f1s)) if f1s else 0.0,
            "field_f1_min": float(np.min(f1s)) if f1s else 0.0,
            "field_f1_max": float(np.max(f1s)) if f1s else 0.0,
            "nted_mean": float(np.mean(nteds)) if nteds else 0.0,
        }

    # Budget curve on validation: 200/400 from budget jobs, 800 = seed-0 main.
    budget_points: dict[str, Any] = {}
    for spec_budget in config.budget_sizes:
        val_map = scores.get(f"budget{spec_budget}", {}).get("validation", {})
        budget_points[str(spec_budget)] = (
            aggregate_scores([val_map[k] for k in sorted(val_map)]) if val_map else None
        )
    val_map800 = scores.get("qlora_sft_seed0", {}).get("validation", {})
    budget_points[str(config.main_budget)] = (
        aggregate_scores([val_map800[k] for k in sorted(val_map800)]) if val_map800 else None
    )
    metrics["budget_curve_validation"] = budget_points

    # Paired bootstrap on test for headline conditions (seed-0 QLoRA),
    # restricted to documents every condition predicted.
    bootstrap_conditions = ["zero_shot", "few_shot", "qlora_sft_seed0"]
    doc_maps = {
        cond: scores.get(cond, {}).get("test", {})
        for cond in bootstrap_conditions
    }
    common_ids = set(test_by_id)
    for doc_map in doc_maps.values():
        common_ids &= set(doc_map)
    test_order = sorted(common_ids)
    stats: dict[str, list[DocStats]] = {}
    for cond, doc_map in doc_maps.items():
        if not doc_map:
            continue
        stats[cond] = [
            DocStats(tp=doc_map[k].tp, fp=doc_map[k].fp, fn=doc_map[k].fn, nted=doc_map[k].nted)
            for k in test_order
        ]
    diff_pairs = [(c, "zero_shot") for c in stats if c != "zero_shot"]
    if "qlora_sft_seed0" in stats and "few_shot" in stats:
        diff_pairs.append(("qlora_sft_seed0", "few_shot"))
    if stats:
        metrics["bootstrap"] = paired_bootstrap(
            stats, diff_pairs, n_boot=config.n_boot, seed=0
        )

    # Slices: OCR-length tertiles and near-dup exclusion, test split.
    tertiles = plan.get("ocr_length_tertiles", [0, 0])
    near_dup_ids = set(plan.get("dedup", {}).get("test_near_dup_ids", []))
    slices: dict[str, Any] = {"ocr_tertiles": {}, "near_dup": {}}
    for cond in ["zero_shot", "few_shot", "qlora_sft_seed0"]:
        doc_map = scores.get(cond, {}).get("test", {})
        if not doc_map:
            continue
        for name, lo, hi in (
            ("tertile_1", 0, tertiles[0]),
            ("tertile_2", tertiles[0], tertiles[1]),
            ("tertile_3", tertiles[1], 10**12),
        ):
            subset = [
                s
                for k, s in doc_map.items()
                if lo <= len(test_by_id[k].ocr_text) < hi
            ]
            if subset:
                slices["ocr_tertiles"].setdefault(name, {})[cond] = aggregate_scores(subset)[
                    "field_f1"
                ]
        novel = [s for k, s in doc_map.items() if k not in near_dup_ids]
        dup = [s for k, s in doc_map.items() if k in near_dup_ids]
        slices["near_dup"][cond] = {
            "novel_field_f1": aggregate_scores(novel)["field_f1"] if novel else None,
            "near_dup_field_f1": aggregate_scores(dup)["field_f1"] if dup else None,
            "n_near_dup": len(dup),
        }
    metrics["slices"] = slices

    # Secondary test metrics excluding near-duplicates of train+validation.
    secondary: dict[str, Any] = {}
    for cond, by_split in scores.items():
        doc_map = by_split.get("test", {})
        if not doc_map:
            continue
        novel = [s for k, s in doc_map.items() if k not in near_dup_ids]
        if novel:
            secondary[cond] = aggregate_scores(novel)
    metrics["test_excluding_near_dups"] = secondary

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    scored_counts = {
        split: {
            cond: len(by_split.get(split, {})) for cond, by_split in scores.items()
        }
        for split in ("test", "validation")
    }
    git_sha = os.environ.get("QLORA_GIT_SHA", "")
    manifest = {
        "protocol": PROTOCOL,
        "git_sha": git_sha,
        "card_write_epoch_ns": plan.get("card_write_epoch_ns"),
        "first_test_read_epoch_ns": first_test_read_epoch_ns,
        "ordering_ok": bool(
            plan.get("card_write_epoch_ns")
            and plan["card_write_epoch_ns"] < first_test_read_epoch_ns
        ),
        "conditions_run": sorted(scores),
        "n_scored": scored_counts,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"metrics": metrics, "manifest": manifest}


def run_benchmark(
    out_dir: Path,
    train_docs: list[CordDocument],
    validation_docs: list[CordDocument],
    test_docs: list[CordDocument],
    tokenizer: ChatTemplateTokenizer,
    config: BenchmarkConfig,
    executor: JobExecutor,
) -> dict[str, Any]:
    """Full pipeline: prepare, run jobs, collect. Returns summary dict."""
    plan = prepare(out_dir, train_docs, validation_docs, test_docs, tokenizer, config)
    jobs_dir = out_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    for spec in job_specs(config, plan):
        start = time.monotonic()
        executor(spec, out_dir)
        timings[spec["name"]] = time.monotonic() - start
    result = collect(out_dir, test_docs, validation_docs, config)
    result["timings"] = timings
    result["plan"] = plan
    return result
