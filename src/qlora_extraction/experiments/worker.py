"""Single-job worker for Kaggle subprocess isolation.

Each job runs in a fresh interpreter: ``python -m
qlora_extraction.experiments.worker --spec jobs/spec.json``. The worker
reads ``analysis_plan.json`` for every frozen parameter, loads only the
CORD splits it needs, trains or prompts as specified, and writes
``jobs/<name>/predictions.jsonl`` plus ``train_result.json`` for QLoRA
jobs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from qlora_extraction.data.cord import CordDocument, load_cord_split


def _select_by_ids(docs: list[CordDocument], ids: list[str]) -> list[CordDocument]:
    by_id = {d.doc_id: d for d in docs}
    return [by_id[i] for i in ids if i in by_id]


def _budget_subset(
    train_docs: list[CordDocument], budget: int, plan: dict[str, Any]
) -> list[CordDocument]:
    """Nested seeded prefix subset shared with ``subsample_docs``."""
    from qlora_extraction.experiments.runner import subsample_docs

    return subsample_docs(
        train_docs, min(budget, len(train_docs)), int(plan["budget_rule"]["seed"])
    )


def run_job(spec: dict[str, Any], work_dir: Path) -> dict[str, Any]:
    """Execute one job spec; returns a small result summary."""
    plan = json.loads((work_dir / "analysis_plan.json").read_text(encoding="utf-8"))
    splits_needed = set(spec.get("splits") or ["test"])
    docs_by_split: dict[str, list[CordDocument]] = {}
    for split in ("train", "validation", "test"):
        if split in splits_needed or spec["kind"] == "qlora":
            docs_by_split[split] = load_cord_split(split)

    # Restrict to the frozen plan subsets (smoke subsamples in smoke runs).
    for split in ("train", "validation", "test"):
        if split in docs_by_split:
            ids = plan.get(f"{split}_doc_ids")
            if ids is not None:
                docs_by_split[split] = _select_by_ids(docs_by_split[split], ids)

    job_dir = work_dir / "jobs" / spec["name"]
    job_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    adapter_dir: str | None = None
    train_info: dict[str, Any] | None = None
    if spec["kind"] == "qlora":
        from qlora_extraction.training import SftConfig, train_qlora_sft

        train_docs = docs_by_split["train"]
        budget = spec.get("budget")
        if budget is not None and budget < len(train_docs):
            train_docs = _budget_subset(train_docs, int(budget), plan)
        epochs = int(plan.get("max_epochs", 5))
        cfg = SftConfig(
            seed=int(spec.get("seed", 0)),
            max_epochs=epochs,
            max_seq_len=int(plan["max_seq_len"]),
        )
        result = train_qlora_sft(
            train_docs, docs_by_split["validation"], job_dir, cfg
        )
        adapter_dir = result.adapter_dir
        train_info = {
            "best_epoch": result.best_epoch,
            "n_train": result.n_train,
            "n_val": result.n_val,
            "n_dropped_overlength": result.n_dropped_overlength,
            "train_time_s": result.train_time_s,
            "peak_gpu_mem_bytes": result.peak_gpu_mem_bytes,
            "n_trainable_params": result.n_trainable_params,
            "n_total_params": result.n_total_params,
            "adapter_bytes": result.adapter_bytes,
            "batch_config": result.batch_config,
            "history": result.history,
        }
        (job_dir / "train_result.json").write_text(
            json.dumps(train_info, indent=2), encoding="utf-8"
        )

    from qlora_extraction.inference import (
        generate_texts,
        generation_prompts_for_condition,
        load_model_for_inference,
    )
    from qlora_extraction.training import BASE_MODEL, BASE_MODEL_REVISION

    model, tokenizer = load_model_for_inference(
        adapter_dir,
        base_model=BASE_MODEL,
        base_revision=BASE_MODEL_REVISION,
    )
    demonstrations = plan.get("demonstrations")
    rows: list[dict[str, Any]] = []
    for split in spec.get("splits") or ["test"]:
        eval_docs = docs_by_split.get(split, [])
        if not eval_docs:
            continue
        prompts = generation_prompts_for_condition(
            "zero_shot" if spec["kind"] == "qlora" else spec["condition"],
            eval_docs,
            tokenizer,
            demonstrations=demonstrations,
        )
        gen = generate_texts(
            model,
            tokenizer,
            prompts,
            max_new_tokens=int(plan["max_new_tokens"]),
        )
        for doc, text, in_tok, out_tok in zip(
            eval_docs, gen.texts, gen.input_token_counts, gen.output_token_counts,
            strict=True,
        ):
            rows.append(
                {
                    "doc_id": doc.doc_id,
                    "split": split,
                    "raw_text": text,
                    "prompt_tokens": in_tok,
                    "output_tokens": out_tok,
                }
            )
    with (job_dir / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "name": spec["name"],
        "n_predictions": len(rows),
        "elapsed_s": time.monotonic() - started,
    }
    (job_dir / "job_result.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, help="Path to job spec JSON")
    parser.add_argument("--work-dir", default=".", help="Benchmark output dir")
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    summary = run_job(spec, Path(args.work_dir))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
