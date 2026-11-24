> This document is a roadmap. See README.md for implemented behavior,
> measured results, and verification limits; planned capabilities are not
> measured results.

# Build Plan: QLoRA Fine-Tuning for Structured JSON Extraction

## Goal

Build a reproducible study of whether supervised 4-bit QLoRA improves
extraction of schema-valid receipt JSON from noisy OCR text over
zero-shot and few-shot prompting, measured on real held-out CORD-v2
documents under a frozen, leakage-controlled protocol.

## Status

- **Part A: complete.** Zero-shot, few-shot, QLoRA seed 0, and the
  200-receipt budget point ran on Kaggle; curated output is in
  `results/` and `image.png` renders from it.
- **Part B: planned future work.** QLoRA seeds 1 and 2 on test and the
  400-receipt budget point on validation are fully specified in
  `kaggle_b/` and frozen by the same `plan_sha256`; they merge into the
  results only through `scripts/merge_results.py`.
- **Out of scope per the settled brief:** the synthetic-data ablation,
  DPO/preference tuning, constrained-decoding baselines, and any
  full-precision or larger-model comparisons.

## Scope

### Included

- CORD-v2 acquisition over HTTPS column reads (official splits 800/100/100).
- OCR reconstruction from `valid_line` word boxes in reading order.
- Exact and near-duplicate audit before any split-dependent choice.
- Versioned `CordParseV1` schema frozen from a train-only key inventory.
- Deterministic zero-shot and few-shot prompted baselines.
- 4-bit NF4 QLoRA SFT with masked prompt loss and validation-loss
  checkpoint selection.
- A training-budget point (200 receipts) on validation.
- Structural, field-level, nTED, exact-match, hallucination/omission,
  slice, and paired-bootstrap metrics.
- Two Kaggle kernel parts sharing one frozen `analysis_plan.json`.

### Explicitly outside

- Training a base model from scratch or an OCR/vision model.
- Using test documents for prompts, demonstrations, token limits,
  synthetic generation, or checkpoint selection.
- Synthetic training data and preference tuning (future briefs only).
- Arbitrary schemas beyond `CordParseV1`, hosted services, or claims of
  correctness from JSON validity alone.

## Architecture

```text
CORD-v2 ground_truth (HTTPS column reads)
        |
        v
schema inventory -> dedup audit -> analysis_plan.json (frozen)
        |
        v
canonical CordParseV1 targets + chat-template prompts
        |
        +----------------+----------------+
        |                                 |
        v                                 v
zero_shot / few_shot              QLoRA SFT (seeds, budgets)
        |                                 |
        +----------------+----------------+
                         v
          predictions.jsonl per job
                         |
                         v
   score_doc / parse_model_output / nTED / bootstrap
                         |
                         v
   metrics.json + manifest.json (ordering_ok, git SHA)
```

### Components

- `src/qlora_extraction/data/`: CORD loading (`cord.py`), dedup audit
  (`dedup.py`), schema-gap inventory (`inventory.py`).
- `src/qlora_extraction/schema.py`: `CordParseV1` Pydantic models,
  canonicalization (dict-or-list `menu`/`sub`/`void_menu` to lists),
  forbidden-key detection.
- `src/qlora_extraction/prompting/`: chat-template instruction, minified
  JSON Schema, fixed few-shot demonstrations.
- `src/qlora_extraction/training/sft.py`: 4-bit QLoRA SFT, masked prompt
  loss, validation early stop (GPU extra only).
- `src/qlora_extraction/inference/`: batched greedy generation and
  balanced-JSON output parsing.
- `src/qlora_extraction/evaluation/`: `score_doc`, field F1, Donut nTED
  port, structural rates, slices, paired bootstrap.
- `src/qlora_extraction/experiments/`: prepare/jobs/collect runner and
  the Kaggle worker; `merge.py` merges part B into part A.
- `kaggle/`, `kaggle_b/`, `kaggle_smoke/`: the three kernel entrypoints.
- `scripts/kaggle_run.sh`: archive HEAD, version the source dataset,
  push, poll, download with a git-SHA guard. `scripts/merge_results.py`:
  part B merge checks. `scripts/render_figure.py`: `image.png` from
  `results/` only, with a scorer reproduction gate.

## Protocol

The benchmark runs as two Kaggle GPU T4 kernels because the weekly GPU
quota cannot hold the whole protocol in one run:

| Part | Kernel | Jobs | Status |
| :--- | :--- | :--- | :--- |
| A | `qlora-json-extraction-benchmark` | `zero_shot`, `few_shot`, `qlora_sft_seed0` (val+test), `budget200` (val only) | complete |
| B | `qlora-json-extraction-benchmark-b` | `qlora_sft_seed1`, `qlora_sft_seed2` (test only), `budget400` (val only) | planned |

Both parts run `prepare` independently and freeze the identical plan;
`analysis_plan.json` carries `plan_sha256`, the sha256 of the plan with
volatile fields removed, proving the parts share one configuration.
Each part writes its own `metrics.json` and `manifest.json`;
`scripts/merge_results.py` asserts the same git SHA, the same
`plan_sha256`, disjoint job sets, and `ordering_ok` in both manifests
before writing the merged result. Headline comparisons and bootstraps
come from part A; QLoRA per-seed numbers, mean, sd and range aggregate
seeds 0-2 after the merge; the budget curve is 200 (A), 400 (B), 800
(A seed 0).

## Data and evaluation

- CORD-v2 official splits: train 800, validation 100, test 100.
- Dedup: normalized-OCR sha256 plus char_wb 5-gram Jaccard >= 0.8,
  test vs train+validation and validation vs train; 24 of 100 test
  receipts flagged (8 exact-hash pairs).
- Schema: `CordParseV1` frozen from train-only keys; `extra="forbid"`.
- Scoring: parse rate, schema-valid rate, forbidden-key rate,
  extra-commentary rate, field-level micro P/R/F1, Donut nTED accuracy,
  exact match, per-group F1, hallucination/omission, OCR-tertile and
  near-duplicate slices, paired document bootstrap (2000 resamples,
  seed 0, 95% percentile CIs).
- `max_new_tokens` 684 and `max_seq_len` 2076 from train-only p99
  statistics; over-length sequences dropped and counted, never
  truncated.

## Phases

### Phase 1: Package and CPU-safe core (Status: complete)

Package, pinned deps, `CordParseV1` schema, CORD loading, dedup, prompt
construction, parsing, metrics, bootstrap, experiments runner, CLI,
CPU tests, CI.

### Phase 2: Kaggle kernels and frozen protocol (Status: complete)

Part A/B/smoke kernels, shared `analysis_plan.json` with
`plan_sha256`, ordering manifest, resource logging, smoke diagnostics.

### Phase 3: Part A benchmark run (Status: complete)

Part A executed on Kaggle at git `9102c11`; outputs curated unedited
into `results/`; `image.png` rendered via `scripts/render_figure.py`
with a reproduction gate matching `metrics.json` to four decimals.

### Phase 4: Part B benchmark run (Status: planned)

QLoRA seeds 1 and 2 on test plus the 400-receipt budget point on
validation, launched when the weekly GPU quota allows; merged only
when `plan_sha256` and ordering checks pass.

### Phase 5: Reporting (Status: in progress)

README results/findings, limitations, and reproduction docs track the
merged state; until part B lands, QLoRA numbers are single-seed and the
budget curve has two points.
