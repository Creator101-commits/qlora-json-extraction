![Banner](./image.png)

# QLoRA JSON Extraction

> A reproducible benchmark of prompted baselines versus 4-bit QLoRA
> supervised fine-tuning for schema-valid receipt JSON extraction from
> noisy OCR text.

> **Status: part A complete; part B is planned future work.** All numbers
> below come from the part A run (`results/`); the part B jobs are fully
> specified but have not been executed.

![GitHub last commit](https://img.shields.io/github/last-commit/Creator101-commits/qlora-json-extraction)
![GitHub issues](https://img.shields.io/github/issues-raw/Creator101-commits/qlora-json-extraction)
![GitHub pull requests](https://img.shields.io/github/issues-pr/Creator101-commits/qlora-json-extraction)
![GitHub](https://img.shields.io/github/license/Creator101-commits/qlora-json-extraction)

This project asks whether supervised QLoRA improves structured extraction
over zero-shot and few-shot prompting. All conditions share the same base
model (`Qwen/Qwen2.5-3B-Instruct`, revision-pinned), the same chat-template
prompt family, greedy decoding, and the same 4-bit NF4 quantized base, so
differences come from fine-tuning rather than quantization. The benchmark
runs on private Kaggle GPU T4 kernels; local development is CPU-safe and
uses fixtures only.

## Table of Contents

- [Features](#features)
- [Quickstart](#quickstart)
- [Installation](#installation)
- [Methodology](#methodology)
- [Results](#results)
- [Findings](#findings)
- [Future plan](#future-plan)
- [Limitations](#limitations)
- [Reproduction](#reproduction)
- [Development and Testing](#development-and-testing)
- [Provenance](#provenance)
- [Reproducing the figure](#reproducing-the-figure)
- [License](#license)

## Features

- **Real CORD-v2 benchmark**: official train/validation/test splits
  (800/100/100), OCR text reconstructed from `valid_line` word boxes in
  reading order, targets canonicalized to the versioned `CordParseV1`
  schema.
- **Leakage controls**: exact and near-duplicate audit before any
  split-dependent choice; `analysis_plan.json` and `manifest.json` record
  `ordering_ok` (`card_write_epoch < first_test_read_epoch`); test targets
  are never read before the plan is frozen.
- **Three evaluated conditions**: zero-shot, few-shot (3 fixed train
  demonstrations), and QLoRA SFT, plus a training-budget point at 200
  receipts on validation.
- **Donut-style scoring**: field-level micro precision/recall/F1 over
  (key path, normalized value) multisets, nTED accuracy ported from
  `clovaai/donut`, structural rates, hallucination/omission, paired
  document bootstrap CIs.
- **Two-part Kaggle protocol**: part A (this release) covers zero-shot,
  few-shot, QLoRA seed 0, and the 200-receipt budget point; part B is
  frozen by the same `plan_sha256` and merges only when it matches.

## Quickstart

```bash
uv sync                 # CPU-light core only
uv sync --extra gpu     # adds torch, transformers, peft, bitsandbytes
uv run pytest           # CPU-safe tests, no network, no model downloads
```

The package is a benchmark harness, not a service: the only CLI command
is `qlora-extract version`. All real runs happen inside the Kaggle
kernels described under [Reproduction](#reproduction).

## Installation

Requires Python 3.12-3.13 and `uv`:

```bash
git clone https://github.com/Creator101-commits/qlora-json-extraction.git
cd qlora-json-extraction
uv sync
```

CORD-v2 `ground_truth` columns are read over HTTPS range requests inside
the kernels; the 2.3 GB image payloads are never transferred. Model
weights are fetched inside the kernel as well, so nothing large is
committed.

## Methodology

**Data and splits.** CORD-v2 (`naver-clova-ix/cord-v2` @ `41b07579`),
Indonesian receipt images with OCR word boxes and `gt_parse`
annotations. The official splits are kept: train 800, validation 100,
test 100. `gt_parse` is canonicalized into `CordParseV1` (menu items with
`nm`/`cnt`/`price`/`unitprice` and optional `sub` items, `sub_total`,
`total`, `void_menu`; `extra="forbid"` rejects unknown keys). CORD stores
`menu`, `sub` and `void_menu` inconsistently as dict or list; the
canonical form always uses lists. The schema was frozen from a
train-only key inventory; keys seen only in validation or test are
reported as schema gaps, never added.

**Deduplication.** Before any split-dependent choice, a pair is flagged
when normalized-OCR sha256 matches exactly or char_wb 5-gram Jaccard
reaches 0.8. The audit runs test vs train+validation and validation vs
train. 24 of 100 test receipts are flagged (8 exact-hash pairs); the
official splits stay primary and near-duplicate-excluded metrics are
reported alongside.

**Prompts.** All conditions use the model's chat template. Zero-shot
sends an instruction plus the minified `CordParseV1` JSON Schema and the
OCR text. Few-shot adds 3 fixed training demonstrations selected greedily
by schema-key coverage, tie-broken by shorter OCR then lower index, from
train data only. QLoRA trains on the zero-shot prompt with prompt tokens
masked from the loss.

**Training.** `Qwen/Qwen2.5-3B-Instruct` @ `aa8e7253`, 4-bit NF4 with
double quantization, fp16 compute, LoRA r=16 alpha=32 dropout=0.05 on all
linear projections, lr 2e-4 cosine with 3% warmup, effective batch 16
(2 x 8 accumulation), grad clip 1.0, up to 5 epochs with patience 1 on
validation loss. `max_new_tokens` 684 and `max_seq_len` 2076 come from
train-only p99 statistics; over-length sequences are dropped and counted,
never truncated (8 of 800 train receipts for seed 0, 4 of 200 for the
budget point). Checkpoint selection uses validation loss only; the test
split is evaluated exactly once after everything is frozen.

**Scoring.** Parsing takes the first balanced top-level JSON object;
non-whitespace text outside it counts as extra commentary and a missing
or unbalanced object is a parse failure. Metrics: parse rate,
schema-valid rate, forbidden-key rate, extra-commentary rate; field-level
micro precision/recall/F1 over (key path, normalized value) multisets;
Donut nTED accuracy `max(0, 1 - TED(pred, ref) / TED(empty, ref))` via
`zss` (port of `clovaai/donut` `JSONParseEvaluator.cal_acc`, MIT);
document exact match; per-group F1 (menu, sub_total, total);
hallucination and omission rates; paired document bootstrap (2000
resamples, seed 0, percentile 95% CIs); OCR-length tertile and
near-duplicate slices.

**Ordering.** `analysis_plan.json` is persisted before the first test
target is read; `manifest.json` records `ordering_ok` and the git SHA.
`plan_sha256` (the plan minus volatile epochs/timestamps) proves the two
Kaggle parts share one frozen configuration.

## Results

Part A run at git `9102c11` on a Tesla T4 (kernel
`qlora-json-extraction-benchmark`), `ordering_ok: true`, smoke `false`.
Test split, n = 100:

| Condition | Parse | Schema-valid | Field F1 [95% CI] | nTED | Exact match | Forbidden key | Commentary |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero-shot | 0.94 | 0.78 | 0.329 [0.290, 0.365] | 0.489 | 0.00 | 0.16 | 0.04 |
| Few-shot (3 demos) | 0.83 | 0.62 | 0.412 [0.355, 0.471] | 0.510 | 0.00 | 0.21 | 0.28 |
| QLoRA seed 0 | 0.99 | 0.97 | 0.901 [0.855, 0.938] | 0.926 | 0.62 | 0.02 | 0.01 |

Paired bootstrap differences in field F1 (same 100 documents):

| Comparison | Mean delta [95% CI] |
| :--- | ---: |
| QLoRA - few-shot | +0.489 [+0.430, +0.552] |
| QLoRA - zero-shot | +0.573 [+0.517, +0.623] |
| few-shot - zero-shot | +0.084 [+0.010, +0.160] |

Near-duplicate slice (24 flagged test receipts): QLoRA field F1 is 0.904
on the 76 novel receipts versus 0.891 on flagged ones, so memorization of
near-duplicates does not explain the gain.

Validation split: QLoRA seed 0 field F1 0.947; the 200-receipt budget
point reaches 0.866. The budget curve therefore has two points (200 and
the 800-receipt seed-0 run); 400 is part B.

Operational: 29,933,568 trainable LoRA parameters; adapter 135.7 MB;
peak GPU memory about 10 GB; seed-0 training 12,050 s and budget-200
training 3,325 s; all four jobs about 5.3 GPU-hours on one T4. Seed 0
selected epoch 4 of 5; budget 200 selected epoch 5, the cap.

## Findings

- **QLoRA transforms the task.** Field F1 rises from 0.329 (zero-shot)
  and 0.412 (few-shot) to 0.901, exact match goes from 0 to 0.62, and
  schema validity from 0.78 to 0.97. Hallucination drops from 0.62 to
  0.07 and omission from 0.69 to 0.08.
- **The gain is not memorization.** On the 76 test receipts with no
  near-duplicate in train/validation, QLoRA still scores 0.904 field F1.
- **Few-shot prompting backfires on format.** Adding demonstrations
  lowers schema validity (0.62 vs 0.78), raises forbidden keys (0.21 vs
  0.16) and extra commentary (0.28 vs 0.04); it buys a small field-F1
  gain (+0.084, CI excludes zero) at a real compliance cost.
- **Most of the data-efficiency curve is already visible.** 200 training
  receipts reach validation F1 0.866 versus 0.947 at 800, though the
  budget point hit the 5-epoch cap and may be undertrained.
- **Residual errors track OCR difficulty.** QLoRA test F1 is 0.957 /
  0.944 / 0.847 across OCR-length tertiles; the hardest third costs about
  0.11 F1, the main remaining weakness.

## Future plan

Part B (`kaggle_b/`, kernel `qlora-json-extraction-benchmark-b`) is fully
specified and frozen by the same `plan_sha256`; it was not launched for
this release:

- `qlora_sft_seed1` and `qlora_sft_seed2`: test generation only (their
  validation outputs drive nothing), giving the QLoRA mean/sd/range over
  3 seeds.
- `budget400`: the 400-receipt budget point on validation, completing the
  200/400/800 curve.
- `scripts/merge_results.py` merges part B into the part A results only
  when the git SHA, `plan_sha256`, disjoint job sets, and `ordering_ok`
  all match.

Because the merge requires an identical git SHA, launch part B from the
part A commit `9102c11` (for example from a `git worktree` checkout)
with `scripts/kaggle_run.sh --part b`.

Until part B lands, every QLoRA number here is a single seed and the
budget curve has one interior point.

## Limitations

- **Single seed.** QLoRA test numbers are seed 0 only; seed variance is
  unmeasured until part B.
- **Small test set.** n = 100 gives wide bootstrap intervals; slice
  numbers (tertiles, near-duplicates) are noisier still.
- **Near-duplicates.** 24% of test receipts have a near-duplicate in
  train/validation; the novel-only F1 mitigates but does not eliminate
  the concern.
- **One GPU.** All runs are Tesla T4; no cross-hardware or larger-model
  comparison exists.
- **4-bit only.** No fp16/bf16 QLoRA or full-finetuning comparison;
  quantization effects are unmeasured.
- **Budget cap.** The 200-receipt run selected the last epoch, so its
  validation score may underestimate that budget.
- **Domain.** CORD-v2 is Indonesian receipts with masked store
  information; results may not transfer to other languages or schemas.
- **Smoke diagnostics.** The smoke kernel scored a handful of test
  documents with toy models; no configuration was selected from those
  scores (only engineering fixes: OOM micro-batch, dataset mount layout).

## Reproduction

Part A (this release):

```bash
scripts/kaggle_run.sh          # full part A run on Kaggle GPU T4
scripts/kaggle_run.sh --smoke  # both parts on tiny subsets + merge check
```

Part B (after the weekly GPU quota resets):

```bash
scripts/kaggle_run.sh --part b
```

The script archives `HEAD` into the private Kaggle source dataset
(`sreeharshak101/qlora-json-extraction-src`), pushes the kernel id from
the matching `kernel-metadata.json`, polls to completion, and downloads
outputs to `artifacts/kaggle/<timestamp>/` with a git-SHA guard. Curate
the listed outputs into `results/` unedited; `results/` here is exactly
that copy. `scripts/merge_results.py` performs the part B merge checks.

Kernels: part A
`kaggle.com/code/sreeharshak101/qlora-json-extraction-benchmark`, part B
`.../qlora-json-extraction-benchmark-b`, smoke
`.../qlora-json-extraction-smoke`.

## Development and Testing

```bash
uv sync --locked --all-groups   # install with dev dependencies
uv run pytest                   # CPU-safe suite; fakes, no network
uv run ruff check .             # lint, including Kaggle scripts
uv run mypy src                 # strict type check
```

Tests never import the `gpu` extra and never touch the network; they use
tiny synthetic fixtures and injected fake generators. GPU paths
(QLoRA training, batched generation) are exercised only inside the
Kaggle kernels.

## Provenance

- Dataset: `naver-clova-ix/cord-v2` @ `41b075798d0d9da590a950dac1e3d4a21826d4fd` (CC BY 4.0).
- Base model: `Qwen/Qwen2.5-3B-Instruct` @ `aa8e7253` (Qwen Research license); adapters are not published.
- nTED: port of `clovaai/donut` `JSONParseEvaluator.cal_acc` (MIT).
- Kernel run: git `9102c11`, `plan_sha256` `f0cbe101...` (full value in `results/analysis_plan.json`), `ordering_ok: true`.
- `results/` holds unedited kernel output; `image.png` renders from `results/` only.

## Reproducing the figure

`image.png` is rendered by `scripts/render_figure.py` from `results/`
only. Panel (a) rescores the saved predictions with the repo's own
scorer and a reproduction gate aborts the render if the recomputed micro
field F1 does not match `results/metrics.json` to four decimals. It needs
network access once to read the CORD-v2 `ground_truth` column.

```bash
uv run --with matplotlib python scripts/render_figure.py
```

## License

MIT. See [LICENSE](./LICENSE). CORD-v2 is CC BY 4.0; Qwen2.5-3B-Instruct
weights are Qwen Research licensed and adapters are not published.
