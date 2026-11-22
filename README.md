# QLoRA JSON Extraction

A reproducible benchmark comparing prompted baselines with 4-bit QLoRA
supervised fine-tuning for extracting schema-valid receipt JSON from noisy
OCR text on the CORD-v2 dataset.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Benchmark Design](#benchmark-design)
- [Metrics](#metrics)
- [Kaggle Execution](#kaggle-execution)
- [Development and Testing](#development-and-testing)
- [Results](#results)
- [License](#license)

## Overview

The benchmark asks whether supervised QLoRA improves structured extraction
over zero-shot and few-shot prompting. All conditions share the same base
model (`Qwen/Qwen2.5-3B-Instruct`, revision-pinned, Qwen Research license),
the same chat-template prompt family, greedy decoding, and the same
4-bit NF4 quantized base, so differences come from fine-tuning rather than
quantization.

Input is OCR text reconstructed from CORD-v2 `valid_line` word boxes in
reading order; targets are canonical `gt_parse` JSON validated against the
versioned `CordParseV1` schema. Official splits are train 800, validation
100, test 100. Exact and near-duplicate documents are audited before any
split-dependent choice; test documents never drive prompts, demonstrations,
token limits, or checkpoint selection.

The earlier regex-heuristic path on five hand-written synthetic receipts
has been retired; it was unrelated to the real task.

## Installation

Requires Python 3.12-3.13 and `uv`:

```bash
uv sync                 # CPU-light core only
uv sync --extra gpu     # adds torch, transformers, peft, bitsandbytes
```

## Benchmark Design

Conditions:

1. `zero_shot`: chat-template instruction with the minified `CordParseV1`
   JSON Schema and the OCR text.
2. `few_shot`: the zero-shot prompt plus 3 fixed training demonstrations,
   selected greedily by schema-key coverage, tie-broken by shorter OCR then
   lower index, from train data only.
3. `qlora_sft`: QLoRA on the same zero-shot prompt (prompt tokens masked
   from the loss). NF4 4-bit with double quantization, fp16 compute, LoRA
   r=16 alpha=32 dropout=0.05 on all linear projections, lr 2e-4 cosine
   with 3% warmup, effective batch 16, grad clip 1.0, up to 5 epochs with
   patience 1 on validation loss. Seeds 0, 1, 2.
4. Budget curve: QLoRA at 200 and 400 training receipts (nested subsets,
   seed 0) plus the 800-receipt seed-0 run, evaluated on validation only.

`max_new_tokens` is `ceil(1.25 x p99)` of train target token length;
`max_seq_len` is the p99 of train prompt+target length and over-length
sequences are dropped and counted, never truncated.

Parsing takes the first balanced top-level JSON object; non-whitespace text
outside it counts as extra commentary and a missing or unbalanced object is
a parse failure.

## Metrics

- Structural: JSON parse rate, `CordParseV1` schema-valid rate,
  forbidden-key rate, extra-commentary rate.
- Semantic: Donut-style field-level micro precision/recall/F1 over
  (key path, normalized value) multisets, nTED tree-edit-distance accuracy
  via `zss`, document-level exact match, per-top-level-group F1
  (menu, sub_total, total), hallucination and omission rates.
- Uncertainty: document-level paired bootstrap, 2000 resamples, seed 0,
  percentile 95% CIs; QLoRA reports the mean and range over 3 seeds.
- Operational: train time, peak GPU memory, trainable vs total params,
  adapter size, generation latency and token counts.
- Slices: OCR length tertiles (train cut points) and near-duplicate vs
  novel test documents.

Normalization (frozen before test): strip whitespace, collapse internal
spaces; case and digits are untouched.

The test set is evaluated exactly once after configs and checkpoints are
frozen; `analysis_plan.json` and `manifest.json` record
`card_write_epoch < first_test_read_epoch` as `ordering_ok`.

## Kaggle Execution

The full benchmark runs on a private Kaggle GPU T4 kernel:

```bash
scripts/kaggle_run.sh          # full run
scripts/kaggle_run.sh --smoke  # reduced diagnostic run
```

Each condition/seed runs in its own subprocess with resource logging.
Kernel outputs land in `artifacts/kaggle/<timestamp>/`. The smoke kernel
uses about 16 train / 8 validation / 8 test docs, one epoch, and all
conditions; its outputs are diagnostics only and are never committed as
results.

## Development and Testing

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```

Tests never import the `gpu` extra and never touch the network; they use
tiny synthetic fixtures and injected fake generators.

## Results

Pending the Kaggle run; `results/` will hold unedited kernel output once
the full benchmark completes.

## License

MIT. See [LICENSE](./LICENSE). CORD-v2 is CC BY 4.0; Qwen2.5-3B-Instruct
weights are Qwen Research licensed and adapters are not published.
