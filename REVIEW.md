# Review and verification

Scope: source modules, schema and leakage controls, evaluation
mathematics, Kaggle scripts, curated results, and README claims.

## Verified locally

- 65 tests passed on the locked environment (Python 3.12/3.13 CI matrix).
- `ruff check .` passed, including Kaggle scripts and
  `scripts/render_figure.py`.
- `mypy src` passed (24 source files).

## Architecture notes

- The retired regex-heuristic path and its synthetic receipts are gone;
  the only remaining CLI command is `version`.
- `CordParseV1` was frozen from train-only key inventory; keys observed
  only in validation or test are reported as schema gaps, not added.
- `menu` is canonicalized to a list whether CORD stores it as dict or list.
- Dedup audit uses normalized-OCR sha256 plus char_wb 5-gram Jaccard >= 0.8;
  official splits stay primary and secondary metrics exclude
  near-duplicates (24 of 100 test receipts flagged, 8 exact-hash pairs).
- The runner writes inventory and dedup artifacts plus
  `analysis_plan.json` before any test target is read, and
  `ordering_ok` requires `card_write_epoch < first_test_read_epoch`.
- The protocol runs as two Kaggle kernels (parts A and B) because the
  weekly GPU quota cannot hold it in one run. Both parts freeze the same
  plan; `plan_sha256` (plan minus epochs and timestamps) proves it, and
  `scripts/merge_results.py` asserts matching git SHA, plan hash,
  disjoint job sets and `ordering_ok` in both parts before merging.
  Part A's test numbers exist before part B runs; part B is fully frozen
  by the shared plan hash and is planned future work, not yet executed.
- QLoRA loss masks prompt tokens; only assistant target tokens plus EOS
  carry loss. Over-length sequences are dropped and counted, never
  truncated.
- `results/` holds the unedited part A kernel output (verified
  byte-identical to `artifacts/kaggle/20261122-174100`); `image.png` is
  rendered from `results/` only by `scripts/render_figure.py`, whose
  reproduction gate recomputes per-document field F1 with the repo's own
  scorer and aborts unless it matches `metrics.json` to four decimals
  (0.3285 / 0.4122 / 0.9013 for zero-shot / few-shot / QLoRA seed 0).

## Limits

GPU paths (QLoRA training, batched generation) are exercised only inside
the Kaggle kernels; local tests inject fake generators. The smoke kernel
scores a few test documents with toy models (16 train docs, 1 epoch) and
no configuration was changed based on those scores; only engineering
fixes (OOM micro-batch, dataset mount layout) came from smoke evidence.
Part A test metrics exist in `results/`; part B metrics do not exist
until that kernel runs. CORD-v2 is Indonesian receipts with masked store
information and a test set of 100 documents, so bootstrap intervals are
wide, and QLoRA test numbers are a single seed until part B lands.
