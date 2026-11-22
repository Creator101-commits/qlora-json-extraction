# Review and verification

Scope: source modules, schema and leakage controls, evaluation mathematics,
Kaggle scripts, and README claims.

## Verified locally

- 55 tests passed on the locked environment (Python 3.12/3.13 CI matrix).
- `ruff check .` passed, including Kaggle scripts.
- `mypy src` passed (23 source files).

## Architecture notes

- The retired regex-heuristic path and its synthetic receipts are gone;
  the only remaining CLI command is `version`.
- `CordParseV1` was frozen from train-only key inventory; keys observed
  only in validation or test are reported as schema gaps, not added.
- `menu` is canonicalized to a list whether CORD stores it as dict or list.
- Dedup audit uses normalized-OCR sha256 plus char_wb 5-gram Jaccard >= 0.8
  in both directions (test vs train+val, val vs train); official splits
  stay primary and secondary metrics exclude near-duplicates.
- The runner writes inventory and dedup artifacts plus
  `analysis_plan.json` before any test target is read, and
  `ordering_ok` requires `card_write_epoch < first_test_read_epoch`.
- QLoRA loss masks prompt tokens; only assistant target tokens plus EOS
  carry loss. Over-length sequences are dropped and counted, never
  truncated.

## Limits

GPU paths (QLoRA training, batched generation) are exercised only inside
the Kaggle kernels; local tests inject fake generators. No test metrics
exist until the Kaggle run completes. CORD-v2 is Indonesian receipts with
masked store information and a test set of 100 documents, so bootstrap
intervals are wide.
