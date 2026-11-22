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
- The protocol runs as two Kaggle kernels (parts A and B) because the
  weekly GPU quota cannot hold it in one run. Both parts freeze the same
  plan; `plan_sha256` (plan minus epochs and timestamps) proves it, and
  `scripts/merge_results.py` asserts matching git SHA, plan hash,
  disjoint job sets and `ordering_ok` in both parts before merging.
  Part A's test numbers exist before part B runs; part B is fully frozen
  by the shared plan hash.
- QLoRA loss masks prompt tokens; only assistant target tokens plus EOS
  carry loss. Over-length sequences are dropped and counted, never
  truncated.

## Limits

GPU paths (QLoRA training, batched generation) are exercised only inside
the Kaggle kernels; local tests inject fake generators. The smoke kernel
scores a few test documents with toy models (16 train docs, 1 epoch) and
no configuration was changed based on those scores; only engineering
fixes (OOM micro-batch, dataset mount layout) came from smoke evidence.
No full-run test metrics exist until the Kaggle run completes. CORD-v2 is
Indonesian receipts with masked store information and a test set of 100
documents, so bootstrap intervals are wide.
