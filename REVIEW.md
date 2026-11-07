# Review and verification

Scope: source modules, CLI behavior, evaluation mathematics, project claims, Kaggle scripts where present, and README figures.

Separated regex heuristics from QLoRA claims, corrected native CORD OCR inputs, exact scalar and multiset line matching, JSON repair, strict dates and numbers, and missing-value handling.

## Verified locally

- 30 tests passed with the repository's existing Python environment.
- `ruff check .` passed, including tests, figure source, and Kaggle scripts.
- `mypy src` passed.
- The README image was rendered from code using Matplotlib and visually inspected. Its inputs/results and a source fingerprint are in `scripts/figure-data.json`.

## Limits

GPU training targets native CORD gt_parse, not the CPU ReceiptSchema. Adapter inference, near-duplicate audit, and a fair prompted-versus-trained test benchmark remain unimplemented. GPU execution could not be verified here.

Kaggle API access failed because the execution environment could not resolve its hostname. The private notebook was also unavailable through the browser. No GPU training or full-scale external benchmark result is claimed by this review. New CI workflows are configuration changes; their hosted execution must be checked after publication.
