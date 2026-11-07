# Review and verification

Scope: source modules, CLI behavior, evaluation mathematics, project claims, Kaggle scripts where present, and README figures.

Separated regex heuristics from QLoRA claims, corrected native CORD OCR inputs, exact scalar and multiset line matching, JSON repair, strict dates and numbers, and missing-value handling.

## Verified locally

- 30 tests passed with the repository's existing Python environment.
- `ruff check .` passed, including tests, figure source, and Kaggle scripts.
- `mypy src` passed.
- The README image was rendered from code using Matplotlib and visually inspected. Its inputs/results and a source fingerprint are in `scripts/figure-data.json`.

## Limits

GPU training targets native CORD gt_parse, not the CPU ReceiptSchema. On November 6, 2026, the revised Kaggle script completed 50 CUDA training steps using 768 training and 89 validation examples, and saved an adapter and run manifest. The selected validation loss was approximately 0.0803. No test accuracy was measured. Adapter inference, near-duplicate audit, and a fair prompted-versus-trained test benchmark remain unimplemented.

Kaggle access was unavailable during the initial review and restored for the later pilot. New CI workflows are configuration changes; their hosted execution must be checked after publication.
