![Banner](./image.png)

# QLoRA JSON Extraction

> CPU receipt-extraction heuristics and a separate CUDA QLoRA pilot for native CORD JSON.

![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/Creator101-commits/qlora-json-extraction?include_prereleases)
![GitHub last commit](https://img.shields.io/github/last-commit/Creator101-commits/qlora-json-extraction)
![GitHub issues](https://img.shields.io/github/issues-raw/Creator101-commits/qlora-json-extraction)
![GitHub pull requests](https://img.shields.io/github/issues-pr/Creator101-commits/qlora-json-extraction)
![GitHub](https://img.shields.io/github/license/Creator101-commits/qlora-json-extraction)

The CPU CLI runs deterministic regex heuristics on five hand-written synthetic receipts. It does not load an LLM, prompts, quantized weights, or a trained adapter. The separate Kaggle script trains TinyLlama with NF4 LoRA on actual CORD OCR words and native `gt_parse` targets. This native target is different from the local `ReceiptSchema`; adapter inference and a fair prompted-versus-fine-tuned benchmark remain unimplemented.

## Table of Contents

- [Features](#features)
- [Quickstart](#quickstart)
- [Installation](#installation)
- [Kaggle GPU Execution](#kaggle-gpu-execution)
- [CLI Reference](#cli-reference)
- [Benchmark Results](#benchmark-results)
- [Development and Testing](#development-and-testing)
- [License](#license)

## Features

- Receipt parsing with forbidden-key checks, finite numeric values, canonical calendar dates, and nullable missing dates.
- Conservative JSON repair that respects quoted strings and bracket nesting.
- Exact normalized merchant scoring and line-item multiset matching including quantity and price.
- A NumPy LoRA algebra demonstration, separate from GPU training.
- A CUDA-only Kaggle training pilot with prompt loss masking, validation checkpoint selection, complete target sequences, exact OCR duplicate filtering, and a run manifest. Near-duplicate auditing remains required.

## Quickstart

Run a benchmark comparison across three heuristic variants:

```bash
uv run qlora-extract compare
```

Extract structured JSON from any OCR transcript file:

```bash
uv run qlora-extract extract path/to/ocr_receipt.txt --condition heuristic_items
```

## Installation

Ensure Python 3.12+ and `uv` are installed:

```bash
git clone https://github.com/Creator101-commits/qlora-json-extraction.git
cd qlora-json-extraction
uv sync
```

## Kaggle GPU Execution

For full 4-bit QLoRA fine-tuning with PEFT and BitsAndBytes on a GPU without straining local hardware, push the included Kaggle kernel:

```bash
kaggle kernels push -p kaggle
```

Check job execution status:

```bash
kaggle kernels status sreeharshak101/qlora-receipt-json-extraction
```

## CLI Reference

### 1. Version Check
```bash
uv run qlora-extract version
```

### 2. Extract from OCR Text
```bash
uv run qlora-extract extract sample_receipt.txt --condition heuristic_items
```

Conditions: `heuristic_minimal`, `heuristic_totals`, `heuristic_items`. Legacy identifiers `zero_shot`, `few_shot`, and `qlora` remain aliases for these regex variants; they do not select models.

### 3. Benchmark Comparison
```bash
uv run qlora-extract compare
```

### 4. Markdown Report
```bash
uv run qlora-extract report --output artifacts/qlora_benchmark_report.md
```

## Benchmark Results

No real CORD test or QLoRA inference accuracy is established. `compare` measures only two synthetic test receipts. Prior claimed QLoRA, VRAM, and learning-curve results were unsupported and have been removed. Raw JSON parsing and schema validity are distinct; evaluation accepts separate parse flags when raw generations are available. The four-component macro score averages merchant match, date match, total match, and line-item F1; it is not an exhaustive field-level F1.

The CUDA runner uses [CORD v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2), extracting input text only from `valid_line[].words[].text` and output labels only from `gt_parse`. It loads train and validation partitions, never test. Its exported adapter is not wired to the CPU CLI. See [QLoRA](https://arxiv.org/abs/2305.14314) for the method; the paper does not establish results for this repository.

## Development and Testing

```bash
# Run pytest with branch coverage
uv run pytest

# Run Ruff linter
uv run ruff check .

# Run static type checker
uv run mypy src
```

## License

Distributed under the MIT License. See [LICENSE](./LICENSE) for details.

## Reproducing the figure

`image.png` contains only Matplotlib graphs. Run `PYTHONPATH=src python scripts/render_figure.py` in a Python environment containing the project dependencies and Matplotlib. The plotting environment is separate from the core runtime when Matplotlib is absent. `scripts/figure-data.json` stores the plotted measurements or analytical inputs. Captions distinguish real packaged datasets, synthetic fixtures, analytical calculations, and local timings. No GPU performance is inferred from CPU demonstrations.
