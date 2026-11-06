![Banner](./image.png)

# QLoRA JSON Extraction

> Supervised 4-bit QLoRA fine-tuning for schema-valid receipt information extraction from noisy OCR.

![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/Creator101-commits/qlora-json-extraction?include_prereleases)
![GitHub last commit](https://img.shields.io/github/last-commit/Creator101-commits/qlora-json-extraction)
![GitHub issues](https://img.shields.io/github/issues-raw/Creator101-commits/qlora-json-extraction)
![GitHub pull requests](https://img.shields.io/github/issues-pr/Creator101-commits/qlora-json-extraction)
![GitHub](https://img.shields.io/github/license/Creator101-commits/qlora-json-extraction)

A reproducible study comparing zero-shot prompting, few-shot in-context learning, and 4-bit Quantized Low-Rank Adaptation (QLoRA) for extracting strict, schema-valid JSON receipt records from noisy CORD OCR transcripts. Evaluates JSON parse reliability, Pydantic schema validation, field-level exact match, line-item F1 scores, hallucination rates, and parameter efficiency.

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

- **Strict Schema Enforcement**: Pydantic v2 schemas validating vendor name, transaction date, subtotal, tax, grand total, and itemized lines.
- **Robust Error Recovery**: Heuristic JSON repair pipeline that balances unclosed brackets and strips markdown fences.
- **4-Bit QLoRA Fine-Tuning**: Parameter-efficient adaptation updating under 1.5% of parameters with NF4 base quantization.
- **Granular Field Evaluation**: Per-field exact match, line-item F1, hallucination tracking, and omission penalties.
- **Dual Execution Modes**: Fast CPU smoke verification for local development plus Kaggle GPU batch training (`kaggle kernels push`).

## Quickstart

Run a benchmark comparison across zero-shot, few-shot, and QLoRA conditions:

```bash
uv run qlora-extract compare
```

Extract structured JSON from any OCR transcript file:

```bash
uv run qlora-extract extract path/to/ocr_receipt.txt --condition qlora
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
uv run qlora-extract extract sample_receipt.txt --condition qlora
```

Available conditions: `zero_shot`, `few_shot`, `qlora`.

### 3. Benchmark Comparison
```bash
uv run qlora-extract compare
```

### 4. Markdown Report
```bash
uv run qlora-extract report --output artifacts/qlora_benchmark_report.md
```

## Benchmark Results

| Condition | Parse Rate | Schema Valid | Merchant EM | Total EM | Line Item F1 | Macro Field F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Zero-Shot Prompting` | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 75.0% |
| `Few-Shot Prompting` | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 75.0% |
| `4-Bit QLoRA SFT` | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

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
