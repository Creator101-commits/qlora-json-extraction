# Build Plan: QLoRA Fine-Tuning for Structured JSON Extraction

## Goal

Build a reproducible study of whether supervised QLoRA improves extraction of schema-valid receipt information from noisy OCR text over zero-shot and few-shot prompting. The project will measure structural validity, field accuracy, hallucination, omission, data efficiency, robustness, inference cost, and variability on real held-out documents, with synthetic examples restricted to controlled training ablations.

## Scope

### Included in v1

- Acquire and validate the CORD receipt dataset or an integrity-verifiable equivalent with OCR text and structured annotations.
- Define a narrow, versioned JSON schema covering merchant, transaction, and total fields that are reliably supported by the source annotations.
- Convert document annotations into instruction/input/output examples without mixing documents across partitions.
- Detect exact and near-duplicate documents before finalizing splits.
- Build deterministic zero-shot and few-shot prompted baselines.
- Build one constrained-decoding or repair baseline that does not update model weights.
- Fine-tune a compact instruction model with 4-bit QLoRA and supervised fine-tuning.
- Compare at least two controlled training-data sizes.
- Generate optional synthetic training examples and measure their incremental effect separately.
- Validate every output through a strict parser and Pydantic/JSON Schema model.
- Evaluate JSON parse rate, schema validity, normalized exact match, per-field precision/recall/F1, hallucination, omission, latency, tokens, memory, and adapter size.
- Analyze performance by OCR noise, field type, document length, and seen/unseen layout groups.
- Track model revisions, data hashes, prompts, adapter settings, training curves, seeds, and artifacts with MLflow.
- Add an optional preference-tuning phase only if reviewed chosen/rejected pairs are available and SFT error analysis supports it.
- Provide CPU-safe tests and GPU training commands.

### Explicitly outside v1

- Training a base language model from scratch.
- Full-document OCR model training or image-to-text modeling.
- Using test documents to generate synthetic examples or select checkpoints.
- Claiming that synthetic test examples demonstrate real-world generalization.
- Extraction of arbitrary contract or invoice schemas unrelated to the training labels.
- Production handling of confidential financial documents.
- A hosted document-upload service.
- Full-precision fine-tuning of all base-model parameters.
- Automatic DPO claims without independently checked preference data.
- Guaranteed correctness based only on JSON validity.

## Research questions

1. How much does QLoRA improve field-level extraction over zero-shot, few-shot, and constrained-prompting baselines?
2. Does QLoRA primarily improve JSON structure, semantic field correctness, or both?
3. How does training-data size affect generalization to held-out receipt layouts?
4. Does synthetic training data improve rare-field recall without increasing hallucinated fields?
5. Which error types remain after fine-tuning, and are they related to OCR noise, schema ambiguity, or model generation behavior?

## Stack

- **Language:** Python 3.11
- **Packaging:** `uv`, `pyproject.toml`, and a locked local environment
- **Training:** PyTorch, Transformers, PEFT, TRL, Accelerate, and bitsandbytes
- **Data:** Hugging Face Datasets, pandas, NumPy, PyArrow, JSONL, and Parquet
- **Validation:** Pydantic and JSON Schema
- **Evaluation:** scikit-learn plus verified internal structural and field metrics
- **Tracking:** MLflow; optional Weights & Biases support disabled by default
- **Visualization:** Matplotlib and seaborn
- **CLI and quality:** Typer, pytest, Ruff, mypy, and GitHub Actions
- **GPU environment:** one documented CUDA environment, with a Colab-compatible notebook or script as a secondary entry point rather than the source of core logic

## Architecture

```text
CORD documents + annotations
              |
              v
validation -> deduplication -> grouped train/validation/test split
              |
              v
OCR text + canonical schema-valid JSON
              |
       +------+--------------------+
       |                           |
       v                           v
prompted baselines          training-data builder
                                    |
                           +--------+---------+
                           |                  |
                           v                  v
                      real examples     optional synthetic
                           |                  |
                           +--------+---------+
                                    |
                             QLoRA SFT trainer
                                    |
                         adapter + training manifest
                                    |
       +----------------------------+------------------+
       |                                               |
       v                                               v
baseline inference                              adapter inference
       |                                               |
       +----------------------------+------------------+
                                    |
                     parse -> schema validate -> normalize
                                    |
                                    v
                     field metrics + errors + report
```

### Components

#### `src/json_extractor/data/`

Acquires the benchmark, validates document and annotation records, reconstructs OCR text, detects duplicates, defines grouped partitions, converts annotations to the target schema, creates prompts, and builds optional synthetic training data. Every output carries a source document ID and transformation version.

#### `src/json_extractor/schemas/`

Contains the canonical Pydantic models, JSON Schema export, normalization rules, missing-value policy, field aliases, and schema-version migration helpers. Scoring uses parsed canonical values rather than raw generated string formatting alone.

#### `src/json_extractor/training/`

Loads the pinned base model and tokenizer, applies quantization, attaches LoRA adapters, tokenizes prompt/target pairs, masks prompt tokens correctly, trains with validation checkpointing, and writes adapters plus a complete training manifest. Optional DPO remains a separate trainer and experiment family.

#### `src/json_extractor/inference/`

Implements zero-shot, few-shot, constrained/repair, base-model, and adapter-model inference behind one protocol. It records raw output, parsed output, validation errors, decoding settings, token counts, and latency without altering reference labels.

#### `src/json_extractor/evaluation/`

Normalizes dates, currency strings, whitespace, punctuation, and numeric formatting using frozen rules; then calculates structural, exact, field-level, and error-category metrics. It preserves both strict and normalized results.

#### `src/json_extractor/reporting/`

Produces training curves, overall comparisons, field-level tables, data-efficiency curves, hallucination/omission plots, robustness slices, compute summaries, and representative errors selected by deterministic rules.

#### `src/json_extractor/cli.py`

Exposes thin commands for acquisition, preparation, baseline inference, training, evaluation, comparison, and report generation.

## Data

### Primary dataset

Use **CORD**, a receipt-understanding dataset containing receipt images, OCR-level text information, and structured semantic annotations. V1 uses the provided OCR/annotation information to create text-to-JSON examples; it does not train an OCR or vision model.

The acquisition step must verify the actual released schema before implementation because dataset wrappers may expose different nested field names. No conversion logic may silently assume an undocumented wrapper layout.

### Target JSON schema

The initial schema will include only fields consistently represented in the selected release:

```json
{
  "merchant": {
    "name": "string or null",
    "address": "string or null",
    "phone": "string or null"
  },
  "transaction": {
    "date": "ISO date string or null",
    "time": "normalized time string or null"
  },
  "totals": {
    "subtotal": "decimal string or null",
    "tax": "decimal string or null",
    "total": "decimal string or null"
  }
}
```

Decimal values remain strings in JSON to avoid binary floating-point ambiguity. A field absent from the document must be `null`; unsupported keys are forbidden.

### Example schema

- `example_id`
- `document_id`
- `layout_group_id`
- `ocr_text`
- `target_json`
- `target_schema_version`
- `split`
- `source_dataset_version`
- `ocr_noise_features`
- `synthetic`: boolean
- `provenance`

### Acquisition and integrity

- Use an official or documented Hugging Face distribution with recorded dataset revision and terms.
- Save source/revision identifiers, acquisition date, record counts, feature schema, and file hashes where available.
- Validate image/document IDs, OCR text availability, annotation parseability, and field coverage.
- Preserve source downloads in cache or `data/raw/`; do not commit them.
- Create a data card describing language, receipt domain, annotation limitations, and potential geographic or merchant bias.

### Deduplication and splits

- Hash normalized OCR text for exact duplicates.
- Use image perceptual hashes or layout/text similarity to flag near duplicates when images are available.
- Group related documents before splitting.
- Prefer official train/validation/test partitions when they already prevent leakage; audit rather than blindly trust them.
- If regrouping is necessary, use a deterministic grouped split and document deviations from the benchmark.
- Keep test documents completely unavailable to example selection, synthetic generation, prompt demonstration selection, and checkpoint choice.

### Synthetic data

- Generate examples only after the real-data split is frozen.
- Build generation templates or LLM prompts from training schema and training-only error patterns.
- Attach generator model/prompt revision and seed to every example.
- Validate JSON automatically and manually review a stratified sample.
- Run real-only and real-plus-synthetic conditions at matched real-data budgets.
- Never report synthetic validation/test performance as the main result.

## Model and training design

### Baselines

- Zero-shot instruction with schema included
- Few-shot instruction using training-only examples selected by a fixed rule
- Zero-shot plus constrained structured-output or deterministic repair, when supported
- Frozen base model evaluated with the same maximum tokens and deterministic decoding

### Base model

Start with a compact instruction model that supports the target language and fits a 16 GB-class GPU under 4-bit quantization. A Qwen 2.5 Instruct model in the 3B range is the initial candidate; a 7B model is an optional scale experiment, not required for v1.

The exact model and tokenizer revisions must be pinned. Model license and allowed redistribution of adapters must be documented before publishing artifacts.

### QLoRA configuration

- 4-bit NF4 quantization with an explicitly recorded compute dtype
- LoRA adapters on a predeclared set of attention projections
- Fixed initial rank/alpha/dropout configuration, followed by a small validation-only ablation if compute permits
- Supervised fine-tuning with prompt tokens masked from target loss where appropriate
- Gradient accumulation, clipping, mixed precision, checkpointing, and validation-based early stopping
- Maximum sequence length chosen after measuring training-text lengths
- At least three final seeds if compute permits; otherwise bootstrap documents and state the seed limitation

### Optional preference tuning

DPO is permitted only after SFT error analysis identifies recurring generation failures and a reviewed preference set can be produced. Chosen and rejected outputs must share the same prompt, represent meaningful alternatives, and remain separate from final test labels. The QLoRA-only model remains the primary v1 result.

## Evaluation design

### Structural metrics

- JSON parse rate
- schema-valid rate
- forbidden-key rate
- correct required-key structure rate
- extra-commentary rate

### Semantic metrics

- strict document-level exact match
- normalized document-level exact match
- micro and macro field-level precision, recall, and F1
- per-field exact match
- hallucination rate for fields whose reference is null
- omission rate for fields present in the reference

### Robustness slices

- short versus long OCR text
- low versus high OCR noise proxy
- common versus rare fields
- seen-like versus novel layout groups
- numeric, date/time, and free-text fields
- real-only versus real-plus-synthetic training

### Operational metrics

- training duration and peak GPU memory
- trainable versus total parameter count
- adapter size
- inference latency and throughput
- input/output token counts
- malformed-output retry count if repair is enabled

### Statistical reporting

- Bootstrap documents, not individual fields, for confidence intervals.
- Use paired document bootstrap differences because every model sees the same test documents.
- Report exact counts alongside percentages on small slices.
- Freeze normalization rules before final evaluation.

## Milestones

### 1. Reproducible skeleton

Create the package, locked environment, schemas, configuration system, CLI, CPU-safe tests, linting, typing, CI, and a documented GPU environment check. Ship schema-validation examples and a deterministic fake model for pipeline tests.

### 2. Data acquisition and schema audit

Implement dataset loading with pinned revision, source-schema inspection, annotation validation, field-coverage analysis, and a data card. Ship a report that justifies the exact v1 output fields.

### 3. Deduplicated preparation pipeline

Implement OCR reconstruction, target conversion, normalization, duplicate checks, grouped splits, prompt formatting, manifests, and Parquet/JSONL output. Ship leakage tests and deterministic output hashes.

### 4. Evaluation harness and prompted baselines

Implement strict parsing, schema validation, normalization, field metrics, error categorization, and zero/few-shot inference. Verify metrics with hand-authored fixtures. Ship complete baseline results before fine-tuning.

### 5. QLoRA training pipeline

Implement quantized loading, adapters, target masking, batching, checkpointing, tracking, and resume behavior. Prove the model can overfit a tiny training fixture and produce a loadable adapter before a full run.

### 6. Supervised fine-tuning experiments

Run bounded training-data-size and adapter-setting experiments using validation data. Select one final configuration without test access. Ship curves, manifests, checkpoints, and validation comparisons.

### 7. Synthetic-data ablation

Create and review training-only synthetic examples, then run real-only versus real-plus-synthetic conditions with matched real examples and training budgets. Ship field-specific benefits, regressions, and hallucination analysis.

### 8. Frozen test evaluation

Freeze schema, prompts, decoding, adapters, normalization, and metric code. Evaluate baselines and selected QLoRA conditions on the held-out test documents, calculate paired intervals, and preserve raw outputs for audit.

### 9. Optional preference experiment

If justified, build reviewed preference pairs from training/development failures, run a small DPO experiment on top of QLoRA, and evaluate it under the same frozen protocol. Report it as optional and neutral if it does not improve the primary metric.

### 10. Final research artifact

Write the report with data, leakage controls, schema, baselines, training design, ablations, results, compute, error analysis, limitations, and reproducibility instructions. Verify clean local setup and the documented GPU workflow before tagging v1.

## Open questions

1. **Dataset release:** Confirm the exact CORD distribution, revision, license, OCR fields, and structured annotation format.
2. **Output fields:** Finalize fields only after coverage analysis; remove fields that are too sparse or inconsistently annotated.
3. **Base-model size:** Choose 3B as the default unless measured GPU headroom justifies a 7B experiment.
4. **Model revision:** Select a revision that can be pinned and whose license permits the planned use and adapter publication.
5. **Layout grouping:** Determine whether merchant labels, perceptual hashes, or clustering provides the most defensible group split.
6. **Normalization:** Freeze date, time, phone, address, and decimal normalization rules before final evaluation.
7. **Constrained output:** Decide whether provider/model-specific constrained generation is a separate baseline or part of every inference condition.
8. **Synthetic generator:** Choose template generation or an LLM only after defining provenance, cost, review, and contamination checks.
9. **Compute budget:** Predeclare maximum runs, epochs, sequence length, and GPU hours before tuning.
10. **DPO gate:** Require a minimum reviewed preference-set size and a validation improvement threshold before spending compute on DPO.
