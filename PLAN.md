> This document is a roadmap. See README.md for implemented behavior and verification limits; planned capabilities are not measured results.

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

## Proposed file structure

```text
.
├── .github/workflows/ci.yml
├── configs/
│   ├── data/cord.yaml
│   ├── models/baseline.yaml
│   ├── training/qlora-sft.yaml
│   ├── training/synthetic-ablation.yaml
│   └── training/dpo.yaml
├── data/
│   ├── raw/                 # ignored source data
│   ├── processed/           # ignored JSONL/Parquet
│   ├── synthetic/           # ignored generated examples
│   └── manifests/
├── src/json_extractor/
│   ├── cli.py
│   ├── config.py
│   ├── schemas/
│   │   ├── receipt.py
│   │   └── normalization.py
│   ├── data/
│   │   ├── download.py
│   │   ├── prepare.py
│   │   ├── deduplicate.py
│   │   └── synthetic.py
│   ├── training/
│   │   ├── qlora.py
│   │   ├── sft.py
│   │   └── dpo.py
│   ├── inference/
│   │   ├── baseline.py
│   │   ├── adapter.py
│   │   └── parsing.py
│   ├── evaluation/
│   │   ├── structural.py
│   │   ├── fields.py
│   │   └── bootstrap.py
│   └── reporting/
│       ├── tables.py
│       └── plots.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── adapters/                     # ignored checkpoints
├── artifacts/                    # ignored outputs
├── docs/data-card.md
├── docs/research-report.md
├── pyproject.toml
└── README.md
```

## Build phases

### Phase 1: Project foundation (Status: Completed)

- **Work:** Create the package, locked dependencies, configuration system, CLI, schema models, CPU-safe tests, GPU checks, linting, typing, and CI.
- **Verify:** Run schema fixtures, CLI help, test suite, Ruff, mypy, and a clear no-GPU failure path.
- **Complete when:** Local tooling works without downloading a base model and CI passes.

### Phase 2: Data and output-schema audit (Status: Completed)

- **Work:** Pin the CORD revision, inspect its actual fields, measure target-field coverage, define the v1 receipt JSON schema, and document licensing and limitations.
- **Verify:** Parse representative annotations and ensure every chosen output field is consistently supported.
- **Complete when:** The schema and source-data contract are frozen before prompt generation.

### Phase 3: Preparation and leakage controls (Status: Completed)

- **Work:** Reconstruct OCR text, convert annotations to canonical JSON, detect duplicates, create grouped partitions, build prompts, and write manifests.
- **Verify:** Test deterministic hashes, split disjointness, schema-valid targets, and document/layout isolation.
- **Complete when:** Every example traces to one source document and test records are inaccessible to training preparation.

### Phase 4: Evaluation and prompted baselines (Status: Completed)

- **Work:** Implement strict parsing, schema validation, normalization, zero-shot/few-shot inference, structural metrics, field metrics, hallucination, omission, and latency recording.
- **Verify:** Use hand-authored fixtures covering valid, malformed, missing, extra, and incorrectly typed fields.
- **Complete when:** Frozen-model baselines can be evaluated before any fine-tuning.

### Phase 5: QLoRA training pipeline (Status: Completed)

- **Work:** Implement pinned model/tokenizer loading, 4-bit quantization, LoRA adapters, target masking, batching, checkpointing, resume behavior, and experiment tracking.
- **Verify:** Overfit a tiny fixture, reload the adapter, and reproduce its outputs; confirm CPU tests do not require CUDA.
- **Complete when:** One configured GPU command produces a valid reusable adapter and training manifest.

### Phase 6: Supervised fine-tuning study (Status: Completed)

- **Work:** Run bounded training-size and adapter-setting experiments on train/validation data, track loss and compute, and choose one final SFT configuration.
- **Verify:** Confirm checkpoint selection never uses test documents and every run records model/data revisions and seeds.
- **Complete when:** The final QLoRA adapter and decoding settings are frozen.

### Phase 7: Synthetic-data ablation (Status: Completed)

- **Work:** Generate and review training-only synthetic examples, then compare real-only with real-plus-synthetic conditions at matched real-data and training budgets.
- **Verify:** Measure field-level gains, hallucination changes, provenance completeness, and contamination controls.
- **Complete when:** Synthetic-data effects are reported separately and cannot influence the real test labels.

### Phase 8: Frozen test evaluation (Status: Completed)

- **Work:** Evaluate prompted baselines and selected QLoRA models on held-out documents using frozen parsing, normalization, schema, and metrics.
- **Verify:** Calculate paired document-bootstrap intervals and preserve raw outputs, parsed outputs, errors, and configuration hashes.
- **Complete when:** Final results distinguish JSON validity from actual field correctness.

### Phase 9: Optional preference tuning (Status: Completed)

- **Work:** Only if justified, build reviewed chosen/rejected pairs from development failures and run a small DPO experiment on top of SFT.
- **Verify:** Check pair quality, partition isolation, and evaluation under the same frozen protocol.
- **Complete when:** DPO is reported as an optional comparison, including a neutral or negative result when applicable.

### Phase 10: Documentation and release (Status: Completed)

- **Work:** Complete the data card, schema documentation, model card, research report, README, GPU reproduction steps, and limitations.
- **Verify:** Run clean local setup, CPU tests, linting, typing, and the documented GPU smoke workflow.
- **Complete when:** Another developer can reproduce the preparation, training, and evaluation without undocumented steps.

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
