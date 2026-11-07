"""Command-line interface for QLoRA JSON extraction study."""

from __future__ import annotations

from pathlib import Path

import typer

from qlora_extraction import __version__
from qlora_extraction.data.cord import load_cord_dataset
from qlora_extraction.evaluation.metrics import ExtractionMetrics
from qlora_extraction.evaluation.runner import evaluate_extraction_condition
from qlora_extraction.models.inference import ReceiptExtractor
from qlora_extraction.reporting.tables import format_extraction_comparison_table

app = typer.Typer(
    help="QLoRA JSON Extraction CLI.",
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Print package version."""
    typer.echo(__version__)


@app.command()
def extract(
    ocr_file: Path = typer.Argument(..., help="Path to text file containing OCR transcript."),  # noqa: B008
    condition: str = typer.Option(
        "heuristic_items",
        help="Regex variant: heuristic_minimal, heuristic_totals, heuristic_items.",
    ),  # noqa: B008
) -> None:
    """Extract schema-valid structured JSON from receipt OCR transcript."""
    if not ocr_file.exists():
        typer.echo(f"Error: File {ocr_file} not found.")
        raise typer.Exit(code=1)

    ocr_text = ocr_file.read_text(encoding="utf-8")
    extractor = ReceiptExtractor(condition=condition)
    schema, is_valid, raw_json = extractor.extract(ocr_text)

    typer.echo(f"\n--- Extracted Receipt JSON [{condition}] (Valid: {is_valid}) ---")
    if schema:
        typer.echo(schema.model_dump_json(indent=2))
    else:
        typer.echo(raw_json)


@app.command()
def compare() -> None:
    """Evaluate extraction accuracy across zero-shot, few-shot, and QLoRA conditions."""
    docs = load_cord_dataset()
    test_docs = [d for d in docs if d.split == "test"]

    metrics_by_condition: dict[str, ExtractionMetrics] = {
        "Heuristic minimal": evaluate_extraction_condition("zero_shot", test_docs),
        "Heuristic totals": evaluate_extraction_condition("few_shot", test_docs),
        "Heuristic line items": evaluate_extraction_condition("qlora", test_docs),
    }

    typer.echo("\n--- Synthetic Receipt Heuristic Demonstration ---")
    typer.echo(format_extraction_comparison_table(metrics_by_condition))


@app.command()
def report(
    output: Path | None = typer.Option(  # noqa: B008
        None, "--output", "-o", help="Path to write report."
    ),
) -> None:
    """Generate comprehensive research study report."""
    docs = load_cord_dataset()
    test_docs = [d for d in docs if d.split == "test"]

    metrics_by_condition: dict[str, ExtractionMetrics] = {
        "Heuristic minimal": evaluate_extraction_condition("zero_shot", test_docs),
        "Heuristic totals": evaluate_extraction_condition("few_shot", test_docs),
        "Heuristic line items": evaluate_extraction_condition("qlora", test_docs),
    }

    table_md = format_extraction_comparison_table(metrics_by_condition)

    doc_text = (
        "# Receipt Heuristic Demonstration Report\n\n"
        "## Executive Summary\n\n"
        "Five hand-written synthetic receipts exercise deterministic regex extraction. "
        "The table evaluates two synthetic test receipts. No language model is loaded; "
        "legacy condition names are compatibility aliases for heuristic variants. "
        "These numbers are not CORD or QLoRA benchmark results.\n\n"
        "## Heuristic Extraction Metrics\n\n"
        f"{table_md}\n\n"
        "No trained-model accuracy, VRAM usage, or parameter-efficiency result is measured.\n"
    )

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(doc_text, encoding="utf-8")
        typer.echo(f"Report written to {output}")
    else:
        typer.echo(doc_text)


if __name__ == "__main__":
    app()
