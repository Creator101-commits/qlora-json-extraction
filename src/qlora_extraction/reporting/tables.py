"""Markdown tables comparing extraction performance across prompting and fine-tuned conditions."""

from __future__ import annotations

from qlora_extraction.evaluation.metrics import ExtractionMetrics


def format_extraction_comparison_table(
    metrics_by_condition: dict[str, ExtractionMetrics],
) -> str:
    """Format extraction results across conditions into a GitHub markdown table."""
    headers = [
        "Condition",
        "Parse Rate",
        "Schema Valid",
        "Merchant EM",
        "Total EM",
        "Line Item F1",
        "Macro Field F1",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join([":---"] * len(headers)) + " |",
    ]

    for name, m in metrics_by_condition.items():
        parse = f"{m.json_parse_rate * 100:.1f}%"
        valid = f"{m.schema_validity_rate * 100:.1f}%"
        merch = f"{m.merchant_exact_match * 100:.1f}%"
        tot = f"{m.total_exact_match * 100:.1f}%"
        line_f1 = f"{m.line_item_f1 * 100:.1f}%"
        macro = f"{m.macro_field_f1 * 100:.1f}%"

        lines.append(f"| `{name}` | {parse} | {valid} | {merch} | {tot} | {line_f1} | {macro} |")

    return "\n".join(lines)
