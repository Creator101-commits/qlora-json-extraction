"""Deterministic regex demonstration; this module does not load a trained model."""

from __future__ import annotations

import json
import re

from qlora_extraction.schema import ReceiptSchema, parse_and_validate_receipt_json


class ReceiptExtractor:
    """Extract dollar-denominated receipt fields with explicit heuristic variants."""

    def __init__(self, condition: str = "heuristic_items") -> None:
        aliases = {
            "heuristic_minimal": "minimal",
            "heuristic_totals": "totals",
            "heuristic_items": "items",
            "zero_shot": "minimal",
            "few_shot": "totals",
            "qlora": "items",
        }
        if condition not in aliases:
            raise ValueError(f"Unknown extraction condition: {condition}")
        self.condition = aliases[condition]

    def extract(self, ocr_text: str) -> tuple[ReceiptSchema | None, bool, str]:
        """Return a validated schema, validity flag, and raw heuristic JSON.

        Missing dates remain null; a missing labeled total fails schema validation.
        No total is inferred from tendered cash, change, or the largest price.
        """
        lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]
        merchant = lines[0].title() if lines else ""
        date_match = re.search(r"\b(\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4})\b", ocr_text)
        date = date_match.group(1) if date_match else None
        total_match = re.search(
            r"^\s*(?:GRAND\s+)?TOTAL(?:\s+(?:AMOUNT|DUE|SALE))?\s*:?\s*\$?\s*(\d+\.\d{2})",
            ocr_text,
            re.IGNORECASE | re.MULTILINE,
        )
        total = float(total_match.group(1)) if total_match else None
        tax_match = re.search(
            r"^\s*TAX\b[^\n]*?\$\s*(\d+\.\d{2})\s*$",
            ocr_text,
            re.IGNORECASE | re.MULTILINE,
        )
        tax = float(tax_match.group(1)) if tax_match else None
        subtotal_match = re.search(
            r"^\s*SUBTOTAL\s*:?\s*\$?\s*(\d+\.\d{2})",
            ocr_text,
            re.IGNORECASE | re.MULTILINE,
        )
        subtotal = float(subtotal_match.group(1)) if subtotal_match else None
        fields: dict[str, object] = {
            "merchant_name": merchant,
            "transaction_date": date,
            "total": total,
            "currency": "USD",
        }
        if self.condition in {"totals", "items"}:
            fields.update(subtotal=subtotal, tax=tax)
        if self.condition == "items":
            items = []
            for line in lines[1:]:
                match = re.search(r"^(.*?)\s+\$\s*(\d+\.\d{2})$", line)
                if match and not re.match(
                    r"^(?:GRAND\s+)?(?:TOTAL|SUBTOTAL|TAX|CASH|CHANGE|TENDER|PAID)\b",
                    line,
                    re.IGNORECASE,
                ):
                    description = match.group(1).strip()
                    quantity_match = re.match(r"^(\d+)\s+(.+)$", description)
                    quantity = int(quantity_match.group(1)) if quantity_match else 1
                    if quantity_match:
                        description = quantity_match.group(2)
                    items.append(
                        {
                            "description": description.title(),
                            "quantity": quantity,
                            "price": float(match.group(2)),
                        }
                    )
            fields["line_items"] = items
        raw_json = json.dumps(fields, ensure_ascii=False)
        schema, is_valid, _ = parse_and_validate_receipt_json(raw_json)
        return schema, is_valid, raw_json
