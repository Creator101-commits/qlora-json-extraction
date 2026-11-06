"""Strict Pydantic schemas and error-recovery JSON parser for receipt extraction."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError


class ReceiptItem(BaseModel):
    """Single line item from an extracted receipt."""

    description: str = Field(..., min_length=1, description="Item or service name")
    quantity: int = Field(default=1, ge=1, description="Quantity purchased")
    price: float = Field(..., ge=0.0, description="Total price for this line item")


class ReceiptSchema(BaseModel):
    """Canonical structured receipt extraction schema."""

    merchant_name: str = Field(..., min_length=1, description="Name of the vendor/store")
    transaction_date: str = Field(..., min_length=4, description="Date in YYYY-MM-DD or DD/MM/YYYY")
    currency: str = Field(default="USD", min_length=1, max_length=5)
    subtotal: float | None = Field(default=None, ge=0.0)
    tax: float | None = Field(default=None, ge=0.0)
    total: float = Field(..., ge=0.0, description="Grand total charged")
    line_items: list[ReceiptItem] = Field(default_factory=list)


def repair_json_string(text: str) -> str:
    """Apply heuristics to repair common JSON malformations from LLM generations."""
    clean = text.strip()

    # Extract content between ```json ... ``` or first { to last }
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, re.DOTALL)
    if json_match:
        clean = json_match.group(1).strip()
    else:
        start_idx = clean.find("{")
        end_idx = clean.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            clean = clean[start_idx : end_idx + 1]

    # Remove trailing commas before closing braces/brackets
    clean = re.sub(r",\s*(\}|\])", r"\1", clean)

    # Balance missing closing braces if truncated
    open_braces = clean.count("{") - clean.count("}")
    if open_braces > 0:
        clean = clean + ("}" * open_braces)

    return clean


def parse_and_validate_receipt_json(
    raw_text: str,
    attempt_repair: bool = True,
) -> tuple[ReceiptSchema | None, bool, str | None]:
    """Parse raw generation string into validated ReceiptSchema with optional repair."""
    text_to_parse = repair_json_string(raw_text) if attempt_repair else raw_text.strip()

    try:
        data: dict[str, Any] = json.loads(text_to_parse)
    except Exception as exc:
        return None, False, f"JSON parse error: {exc}"

    try:
        validated = ReceiptSchema.model_validate(data)
        return validated, True, None
    except ValidationError as v_exc:
        return None, False, f"Schema validation error: {v_exc}"
