"""Strict Pydantic schemas and error-recovery JSON parser for receipt extraction."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class ReceiptItem(BaseModel):
    """Single line item from an extracted receipt."""

    model_config = ConfigDict(
        extra="forbid", strict=True, allow_inf_nan=False, str_strip_whitespace=True
    )

    description: str = Field(..., min_length=1, description="Item or service name")
    quantity: int = Field(default=1, ge=1, description="Quantity purchased")
    price: float = Field(..., ge=0.0, description="Total price for this line item")


class ReceiptSchema(BaseModel):
    """Canonical structured receipt extraction schema."""

    model_config = ConfigDict(
        extra="forbid", strict=True, allow_inf_nan=False, str_strip_whitespace=True
    )

    merchant_name: str = Field(..., min_length=1, description="Name of the vendor/store")
    transaction_date: str | None = Field(..., description="ISO date, or null when absent")
    currency: str = Field(default="USD", min_length=1, max_length=5)
    subtotal: float | None = Field(default=None, ge=0.0)
    tax: float | None = Field(default=None, ge=0.0)
    total: float = Field(..., ge=0.0, description="Grand total charged")
    line_items: list[ReceiptItem] = Field(default_factory=list)

    @field_validator("transaction_date")
    @classmethod
    def canonical_date(cls, value: str | None) -> str | None:
        if value is None:
            return None
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y"):
            try:
                return datetime.strptime(value, fmt).date().isoformat()
            except ValueError:
                continue
        raise ValueError("transaction_date must be a valid calendar date")


def repair_json_string(text: str) -> str:
    """Apply heuristics to repair common JSON malformations from LLM generations."""
    clean = text.strip()

    match = re.search(r"```(?:json)?\s*(.*?)\s*```", clean, re.DOTALL)
    if match:
        clean = match.group(1).strip()
    start = clean.find("{")
    if start < 0:
        return clean
    clean = clean[start:]
    stack: list[str] = []
    output: list[str] = []
    quoted = escaped = False
    for i, char in enumerate(clean):
        if quoted:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if not stack or stack.pop() != char:
                return clean
        elif char == "," and clean[i + 1 :].lstrip().startswith(("}", "]")):
            continue
        output.append(char)
        if not stack and char == "}":
            break
    if quoted:
        return clean
    clean = "".join(output) + "".join(reversed(stack))

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
