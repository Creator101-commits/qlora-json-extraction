"""Extraction of the first balanced top-level JSON object from model output.

The parser scans for the first ``{`` and tracks brace depth while respecting
JSON string literals and escapes. Text outside the balanced object is extra
commentary; a missing or unbalanced object is a parse failure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParsedOutput:
    """Result of parsing one raw model generation."""

    raw_text: str
    json_text: str | None
    obj: Any | None
    has_commentary: bool
    parse_ok: bool


def _find_balanced_object(text: str) -> tuple[int, int] | None:
    """Return ``(start, end)`` of the first balanced top-level ``{...}``."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return start, i + 1
    return None


def extract_first_json_object(text: str) -> tuple[str | None, bool]:
    """Return ``(json_substring, has_commentary)`` for the first object.

    ``json_substring`` is ``None`` when no balanced top-level object exists.
    ``has_commentary`` is True when any non-whitespace text appears outside
    the extracted object (or anywhere, when extraction fails).
    """
    span = _find_balanced_object(text)
    outside = ""
    if span is None:
        outside = text
    else:
        start, end = span
        outside = text[:start] + text[end:]
    has_commentary = bool(outside.strip())
    if span is None:
        return None, has_commentary
    start, end = span
    return text[start:end], has_commentary


def parse_model_output(raw_text: str) -> ParsedOutput:
    """Parse raw generation into JSON, flagging commentary and failures."""
    json_text, has_commentary = extract_first_json_object(raw_text)
    if json_text is None:
        return ParsedOutput(
            raw_text=raw_text,
            json_text=None,
            obj=None,
            has_commentary=has_commentary,
            parse_ok=False,
        )
    try:
        obj = json.loads(json_text)
    except json.JSONDecodeError:
        return ParsedOutput(
            raw_text=raw_text,
            json_text=json_text,
            obj=None,
            has_commentary=has_commentary,
            parse_ok=False,
        )
    return ParsedOutput(
        raw_text=raw_text,
        json_text=json_text,
        obj=obj,
        has_commentary=has_commentary,
        parse_ok=True,
    )
