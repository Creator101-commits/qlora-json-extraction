"""Chat-template prompt construction for the CORD extraction benchmark.

Zero-shot, few-shot, and QLoRA conditions share one prompt family: a system
instruction, then a user turn carrying a minified ``CordParseV1`` JSON
schema and the OCR transcript. Few-shot inserts three fixed train
demonstrations as user/assistant pairs before the final user turn.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from qlora_extraction.schema import CordParseV1

SYSTEM_INSTRUCTION = (
    "You are an information extraction engine. Read the OCR transcript of a "
    "receipt and respond with a single JSON object that follows the provided "
    "JSON schema. Output only the JSON object, with no commentary."
)

USER_TEMPLATE = (
    "Extract the receipt fields from this OCR transcript.\n\n"
    "JSON Schema:\n{schema_json}\n\n"
    "OCR transcript:\n{ocr_text}"
)

SCHEMA_TARGET_GROUPS = ("menu", "sub_total", "total", "void_menu")


class ChatTemplateTokenizer(Protocol):
    """Minimal tokenizer surface used by prompt rendering and token counts."""

    eos_token: str

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> Any: ...

    def encode(self, text: str) -> list[int]: ...


def _strip_annotations(node: Any) -> Any:
    """Remove ``title`` and ``description`` keys recursively for minification."""
    if isinstance(node, dict):
        return {
            key: _strip_annotations(value)
            for key, value in node.items()
            if key not in {"title", "description"}
        }
    if isinstance(node, list):
        return [_strip_annotations(item) for item in node]
    return node


def minified_schema_json() -> str:
    """Compact, annotation-free ``CordParseV1`` JSON Schema string."""
    schema = _strip_annotations(CordParseV1.model_json_schema())
    return json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def schema_token_count(tokenizer: ChatTemplateTokenizer) -> int:
    """Token count of the minified schema JSON under the given tokenizer."""
    return len(tokenizer.encode(minified_schema_json()))


def build_user_content(ocr_text: str, schema_json: str) -> str:
    return USER_TEMPLATE.format(schema_json=schema_json, ocr_text=ocr_text)


def build_messages(
    condition: str,
    ocr_text: str,
    schema_json: str,
    demonstrations: list[tuple[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Build chat messages for ``zero_shot`` or ``few_shot``.

    ``demonstrations`` is a list of ``(ocr_text, target_json)`` pairs; it is
    required for ``few_shot`` and ignored otherwise.
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
    if condition == "few_shot":
        for demo_ocr, demo_target in demonstrations or []:
            messages.append({"role": "user", "content": build_user_content(demo_ocr, schema_json)})
            messages.append({"role": "assistant", "content": demo_target})
    elif condition != "zero_shot":
        raise ValueError(f"unknown prompt condition {condition!r}")
    messages.append({"role": "user", "content": build_user_content(ocr_text, schema_json)})
    return messages


def render_prompt_text(
    tokenizer: ChatTemplateTokenizer,
    messages: list[dict[str, str]],
) -> str:
    """Render messages through the tokenizer's chat template.

    ``add_generation_prompt=True`` appends the assistant header so the model
    continues directly into the target JSON.
    """
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return str(rendered)


def encode_prompt(
    tokenizer: ChatTemplateTokenizer, prompt_text: str
) -> list[int]:
    return tokenizer.encode(prompt_text)
