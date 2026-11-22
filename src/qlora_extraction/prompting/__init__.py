"""Chat-template prompt construction and few-shot demonstration selection."""

from qlora_extraction.prompting.demos import (
    N_DEMONSTRATIONS,
    demonstrations_for_prompt,
    select_demonstrations,
)
from qlora_extraction.prompting.templates import (
    SYSTEM_INSTRUCTION,
    USER_TEMPLATE,
    build_messages,
    build_user_content,
    minified_schema_json,
    render_prompt_text,
    schema_token_count,
)

__all__ = [
    "N_DEMONSTRATIONS",
    "SYSTEM_INSTRUCTION",
    "USER_TEMPLATE",
    "build_messages",
    "build_user_content",
    "demonstrations_for_prompt",
    "minified_schema_json",
    "render_prompt_text",
    "schema_token_count",
    "select_demonstrations",
]
