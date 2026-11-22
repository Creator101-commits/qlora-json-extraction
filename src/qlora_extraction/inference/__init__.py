"""Model inference: JSON extraction parsing and batched generation."""

from qlora_extraction.inference.generate import (
    GENERATION_BATCH_SIZE,
    GenerationResult,
    generate_texts,
    generation_prompts_for_condition,
    load_model_for_inference,
    prompt_texts_for_docs,
)
from qlora_extraction.inference.parse import (
    ParsedOutput,
    extract_first_json_object,
    parse_model_output,
)

__all__ = [
    "GENERATION_BATCH_SIZE",
    "GenerationResult",
    "ParsedOutput",
    "extract_first_json_object",
    "generate_texts",
    "generation_prompts_for_condition",
    "load_model_for_inference",
    "parse_model_output",
    "prompt_texts_for_docs",
]
