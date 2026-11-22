"""Greedy few-shot demonstration selection from the train split only.

Each round picks the train document that adds the most not-yet-covered
schema key paths in its canonicalized target. Ties break toward the
shorter OCR transcript, then the lower train index. The selected
demonstrations are fixed for the whole benchmark and recorded in the
analysis plan.
"""

from __future__ import annotations

from qlora_extraction.data.cord import CordDocument
from qlora_extraction.schema import key_paths

N_DEMONSTRATIONS = 3


def _doc_key_paths(doc: CordDocument) -> set[str]:
    return key_paths(doc.get_target())


def select_demonstrations(
    train_docs: list[CordDocument], n: int = N_DEMONSTRATIONS
) -> list[CordDocument]:
    """Greedily select ``n`` train documents maximizing key coverage."""
    if n <= 0:
        return []
    candidates = list(enumerate(train_docs))
    covered: set[str] = set()
    chosen: list[CordDocument] = []
    for _ in range(n):
        best_key: tuple[int, int, int] | None = None
        best_doc: CordDocument | None = None
        for index, doc in candidates:
            new_keys = len(_doc_key_paths(doc) - covered)
            # Maximize new keys; tie-break shorter OCR, then lower index.
            rank = (new_keys, -len(doc.ocr_text), -index)
            if best_key is None or rank > best_key:
                best_key = rank
                best_doc = doc
        if best_doc is None:
            break
        chosen.append(best_doc)
        covered |= _doc_key_paths(best_doc)
        candidates = [(i, d) for i, d in candidates if d is not best_doc]
    return chosen


def demonstrations_for_prompt(demos: list[CordDocument]) -> list[tuple[str, str]]:
    """Render selected documents as ``(ocr_text, target_json)`` pairs."""
    return [(doc.ocr_text, doc.target_json) for doc in demos]
