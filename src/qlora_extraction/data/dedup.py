"""Near-duplicate audit over normalized OCR text.

Two signals, either of which flags a pair: identical sha256 of the
whitespace-normalized OCR, or character 5-gram Jaccard similarity of at
least ``JACCARD_THRESHOLD``. The 5-gram set follows the sklearn ``char_wb``
convention: each word is padded with a space on both sides and n-grams never
cross word boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

from qlora_extraction.data.cord import CordDocument, normalized_ocr_sha256

NGRAM_SIZE = 5
JACCARD_THRESHOLD = 0.8


def normalize_text(text: str) -> str:
    """Strip and collapse all whitespace runs to single spaces."""
    return " ".join(text.split())


def char_wb_ngrams(text: str, n: int = NGRAM_SIZE) -> set[str]:
    """Character n-grams within word boundaries (space-padded words)."""
    grams: set[str] = set()
    for word in normalize_text(text).split():
        padded = f" {word} "
        for i in range(len(padded) - n + 1):
            grams.add(padded[i : i + n])
    return grams


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity |A n B| / |A u B|; two empty sets give 1.0."""
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


@dataclass(frozen=True)
class DedupReport:
    """Near-duplicate audit result for one direction of comparison."""

    query_split: str
    reference_splits: tuple[str, ...]
    n_query: int
    n_reference: int
    n_exact_hash_pairs: int
    n_jaccard_pairs: int
    flagged_query_ids: tuple[str, ...]
    pairs: tuple[tuple[str, str, str], ...]
    """``(query_id, reference_id, match_kind)`` for every flagged pair."""


def _ngrams_for(doc: CordDocument) -> tuple[str, set[str]]:
    return normalized_ocr_sha256(doc.ocr_text), char_wb_ngrams(doc.ocr_text)


def find_near_duplicates(
    query_docs: list[CordDocument],
    reference_docs: list[CordDocument],
    *,
    jaccard_threshold: float = JACCARD_THRESHOLD,
) -> DedupReport:
    """Flag query docs that exactly hash-match or Jaccard-match a reference.

    Reference docs are indexed once by normalized-OCR hash and by gram set;
    the Jaccard comparison runs only between docs that do not already share
    an exact hash, to avoid double counting.
    """
    ref_hashes: dict[str, list[str]] = {}
    ref_grams: list[tuple[str, set[str]]] = []
    for ref in reference_docs:
        digest, grams = _ngrams_for(ref)
        ref_hashes.setdefault(digest, []).append(ref.doc_id)
        ref_grams.append((ref.doc_id, grams))

    pairs: list[tuple[str, str, str]] = []
    flagged: set[str] = set()
    n_exact = 0
    n_jaccard = 0
    for query in query_docs:
        digest, grams = _ngrams_for(query)
        exact_refs = ref_hashes.get(digest, [])
        for ref_id in exact_refs:
            pairs.append((query.doc_id, ref_id, "exact_hash"))
            flagged.add(query.doc_id)
        n_exact += len(exact_refs)
        for ref_id, ref_grams_set in ref_grams:
            if ref_id in exact_refs:
                continue
            sim = jaccard(grams, ref_grams_set)
            if sim >= jaccard_threshold:
                pairs.append((query.doc_id, ref_id, f"jaccard_{sim:.4f}"))
                flagged.add(query.doc_id)
                n_jaccard += 1

    return DedupReport(
        query_split=query_docs[0].split if query_docs else "",
        reference_splits=tuple(sorted({d.split for d in reference_docs})),
        n_query=len(query_docs),
        n_reference=len(reference_docs),
        n_exact_hash_pairs=n_exact,
        n_jaccard_pairs=n_jaccard,
        flagged_query_ids=tuple(sorted(flagged)),
        pairs=tuple(sorted(pairs)),
    )


def dedup_audit(
    train_docs: list[CordDocument],
    validation_docs: list[CordDocument],
    test_docs: list[CordDocument],
) -> dict[str, DedupReport]:
    """Official-split audit: test vs train+validation, validation vs train."""
    return {
        "test_vs_train_validation": find_near_duplicates(
            test_docs, train_docs + validation_docs
        ),
        "validation_vs_train": find_near_duplicates(validation_docs, train_docs),
    }


def report_to_dict(report: DedupReport) -> dict[str, object]:
    return {
        "query_split": report.query_split,
        "reference_splits": list(report.reference_splits),
        "n_query": report.n_query,
        "n_reference": report.n_reference,
        "n_exact_hash_pairs": report.n_exact_hash_pairs,
        "n_jaccard_pairs": report.n_jaccard_pairs,
        "n_flagged_query": len(report.flagged_query_ids),
        "flagged_query_ids": list(report.flagged_query_ids),
        "pairs": [list(p) for p in report.pairs],
    }
