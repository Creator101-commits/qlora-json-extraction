"""CORD-v2 loading: parquet column reads, OCR reconstruction, split handling.

The HF dataset repo stores full receipt images inside the parquet files
(about 2.3 GB total). Only the ``ground_truth`` JSON column is read, via
range requests, so the image bytes are never transferred.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Protocol

import pyarrow.parquet as pq

from qlora_extraction.schema import canonicalize_cord_parse

CORD_V2_REPO = "naver-clova-ix/cord-v2"
CORD_V2_REVISION = "41b075798d0d9da590a950dac1e3d4a21826d4fd"

_RESOLVE_URL = (
    "https://huggingface.co/datasets/"
    f"{CORD_V2_REPO}/resolve/{CORD_V2_REVISION}/data/{{name}}"
)

SPLIT_FILES: dict[str, tuple[str, ...]] = {
    "train": (
        "train-00000-of-00004-b4aaeceff1d90ecb.parquet",
        "train-00001-of-00004-7dbbe248962764c5.parquet",
        "train-00002-of-00004-688fe1305a55e5cc.parquet",
        "train-00003-of-00004-2d0cd200555ed7fd.parquet",
    ),
    "validation": ("validation-00000-of-00001-cc3c5779fe22e8ca.parquet",),
    "test": ("test-00000-of-00001-9c204eb3f4e11791.parquet",),
}
SPLIT_NAMES = ("train", "validation", "test")


class _Openable(Protocol):
    def __enter__(self) -> Any: ...
    def __exit__(self, *args: Any) -> Any: ...


class FsLike(Protocol):
    def open(self, path: str) -> _Openable: ...


def _default_fs() -> FsLike:
    import fsspec

    return fsspec.filesystem("https")  # type: ignore[no-any-return]


@dataclass
class CordDocument:
    """One CORD-v2 receipt.

    ``gt_parse_raw`` holds the untouched ``gt_parse`` JSON string. The
    canonicalized target is materialized lazily through ``get_target`` so
    callers control when ground truth is read; the runner uses that to keep
    test truth unread until ``analysis_plan.json`` is persisted.
    """

    doc_id: str
    split: str
    ocr_text: str
    gt_parse_raw: str
    target: dict[str, Any] | None = field(default=None, repr=False)

    def get_target(self) -> dict[str, Any]:
        """Return the canonicalized ``gt_parse`` dict, parsing on first use."""
        if self.target is None:
            self.target = canonicalize_cord_parse(json.loads(self.gt_parse_raw))
        return self.target

    @property
    def target_json(self) -> str:
        """Deterministic JSON serialization of the canonicalized target."""
        return json.dumps(self.get_target(), ensure_ascii=False, sort_keys=True)


def _word_position(word: dict[str, Any]) -> tuple[float, float]:
    quad = word.get("quad") or {}
    xs = [quad[k] for k in ("x1", "x2", "x3", "x4") if isinstance(quad.get(k), int | float)]
    ys = [quad[k] for k in ("y1", "y2", "y3", "y4") if isinstance(quad.get(k), int | float)]
    return (min(xs) if xs else 0.0, min(ys) if ys else 0.0)


def ocr_text_from_valid_line(valid_line: list[dict[str, Any]]) -> str:
    """Reconstruct a reading-order OCR transcript from CORD ``valid_line``.

    Each entry is one visual line of word boxes. Lines are sorted top to
    bottom, then left to right, using bounding-box coordinates; words inside
    a line are sorted left to right. Entries without text are dropped.
    """
    rows: list[tuple[float, float, str]] = []
    for entry in valid_line:
        words = [w for w in entry.get("words") or [] if w.get("text")]
        if not words:
            continue
        positioned = sorted((_word_position(w), str(w["text"])) for w in words)
        text = " ".join(t for _, t in positioned)
        xs = [p[0] for p, _ in positioned]
        ys = [p[1] for p, _ in positioned]
        rows.append((min(ys), min(xs), text))
    rows.sort(key=lambda r: (r[0], r[1]))
    return "\n".join(text for _, _, text in rows)


def cord_documents_from_records(
    records: list[dict[str, Any]], split: str
) -> list[CordDocument]:
    """Build documents from ``{"ground_truth": str}`` rows."""
    docs: list[CordDocument] = []
    seen: set[str] = set()
    for offset, record in enumerate(records):
        gt = json.loads(record["ground_truth"])
        meta = gt.get("meta") or {}
        image_id = meta.get("image_id", offset)
        doc_id = f"{split}:{image_id}"
        if doc_id in seen:
            raise ValueError(f"duplicate doc_id {doc_id} in split {split}")
        seen.add(doc_id)
        docs.append(
            CordDocument(
                doc_id=doc_id,
                split=split,
                ocr_text=ocr_text_from_valid_line(gt.get("valid_line") or []),
                gt_parse_raw=json.dumps(gt.get("gt_parse"), ensure_ascii=False),
            )
        )
    return docs


def load_cord_split(split: str, *, fs: FsLike | None = None) -> list[CordDocument]:
    """Load one official CORD-v2 split, reading only the ``ground_truth`` column."""
    if split not in SPLIT_FILES:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLIT_NAMES}")
    filesystem = fs if fs is not None else _default_fs()
    docs: list[CordDocument] = []
    for name in SPLIT_FILES[split]:
        url = _RESOLVE_URL.format(name=name)
        with filesystem.open(url) as handle:
            table = pq.ParquetFile(handle).read(columns=["ground_truth"])
        records = [{"ground_truth": gt} for gt in table.column("ground_truth").to_pylist()]
        docs.extend(cord_documents_from_records(records, split))
    return docs


def normalized_ocr_sha256(ocr_text: str) -> str:
    """sha256 of the whitespace-normalized OCR text (dedup exact match)."""
    normalized = " ".join(ocr_text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
