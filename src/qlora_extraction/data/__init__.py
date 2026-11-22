"""CORD-v2 dataset loading, schema inventory, and near-duplicate audits."""

from qlora_extraction.data.cord import (
    CordDocument,
    cord_documents_from_records,
    load_cord_split,
    normalized_ocr_sha256,
    ocr_text_from_valid_line,
)
from qlora_extraction.data.dedup import (
    DedupReport,
    dedup_audit,
    find_near_duplicates,
    report_to_dict,
)
from qlora_extraction.data.inventory import audit_inventories, inventory_for_docs

__all__ = [
    "CordDocument",
    "DedupReport",
    "audit_inventories",
    "cord_documents_from_records",
    "dedup_audit",
    "find_near_duplicates",
    "inventory_for_docs",
    "load_cord_split",
    "normalized_ocr_sha256",
    "ocr_text_from_valid_line",
    "report_to_dict",
]
