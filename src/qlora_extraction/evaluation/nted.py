"""Normalized tree edit distance (nTED) via Zhang-Shasha ``zss``.

JSON is converted to ordered trees: objects become a ``{}`` node whose
children are key nodes wrapping their values, arrays become ``[]`` nodes
keeping element order, and scalars become leaves labeled by their
normalized value. nTED per document is ``1 - dist / max(size1, size2)``.
"""

from __future__ import annotations

from typing import Any

import zss
from zss.simple_tree import Node

from qlora_extraction.evaluation.normalize import normalize_leaf


def _to_tree(obj: Any) -> Node:
    if isinstance(obj, dict):
        node = Node("{}")
        for key, value in sorted(obj.items()):
            node.addkid(Node(str(key), [_to_tree(value)]))
        return node
    if isinstance(obj, list):
        node = Node("[]")
        for item in obj:
            node.addkid(_to_tree(item))
        return node
    return Node(normalize_leaf(obj))


def _tree_size(node: Node) -> int:
    return 1 + sum(_tree_size(child) for child in node.children)


def nted_score(pred: Any, ref: Any) -> float:
    """Normalized tree edit distance similarity in [0, 1]."""
    t_pred = _to_tree(pred)
    t_ref = _to_tree(ref)
    size = max(_tree_size(t_pred), _tree_size(t_ref))
    if size == 0:
        return 1.0
    distance = float(zss.simple_distance(t_pred, t_ref))
    return max(0.0, 1.0 - distance / size)
