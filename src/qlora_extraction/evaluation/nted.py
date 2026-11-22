"""Donut-style normalized tree-edit-distance (nTED) accuracy.

Port of clovaai/donut ``JSONParseEvaluator.cal_acc`` (MIT License,
https://github.com/clovaai/donut/blob/master/donut/util.py). Leaf text
uses this repo's frozen normalization (strip plus collapse whitespace)
where Donut uses ``str.strip()``; a local Levenshtein DP replaces NLTK.
"""

from __future__ import annotations

from typing import Any

import zss
from zss import Node

_LEAF = "<leaf>"
_SUBTREE = "<subtree>"


def _collapse(value: Any) -> str:
    return " ".join(str(value).split())


def normalize_dict(data: Any) -> Any:
    """Donut ``normalize_dict``: sorted keys, scalars wrapped in lists."""
    if not data:
        return {}
    if isinstance(data, dict):
        new_data: dict[str, Any] = {}
        for key in sorted(data.keys(), key=lambda k: (len(k), k)):
            value = normalize_dict(data[key])
            if value:
                if not isinstance(value, list):
                    value = [value]
                new_data[key] = value
        return new_data
    if isinstance(data, list):
        if all(isinstance(item, dict) for item in data):
            return [
                item
                for item in (normalize_dict(i) for i in data)
                if item
            ]
        return [
            text
            for item in data
            if type(item) in {str, int, float}
            and (text := _collapse(item))
        ]
    return [_collapse(data)]


def construct_tree(data: Any, node_name: str | None = None) -> Node:
    """Donut ``construct_tree_from_dict``: <root>, <subtree>, <leaf>."""
    if node_name is None:
        node_name = "<root>"
    node = Node(node_name)
    if isinstance(data, dict):
        for key, value in data.items():
            node.addkid(construct_tree(value, str(key)))
    elif isinstance(data, list):
        if all(isinstance(item, dict) for item in data):
            for item in data:
                node.addkid(construct_tree(item, _SUBTREE))
        else:
            for item in data:
                node.addkid(Node(f"{_LEAF}{item}"))
    else:
        raise ValueError(f"cannot construct tree node from {data!r}")
    return node


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _is_leaf(label: str) -> bool:
    return _LEAF in label


def _leaf_text(label: str) -> str:
    return label.replace(_LEAF, "")


def update_cost(node1: Node, node2: Node) -> int:
    """Donut ``update_cost``: Levenshtein for leaf pairs."""
    label1, label2 = node1.label, node2.label
    leaf1, leaf2 = _is_leaf(label1), _is_leaf(label2)
    if leaf1 and leaf2:
        return _levenshtein(_leaf_text(label1), _leaf_text(label2))
    if leaf1 and not leaf2:
        return 1 + len(_leaf_text(label1))
    if leaf2 and not leaf1:
        return 1 + len(_leaf_text(label2))
    return int(label1 != label2)


def insert_remove_cost(node: Node) -> int:
    """Donut ``insert_and_remove_cost``: leaf text length, else 1."""
    label = node.label
    if _is_leaf(label):
        return len(_leaf_text(label))
    return 1


def _distance(tree1: Node, tree2: Node) -> float:
    return float(
        zss.distance(
            tree1,
            tree2,
            get_children=zss.Node.get_children,
            insert_cost=insert_remove_cost,
            remove_cost=insert_remove_cost,
            update_cost=update_cost,
            return_operations=False,
        )
    )


def nted_score(pred: Any, ref: Any) -> float:
    """Donut ``cal_acc``: max(0, 1 - TED(pred, ref) / TED(empty, ref))."""
    t_pred = construct_tree(normalize_dict(pred))
    t_ref = construct_tree(normalize_dict(ref))
    denom = _distance(construct_tree(normalize_dict({})), t_ref)
    if denom == 0:
        return 1.0 if _distance(t_pred, t_ref) == 0 else 0.0
    return max(0.0, 1.0 - _distance(t_pred, t_ref) / denom)
