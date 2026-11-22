"""Tests for chat-template prompts, schema minification, and demo selection."""

from __future__ import annotations

import json
from typing import Any

from qlora_extraction.data.cord import CordDocument
from qlora_extraction.prompting import (
    build_messages,
    build_user_content,
    demonstrations_for_prompt,
    minified_schema_json,
    render_prompt_text,
    schema_token_count,
    select_demonstrations,
)


class FakeTokenizer:
    eos_token = "<|endoftext|>"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> Any:
        assert tokenize is False
        parts = [f"<{m['role']}>{m['content']}</" for m in messages]
        if add_generation_prompt:
            parts.append("<|assistant|>")
        return "\n".join(parts)

    def encode(self, text: str) -> list[int]:
        return [len(w) for w in text.split()]


def _doc(doc_id: str, target: dict[str, object], ocr: str = "OCR") -> CordDocument:
    return CordDocument(doc_id, "train", ocr, json.dumps(target))


def test_minified_schema_has_no_annotations() -> None:
    schema = minified_schema_json()
    assert '"title"' not in schema
    assert '"description"' not in schema
    parsed = json.loads(schema)
    assert parsed["additionalProperties"] is False
    assert "menu" in parsed["properties"]


def test_schema_token_count_uses_tokenizer() -> None:
    tok = FakeTokenizer()
    assert schema_token_count(tok) == len(tok.encode(minified_schema_json()))


def test_build_messages_zero_shot_shape() -> None:
    msgs = build_messages("zero_shot", "TEH BOTOL", "SCHEMA")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "SCHEMA" in msgs[1]["content"]
    assert "TEH BOTOL" in msgs[1]["content"]


def test_build_messages_few_shot_inserts_pairs() -> None:
    demos = [("OCR1", '{"menu": []}'), ("OCR2", '{"menu": [], "total": {}}')]
    msgs = build_messages("few_shot", "OCR3", "SCHEMA", demos)
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
    assert msgs[2]["content"] == '{"menu": []}'
    assert "OCR3" in msgs[-1]["content"]
    assert "OCR1" in msgs[1]["content"]


def test_render_prompt_text_appends_generation_prompt() -> None:
    tok = FakeTokenizer()
    msgs = build_messages("zero_shot", "OCR", "SCHEMA")
    text = render_prompt_text(tok, msgs)
    assert text.endswith("<|assistant|>")
    assert "<user>" in text


def test_build_user_content_layout() -> None:
    content = build_user_content("MY OCR", "MYSCHEMA")
    assert content.index("MYSCHEMA") < content.index("MY OCR")


def test_select_demonstrations_greedy_coverage() -> None:
    docs = [
        _doc("train:1", {"menu": [{"nm": "A"}], "total": {"total_price": "1"}}),
        _doc("train:2", {"menu": [], "sub_total": {"etc": "x"}, "total": {}}),
        _doc("train:3", {"menu": [{"nm": "C", "sub": [{"nm": "S"}]}]}),
    ]
    chosen = select_demonstrations(docs, n=3)
    # train:3 has the most key paths (nested sub items); train:2 then adds
    # sub_total keys; train:1 fills in last.
    assert [d.doc_id for d in chosen] == ["train:3", "train:2", "train:1"]


def test_select_demonstrations_tie_break_shorter_ocr() -> None:
    big = {"menu": [{"nm": "A", "cnt": "1", "price": "2"}], "total": {"total_price": "3"}}
    docs = [
        _doc("train:1", big, ocr="x" * 500),
        _doc("train:2", big, ocr="short"),
    ]
    chosen = select_demonstrations(docs, n=1)
    assert chosen[0].doc_id == "train:2"


def test_select_demonstrations_second_pick_covers_new_keys() -> None:
    docs = [
        _doc("train:1", {"menu": [{"nm": "A"}], "total": {"total_price": "1"}}),
        _doc(
            "train:2",
            {"menu": [{"nm": "B"}], "sub_total": {"tax_price": "2"}, "total": {"total_price": "9"}},
        ),
        _doc("train:3", {"menu": [{"nm": "C"}], "void_menu": [{"nm": "V"}]}),
    ]
    chosen = select_demonstrations(docs, n=2)
    # Round 1: train:2 has the most keys (adds sub_total). Round 2: train:3
    # adds the void_menu group while train:1 adds only total.total_price.
    assert chosen[0].doc_id == "train:2"
    assert chosen[1].doc_id == "train:3"


def test_demonstrations_for_prompt_pairs() -> None:
    doc = _doc("train:1", {"menu": []})
    doc.get_target()
    pairs = demonstrations_for_prompt([doc])
    assert pairs == [(doc.ocr_text, doc.target_json)]


def test_demo_selection_deterministic() -> None:
    docs = [
        _doc(f"train:{i}", {"menu": [{"nm": f"I{i}"}], "total": {"total_price": str(i)}})
        for i in range(6)
    ]
    a = [d.doc_id for d in select_demonstrations(docs)]
    b = [d.doc_id for d in select_demonstrations(docs)]
    assert a == b
    assert len(a) == 3
