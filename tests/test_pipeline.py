"""
test_pipeline.py
----------------
Tests unitaires pour les helpers JSONL et la logique du pipeline.
"""

import json
from pathlib import Path

import pytest

from eloquent.pipeline import find_question_field, read_jsonl, write_jsonl


# ---------------------------------------------------------------------------
# Tests helpers JSONL
# ---------------------------------------------------------------------------

class TestReadWriteJsonl:

    def test_write_then_read(self, tmp_path):
        """write_jsonl puis read_jsonl doit restituer les mêmes données."""
        records = [
            {"id": "1", "question": "Bonjour ?", "answer": "Salut"},
            {"id": "2", "question": "Ça va ?",   "answer": "Oui"},
        ]
        path = tmp_path / "test.jsonl"
        write_jsonl(records, path)
        loaded = read_jsonl(path)

        assert len(loaded) == 2
        assert loaded[0]["id"] == "1"
        assert loaded[1]["answer"] == "Oui"

    def test_read_skips_empty_lines(self, tmp_path):
        """Les lignes vides dans un JSONL ne doivent pas planter."""
        path = tmp_path / "sparse.jsonl"
        path.write_text('{"id": "1"}\n\n{"id": "2"}\n', encoding="utf-8")
        records = read_jsonl(path)
        assert len(records) == 2

    def test_unicode_preserved(self, tmp_path):
        """Les caractères non-ASCII (accents, japonais, etc.) sont préservés."""
        records = [{"question": "Quel est le résumé ? 日本語"}]
        path = tmp_path / "unicode.jsonl"
        write_jsonl(records, path)
        loaded = read_jsonl(path)
        assert loaded[0]["question"] == "Quel est le résumé ? 日本語"


class TestFindQuestionField:

    def test_finds_question_key(self):
        assert find_question_field({"question": "test ?"}) == "question"

    def test_finds_query_key(self):
        assert find_question_field({"id": "1", "query": "test ?"}) == "query"

    def test_finds_text_key(self):
        assert find_question_field({"id": "1", "text": "Bonjour"}) == "text"

    def test_returns_none_when_no_string_field(self):
        assert find_question_field({"id": 1, "score": 0.9}) is None

    def test_prioritizes_question_over_query(self):
        record = {"query": "A", "question": "B"}
        assert find_question_field(record) == "question"
