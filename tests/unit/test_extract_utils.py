"""Tests for response JSON extraction."""
from cvinsight.core.utils.extract_utils import extract_json


def test_extract_json_accepts_multiline_json_code_fence():
    assert extract_json('```json\n{"recommendations": ["Use Python"]}\n```') == {
        "recommendations": ["Use Python"]
    }


def test_extract_json_accepts_inline_code_fence():
    assert extract_json('```json {"recommendations": []} ```') == {
        "recommendations": []
    }
