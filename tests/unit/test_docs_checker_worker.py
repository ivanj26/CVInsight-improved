"""Unit tests for the docs checker worker's verdict handling."""
import pytest

from workers.docs_checker_worker import DocsCheckerWorker


@pytest.fixture
def anyio_backend():
    return "asyncio"


class _FakeExtractor:
    def extract(self, file_path: str) -> str:
        return "some document text"


class _FakeAIChecker:
    """Returns a canned AIChecker payload without touching OpenCode."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    async def check(self, text: str) -> dict:
        return self._payload


def _worker(payload: dict) -> DocsCheckerWorker:
    """Build a worker without __init__ — the real one needs a Drive API key."""
    worker = object.__new__(DocsCheckerWorker)
    worker._extractor = _FakeExtractor()
    worker._ai_checker = _FakeAIChecker(payload)
    return worker


def _parse_failure_payload() -> dict:
    """What AIChecker returns when the model replies with unparseable prose."""
    return {
        "parsed": {
            "parse_error": "Could not extract JSON from AI response",
            "raw_content": "This document looks AI generated to me.",
        },
        "raw": {"content": "This document looks AI generated to me.", "usage": {"total_tokens": 10}},
    }


@pytest.mark.anyio
async def test_check_single_file_reports_parse_failure():
    result = await _worker(_parse_failure_payload())._check_single_file(
        1, "doc.pdf", "doc.pdf"
    )

    assert result["status"] == 0
    assert "likelihood_score" in result["error"]
    # Usage must survive so the caller can still write a token-usage record.
    assert result["raw_ai_response"]["usage"]["total_tokens"] == 10


@pytest.mark.anyio
async def test_check_single_file_normalises_string_score():
    payload = {
        "parsed": {"likelihood_score": "80", "reasoning": "robotic"},
        "raw": {"content": "{}", "usage": {"total_tokens": 5}},
    }

    result = await _worker(payload)._check_single_file(1, "doc.pdf", "doc.pdf")

    assert result["status"] == 1
    assert result["result"] == {
        "likelihood_score": 80,
        "reasoning": "robotic",
        "is_ai_generated": True,
    }


def test_coerce_score_accepts_numbers_only():
    assert DocsCheckerWorker._coerce_score(80) == 80
    assert DocsCheckerWorker._coerce_score(80.6) == 81
    assert DocsCheckerWorker._coerce_score(" 85 ") == 85
    assert DocsCheckerWorker._coerce_score(None) is None
    assert DocsCheckerWorker._coerce_score("high") is None
    assert DocsCheckerWorker._coerce_score(True) is None
    assert DocsCheckerWorker._coerce_score(float("nan")) is None
    assert DocsCheckerWorker._coerce_score({"score": 1}) is None


def test_aggregate_survives_file_without_score():
    """The regression: a score-less file used to raise KeyError and kill the job."""
    results = [
        {"file_name": "a.docx", "status": 1, "result": {"likelihood_score": 70, "reasoning": "x"}},
        {"file_name": "b.docx", "status": 0, "error": "AI response contained no usable likelihood_score"},
    ]

    msg = DocsCheckerWorker._aggregate_results(43380, results)

    assert msg["status"] == 1
    assert msg["file_names"] == ["a.docx", "b.docx"]
    assert msg["result"]["likelihood_score"] == 70
    assert msg["result"]["is_ai_generated"] is True
    assert "[a.docx]" in msg["result"]["reasoning"]


def test_aggregate_averages_successful_files():
    results = [
        {"file_name": "a.docx", "status": 1, "result": {"likelihood_score": 70, "reasoning": "a"}},
        {"file_name": "b.docx", "status": 1, "result": {"likelihood_score": 81, "reasoning": "b"}},
    ]

    msg = DocsCheckerWorker._aggregate_results(1, results)

    assert msg["result"]["likelihood_score"] == 76  # round(75.5)
    assert msg["result"]["is_ai_generated"] is True
    assert "[a.docx] a" in msg["result"]["reasoning"]
    assert "[b.docx] b" in msg["result"]["reasoning"]


def test_aggregate_all_failed_returns_first_error():
    results = [
        {"file_name": "a.docx", "status": 0, "error": "download failed"},
        {"file_name": "b.docx", "status": 0, "error": "extraction failed"},
    ]

    msg = DocsCheckerWorker._aggregate_results(7, results)

    assert msg["status"] == 0
    assert msg["file_names"] == ["a.docx", "b.docx"]
    assert msg["error"] == "download failed"
