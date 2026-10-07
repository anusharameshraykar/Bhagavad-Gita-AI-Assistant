from io import BytesIO
import json
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

import pytest
import httpx

from src import config
from src import generate
from src.generate import LLMUnavailable, call_gemini, call_groq


class Response:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def read(self):
        return json.dumps({
            "candidates": [{
                "content": {"parts": [{"text": "Grounded answer [BG 2.47]."}]}
            }]
        }).encode()


def test_call_gemini_converts_common_messages_and_extracts_text(monkeypatch):
    request_data = {}

    def fake_urlopen(request, timeout):
        request_data["request"] = request
        request_data["timeout"] = timeout
        return Response()

    monkeypatch.setattr("src.generate.urllib.request.urlopen", fake_urlopen)
    answer = call_gemini(
        [
            {"role": "system", "content": "Use supplied verses only."},
            {"role": "user", "content": "What does BG 2.47 teach?"},
            {"role": "assistant", "content": "Previous response."},
        ],
        api_key="test-api-key",
        model="gemini-test",
    )

    request = request_data["request"]
    payload = json.loads(request.data)
    assert parse_qs(urlsplit(request.full_url).query) == {"key": ["test-api-key"]}
    assert payload["systemInstruction"]["parts"][0]["text"] == "Use supplied verses only."
    assert [entry["role"] for entry in payload["contents"]] == ["user", "model"]
    assert answer == "Grounded answer [BG 2.47]."


def test_call_gemini_retries_http_503_then_returns_answer(monkeypatch):
    attempts = []
    delays = []

    def fake_urlopen(request, timeout):
        attempts.append(request)
        if len(attempts) == 1:
            raise HTTPError(
                request.full_url,
                503,
                "Unavailable",
                {},
                BytesIO(b'{"error":"high demand"}'),
            )
        return Response()

    monkeypatch.setattr("src.generate.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("src.generate.time.sleep", delays.append)
    monkeypatch.setattr(config, "GEMINI_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(config, "GEMINI_RETRY_DELAY_S", 0.25)

    answer = call_gemini([], api_key="test-api-key", model="gemini-test")

    assert answer == "Grounded answer [BG 2.47]."
    assert len(attempts) == 2
    assert delays == [0.25]


def test_call_gemini_stops_after_max_http_503_attempts(monkeypatch):
    attempts = []
    delays = []

    def fake_urlopen(request, timeout):
        attempts.append(request)
        raise HTTPError(
            request.full_url,
            503,
            "Unavailable",
            {},
            BytesIO(b'{"error":"high demand"}'),
        )

    monkeypatch.setattr("src.generate.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("src.generate.time.sleep", delays.append)
    monkeypatch.setattr(config, "GEMINI_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(config, "GEMINI_RETRY_DELAY_S", 0.25)

    with pytest.raises(LLMUnavailable, match="Gemini returned HTTP 503"):
        call_gemini([], api_key="test-api-key", model="gemini-test")

    assert len(attempts) == 3
    assert delays == [0.25, 0.5]


def test_call_gemini_does_not_retry_non_503_http_errors(monkeypatch):
    attempts = []

    def fake_urlopen(request, timeout):
        attempts.append(request)
        raise HTTPError(request.full_url, 429, "Rate limited", {}, BytesIO(b""))

    monkeypatch.setattr("src.generate.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(config, "GEMINI_MAX_ATTEMPTS", 3)

    with pytest.raises(LLMUnavailable, match="Gemini returned HTTP 429"):
        call_gemini([], api_key="test-api-key", model="gemini-test")

    assert len(attempts) == 1


def test_call_groq_sends_openai_compatible_request_and_extracts_text(monkeypatch):
    request_data = {}

    class FakeGroq:
        def __init__(self, api_key, timeout):
            request_data["api_key"] = api_key
            request_data["timeout"] = timeout
            self.chat = self
            self.completions = self

        def create(self, model, messages, temperature):
            request_data["model"] = model
            request_data["messages"] = messages
            request_data["temperature"] = temperature
            return type(
                "Completion",
                (),
                {
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {"message": type("Message", (), {"content": "Grounded answer [BG 2.47]."})()},
                        )()
                    ]
                },
            )()

    monkeypatch.setattr(generate, "Groq", FakeGroq)
    answer = call_groq(
        [
            {"role": "system", "content": "Use supplied verses only."},
            {"role": "user", "content": "What does BG 2.47 teach?"},
        ],
        api_key="test-api-key",
        model="openai/gpt-oss-20b",
    )

    assert request_data["api_key"] == "test-api-key"
    assert request_data["model"] == "openai/gpt-oss-20b"
    assert request_data["messages"][0]["role"] == "system"
    assert answer == "Grounded answer [BG 2.47]."


def test_call_groq_requires_api_key():
    with pytest.raises(LLMUnavailable, match="GROQ_API_KEY is not configured"):
        call_groq([], api_key="")


def test_call_groq_reports_http_errors(monkeypatch):
    class FakeGroq:
        def __init__(self, api_key, timeout):
            self.chat = self
            self.completions = self

        def create(self, **kwargs):
            response = httpx.Response(
                401,
                request=httpx.Request("POST", "https://api.groq.com"),
            )
            raise generate.APIStatusError(
                "Unauthorized",
                response=response,
                body='{"error":{"message":"invalid api key"}}',
            )

    monkeypatch.setattr(generate, "Groq", FakeGroq)
    with pytest.raises(LLMUnavailable, match="Groq returned HTTP 401"):
        call_groq([], api_key="test-api-key")


def test_call_groq_explains_cloudflare_1010(monkeypatch):
    class FakeGroq:
        def __init__(self, api_key, timeout):
            self.chat = self
            self.completions = self

        def create(self, **kwargs):
            response = httpx.Response(
                403,
                request=httpx.Request("POST", "https://api.groq.com"),
            )
            raise generate.APIStatusError(
                "Forbidden",
                response=response,
                body="error code: 1010",
            )

    monkeypatch.setattr(generate, "Groq", FakeGroq)
    with pytest.raises(LLMUnavailable, match="Cloudflare error 1010"):
        call_groq([], api_key="test-api-key")
