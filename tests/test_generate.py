import json
from urllib.parse import parse_qs, urlsplit

from src.generate import call_gemini


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
