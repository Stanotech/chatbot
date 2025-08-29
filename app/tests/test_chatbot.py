import pytest

from app.services.ollama_llm import OllamaLLM


class FakeResponse:
    def json(self) -> dict:
        return {"response": "Mocked answer"}


def fake_post(url: str, json: dict) -> FakeResponse:
    return FakeResponse()


def test_ollama_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("requests.post", fake_post)

    llm = OllamaLLM(ollama_url="http://fake-url")
    result = llm.invoke("Hello?")
    assert result == "Mocked answer"
