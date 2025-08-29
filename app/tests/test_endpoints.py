import pytest
from fastapi.testclient import TestClient


def test_ask_endpoint(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # Mockujemy qa_chain, żeby nie wywoływać prawdziwego Ollama + Qdrant
    def fake_chain(question: str) -> dict:
        return {
            "result": "To jest testowa odpowiedź",
            "source_documents": [type("Doc", (), {"metadata": {"source": "test.txt"}})()],
        }

    client.app.state.qa_chain = fake_chain

    response = client.post("/ask", json={"question": "Co to jest test?"})
    assert response.status_code == 200

    data = response.json()
    assert data["answer"] == "To jest testowa odpowiedź"
    assert data["sources"] == ["test.txt"]


def test_ask_invalid_request(client: TestClient) -> None:
    response = client.post("/ask", json={})
    assert response.status_code == 422  # validation error (brak pola "question")
