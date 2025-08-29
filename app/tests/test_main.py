from fastapi.testclient import TestClient


def test_healthcheck_like(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 404
