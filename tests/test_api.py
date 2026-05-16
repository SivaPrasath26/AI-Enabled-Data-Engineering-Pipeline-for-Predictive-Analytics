from fastapi.testclient import TestClient

from src.serving.app import app


def test_health_endpoint():
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
