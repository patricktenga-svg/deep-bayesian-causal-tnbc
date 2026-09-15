from fastapi.testclient import TestClient


def test_health(monkeypatch):
    from api.main import app
    client = TestClient(app)
    r = client.get("/api/v1/health")
    assert r.status_code == 200