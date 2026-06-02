"""API: health, bearer auth, and policy-aware /agent/query (mock backend)."""

from fastapi.testclient import TestClient

from api.auth import verify_static_token
from api.main import create_app, get_llm_client
from serving.client import ScriptedLLMClient

AUTH = {"Authorization": "Bearer test-token"}


def _client(monkeypatch) -> TestClient:
    monkeypatch.setenv("API_AUTH_TOKEN", "test-token")
    app = create_app()
    app.dependency_overrides[get_llm_client] = ScriptedLLMClient
    return TestClient(app)


def test_health(monkeypatch) -> None:
    assert _client(monkeypatch).get("/health").json() == {"status": "ok"}


def test_query_requires_auth(monkeypatch) -> None:
    resp = _client(monkeypatch).post("/agent/query", json={"message": "查询订单 A1001"})
    assert resp.status_code == 401


def test_query_refund_allowed(monkeypatch) -> None:
    resp = _client(monkeypatch).post(
        "/agent/query", json={"message": "订单 A1001 我要退款"}, headers=AUTH
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["tool"] == "refund"
    assert body["violations"] == []


def test_query_refund_blocked_by_policy(monkeypatch) -> None:
    resp = _client(monkeypatch).post(
        "/agent/query", json={"message": "订单 A1009 我要退款"}, headers=AUTH
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["tool"] == "refund"
    assert any(v["rule_id"] == "refund_window" for v in body["violations"])


def test_stream_emits_final_event(monkeypatch) -> None:
    resp = _client(monkeypatch).post(
        "/agent/stream", json={"message": "查询订单 A1001"}, headers=AUTH
    )
    assert resp.status_code == 200
    assert "final" in resp.text


def test_static_token(monkeypatch) -> None:
    monkeypatch.setenv("API_AUTH_TOKEN", "secret")
    assert verify_static_token("secret")
    assert not verify_static_token("wrong")
