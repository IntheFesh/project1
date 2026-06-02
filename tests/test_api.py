"""API app factory + auth helper smoke tests."""

from api.auth import verify_static_token
from api.main import create_app


def test_app_exposes_health() -> None:
    app = create_app()
    paths = {route.path for route in app.routes}
    assert "/health" in paths


def test_static_token(monkeypatch) -> None:
    monkeypatch.setenv("API_AUTH_TOKEN", "secret")
    assert verify_static_token("secret")
    assert not verify_static_token("wrong")
