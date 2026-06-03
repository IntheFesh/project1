"""Deploy artifacts exist and the compose files define the app services."""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("compose", ["docker-compose.yml", "docker-compose.app.yml"])
def test_compose_defines_app_services(compose: str) -> None:
    data = yaml.safe_load((ROOT / compose).read_text(encoding="utf-8"))
    assert "agent-api" in data["services"]
    assert "frontend" in data["services"]


def test_dockerfiles_present() -> None:
    assert (ROOT / "Dockerfile").exists()
    assert (ROOT / "frontend" / "Dockerfile").exists()


@pytest.mark.parametrize("script", ["scripts/quickstart.sh", "scripts/deploy.sh"])
def test_scripts_present(script: str) -> None:
    assert (ROOT / script).exists()
