"""Every config file loads and contains its expected top-level keys."""

from pathlib import Path

import pytest
import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

EXPECTED: dict[str, list[str]] = {
    "model.yaml": ["model", "thinking"],
    "lora.yaml": ["method", "base_model", "lora", "train"],
    "retrieval.yaml": ["embedding", "reranker", "vector_store", "retrieval"],
    "server.yaml": ["backend", "sglang", "sampling"],
    "eval.yaml": ["benchmarks", "statistics", "ci_gate"],
}


@pytest.mark.parametrize(("filename", "keys"), list(EXPECTED.items()))
def test_config_has_keys(filename: str, keys: list[str]) -> None:
    data = yaml.safe_load((CONFIG_DIR / filename).read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    for key in keys:
        assert key in data, f"{filename} is missing top-level key: {key}"
