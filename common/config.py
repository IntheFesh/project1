"""Shared configuration: env-driven settings + typed loaders for configs/*.yaml.

All knobs live in YAML under ``configs/`` (no magic constants in code) and all secrets
come from the environment via ``Settings`` (never hardcoded).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


def _expand(value: Any) -> Any:
    """Recursively expand ${ENV_VAR} references in loaded YAML values."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def load_yaml(name: str) -> dict[str, Any]:
    """Load a YAML config by name from ``configs/`` (or an absolute path).

    Environment references such as ``${MILVUS_URI}`` are expanded on load.
    """
    candidate = Path(name)
    if candidate.is_absolute() or candidate.exists():
        path = candidate
    else:
        path = CONFIG_DIR / name
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return _expand(data)


class Settings(BaseSettings):
    """Environment-driven settings (read from ``.env`` / ``os.environ``)."""

    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", case_sensitive=False
    )

    serving_backend: str = "vllm"  # vLLM is the engine used on Blackwell; "mock" off-GPU
    openai_base_url: str = "http://localhost:30000/v1"
    openai_api_key: str = "EMPTY"
    model_id: str = "Qwen/Qwen3-8B"
    vllm_base_url: str = "http://localhost:8000/v1"
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen3:8b"
    milvus_uri: str = "http://localhost:19530"
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    api_auth_token: str = "changeme-dev-token"
    jwt_secret: str = "changeme-dev-secret"
    rate_limit_per_minute: int = 60
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()


class _Cfg(BaseModel):
    """Base for typed configs; ignore unknown YAML keys for forward-compat."""

    model_config = ConfigDict(extra="ignore")


class SamplingConfig(_Cfg):
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1024


class SGLangConfig(_Cfg):
    model_path: str = "Qwen/Qwen3-8B"
    host: str = "0.0.0.0"
    port: int = 30000
    tool_call_parser: str = "qwen25"
    reasoning_parser: str = "qwen3"
    attention_backend: str = "flashinfer"
    grammar_backend: str = "xgrammar"
    mem_fraction_static: float = 0.85
    quantization: str | None = None


class AgentCfg(_Cfg):
    # max_steps=1 keeps the single-tool-per-turn behavior (CI gate, zh eval, BFCL).
    # Set >1 for multi-step agentic loops (tau2-bench).
    max_steps: int = 1


class ServerConfig(_Cfg):
    backend: str = "vllm"
    sglang: SGLangConfig = Field(default_factory=SGLangConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    agent: AgentCfg = Field(default_factory=AgentCfg)


class RetrievalCfg(_Cfg):
    mode: str = "hybrid"
    dense_top_k: int = 20
    bm25_top_k: int = 20
    fusion: str = "rrf"
    final_top_k: int = 5
    require_citations: bool = True


class EmbeddingCfg(_Cfg):
    model: str = "BAAI/bge-m3"
    dim: int = 1024
    normalize: bool = True


class ChunkingCfg(_Cfg):
    chunk_size: int = 512
    chunk_overlap: int = 64


class RerankerCfg(_Cfg):
    model: str = "BAAI/bge-reranker-v2-m3"
    top_n: int = 5


class RetrievalConfig(_Cfg):
    embedding: EmbeddingCfg = Field(default_factory=EmbeddingCfg)
    reranker: RerankerCfg = Field(default_factory=RerankerCfg)
    chunking: ChunkingCfg = Field(default_factory=ChunkingCfg)
    retrieval: RetrievalCfg = Field(default_factory=RetrievalCfg)


class GateThresholds(_Cfg):
    schema_valid_rate: float = 0.95
    tool_accuracy: float = 0.70


class CIGateConfig(_Cfg):
    deterministic: bool = True
    temperature: float = 0.0
    task_slice: str = "smoke"
    thresholds: GateThresholds = Field(default_factory=GateThresholds)


class StatsConfig(_Cfg):
    bootstrap_resamples: int = 10000
    ci: float = 0.95
    seed: int = 42
    multiple_comparison: str = "holm-bonferroni"


class EvalConfig(_Cfg):
    statistics: StatsConfig = Field(default_factory=StatsConfig)
    ci_gate: CIGateConfig = Field(default_factory=CIGateConfig)


def load_server_config() -> ServerConfig:
    """Load and validate ``configs/server.yaml``."""
    return ServerConfig.model_validate(load_yaml("server.yaml"))


def load_retrieval_config() -> RetrievalConfig:
    """Load and validate ``configs/retrieval.yaml``."""
    return RetrievalConfig.model_validate(load_yaml("retrieval.yaml"))


def load_eval_config() -> EvalConfig:
    """Load and validate ``configs/eval.yaml``."""
    return EvalConfig.model_validate(load_yaml("eval.yaml"))
