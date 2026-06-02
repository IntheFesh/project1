"""Lightweight versioned prompt registry (mirrors Langfuse-managed prompts).

Keeps prompt text versioned in code so changes are reviewable; on the deployed box these
can be backed by Langfuse-managed prompts for runtime updates without redeploying.
"""

from __future__ import annotations

from pydantic import BaseModel


class Prompt(BaseModel):
    """A named, versioned prompt template."""

    name: str
    version: int
    template: str


_REGISTRY: dict[str, list[Prompt]] = {
    "system": [
        Prompt(
            name="system",
            version=1,
            template=(
                "你是企业服务台助手。请选择合适的工具完成用户请求，"
                "并严格遵守平台政策（如退款窗口、改单限制）。"
                "涉及知识问答时引用知识库来源。"
            ),
        ),
    ],
}


def get_prompt(name: str, version: int | None = None) -> Prompt:
    """Return a prompt by name (latest version unless one is specified)."""
    versions = _REGISTRY[name]
    if version is None:
        return versions[-1]
    for prompt in versions:
        if prompt.version == version:
            return prompt
    raise KeyError(f"prompt {name!r} v{version} not found")
