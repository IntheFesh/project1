"""OpenAI-compatible LLM client + a deterministic mock for off-GPU development.

``OpenAICompatibleClient`` talks to SGLang / vLLM / Ollama purely by ``base_url``, so the
same code path serves all three. ``ScriptedLLMClient`` is a deterministic, rule-based
stand-in for tests and off-GPU demos — it is NOT a model, and its outputs must never be
reported as model results.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from common.config import Settings, get_settings


class LLMToolCall(BaseModel):
    """A tool call parsed from an LLM response."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """A normalized chat completion: free text and/or structured tool calls."""

    content: str | None = None
    tool_calls: list[LLMToolCall] = Field(default_factory=list)


@runtime_checkable
class LLMClient(Protocol):
    """Minimal chat interface shared by the real client and the mock."""

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Return a normalized response for ``messages`` (optionally tool-calling)."""
        ...


class OpenAICompatibleClient:
    """Thin wrapper over the OpenAI SDK pointed at any OpenAI-compatible server."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        from openai import OpenAI  # lazy: keep import cost out of pure-mock paths

        s = settings or get_settings()
        self.model = model or s.model_id
        self._client = OpenAI(
            base_url=base_url or s.openai_base_url,
            api_key=api_key or s.openai_api_key,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        msg = resp.choices[0].message
        calls: list[LLMToolCall] = []
        for tc in msg.tool_calls or []:
            args = tc.function.arguments or "{}"
            calls.append(
                LLMToolCall(name=tc.function.name, arguments=json.loads(args))
            )
        return LLMResponse(content=msg.content, tool_calls=calls)


# --- deterministic offline stand-in -----------------------------------------

_ORDER_RE = re.compile(r"\b([A-Z]\d{3,})\b")
# Intent keywords -> tool name. Order matters (first match wins).
_INTENT_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("refund", ("退款", "退货", "refund")),
    ("modify_order", ("修改", "改地址", "改单", "modify", "change")),
    ("create_ticket", ("工单", "投诉", "人工", "ticket", "complaint")),
    ("query_order", ("查询", "订单状态", "物流", "query", "status", "order")),
    ("search_kb", ("政策", "怎么", "如何", "faq", "policy", "?", "？")),
]


class ScriptedLLMClient:
    """Deterministic, rule-based LLM stand-in for off-GPU development and tests.

    NOT a model. It inspects the last user message, extracts an order id, and selects a
    tool by keyword. Used so the agent graph / API / eval pipeline run reproducibly
    without a GPU. Never report its outputs as model results.
    """

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        # No tools available, or asked to answer directly -> return templated content.
        if not tools or tool_choice == "none":
            return LLMResponse(content=self._compose_answer(messages))

        forced = self._forced_tool(tool_choice)
        tool_name = forced or self._select_tool(last_user)
        if tool_name is None:
            return LLMResponse(content=self._compose_answer(messages))

        args = self._build_args(tool_name, last_user)
        return LLMResponse(tool_calls=[LLMToolCall(name=tool_name, arguments=args)])

    @staticmethod
    def _forced_tool(tool_choice: str | dict[str, Any] | None) -> str | None:
        if isinstance(tool_choice, dict):
            return tool_choice.get("function", {}).get("name")
        return None

    @staticmethod
    def _select_tool(text: str) -> str | None:
        lowered = text.lower()
        for tool_name, keywords in _INTENT_RULES:
            if any(kw.lower() in lowered for kw in keywords):
                return tool_name
        return None

    @staticmethod
    def _build_args(tool_name: str, text: str) -> dict[str, Any]:
        order_match = _ORDER_RE.search(text)
        order_id = order_match.group(1) if order_match else "A1001"
        if tool_name == "query_order":
            return {"order_id": order_id}
        if tool_name == "modify_order":
            return {"order_id": order_id, "changes": {"address": "（用户提供的新地址）"}}
        if tool_name == "refund":
            return {"order_id": order_id, "amount": None, "reason": text[:80]}
        if tool_name == "create_ticket":
            return {"subject": "用户请求", "description": text[:200], "priority": "normal"}
        if tool_name == "search_kb":
            return {"query": text, "top_k": 5}
        return {}

    @staticmethod
    def _compose_answer(messages: list[dict[str, Any]]) -> str:
        tool_msg = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "tool"),
            None,
        )
        if tool_msg:
            return f"（基于工具结果）{tool_msg}"
        return "已收到您的请求。"


def get_client(backend: str | None = None, settings: Settings | None = None) -> LLMClient:
    """Construct an LLM client for the configured backend.

    ``backend`` is one of ``sglang`` (default), ``vllm``, ``ollama``, or ``mock``
    (the deterministic offline stand-in). When omitted, uses ``SERVING_BACKEND``.
    """
    s = settings or get_settings()
    name = (backend or s.serving_backend).lower()
    if name == "mock":
        return ScriptedLLMClient()
    if name == "vllm":
        return OpenAICompatibleClient(base_url=s.vllm_base_url, settings=s)
    if name == "ollama":
        return OpenAICompatibleClient(
            base_url=s.ollama_base_url, model=s.ollama_model, settings=s
        )
    return OpenAICompatibleClient(settings=s)  # sglang (default)
