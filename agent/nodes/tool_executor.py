"""Tool-executor node: validate arguments against the JSON schema, retry once, execute."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from agent.state import AgentState, Citation, ToolCall
from agent.tools.registry import execute, is_valid
from agent.tools.schemas import openai_tools
from agent.tools.services import ServiceDesk
from serving.client import LLMClient


def tool_executor(state: AgentState, client: LLMClient, services: ServiceDesk) -> dict[str, Any]:
    """Run the selected tool with schema validation and a single retry-on-invalid."""
    tool = state.selected_tool
    if tool is None:
        return {}

    if not is_valid(tool.name, tool.arguments):
        tool = _retry_invalid(state, client, tool.name)

    if tool is None or not is_valid(tool.name, tool.arguments):
        return {
            "selected_tool": tool,
            "tool_result": {"ok": False, "reason": "invalid_arguments"},
        }

    result = execute(services, tool.name, tool.arguments)
    updates: dict[str, Any] = {"selected_tool": tool, "tool_result": result}
    citations = _citations_from(result)
    if citations:
        updates["citations"] = citations
    return updates


def _retry_invalid(state: AgentState, client: LLMClient, tool_name: str) -> ToolCall | None:
    """Re-ask the model for valid arguments, forcing the same tool."""
    messages = [m.model_dump(exclude_none=True) for m in state.messages]
    messages.append(
        {
            "role": "system",
            "content": f"上一次为工具 {tool_name} 生成的参数不符合 schema，请重新生成合法参数。",
        }
    )
    forced = {"type": "function", "function": {"name": tool_name}}
    resp = client.chat(messages, tools=openai_tools(), tool_choice=forced, temperature=0.0)
    if resp.tool_calls:
        call = resp.tool_calls[0]
        return ToolCall(name=call.name, arguments=call.arguments)
    return None


def _citations_from(result: dict[str, Any]) -> list[Citation]:
    """Convert any citation dicts in a tool result into Citation models."""
    out: list[Citation] = []
    for item in result.get("citations", []) or []:
        try:
            out.append(Citation.model_validate(item))
        except ValidationError:
            continue  # skip malformed citation entries
    return out
