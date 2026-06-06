"""Tool-select node: ask the model (via the server-side tool parser) for a tool call."""

from __future__ import annotations

from typing import Any

from agent.state import AgentState, ToolCall
from agent.tools.schemas import openai_tools
from serving.client import LLMClient


def tool_select(state: AgentState, client: LLMClient) -> dict[str, Any]:
    """Select the next tool call; if the model answers directly, set final_answer.

    Honors per-turn ``tool_choice_override`` on state (set by the eval scorer to switch
    between 'auto' and 'required' by task category); defaults to 'auto'.
    """
    messages = [m.model_dump(exclude_none=True) for m in state.messages] + state.tool_history
    tc = state.tool_choice_override or "auto"
    resp = client.chat(messages, tools=openai_tools(), tool_choice=tc, temperature=0.0)
    if resp.tool_calls:
        call = resp.tool_calls[0]
        return {"selected_tool": ToolCall(name=call.name, arguments=call.arguments)}
    return {"selected_tool": None, "final_answer": resp.content or ""}
