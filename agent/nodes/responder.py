"""Responder node: compose the final, policy-checked, optionally cited answer."""

from __future__ import annotations

from typing import Any

from agent.state import AgentState


def responder(state: AgentState) -> dict[str, Any]:
    """Render the final answer from violations / tool results / citations."""
    if state.final_answer:
        return {}  # tool_select already produced a direct answer

    if state.violations:
        reasons = "；".join(v.message for v in state.violations)
        return {
            "final_answer": (
                f"很抱歉，根据平台政策无法直接执行该操作：{reasons}。"
                "如有需要，我可以为您创建人工工单进一步处理。"
            )
        }

    tool = state.selected_tool
    result = state.tool_result or {}
    if tool is None:
        return {"final_answer": "已收到您的请求。"}

    citation_note = ""
    if tool.name == "search_kb" and state.citations:
        refs = "，".join(c.doc_id for c in state.citations)
        citation_note = f"（引用：{refs}）"

    return {"final_answer": f"已通过 {tool.name} 为您处理，结果：{result}{citation_note}"}
