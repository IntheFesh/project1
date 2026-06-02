"""Gradio chat UI for the PolicyArena agent (CORE demo, Phase 2).

Shows the live tool call + policy outcome alongside the final answer. State is held in
app/server memory only — no browser localStorage/sessionStorage. Talks to the agent graph
directly via the configured backend (set SERVING_BACKEND=mock to run off-GPU).

Run:  uv run --extra ui python frontend/app.py
"""

from __future__ import annotations

import json

from agent.graph import build_graph
from agent.state import AgentState
from agent.tools.services import ServiceDesk
from rag.pipeline import build_default_kb_search
from serving.client import get_client

_CLIENT = get_client()
_SERVICES = ServiceDesk(kb_search=build_default_kb_search())
_GRAPH = build_graph(_CLIENT, _SERVICES)


def _run(message: str, thread_id: str) -> AgentState:
    result = _GRAPH.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result if isinstance(result, AgentState) else AgentState.model_validate(result)


def respond(message: str, history: list[dict[str, str]]) -> tuple[list[dict[str, str]], str]:
    """Gradio handler: run a turn and return updated history + a trace panel."""
    state = _run(message, thread_id="ui")
    trace = {
        "plan": state.plan,
        "tool": state.selected_tool.name if state.selected_tool else None,
        "arguments": state.selected_tool.arguments if state.selected_tool else None,
        "tool_result": state.tool_result,
        "violations": [v.model_dump() for v in state.violations],
        "policy_ok": state.policy_ok,
    }
    history = (history or []) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": state.final_answer or ""},
    ]
    return history, json.dumps(trace, ensure_ascii=False, indent=2)


def build_ui():  # type: ignore[no-untyped-def]
    """Construct the Gradio Blocks UI (gradio imported lazily)."""
    import gradio as gr

    with gr.Blocks(title="PolicyArena") as demo:
        gr.Markdown("# PolicyArena — 企业服务台 Agent\n选择/调用工具并做政策合规校验。")
        chatbot = gr.Chatbot(type="messages", height=420)
        trace_box = gr.Code(label="trace (plan / tool / policy)", language="json")
        msg = gr.Textbox(label="输入", placeholder="例如：订单 A1009 我要退款")

        def _on_submit(message: str, history: list[dict[str, str]]):
            new_history, trace = respond(message, history)
            return new_history, trace, ""

        msg.submit(_on_submit, [msg, chatbot], [chatbot, trace_box, msg])
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7860)
