"""Serving client: deterministic offline mock behaviour + factory wiring."""

from agent.tools.registry import is_valid
from agent.tools.schemas import openai_tools
from serving.client import LLMResponse, ScriptedLLMClient, get_client

TOOLS = openai_tools()


def _user(text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": text}]


def test_mock_selects_refund_with_order_id() -> None:
    resp = ScriptedLLMClient().chat(_user("我想给订单 A1009 退款"), tools=TOOLS)
    assert len(resp.tool_calls) == 1
    call = resp.tool_calls[0]
    assert call.name == "refund"
    assert call.arguments["order_id"] == "A1009"
    assert is_valid(call.name, call.arguments)


def test_mock_selects_query_order() -> None:
    resp = ScriptedLLMClient().chat(_user("帮我查询订单 A1001 的状态"), tools=TOOLS)
    assert resp.tool_calls[0].name == "query_order"


def test_mock_honors_forced_tool_choice() -> None:
    choice = {"type": "function", "function": {"name": "create_ticket"}}
    resp = ScriptedLLMClient().chat(_user("随便说点什么"), tools=TOOLS, tool_choice=choice)
    assert resp.tool_calls[0].name == "create_ticket"


def test_mock_answers_without_tools() -> None:
    resp = ScriptedLLMClient().chat(_user("你好"), tools=None)
    assert isinstance(resp, LLMResponse)
    assert resp.content and not resp.tool_calls


def test_get_client_mock() -> None:
    assert isinstance(get_client("mock"), ScriptedLLMClient)
