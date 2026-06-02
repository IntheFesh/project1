"""Tool schema + registry behaviour."""

from agent.tools.registry import is_valid
from agent.tools.schemas import TOOL_SPECS, openai_tools


def test_five_tools_exposed() -> None:
    tools = openai_tools()
    assert len(tools) == 5
    names = {t["function"]["name"] for t in tools}
    assert names == set(TOOL_SPECS)
    for tool in tools:
        params = tool["function"]["parameters"]
        assert params["type"] == "object"


def test_registry_validation() -> None:
    assert is_valid("query_order", {"order_id": "A1001"})
    assert not is_valid("query_order", {})        # missing required field
    assert not is_valid("unknown_tool", {})       # unknown tool name
