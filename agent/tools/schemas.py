"""Pydantic argument schemas for the five enterprise service-desk tools.

Each model doubles as a JSON-Schema source for an OpenAI-compatible ``tools=[...]`` array.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Priority = Literal["low", "normal", "high", "urgent"]


class QueryOrderArgs(BaseModel):
    """Look up an order by id."""

    order_id: str = Field(description="Order identifier, e.g. 'A1001'.")


class ModifyOrderArgs(BaseModel):
    """Modify mutable fields on an existing order."""

    order_id: str
    changes: dict[str, Any] = Field(description="Mapping of field -> new value.")


class RefundArgs(BaseModel):
    """Issue a (possibly partial) refund against an order."""

    order_id: str
    amount: float | None = Field(default=None, description="None means a full refund.")
    reason: str


class CreateTicketArgs(BaseModel):
    """Open a support ticket."""

    subject: str
    description: str
    priority: Priority = "normal"


class SearchKBArgs(BaseModel):
    """Search the FAQ / knowledge base."""

    query: str
    top_k: int = 5


# tool name -> (argument model, human-readable description)
TOOL_SPECS: dict[str, tuple[type[BaseModel], str]] = {
    "query_order": (QueryOrderArgs, "Query an order's status and details by id."),
    "modify_order": (ModifyOrderArgs, "Modify fields of an existing order."),
    "refund": (RefundArgs, "Issue a full or partial refund for an order."),
    "create_ticket": (CreateTicketArgs, "Create a support ticket."),
    "search_kb": (SearchKBArgs, "Search the FAQ / knowledge base."),
}


def openai_tools() -> list[dict[str, Any]]:
    """Render all tools as an OpenAI-compatible ``tools`` array."""
    tools: list[dict[str, Any]] = []
    for name, (model, description) in TOOL_SPECS.items():
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": model.model_json_schema(),
                },
            }
        )
    return tools
