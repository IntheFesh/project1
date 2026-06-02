"""Typed state schema threaded through the LangGraph nodes."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]


class Message(BaseModel):
    """A single chat message in the running conversation."""

    role: Role
    content: str
    name: str | None = None


class ToolCall(BaseModel):
    """A structured tool call proposed by the model."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class PolicyViolation(BaseModel):
    """A recorded policy violation; any violation makes the turn a FAILURE."""

    rule_id: str
    message: str


class Citation(BaseModel):
    """A grounding citation returned by the RAG subgraph."""

    doc_id: str
    score: float
    snippet: str


class AgentState(BaseModel):
    """End-to-end state for one agent turn (planner -> ... -> responder)."""

    messages: list[Message] = Field(default_factory=list)
    plan: str | None = None
    selected_tool: ToolCall | None = None
    tool_result: dict[str, Any] | None = None
    citations: list[Citation] = Field(default_factory=list)
    violations: list[PolicyViolation] = Field(default_factory=list)
    final_answer: str | None = None

    @property
    def policy_ok(self) -> bool:
        """True iff no policy violation has been recorded this turn."""
        return len(self.violations) == 0

    def last_user_text(self) -> str:
        """Return the most recent user message content (empty string if none)."""
        for message in reversed(self.messages):
            if message.role == "user":
                return message.content
        return ""
