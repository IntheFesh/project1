"""In-memory service-desk backend — mock side effects for off-GPU demos and tests.

Deterministic sample data; NOT a real datastore. The five tools read/write this store so
the agent graph runs end-to-end without external systems. ``search_kb`` delegates to a
RAG callable injected in Phase 3 (until then it returns an empty, clearly-labeled result).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from agent.tools.order_data import all_orders


class Order(BaseModel):
    """A sample order record."""

    order_id: str
    status: str
    amount: float
    days_since_purchase: int
    shipped: bool
    items: list[str] = Field(default_factory=list)


def _sample_orders() -> dict[str, Order]:
    """Build Order objects from the union of the disjoint train + eval pools."""
    return {oid: Order(order_id=oid, **data) for oid, data in all_orders().items()}


KBSearch = Callable[[str, int], dict[str, Any]]


class ServiceDesk:
    """Holds the mock order store and exposes the five service-desk tools."""

    def __init__(self, kb_search: KBSearch | None = None) -> None:
        self._orders = _sample_orders()
        self._ticket_seq = itertools.count(1001)
        self.kb_search = kb_search  # injected by the RAG subgraph in Phase 3

    def get_order(self, order_id: str) -> Order | None:
        """Return the order, or None if unknown."""
        return self._orders.get(order_id)

    # --- tools -------------------------------------------------------------
    def query_order(self, order_id: str) -> dict[str, Any]:
        order = self.get_order(order_id)
        if order is None:
            return {"found": False, "order_id": order_id}
        return {"found": True, **order.model_dump()}

    def modify_order(self, order_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        order = self.get_order(order_id)
        if order is None:
            return {"ok": False, "reason": "order_not_found", "order_id": order_id}
        return {"ok": True, "order_id": order_id, "changes": changes, "shipped": order.shipped}

    def refund(
        self, order_id: str, amount: float | None = None, reason: str = ""
    ) -> dict[str, Any]:
        order = self.get_order(order_id)
        if order is None:
            return {"ok": False, "reason": "order_not_found", "order_id": order_id}
        return {
            "ok": True,
            "order_id": order_id,
            "amount": order.amount if amount is None else amount,
            "days_since_purchase": order.days_since_purchase,
            "reason": reason,
        }

    def create_ticket(
        self, subject: str, description: str, priority: str = "normal"
    ) -> dict[str, Any]:
        return {
            "ok": True,
            "ticket_id": f"T{next(self._ticket_seq)}",
            "subject": subject,
            "priority": priority,
        }

    def search_kb(self, query: str, top_k: int = 5) -> dict[str, Any]:
        if self.kb_search is not None:
            return self.kb_search(query, top_k)
        return {"results": [], "citations": [], "note": "RAG not wired yet (Phase 3)."}
