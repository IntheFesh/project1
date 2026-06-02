"""In-memory service-desk backend behaviour."""

from agent.tools.services import ServiceDesk


def test_query_known_and_unknown() -> None:
    s = ServiceDesk()
    assert s.query_order("A1001")["found"] is True
    assert s.query_order("ZZZ")["found"] is False


def test_refund_reports_days_since_purchase() -> None:
    result = ServiceDesk().refund("A1009")
    assert result["ok"] is True
    assert result["days_since_purchase"] == 30


def test_tickets_get_unique_ids() -> None:
    s = ServiceDesk()
    first = s.create_ticket("主题", "描述")["ticket_id"]
    second = s.create_ticket("主题", "描述")["ticket_id"]
    assert first != second


def test_search_kb_stub_until_phase3() -> None:
    assert ServiceDesk().search_kb("退款政策")["citations"] == []
