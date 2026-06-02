"""Render a markdown results table from run outputs (missing values become TBD)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def render_table(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> str:
    """Render ``rows`` as a markdown table over ``columns``; blanks/None -> ``TBD``."""
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col)
            cells.append("TBD" if value is None or value == "" else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
