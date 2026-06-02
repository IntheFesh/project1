"""Runtime tool registry: schema-validated argument dispatch.

Concrete side effects (order lookups, refunds) are mocked in Phase 2; this module
provides the validation plumbing so the package imports and tests cleanly now.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

from agent.tools.schemas import TOOL_SPECS


def validate_args(tool_name: str, raw_args: dict[str, Any]) -> BaseModel:
    """Validate raw tool arguments against the tool's pydantic schema.

    Raises:
        KeyError: if ``tool_name`` is unknown.
        pydantic.ValidationError: if ``raw_args`` do not satisfy the schema.
    """
    model, _ = TOOL_SPECS[tool_name]
    return model.model_validate(raw_args)


def is_valid(tool_name: str, raw_args: dict[str, Any]) -> bool:
    """Return True iff ``raw_args`` validate for ``tool_name``."""
    try:
        validate_args(tool_name, raw_args)
    except (KeyError, ValidationError):
        return False
    return True
