"""Bearer-token / JWT auth helpers (wired into routes in Phase 2). Keys via os.environ."""

from __future__ import annotations

import os
from typing import Any

import jwt


def verify_static_token(token: str) -> bool:
    """Return True iff ``token`` matches API_AUTH_TOKEN from the environment."""
    expected = os.environ.get("API_AUTH_TOKEN", "")
    return bool(expected) and token == expected


def decode_jwt(token: str) -> dict[str, Any]:
    """Decode a JWT using JWT_SECRET (HS256). Raises on an invalid token."""
    secret = os.environ["JWT_SECRET"]
    return jwt.decode(token, secret, algorithms=["HS256"])
