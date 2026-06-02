"""Rate limiting via SlowAPI (wired into the app in Phase 2)."""

from __future__ import annotations

import os

from slowapi import Limiter
from slowapi.util import get_remote_address


def build_limiter() -> Limiter:
    """Construct a SlowAPI limiter using RATE_LIMIT_PER_MINUTE from the environment."""
    per_minute = os.environ.get("RATE_LIMIT_PER_MINUTE", "60")
    return Limiter(key_func=get_remote_address, default_limits=[f"{per_minute}/minute"])
