"""Build the SFT dataset (tool-calling / policy / citation samples).

Sources (check each license before use — some are research-only — and record it):
  - ToolACE (arXiv 2409.00920)
  - APIGen-MT / xLAM subsets
  - The self-built Chinese enterprise service-desk domain, with REAL Chinese
    trajectories (Chinese-LLM generation + rule filtering), NOT English-with-augmentation.
Converts to Qwen3 chat/tool format and rule-filters. Implemented in Phase 7.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class SFTSample(BaseModel):
    """One supervised fine-tuning sample in Qwen3 chat/tool format."""

    messages: list[dict[str, Any]]
    source: str
    language: str  # "zh" or "en"


def build_sft_data(out_path: str) -> int:
    """Build 8k-15k samples, write them to ``out_path``, return the count (Phase 7)."""
    raise NotImplementedError("build_sft_data is implemented in Phase 7.")
