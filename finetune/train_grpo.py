"""GRPO training (STRETCH).

Verifiable reward = tool-correct + args-correct + policy-not-violated. A single RTX 5090
cannot fit rollout + train — run only on an RTX PRO 6000 / larger card, and only after the
owner confirms. Budget <= 1.5 GPU-days. Implemented in Phase 7 (STRETCH).
"""

from __future__ import annotations


def train(config_path: str = "configs/lora.yaml") -> None:
    """Run GRPO with a verifiable reward (STRETCH; implemented in Phase 7)."""
    raise NotImplementedError("train_grpo is STRETCH; see Phase 7.")
