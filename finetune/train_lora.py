"""QLoRA-SFT training (CORE; default on a single RTX 5090).

Defaults: r=16, alpha=32, target=all-linear, 4-bit base, bf16 compute; save the adapter
only. On an RTX PRO 6000 you may switch to LoRA + bf16 with a larger batch. Requires the
CUDA 12.8+ training env (see requirements/train.txt). Implemented in Phase 7.
"""

from __future__ import annotations


def train(config_path: str = "configs/lora.yaml") -> None:
    """Run QLoRA-SFT from a YAML config (implemented in Phase 7)."""
    raise NotImplementedError("train_lora is implemented in Phase 7.")
