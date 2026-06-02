"""GRPO training (STRETCH).

The verifiable reward below is implemented and tested (it is also reusable for analysis):
a policy violation forces reward 0 (policy is a hard failure); otherwise the reward rewards
selecting the correct tool with valid arguments.

The GRPO training loop itself is STRETCH and is NOT run here. A single RTX 5090 cannot fit
rollout + train; on the RTX PRO 6000 it is feasible but must NOT be started without the
owner's go-ahead (rollout via SGLang/TRL, budget <= 1.5 GPU-days).
"""

from __future__ import annotations


def verifiable_reward(*, tool_correct: bool, args_valid: bool, policy_ok: bool) -> float:
    """Verifiable reward in [0, 1]: policy-violation => 0; else 0.6*tool + 0.4*args."""
    if not policy_ok:
        return 0.0
    return 0.6 * float(tool_correct) + 0.4 * float(args_valid)


def train(config_path: str = "configs/lora.yaml") -> None:
    """Run GRPO with the verifiable reward (STRETCH; ask the owner before running)."""
    raise NotImplementedError(
        "GRPO is STRETCH and requires the RTX PRO 6000 (rollout + train won't fit a 5090). "
        "Ask the owner before running; rollout via SGLang/TRL, budget <= 1.5 GPU-days."
    )
