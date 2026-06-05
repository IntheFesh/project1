"""Data-hygiene guards: the SFT train pool and the held-out eval pool must not overlap.

A reviewer's first trust test is leakage. These assertions fail CI if training data ever
references an evaluation order id (or vice versa), so reported accuracy cannot be inflated
by training on the evaluation prompts/orders.
"""

from __future__ import annotations

import json
import re

from agent.tools.order_data import eval_order_ids, train_order_ids
from finetune.build_sft_data import build_chinese_trajectories

# Order ids use the A- (train) / E- (eval) prefixes; ticket ids (T...) are not orders.
_ORDER_RE = re.compile(r"\b([AE]\d{3,})\b")


def _sample_text(sample: object) -> str:
    """All user/assistant/tool text + tool-call arguments in one SFT sample."""
    parts: list[str] = []
    for message in sample.messages:  # type: ignore[attr-defined]
        if message.get("content"):
            parts.append(str(message["content"]))
        for call in message.get("tool_calls") or []:
            parts.append(call["function"]["arguments"])
    return "\n".join(parts)


def test_train_and_eval_order_pools_are_disjoint() -> None:
    assert train_order_ids().isdisjoint(eval_order_ids())


def test_sft_data_references_only_train_orders() -> None:
    train_ids, eval_ids = train_order_ids(), eval_order_ids()
    for sample in build_chinese_trajectories():
        ids = set(_ORDER_RE.findall(_sample_text(sample)))
        assert ids.isdisjoint(eval_ids), f"SFT sample leaks eval order(s): {ids & eval_ids}"
        assert ids <= train_ids, f"SFT sample references unknown order(s): {ids - train_ids}"


def test_sft_arguments_are_valid_json() -> None:
    for sample in build_chinese_trajectories():
        for message in sample.messages:
            for call in message.get("tool_calls") or []:
                json.loads(call["function"]["arguments"])  # raises on malformed args
