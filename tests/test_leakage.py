"""Data-hygiene guards: the SFT train pool and the held-out eval pool must not overlap.

A reviewer's first trust test is leakage. These assertions fail CI if training data ever
references an evaluation order id (or vice versa), so reported accuracy cannot be inflated
by training on the evaluation prompts/orders.
"""

from __future__ import annotations

import json
import re

from agent.tools.order_data import eval_order_ids, train_order_ids
from eval.zh_service_desk import load_tasks
from finetune.build_sft_data import build_chinese_trajectories

# Order ids use the A- (train) / E- (eval) prefixes; ticket ids (T...) are not orders.
_ORDER_RE = re.compile(r"\b([AE]\d{3,})\b")


def _normalize(text: str) -> str:
    """Lowercase and strip whitespace/punctuation for prompt-overlap comparison."""
    return re.sub(r"[\s\W_]+", "", text.lower())


def _sample_text(sample: object) -> str:
    """All user/assistant/tool text + tool-call arguments in one SFT sample."""
    parts: list[str] = []
    for message in sample.messages:  # type: ignore[attr-defined]
        if message.get("content"):
            parts.append(str(message["content"]))
        for call in message.get("tool_calls") or []:
            parts.append(call["function"]["arguments"])
    return "\n".join(parts)


def _sft_user_prompts() -> set[str]:
    return {
        _normalize(m["content"])
        for sample in build_chinese_trajectories()
        for m in sample.messages
        if m["role"] == "user" and m.get("content")
    }


def test_train_and_eval_order_pools_are_disjoint() -> None:
    assert train_order_ids().isdisjoint(eval_order_ids())


def test_sft_data_references_only_train_orders() -> None:
    train_ids, eval_ids = train_order_ids(), eval_order_ids()
    for sample in build_chinese_trajectories():
        ids = set(_ORDER_RE.findall(_sample_text(sample)))
        assert ids.isdisjoint(eval_ids), f"SFT sample leaks eval order(s): {ids & eval_ids}"
        assert ids <= train_ids, f"SFT sample references unknown order(s): {ids - train_ids}"


def test_eval_set_references_only_eval_orders() -> None:
    train_ids, eval_ids = train_order_ids(), eval_order_ids()
    for task in load_tasks():
        ids = set(_ORDER_RE.findall(task.prompt))
        order_id = task.arg_constraints.get("order_id")
        if isinstance(order_id, str):
            ids.add(order_id)
        assert ids.isdisjoint(train_ids), f"{task.id} leaks train order(s): {ids & train_ids}"
        assert ids <= eval_ids, f"{task.id} references unknown order(s): {ids - eval_ids}"


def test_sft_and_eval_prompts_are_disjoint() -> None:
    sft_prompts = _sft_user_prompts()
    eval_prompts = {_normalize(t.prompt) for t in load_tasks()}
    overlap = sft_prompts & eval_prompts
    assert not overlap, f"SFT and eval prompts overlap: {overlap}"


def test_sft_arguments_are_valid_json() -> None:
    for sample in build_chinese_trajectories():
        for message in sample.messages:
            for call in message.get("tool_calls") or []:
                json.loads(call["function"]["arguments"])  # raises on malformed args
