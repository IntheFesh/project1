"""Build the SFT dataset in Qwen3 chat/tool format.

This generates REAL Chinese service-desk trajectories from deterministic templates over the
in-memory ServiceDesk + RAG (a policy-compliant seed set: query, allowed/denied refund,
modify, ticket, cited knowledge). On the GPU box this is augmented with ToolACE
(arXiv 2409.00920) + APIGen-MT / xLAM subsets — check each license before use and record it.
Samples are rule-filtered (valid tool args, non-empty final answer) before writing JSONL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from agent.tools.registry import is_valid
from agent.tools.services import ServiceDesk
from observability.prompts import get_prompt
from rag.pipeline import build_default_kb_search

SYSTEM_PROMPT = get_prompt("system").template
DEFAULT_OUT = "outputs/sft/zh_service_desk.jsonl"


class SFTSample(BaseModel):
    """One supervised fine-tuning sample in Qwen3 chat/tool format."""

    messages: list[dict[str, Any]]
    source: str
    language: str  # "zh" or "en"


def _tool_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call_{name}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            }
        ],
    }


def _tool_result(name: str, result: dict[str, Any]) -> dict[str, Any]:
    return {"role": "tool", "name": name, "content": json.dumps(result, ensure_ascii=False)}


def _trajectory(user: str, tool: str, args: dict[str, Any], result: dict[str, Any], final: str) -> SFTSample:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        _tool_call(tool, args),
        _tool_result(tool, result),
        {"role": "assistant", "content": final},
    ]
    return SFTSample(messages=messages, source="zh-service-desk-template", language="zh")


def build_chinese_trajectories() -> list[SFTSample]:
    """Generate the policy-compliant Chinese service-desk seed trajectories."""
    svc = ServiceDesk(kb_search=build_default_kb_search())
    samples: list[SFTSample] = []

    # Order lookups.
    for oid in ("A1001", "A1002", "A1009"):
        res = svc.query_order(oid)
        final = f"订单 {oid} 当前状态为 {res.get('status')}，金额 {res.get('amount')} 元。"
        samples.append(_trajectory(f"帮我查一下订单 {oid} 的状态", "query_order", {"order_id": oid}, res, final))

    # Refund within window -> execute.
    res = svc.refund("A1001", None, "用户申请退款")
    samples.append(
        _trajectory(
            "订单 A1001 我要退款", "refund",
            {"order_id": "A1001", "amount": None, "reason": "用户申请退款"}, res,
            f"已为订单 A1001 提交退款，金额 {res['amount']} 元，将原路退回。",
        )
    )

    # Refund past window -> check first, then refuse (policy-compliant target).
    res = svc.query_order("A1009")
    samples.append(
        _trajectory(
            "订单 A1009 我要退款", "query_order", {"order_id": "A1009"}, res,
            "很抱歉，订单 A1009 已超过 7 天退款窗口，无法直接退款。如有需要，我可以为您创建人工工单。",
        )
    )

    # Modify shipped order -> refuse and offer a ticket.
    res = svc.query_order("A1002")
    samples.append(
        _trajectory(
            "把订单 A1002 的地址改一下", "query_order", {"order_id": "A1002"}, res,
            "订单 A1002 已发货，无法直接修改收货信息，建议创建工单由人工处理，是否需要？",
        )
    )

    # Modify unshipped order -> allowed.
    res = svc.modify_order("A1001", {"address": "新地址"})
    samples.append(
        _trajectory(
            "订单 A1001 还没发货，帮我改下收货地址", "modify_order",
            {"order_id": "A1001", "changes": {"address": "新地址"}}, res,
            "已更新订单 A1001 的收货地址。",
        )
    )

    # Create ticket.
    res = svc.create_ticket("物流投诉", "用户投诉物流太慢", "high")
    samples.append(
        _trajectory(
            "我要投诉物流太慢，帮我建个工单", "create_ticket",
            {"subject": "物流投诉", "description": "用户投诉物流太慢", "priority": "high"}, res,
            f"已为您创建工单 {res['ticket_id']}，将优先处理。",
        )
    )

    # Cited knowledge answer.
    kb = svc.search_kb("运费怎么算", 3)
    cites = "，".join(c["doc_id"] for c in kb["citations"])
    samples.append(
        _trajectory(
            "请问运费是怎么计算的？", "search_kb", {"query": "运费怎么算", "top_k": 3}, kb,
            f"运费按重量与配送地区计算，订单满 99 元包邮，未满收取 10 元。（来源：{cites}）",
        )
    )
    return samples


def rule_filter(samples: list[SFTSample]) -> list[SFTSample]:
    """Drop samples without a user turn, without a final answer, or with invalid tool args."""
    kept: list[SFTSample] = []
    for sample in samples:
        roles = [m["role"] for m in sample.messages]
        has_final = any(m["role"] == "assistant" and m.get("content") for m in sample.messages)
        if "user" not in roles or not has_final:
            continue
        if _tool_calls_valid(sample):
            kept.append(sample)
    return kept


def _tool_calls_valid(sample: SFTSample) -> bool:
    for message in sample.messages:
        for call in message.get("tool_calls") or []:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"]["arguments"])
            except json.JSONDecodeError:
                return False
            if not is_valid(name, args):
                return False
    return True


def build_sft_data(out_path: str = DEFAULT_OUT) -> int:
    """Build + filter trajectories, write JSONL to ``out_path``, return the sample count."""
    samples = rule_filter(build_chinese_trajectories())
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.model_dump(), ensure_ascii=False) + "\n")
    return len(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Chinese service-desk SFT dataset.")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    count = build_sft_data(args.out)
    print(f"wrote {count} samples to {args.out}")


if __name__ == "__main__":
    main()
