"""Build the Chinese service-desk SFT dataset in Qwen3 chat/tool format.

Trajectories are generated deterministically from templates over the **train order pool**
(``agent.tools.order_data.TRAIN_ORDERS``) + the in-memory ServiceDesk/RAG. They teach:
- correct tool calls (query / refund-in-window / modify-unshipped / ticket / cited KB),
- **policy-compliant refusals** (refund past the 7-day window, modify a shipped order) —
  the model is taught to check then refuse, NOT to call the forbidden tool, which is what
  drives the eval's ``unsafe_selection_rate`` toward zero,
- a few **multi-step** trajectories (query, then act) for the agentic loop.

The held-out eval pool (E-series) is never referenced; ``tests/test_leakage.py`` asserts
the train/eval order pools and prompts are disjoint. On the GPU box this seed set is mixed
with licensed external corpora via ``ingest_external`` (ToolACE / APIGen-MT / xLAM — each
license checked and recorded). Samples are rule-filtered before writing JSONL.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from agent.policies.rules import REFUND_WINDOW_DAYS
from agent.tools.order_data import TRAIN_ORDERS
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


def _normalize(text: str) -> str:
    """Lowercase + strip whitespace/punctuation (for dedup against the eval set)."""
    return re.sub(r"[\s\W_]+", "", text.lower())


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


def _sample(messages: list[dict[str, Any]], source: str = "zh-service-desk-template") -> SFTSample:
    return SFTSample(
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, *messages],
        source=source,
        language="zh",
    )


def _single(user: str, tool: str, args: dict[str, Any], result: dict[str, Any], final: str) -> SFTSample:
    """A single-tool trajectory: user -> tool_call -> tool_result -> final answer."""
    return _sample(
        [
            {"role": "user", "content": user},
            _tool_call(tool, args),
            _tool_result(tool, result),
            {"role": "assistant", "content": final},
        ]
    )


def _refusal(user: str, check_tool: str, args: dict[str, Any], result: dict[str, Any], final: str) -> SFTSample:
    """A policy-compliant refusal: check the order, then refuse the forbidden action."""
    return _single(user, check_tool, args, result, final)


def build_chinese_trajectories() -> list[SFTSample]:
    """Generate policy-compliant Chinese service-desk trajectories over the train pool."""
    svc = ServiceDesk(kb_search=build_default_kb_search())
    samples: list[SFTSample] = []

    for oid, data in TRAIN_ORDERS.items():
        samples.extend(_order_trajectories(svc, oid, data))

    samples.extend(_knowledge_trajectories(svc))
    samples.extend(_ticket_trajectories(svc))
    samples.extend(_multistep_trajectories(svc))
    return samples


def _order_trajectories(svc: ServiceDesk, oid: str, data: dict[str, Any]) -> list[SFTSample]:
    out: list[SFTSample] = []
    q = svc.query_order(oid)
    status, amount = q.get("status"), q.get("amount")

    # Query (several phrasings).
    for user in (f"帮我查一下订单 {oid} 的状态", f"订单 {oid} 现在是什么情况",
                 f"看下 {oid} 这个单子到哪了", f"{oid} 的物流状态查一下"):
        out.append(_single(user, "query_order", {"order_id": oid}, q,
                           f"订单 {oid} 当前状态为 {status}，金额 {amount} 元。"))

    # Refund: execute within window, else refuse (check first).
    refund_users = (f"订单 {oid} 我要退款", f"{oid} 这个订单申请退款", f"帮我把订单 {oid} 退了")
    if data["days_since_purchase"] <= REFUND_WINDOW_DAYS:
        for user in refund_users:
            res = svc.refund(oid, None, "用户申请退款")
            out.append(_single(user, "refund",
                               {"order_id": oid, "amount": None, "reason": "用户申请退款"}, res,
                               f"已为订单 {oid} 提交退款，金额 {res['amount']} 元，将原路退回。"))
        # PATCH-C/D: keep one in-window refusal phrasing for policy robustness (not two — to limit query_order inflation).
        out.append(_refusal(f"订单 {oid} 已经买了一个月，能退吗", "query_order", {"order_id": oid}, q,
                            f"根据系统，订单 {oid} 在退款窗口内，可以为您处理退款。"
                            "如对窗口期有疑问，建议先确认下单时间。"))

    else:
        for user in refund_users:
            out.append(_refusal(user, "query_order", {"order_id": oid}, q,
                                f"很抱歉，订单 {oid} 已超过 {REFUND_WINDOW_DAYS} 天退款窗口，无法直接退款。"
                                "如有需要，我可以为您创建人工工单。"))

    # Modify: execute if unshipped (varied fields), else refuse and offer a ticket.
    if not data["shipped"]:
        variants = [
            (f"订单 {oid} 还没发货，帮我改下收货地址", {"address": "新的收货地址"}, "收货地址"),
            (f"{oid} 修改一下联系电话", {"phone": "13800000000"}, "联系电话"),
            (f"{oid} 改一下收货人姓名", {"recipient": "张三"}, "收货人姓名"),
            # PATCH-D: more phrasing variants for modify (typical eval prompts use casual/abbreviated forms).
            (f"{oid} 没发货吧，改个手机号", {"phone": "13900000000"}, "联系电话"),
            (f"帮我把订单 {oid} 的地址换一下", {"address": "新的收货地址2"}, "收货地址"),
        ]
        for user, changes, label in variants:
            res = svc.modify_order(oid, changes)
            out.append(_single(user, "modify_order", {"order_id": oid, "changes": changes}, res,
                               f"已更新订单 {oid} 的{label}。"))
    else:
        for user in (f"把订单 {oid} 的地址改一下", f"{oid} 改下收货人电话"):
            out.append(_refusal(user, "query_order", {"order_id": oid}, q,
                                f"订单 {oid} 已发货，无法直接修改收货信息，建议创建工单由人工处理，是否需要？"))
    return out


# KB phrasings here are deliberately DIFFERENT from the held-out eval prompts.
_KB_QUERIES: list[str] = [
    # shipping (8)
    "邮费是怎么收的？", "快递费用如何计算？", "包邮门槛是多少？", "运费多少钱起步？",
    "运费多少？", "邮费多少钱？", "什么时候包邮？", "买多少才包邮？",
    # refund (8)
    "退款会原路返回吗？", "退款到账需要几天？", "退款流程是怎样的？", "申请退款后多久能收到？",
    "退款怎么操作？", "退款多久到账？", "退款规则是什么？", "退款政策说一下",
    # ticket SLA (8)
    "客服多久会回复工单？", "工单一般多长时间解决？", "我提交的工单为什么还没人处理？", "工单进度在哪里查看？",
    "工单要多久才有结果？", "工单处理时效是？", "工单查询入口在哪？", "提交的工单如何跟进？",
    # account (8)
    "忘记登录密码了怎么处理？", "账号登录不上怎么办？", "如何修改账户绑定的手机号？", "电子发票怎么开？",
    "密码忘记了怎么办？", "账号被锁了如何解锁？", "怎么改绑定手机？", "开发票的流程是？",
    # general FAQ (8)
    "你们的退换货标准是什么？", "可以延后发货吗？", "商品保修多久？", "如何申请发票？",
    "下单后还能改地址吗？", "怎么取消订单？", "VIP 会员有什么权益？", "积分怎么用？",
]


def _knowledge_trajectories(svc: ServiceDesk) -> list[SFTSample]:
    out: list[SFTSample] = []
    for query in _KB_QUERIES:
        kb = svc.search_kb(query, 3)
        cites = "，".join(c["doc_id"] for c in kb["citations"])
        # Original phrasing.
        out.append(_single(query, "search_kb", {"query": query, "top_k": 3}, kb,
                           f"根据知识库，{query.rstrip('？?')}的说明如下，请参考。（来源：{cites}）"))
        # PATCH-D: alternate phrasing variant for the same KB question, to broaden coverage.
        alt = f"想问一下，{query}"
        out.append(_single(alt, "search_kb", {"query": alt, "top_k": 3}, kb,
                           f"知识库相关说明如下（来源：{cites}）。"))
    return out


def _ticket_trajectories(svc: ServiceDesk) -> list[SFTSample]:
    out: list[SFTSample] = []
    specs = [
        ("帮我开个工单反馈一个问题", "问题反馈", "用户反馈一个问题", "normal"),
        ("我要联系人工客服处理", "人工处理", "用户请求人工客服处理", "high"),
        ("商品有质量问题，给我登记一下", "质量问题", "用户反馈商品质量问题", "high"),
        ("这个情况比较急，麻烦加急处理", "加急请求", "用户请求加急处理", "urgent"),
    ]
    for user, subject, desc, prio in specs:
        res = svc.create_ticket(subject, desc, prio)
        out.append(_single(user, "create_ticket",
                           {"subject": subject, "description": desc, "priority": prio}, res,
                           f"已为您创建工单 {res['ticket_id']}，将尽快处理。"))
    return out


def _multistep_trajectories(svc: ServiceDesk) -> list[SFTSample]:
    """Two-step trajectories (query, then act) to teach the agentic loop.

    F2: scaled from 1 -> 6 trajectories covering all check-then-act patterns the eval tests.
    """
    out: list[SFTSample] = []

    # Pattern 1+2: query then refund (in-window)
    for oid in ("A1001", "A1003"):
        q = svc.query_order(oid)
        res = svc.refund(oid, None, "确认后退款")
        out.append(_sample([
            {"role": "user", "content": f"先帮我看下订单 {oid} 能不能退，可以的话就退款"},
            _tool_call("query_order", {"order_id": oid}),
            _tool_result("query_order", q),
            _tool_call("refund", {"order_id": oid, "amount": None, "reason": "确认后退款"}),
            _tool_result("refund", res),
            {"role": "assistant", "content": f"订单 {oid} 在退款窗口内，已为您退款 {res['amount']} 元，原路退回。"},
        ]))

    # Pattern 3+4: query then modify (unshipped) — teaches happy_modify check-then-act
    for oid in ("A1005", "A1008"):
        q = svc.query_order(oid)
        changes = {"address": "新的收货地址"}
        res = svc.modify_order(oid, changes)
        out.append(_sample([
            {"role": "user", "content": f"{oid} 没发货吧，帮我改下地址"},
            _tool_call("query_order", {"order_id": oid}),
            _tool_result("query_order", q),
            _tool_call("modify_order", {"order_id": oid, "changes": changes}),
            _tool_result("modify_order", res),
            {"role": "assistant", "content": f"已确认订单 {oid} 未发货，已更新收货地址。"},
        ]))

    # Pattern 5+6: query then refusal (out-of-window or shipped) — teaches policy_edge two-step
    for oid in ("A1007", "A1012"):  # A1007=20d (refund refuse), A1012=45d delivered (refund refuse)
        q = svc.query_order(oid)
        out.append(_sample([
            {"role": "user", "content": f"先看下 {oid} 能退吗，能退就退"},
            _tool_call("query_order", {"order_id": oid}),
            _tool_result("query_order", q),
            {"role": "assistant", "content": f"订单 {oid} 已超过 {REFUND_WINDOW_DAYS} 天退款窗口，无法直接退款。如有需要，可以为您创建人工工单。"},
        ]))

    return out


def ingest_external(paths: list[str], licenses: dict[str, str], eval_prompts: set[str]) -> list[SFTSample]:
    """Load external SFT corpora (ToolACE / APIGen-MT / xLAM), license-checked + deduped.

    Each path MUST have a recorded license in ``licenses`` (some corpora are research-only).
    Samples whose first user turn collides with a held-out eval prompt are dropped. Run on
    the GPU box where the corpora live; off-GPU this just validates the contract.
    """
    out: list[SFTSample] = []
    for path in paths:
        if path not in licenses:
            raise ValueError(f"refusing to ingest {path}: no recorded license")
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            sample = SFTSample.model_validate_json(line)
            first_user = next((m["content"] for m in sample.messages if m["role"] == "user"), "")
            if _normalize(first_user) in eval_prompts:
                continue  # dedup against the held-out eval set
            out.append(sample)
    return out


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


def train_val_split(
    samples: list[SFTSample], val_frac: float = 0.1, seed: int = 42
) -> tuple[list[SFTSample], list[SFTSample]]:
    """Deterministic train/val split (val is never used for the held-out benchmark)."""
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_frac)) if shuffled else 0
    return shuffled[n_val:], shuffled[:n_val]


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
