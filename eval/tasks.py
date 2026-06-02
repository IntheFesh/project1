"""Fixed, deterministic task slices for off-GPU pipeline checks and the CI gate.

These are SYNTHETIC smoke tasks (one per tool), NOT a benchmark. Real evaluation uses
tau2-bench / BFCL-V4 on the served model (see eval/run_tau2.py, eval/run_bfcl.py).
"""

from __future__ import annotations

SMOKE_TASKS: list[dict[str, str]] = [
    {"prompt": "查询订单 A1001 的状态", "expected_tool": "query_order"},
    {"prompt": "订单 A1001 我要退款", "expected_tool": "refund"},
    {"prompt": "把订单 A1001 的地址改一下", "expected_tool": "modify_order"},
    {"prompt": "帮我创建一个工单投诉", "expected_tool": "create_ticket"},
    {"prompt": "请问运费是怎么计算的？", "expected_tool": "search_kb"},
]
