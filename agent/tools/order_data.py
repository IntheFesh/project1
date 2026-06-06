"""Disjoint train / eval order pools — the data-hygiene foundation.

SFT trajectories (``finetune/build_sft_data.py``) draw ONLY from ``TRAIN_ORDERS``; the
held-out service-desk benchmark (``eval/datasets/zh_service_desk_eval.jsonl``) references
ONLY ``EVAL_ORDERS``. The two id sets are disjoint by construction and asserted by
``tests/test_leakage.py`` — so reported accuracy can never be inflated by training on the
evaluation prompts/orders.

Each entry is plain data (no dependency on ``services``) so this module imports nothing
from the agent; ``services`` builds ``Order`` objects from the union of both pools.
"""

from __future__ import annotations

from typing import Any

# Train pool (A-series): seen during SFT. Covers refund-in-window, refund-out-of-window,
# modify-unshipped (allowed), modify-shipped (refused).
TRAIN_ORDERS: dict[str, dict[str, Any]] = {
    "A1001": {"status": "paid", "amount": 199.0, "days_since_purchase": 2, "shipped": False, "items": ["蓝牙耳机"]},
    "A1002": {"status": "shipped", "amount": 560.0, "days_since_purchase": 4, "shipped": True, "items": ["机械键盘"]},
    "A1003": {"status": "paid", "amount": 320.0, "days_since_purchase": 1, "shipped": False, "items": ["显示器"]},
    "A1004": {"status": "shipped", "amount": 89.0, "days_since_purchase": 10, "shipped": True, "items": ["充电器"]},
    "A1005": {"status": "paid", "amount": 75.0, "days_since_purchase": 3, "shipped": False, "items": ["鼠标垫"]},
    "A1006": {"status": "shipped", "amount": 420.0, "days_since_purchase": 2, "shipped": True, "items": ["显示器支架"]},
    "A1007": {"status": "delivered", "amount": 35.0, "days_since_purchase": 20, "shipped": True, "items": ["贴纸"]},
    "A1008": {"status": "paid", "amount": 260.0, "days_since_purchase": 5, "shipped": False, "items": ["键帽套装"]},
    "A1009": {"status": "delivered", "amount": 88.0, "days_since_purchase": 30, "shipped": True, "items": ["数据线"]},
    "A1010": {"status": "shipped", "amount": 150.0, "days_since_purchase": 6, "shipped": True, "items": ["鼠标"]},
    "A1011": {"status": "shipped", "amount": 130.0, "days_since_purchase": 7, "shipped": True, "items": ["蓝牙音箱"]},
    "A1012": {"status": "delivered", "amount": 999.0, "days_since_purchase": 45, "shipped": True, "items": ["平板电脑"]},
    "A1013": {"status": "paid", "amount": 50.0, "days_since_purchase": 0, "shipped": False, "items": ["硅胶手机壳"]},
    "A1014": {"status": "shipped", "amount": 310.0, "days_since_purchase": 9, "shipped": True, "items": ["无线路由器"]},
}

# Eval pool (E-series): NEVER seen during training. Disjoint ids; covers the same policy
# edges plus the 7-day refund boundary (7d allowed vs 8d refused).
EVAL_ORDERS: dict[str, dict[str, Any]] = {
    "E9001": {"status": "paid", "amount": 249.0, "days_since_purchase": 2, "shipped": False, "items": ["无线鼠标"]},
    "E9002": {"status": "shipped", "amount": 480.0, "days_since_purchase": 3, "shipped": True, "items": ["曲面显示器"]},
    "E9003": {"status": "delivered", "amount": 120.0, "days_since_purchase": 30, "shipped": True, "items": ["手机壳"]},
    "E9004": {"status": "paid", "amount": 99.0, "days_since_purchase": 7, "shipped": False, "items": ["薄膜键盘"]},
    "E9005": {"status": "shipped", "amount": 300.0, "days_since_purchase": 8, "shipped": True, "items": ["头戴耳机"]},
    "E9006": {"status": "shipped", "amount": 1200.0, "days_since_purchase": 5, "shipped": True, "items": ["笔记本电脑"]},
    "E9007": {"status": "delivered", "amount": 60.0, "days_since_purchase": 15, "shipped": True, "items": ["数据线"]},
    "E9008": {"status": "paid", "amount": 45.0, "days_since_purchase": 0, "shipped": False, "items": ["钢化膜"]},
}


def train_order_ids() -> set[str]:
    """Order ids permitted in SFT data."""
    return set(TRAIN_ORDERS)


def eval_order_ids() -> set[str]:
    """Order ids permitted in the held-out benchmark."""
    return set(EVAL_ORDERS)


def all_orders() -> dict[str, dict[str, Any]]:
    """Union of both pools (the store holds every order; tasks reference disjoint subsets)."""
    return {**TRAIN_ORDERS, **EVAL_ORDERS}
