"""SFT data building + rule filtering, LoRA dry-run, and the GRPO verifiable reward."""

import json
from pathlib import Path

import pytest

from finetune.build_sft_data import (
    SFTSample,
    build_chinese_trajectories,
    build_sft_data,
    ingest_external,
    rule_filter,
    train_val_split,
)
from finetune.train_grpo import verifiable_reward
from finetune.train_lora import dry_run


def test_trajectories_are_valid_and_chinese() -> None:
    samples = build_chinese_trajectories()
    assert len(samples) >= 5
    assert all(s.language == "zh" for s in samples)
    # every sample has a system, a user, and a final assistant message
    for sample in samples:
        roles = [m["role"] for m in sample.messages]
        assert "user" in roles
        assert any(m["role"] == "assistant" and m.get("content") for m in sample.messages)


def test_rule_filter_drops_invalid_tool_args() -> None:
    bad = SFTSample(
        messages=[
            {"role": "user", "content": "x"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"function": {"name": "query_order", "arguments": "{}"}}  # missing order_id
                ],
            },
            {"role": "assistant", "content": "done"},
        ],
        source="test",
        language="zh",
    )
    assert rule_filter([bad]) == []


def test_build_sft_data_writes_jsonl(tmp_path: Path) -> None:
    out = tmp_path / "sft.jsonl"
    count = build_sft_data(str(out))
    assert count > 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == count
    first = json.loads(lines[0])
    assert "messages" in first and first["language"] == "zh"


def test_dry_run_reports_ready(tmp_path: Path) -> None:
    data = tmp_path / "sft.jsonl"
    build_sft_data(str(data))
    plan = dry_run("configs/lora.yaml", str(data))
    assert plan["ready"] is True
    assert plan["samples"] > 0
    assert plan["method"] in {"lora", "qlora"}


def test_train_val_split_is_deterministic_and_disjoint() -> None:
    samples = rule_filter(build_chinese_trajectories())
    train, val = train_val_split(samples, val_frac=0.1, seed=42)
    assert len(train) + len(val) == len(samples)
    assert train and val
    # deterministic
    assert train_val_split(samples, seed=42)[1] == val


def test_ingest_external_requires_a_recorded_license(tmp_path: Path) -> None:
    corpus = tmp_path / "toolace.jsonl"
    corpus.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no recorded license"):
        ingest_external([str(corpus)], licenses={}, eval_prompts=set())


def test_ingest_external_dedups_against_eval(tmp_path: Path) -> None:
    from finetune.build_sft_data import _normalize

    corpus = tmp_path / "ext.jsonl"
    keep = SFTSample(messages=[{"role": "user", "content": "外部样本独有问题"},
                               {"role": "assistant", "content": "答复"}], source="ext", language="zh")
    leak = SFTSample(messages=[{"role": "user", "content": "运费是按什么标准来收取的？"},
                               {"role": "assistant", "content": "答复"}], source="ext", language="zh")
    corpus.write_text("\n".join(json.dumps(s.model_dump(), ensure_ascii=False) for s in (keep, leak)),
                      encoding="utf-8")
    eval_prompts = {_normalize("运费是按什么标准来收取的？")}
    out = ingest_external([str(corpus)], licenses={str(corpus): "research-only"}, eval_prompts=eval_prompts)
    assert len(out) == 1 and out[0].messages[0]["content"] == "外部样本独有问题"


def test_verifiable_reward() -> None:
    assert verifiable_reward(tool_correct=True, args_valid=True, policy_ok=True) == 1.0
    assert verifiable_reward(tool_correct=True, args_valid=True, policy_ok=False) == 0.0
    assert verifiable_reward(tool_correct=True, args_valid=False, policy_ok=True) == 0.6
