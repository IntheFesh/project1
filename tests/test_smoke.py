"""Smoke tests: every package/module imports cleanly."""

import importlib

import pytest

MODULES = [
    "agent",
    "agent.state",
    "agent.graph",
    "agent.nodes.planner",
    "agent.nodes.tool_select",
    "agent.nodes.tool_executor",
    "agent.nodes.policy_check",
    "agent.nodes.responder",
    "agent.tools.schemas",
    "agent.tools.registry",
    "agent.policies.rules",
    "rag.ingest",
    "rag.index",
    "rag.retrieve",
    "rag.rerank",
    "api.main",
    "api.auth",
    "api.ratelimit",
    "finetune.build_sft_data",
    "finetune.train_lora",
    "finetune.train_grpo",
    "eval.passk",
    "eval.bootstrap",
    "eval.run_tau2",
    "eval.run_bfcl",
    "eval.rag_triad",
]


@pytest.mark.parametrize("module", MODULES)
def test_import(module: str) -> None:
    importlib.import_module(module)
