# Contributing to PolicyArena

Thanks for your interest. This is a research + portfolio repository with a hard rule:
**no fabricated metrics** — every reported number must be reproducible from the code and a
saved `results/*.json`.

## Development setup

```bash
# uv (https://astral.sh/uv) + Python 3.11+
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync                 # light, GPU-free runtime + dev tools
make check              # ruff + pytest (128 tests) + deterministic eval gate
```

Everything off-GPU runs against the deterministic mock backend (`ScriptedLLMClient`) — no
model or GPU is required to develop or test. GPU-only steps (training, vLLM serving, real
evals) are documented in the README "Reproduce" section and `BLACKWELL_NOTES.md`.

## Before opening a PR

```bash
make check              # must be green: ruff clean, all tests pass, [eval-gate] PASS
```

- **Style.** `ruff` (config in `pyproject.toml`, `target-version = py311`). Match the
  surrounding code: typed, small functions, docstrings, no magic constants (knobs live in
  `configs/*.yaml`).
- **Tests.** Add or update tests under `tests/` for any behaviour change. The data-leakage
  guards (`tests/test_leakage.py`) and the deterministic eval gate must stay green.
- **Honesty.** If a change could alter a reported number, say so in the PR and re-run the
  affected evaluation; do not edit a results table by hand. Mark anything not actually run
  (e.g. τ²-bench, BFCL-V4) as future work.
- **Branches.** Develop on a feature branch; never force-push shared branches.

## Project layout

See the README "Repository layout". The agent graph is `agent/`, retrieval is `rag/`,
evaluation + statistics are `eval/`, training is `finetune/`, and the narrative write-ups
are in `report/`.
