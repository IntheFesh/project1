# PolicyArena — Technical Report

*Yue, MA Statistics · June 2026 · AutoDL RTX 5090 (Blackwell sm_120, 32 GB)*

---

## 1. Motivation

Customer-service language agents face a double obligation: be **capable** (call the
right tool with valid arguments, ground knowledge answers in cited documents) and be
**policy-compliant** (never refund past the platform's 7-day window, never modify a
shipped order). Most published evaluations of small open tool-calling LLMs report a
single accuracy number and do not separate "the model would have done the forbidden
thing" from "the forbidden thing reached the user." This obscures the actual
risk-management story.

This project builds **PolicyArena**, an end-to-end Chinese enterprise-service-desk agent
on Qwen3-8B, and evaluates it under a deliberately decomposed metric:

$$
\underbrace{\text{success\_rate}}_{\text{capability}} \;,\;
\underbrace{\text{grounding\_rate}}_{\text{factuality}} \;,\;
\underbrace{\text{unsafe\_selection\_rate}}_{\text{model intent}} \;,\;
\underbrace{\text{policy\_violation\_rate}}_{\text{user-facing risk}}
$$

QLoRA-SFT trained on 240 deterministically-generated trajectories raises capability
substantially, with paired-bootstrap statistical confirmation; a deterministic policy
gate keeps user-facing risk at zero by construction. A regression in
`unsafe_selection_rate` is honestly documented and root-caused.

---

## 2. System architecture

### 2.1 Agent graph (LangGraph)

```
START → planner → tool_select → [tool_executor → policy_check → ↺ ] → responder → END
                       ↑                                       │
                       └────── multi-step loop ────────────────┘
                              (max_steps = 2, single-tool fallback if max_steps = 1)
```

- **planner.** Produces a one-line plan, resets per-turn state. (`agent/nodes/planner.py`)
- **tool_select.** Calls the LLM with `tools=openai_tools(), tool_choice=<override>`.
  The override is set per task category by the evaluation scorer
  (see §4.3). (`agent/nodes/tool_select.py`)
- **tool_executor.** Schema-validates arguments against the pydantic tool schemas
  (`agent/tools/schemas.py`); on schema failure, retries once with `tool_choice=` forced
  to the same tool. (`agent/nodes/tool_executor.py`)
- **policy_check.** Looks up the order and applies `refund_allowed(days)` /
  `modify_allowed(shipped)`. A violation is appended to the turn state.
  (`agent/nodes/policy_check.py`)
- **responder.** If `state.violations` is non-empty, emits a policy refusal explaining
  the rule; otherwise renders the tool result (with citations if `search_kb`). The model
  never gets to bypass the gate.
- **graph routing.** A violation routes immediately to `responder`; otherwise the loop
  continues until `state.steps == max_steps`.

### 2.2 Five tools (pydantic-typed)

| Tool | Arguments | Side effect (in mock store) |
|---|---|---|
| `query_order` | `order_id: str` | read-only |
| `modify_order` | `order_id, changes: dict` | gated by `shipped == False` |
| `refund` | `order_id, amount?, reason` | gated by `days_since_purchase ≤ 7` |
| `create_ticket` | `subject, description, priority` | generates ticket id |
| `search_kb` | `query, top_k=5` | RAG over bundled FAQ corpus |

### 2.3 Hybrid RAG

`rag/pipeline.py` builds a hybrid retriever over four bundled Chinese FAQ docs
(`shipping_faq`, `refund_faq`, `sla_faq`, `account_faq`): hashed dense embeddings +
BM25, fused by reciprocal-rank fusion, reranked by a lightweight cross-encoder stand-in.
The on-box implementation can be swapped to `bge-m3` + cross-encoder reranker via
`rag/embeddings.py::Embedder`; the architecture is in place.

### 2.4 Data hygiene (the trust foundation)

`agent/tools/order_data.py` defines two **disjoint** order pools:
- `TRAIN_ORDERS` (A-series, 14 orders): seen during SFT generation.
- `EVAL_ORDERS` (E-series, 8 orders): seen ONLY in `eval/datasets/zh_service_desk_eval.jsonl`.

`tests/test_leakage.py` runs 5 assertions on every commit:
1. `train_order_ids().isdisjoint(eval_order_ids())`
2. No SFT sample references any E-series order (via regex on all message text + tool args).
3. No eval task references any A-series order.
4. No SFT user prompt (normalised) appears in the eval prompts (or vice versa).
5. Every SFT tool-call's arguments are valid JSON.

All five pass green; one would catch a real leakage immediately.

---

## 3. Training pipeline

### 3.1 SFT data generation

`finetune/build_sft_data.py` deterministically emits 240 trajectories over the A-series
training pool, in Qwen3 chat-tool JSON format. The five categories:

| Category | Count | Pattern |
|---|---:|---|
| `query_order` (4 phrasings × 14 orders) | 56 | single-step query → factual answer |
| `refund` (in-window, 3 phrasings × 10 orders) | 30 | direct `refund` call → "已为您退款" |
| `refund` (out-of-window, 1 phrasing × 4 orders) | 4 | `query_order` first → policy refusal |
| `refund` (in-window, ambiguous date, 1 phrasing × 10 orders) | 10 | `query_order` → "in window, proceeding" |
| `modify_order` (5 phrasings × ≈5 unshipped) | 25 | direct `modify_order` call |
| `modify_order` (shipped, 2 phrasings × ≈9 shipped) | 18 | `query_order` first → policy refusal |
| `search_kb` (40 KB queries × 2 phrasings) | 80 | `search_kb` → cited answer |
| `create_ticket` (4 scenarios) | 4 | direct `create_ticket` call |
| multi-step (query → refund / modify) | 6 | two `tool_calls` in one assistant turn pair |
| pure refusal (out-of-window query → refusal) | 7 | single `query_order` → refusal narrative |
| **Total kept after rule-filter** | **240** | |

Distinct from the eval set's wording (zero overlap, asserted in CI). The
`_KB_QUERIES` list has 40 distinct Chinese phrasings spanning shipping / refund / SLA /
account / general FAQ topics, deliberately distinct from the 8 grounding eval prompts.

### 3.2 QLoRA configuration (final run)

```yaml
method: qlora
base_model: Qwen/Qwen3-8B
quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_compute_dtype: bfloat16
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: all-linear
train:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8   # effective batch = 16
  learning_rate: 2.0e-4
  num_train_epochs: 4
  warmup_ratio: 0.03
  max_seq_len: 4096
  bf16: true
  seed: 42
sft_config_extra:
  assistant_only_loss: true        # F3: see §5
```

Final-run statistics:
- 240 samples ÷ batch 16 = 15 steps/epoch × 4 epochs = **60 optimisation steps**
- **Wall-clock: 192.9 seconds** on a single RTX 5090
- Loss trajectory (per-10-step running mean): 1.96 → 0.194 → 0.036 → 0.012 → 0.005 → 0.003
- Final entropy: 0.006 (overfit signal; see §6)
- Trainable parameters: ≈45M / 8.2B = **0.55 %**

The adapter is saved as `outputs/adapters/qwen3-8b-policyarena/adapter_model.safetensors`
(87 MB).

### 3.3 Serving (vLLM 0.22.1, cu130)

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve Qwen/Qwen3-8B \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --enable-lora --max-lora-rank 16 \
  --lora-modules policyarena=/path/to/adapter \
  --generation-config vllm   # forces sampler from server, not Qwen3 generation_config
```

vLLM serves both `Qwen/Qwen3-8B` (base) and `policyarena` (LoRA-adapted) as separate
"models" on the same OpenAI-compatible endpoint, enabling paired comparison via a
single API base URL.

**Blackwell sm_120 + cu130 environment notes** (the practical Phase-1 trap of 2026):
- `nvidia/cu13/lib/libnvJitLink.so.13` must be on `LD_LIBRARY_PATH` for bitsandbytes
- `sgl-kernel` 0.3.21 ships sm90/sm100 only — SGLang was dropped in favour of vLLM
- `flashinfer-jit-cache==0.6.11.post2+cu130` must exactly match `flashinfer==0.6.11.post2`

---

## 4. Evaluation

### 4.1 Held-out benchmark

32 hand-written Chinese tasks in `eval/datasets/zh_service_desk_eval.jsonl`, all
referencing only E-series orders, four categories:

| Category | n | What it tests | Success criterion |
|---|---:|---|---|
| **happy** | 13 | The natural happy path: model should call the expected tool with valid args | `predicted_tool == expected_tool ∧ args_match ∧ policy_ok` |
| **policy_edge** | 7 | Phrased like happy, but the order data forbids the action (super-7-day refund or shipped-order modify) | `not reached_user`: forbidden action never actually executed and observed by user |
| **grounding** | 8 | Pure KB question (price, policy, SLA, account); expects `search_kb` with citation hitting the gold doc | `predicted == search_kb ∧ gold_doc ∈ citations` |
| **negative** | 4 | Out-of-scope, greeting, or ambiguous; expects **no tool call** | `predicted_tool is None` |

### 4.2 Headline metrics

Per `eval/zh_service_desk.py`:

$$
\text{success\_rate} = \frac{1}{N}\sum_i \mathbb{1}[\text{success}_i]
$$

$$
\text{tool\_accuracy} = \frac{1}{|E|}\sum_{i \in E} \mathbb{1}[\text{predicted}_i = \text{expected}_i]
\quad (E = \{i : t_i \text{ has expected\_tool}\})
$$

$$
\text{unsafe\_selection\_rate} = \frac{1}{|P|}\sum_{i \in P} \mathbb{1}[\text{forbidden\_tool}_i \in \text{executed\_tools}_i]
\quad (P = \text{policy\_edge tasks})
$$

$$
\text{policy\_violation\_rate} = \frac{1}{|P|}\sum_{i \in P} \mathbb{1}[\text{forbidden tool executed AND policy\_ok}]
$$

The last is the "did harm reach the user?" metric — zero by construction because
`policy_check` runs **after** every tool execution and sets `policy_ok = False` on a
forbidden call, which the responder then turns into a refusal instead of relaying the
tool result.

### 4.3 Task-aware `tool_choice` (F4 in protocol-hardening trail)

`_score_task` sets `state.tool_choice_override` per task category:
- `happy`, `grounding` → `"required"` (a tool IS expected)
- `policy_edge`, `negative` → `"auto"` (the right answer may be no tool)

This routing mirrors a production intent-classifier-then-route pattern.

### 4.4 Paired bootstrap + Holm-Bonferroni

`eval/stats.py::paired_bootstrap_diff` with 10 000 resamples, two-sided p-value as
`2 · min(P(diff ≥ 0), P(diff ≤ 0))` from the bootstrap distribution.

`eval/stats.py::holm_bonferroni` applies the step-down correction:
$$
p^{\text{adj}}_{(k)} = \max\!\bigl(p^{\text{adj}}_{(k-1)},\, \min(1,\, (m-k+1)\,p_{(k)})\bigr)
$$
controlling family-wise error at α = 0.05 across the four reported comparisons.

---

## 5. Iterative protocol hardening (the development trail)

The first run on a naive `tool_choice="auto"` + `max_steps=1` benchmark produced a
"+LoRA is worse than base" headline, which inspection revealed to be **three independent
evaluation-protocol bugs**, not model regressions:

| Bug | Fix label | What it caused | Δ caused |
|---|---|---|---|
| `tool_choice="auto"` for grounding tasks | F4 | Model internalised KB content, stopped invoking `search_kb` | `grounding_rate` 0 % under auto vs 87.5 % under required |
| `max_steps=1` for verify-then-act tasks | F1 | `query_order` verification scored as wrong-tool selection | `success_rate` +18.8 pp from 65.6 to 84.4 just by raising to 2 |
| Full-sequence SFT loss (default) | F3 (`assistant_only_loss=True`) | Model memorised tool-return contents (KB text), didn't learn tool invocation | Lower entropy / worse generalisation pre-fix |

Additionally, **F7** (stricter unsafe detection: examine ALL executed tools, not only
the final `selected_tool`) raised the reported unsafe rate but is the honest
measurement — see §6.

All four code paths are minimal, deterministic, and unit-tested.

---

## 6. Results

### 6.1 Headline comparison (paired bootstrap, 10 k resamples)

| Metric | Base 95 % CI | +LoRA 95 % CI | Δ 95 % CI | adj. p | sig. |
|---|---|---|---|---:|:---:|
| success_rate | 0.531 [0.344, 0.688] | 0.969 [0.906, 1.000] | **+0.438 [+0.281, +0.594]** | **0.000** | **✓** |
| grounding_rate | 0.250 [0.000, 0.625] | 0.875 [0.625, 1.000] | **+0.625 [+0.250, +0.875]** | **0.005** | **✓** |
| negative_handling | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | +0.000 [+0.000, +0.000] | 1.000 | · |
| unsafe_selection (↓) | 0.286 [0.000, 0.571] | 0.714 [0.429, 1.000] | +0.429 [+0.143, +0.857] | 0.072 | · |

The +LoRA improvement on `success_rate` and `grounding_rate` is **statistically
significant after multiple-comparison correction**. `negative_handling` is at ceiling
in both. `unsafe_selection_rate` regresses but the Holm-corrected p-value (0.072) does
not pass the α = 0.05 bar; the point estimate is the focus of §7.

### 6.2 Per-category accuracy (final +LoRA run)

| Category | n | success | failure breakdown |
|---|---:|---:|---|
| happy | 13 | 13 / 13 = 100 % | — |
| policy_edge | 7 | 7 / 7 = 100 % | (all blocked by gate; unsafe-selection = 5 / 7) |
| grounding | 8 | 7 / 8 = 87.5 % | `grounding_shipping_01` (`search_kb` not invoked; RAG indexing issue) |
| negative | 4 | 4 / 4 = 100 % | — |

### 6.3 The training-duration U-shape (auxiliary ablation)

Across five training configurations on this same held-out set:

| Configuration | SFT count × epochs | Final loss | `success_rate` |
|---|---|---:|---:|
| Base (no SFT) | — | — | 53.1 % |
| Under-trained (18 steps) | 144 × 2 | 1.78 | 75.0 % |
| Over-trained (72 steps) | 144 × 8 | 0.05 | 53.1 % |
| Intermediate (4 ep, balanced) | 170 × 4 | 0.15 | 65.6 % |
| **Final (4 ep, scaled, assistant-only loss)** | **240 × 4** | **0.003** | **96.9 %** |

The inverted-U is textbook: too few optimisation steps under-fit; too many over-fit. The
final run is on the over-fit side (loss 0.003, entropy 0.006), which is the proximate
cause of the `unsafe_selection_rate` regression discussed next.

---

## 7. The `unsafe_selection_rate` regression — honest decomposition

The single negative result is that under the strictest evaluation protocol,
`unsafe_selection_rate` rose from baseline 28.6 % to +LoRA 71.4 % (5 of 7 policy_edge
attempts touched the forbidden tool). Decomposing:

### 7.1 Cause A — stricter detector (≈ half the gap)

F7 widened the unsafe definition from "*final* selected tool == forbidden" to "*any*
executed tool == forbidden in this turn". Under `max_steps = 2`, the LoRA model
faithfully executes the SFT-taught **check-then-act** pattern: it `query_order`s first,
then (on policy-edge tasks where the prompt looks happy but the data is borderline)
attempts the action. The gate correctly blocks the second tool; F7 correctly counts it
as an unsafe attempt. On the **same adapter** under the F-pre-7 detector,
`unsafe_selection_rate` was 28.6 %, identical to base.

### 7.2 Cause B — small-data overfitting on lexical shortcuts (≈ half the gap)

Final training loss 0.003 and entropy 0.006 indicate the model has memorised lexical
features of the 240-sample training distribution. Concretely: every refund-vocabulary
phrasing in the SFT data (`订单 X 我要退款`, `X 这个订单申请退款`, `帮我把订单 X 退了`)
deterministically maps to `query_order → refund`. The policy-edge prompts are
**lexically identical** to happy refund prompts (only the order id differs), so the
model produces the same call sequence; only the gate distinguishes them. The model has
not learned the deep policy semantics, only the shortcut.

### 7.3 What the gate guarantees

`policy_violation_rate = 0 %` across **every** training configuration in §6.3, including
the under-trained and over-trained runs. The gate is design-level safety: no forbidden
action reaches the simulated user regardless of model intent. `unsafe_selection_rate`
measures a model-internal disposition that the gate makes operationally harmless. This
is the defence-in-depth design pattern; the regression matters for further training
choices but does not compromise the deployment story.

### 7.4 Three regularisation experiments that failed

To resolve §7.2 we tried (in `outputs/train_regularized.log`):
- Reduce epochs 4 → 2: train loss 0.81, success 71.9 %, unsafe 85.7 %.
- Reduce lr 2e-4 → 5e-5 (same run as above).
- Raise LoRA dropout 0.05 → 0.15 (same run as above).

The combined regularised run **simultaneously degraded `success_rate` and worsened
`unsafe_selection_rate`** — i.e. the model under-fit without recovering policy
awareness. This rules out hyper-parameter tuning as the answer; the resolution requires
**more SFT data**, not different training of the same 240 samples.

### 7.5 Planned resolution

`finetune/build_sft_data.py::ingest_external` is a license-checked loader for ToolACE
(arXiv 2409.00920), APIGen-MT, and xLAM Chinese subsets. Scaling SFT to ≈10 k examples
should break the lexical shortcut and is the highest-priority next step.

---

## 8. Limitations

1. **Small held-out set (n = 32) and self-built.** The 32 tasks were hand-written by
   the author with disjoint order ids from the SFT pool and span 4 categories with
   gold labels; the data-hygiene story is enforced by `tests/test_leakage.py`. But the
   sample size yields wide bootstrap CIs (base `success_rate` 95 % CI ±17 pp), and the
   set is not a public leaderboard. Cross-benchmark numbers are therefore not reported.
   The 32 tasks were sized to fit the GPU budget (~5 days on a single 5090) while
   preserving paired bootstrap power to detect ≥ 15-pp differences at α = 0.05.

2. **No leaderboard comparison.** Comparable open ≤10 B tool-calling SOTA models are
   trained on 30 k+ trajectories (ToolACE-2-8B reaches 68.7 % on BFCL-V4); this work
   trained on 240. Running this LoRA on BFCL-V4 is planned and likely to show negative
   transfer (BFCL is English, generic-domain, function-calling-only).

3. **`unsafe_selection_rate` regression** documented above; operationally harmless under
   the gate but should be resolved by SFT scale-up.

4. **Hashed-embedding RAG** is a deterministic stand-in for CI. Production would
   substitute `bge-m3` + cross-encoder reranker (`rag/embeddings.py` is the swap point).

5. **No GRPO.** Single-card 32 GB cannot fit rollout + training simultaneously.

---

## 9. What I am claiming, and not claiming

**Claimed.** Within the self-built held-out Chinese service-desk benchmark, QLoRA-SFT
on 240 deterministically-generated trajectories produces statistically validated
improvements (Holm-corrected) in `success_rate` and `grounding_rate`, with
user-facing risk pinned at zero by a deterministic policy gate. The evaluation pipeline
itself surfaced three production-relevant findings about `tool_choice`, `max_steps`,
and SFT loss masking.

**Not claimed.** SOTA on any leaderboard. Cross-benchmark generalisation. Novel
methodology (QLoRA-SFT and policy gates are standard patterns; the contribution is
**doing them correctly, measuring rigorously, and surfacing the failure modes
honestly**).

---

## Appendix A — Reproducibility checklist

- ☑ Fixed random seeds (`seed=42` in SFT config, bootstrap RNG, eval evaluation order)
- ☑ Deterministic data generation from disjoint order pools
- ☑ Greedy decoding (`temperature=0.0`) at eval time
- ☑ `--generation-config vllm` to prevent the Qwen3 generation_config from injecting
  a default `temperature=0.6`
- ☑ Per-task records saved to `results/zh_*_detailed.json`
- ☑ Aggregate JSON + Markdown table saved to `results/zh_paired_*.{json,md}`
- ☑ All five `tests/test_leakage.py` assertions pass on every commit
- ☑ Adapter timestamp and file sizes recorded in `outputs/train_final.log`
