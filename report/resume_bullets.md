# PolicyArena · 简历 Bullets（中英双语）

> 项目用途：中国求职版（LLM/Agent/RAG 工程师）+ 海外 PhD/RA 申请版

---

## 中文版（国内求职 — JD 高频关键词对齐）

### 项目名称
**PolicyArena** · 策略合规的工具调用 Agent 平台（Qwen3-8B · vLLM · LangGraph · QLoRA-SFT · 统计严格评测）

### 简历正文（3-4 条，可裁剪）

- **基于 Qwen3-8B 构建端到端可上线的中文企业服务台 Agent 平台**：LangGraph 多步状态机编排（planner → tool_select → tool_executor → policy_check → responder，max_steps=2 多步循环）+ vLLM/Blackwell 部署（cu130, sm_120）+ hybrid RAG（BM25 + dense + RRF + rerank）+ 工具 JSON Schema 校验 + 全链路 Langfuse tracing；策略门 (policy gate) **保证违禁动作 0% 触达用户**。

- **完成 QLoRA-SFT 微调（r=16, all-linear, 4-bit nf4, bf16 计算）并提供统计严格的对比评测**：在自建 32 题中文服务台 held-out benchmark 上，paired bootstrap + Holm-Bonferroni 校正下取得 **success_rate +43.8 pp [+28.1, +59.4], adj. p < 0.001** 与 **grounding_rate +62.5 pp [+25.0, +87.5], adj. p = 0.005** 的统计显著提升；policy_violation_rate 由 deterministic gate 强制为 0%。

- **设计严格的数据卫生与评测协议**：训练集（A 系列订单）与评测集（E 系列订单）完全 disjoint，由 `tests/test_leakage.py` 在 CI 中强制 5 项断言；评测指标解耦为 `unsafe_selection_rate`（模型意图）vs `policy_violation_rate`（用户实际风险）——后者由 gate 保证为 0，前者作为微调学习信号，体现 defense-in-depth 设计。

- **通过评测协议迭代发现三个生产级 finding**：(a) `tool_choice="auto"` vs `"required"` 在 grounding 与 negative-handling 任务上的反向 tradeoff，须按任务类型路由；(b) `max_steps=1` 单步评测错误地把"check-then-act" 模式判为工具选错（修复后 success_rate +18.8 pp）；(c) SFT 默认对全序列计算 loss，导致模型内化工具返回的 KB 内容、反而不调 `search_kb`，须 `assistant_only_loss=True`。所有 finding 已在代码中固化并文档化。

### 一句话版（简历顶部 highlight）

> 基于 Qwen3-8B + QLoRA-SFT + LangGraph 构建端到端策略合规工具调用 Agent，自建中文 service-desk benchmark 上 success_rate **53.1% → 96.9% (p < 0.001, paired bootstrap + Holm-Bonferroni)**，policy_violation_rate gate-guaranteed 0%。

### 技术栈关键词（JD 对齐）

`Python / PyTorch / Transformers / TRL / PEFT (LoRA, QLoRA) / vLLM 部署 / LangGraph 编排 / RAG (BM25, BGE, RRF, rerank) / OpenAI 兼容 API / pydantic schema / FastAPI / Docker Compose / GitHub Actions CI / Langfuse 观测 / 统计严谨（bootstrap CI, paired bootstrap, Holm-Bonferroni, pass^k） / Chinese tool-calling SFT data generation`

---

## English version (PhD / RA applications — embodied AI / LLM agents labs)

### Project name
**PolicyArena** — A statistically-validated, policy-compliant tool-calling agent on Qwen3-8B.

### Resume bullets (3 lines, for PhD / RA CV)

- **Built an end-to-end Chinese enterprise service-desk agent on Qwen3-8B** with a
  LangGraph multi-step state machine (planner → tool_select → tool_executor →
  policy_check → responder, `max_steps=2` agentic loop), vLLM/Blackwell serving
  (sm_120, cu130), hybrid RAG (BM25 + dense + RRF + rerank), and a deterministic
  policy gate that guarantees **`policy_violation_rate = 0%`** by construction.

- **Trained QLoRA-SFT (r=16, all-linear, 4-bit NF4, bf16) on 240 deterministically-generated trajectories** with disjoint train/eval order pools (CI-enforced data
  hygiene); on a 32-task held-out Chinese service-desk benchmark, **achieved
  statistically significant improvements** (paired bootstrap, 10k resamples,
  Holm-Bonferroni corrected): `success_rate` +43.8 pp [+28.1, +59.4], **adj. p < 0.001**;
  `grounding_rate` +62.5 pp [+25.0, +87.5], **adj. p = 0.005**.

- **Surfaced three production-relevant findings through iterative evaluation-protocol
  hardening**: (a) `tool_choice="auto"` vs `"required"` induces a reverse trade-off
  between grounding and negative-input handling — task-aware routing is needed; (b)
  `max_steps=1` mis-scores "check-then-act" SFT patterns (fixing → +18.8 pp); (c)
  default SFT loss over full sequences trains the model to *memorise* tool returns
  (KB content), defeating retrieval — `assistant_only_loss=True` is required. Each
  finding is documented in code with a clean repro path.

### One-line tagline (for cold-email subject line / CV header)

> *Built a policy-compliant tool-calling agent on Qwen3-8B with QLoRA-SFT; paired-bootstrap-validated +43.8 pp success on a held-out Chinese service-desk benchmark; deterministic policy gate guarantees zero policy violations reaching the user.*

### Skill keywords (PhD-app-friendly)

`Language agents · LLM tool-use · QLoRA / PEFT · LangGraph · vLLM · RAG · Statistical evaluation (bootstrap CI, paired bootstrap, Holm-Bonferroni, pass^k) · Reproducibility (CI-enforced data hygiene, leakage tests) · Production agent design patterns (defence-in-depth policy gate, task-aware tool_choice routing) · Failure-mode analysis (U-shape under/overfit ablation)`

---

## 邮件模板（给目标实验室教授）

### 给国内 LLM/Agent 工程师岗（JD 直接对应）

> 您好 X 团队 / 招聘官：
>
> 我是统计学背景的应届硕士，最近在 GitHub 开源了一个端到端策略合规 Agent 平台
> **PolicyArena**（Qwen3-8B + QLoRA-SFT + LangGraph + vLLM），对应贵司 JD 中的「RAG
> 链路工程化 / 函数调用微调 / 评测集自动化回归 / 可观测性 / SGLang/vLLM 部署」需求。
>
> 主要数字（自建 32 题中文 service-desk benchmark，paired bootstrap + Holm 校正）：
> success_rate **53.1% → 96.9% (p < 0.001)**，grounding_rate 25% → 87.5% (p = 0.005)，
> policy_violation_rate 由 deterministic gate 保证为 **0%**。
>
> 仓库：https://github.com/{user}/policyarena · 报告：`report/case_study.md` ·
> 一页 PDF：`report/one_pager.pdf`。我也准备好任何时段做 60 分钟现场技术讲解。
>
> 期待您的回复。

### To PhD/RA in HK / Singapore / Europe (embodied AI labs)

> Dear Prof. [Last name],
>
> I came across your recent paper on [specific paper] — particularly your observation
> that [one specific technical point from their paper]. I am writing to ask whether you
> are taking PhD students (or RAs) for the [fall 2026 / spring 2027] cohort.
>
> My background is an MA in Statistics, and I have been building open-source
> infrastructure to learn agent / VLA evaluation rigorously. The most recent project,
> **PolicyArena**, is a small open-model (Qwen3-8B) policy-compliant tool-calling agent
> with **paired-bootstrap-validated +43.8 pp `success_rate` and +62.5 pp `grounding_rate`
> gains under Holm-Bonferroni correction**, and a deterministic policy gate that
> guarantees zero policy violations reach the user. The repository is at
> https://github.com/{user}/policyarena and the case study is `report/case_study.md`.
>
> What I would bring to your group is **statistical rigour in evaluation** (paired
> bootstrap, Holm correction, pass^k — none of which is standard in current LLM-agent
> papers) and a track record of completing real end-to-end systems. I would be grateful
> for any pointer toward your application process.
>
> Best regards,
> Yue
> [Statistics MA, US] · [GitHub] · [CV PDF link]

---

## 项目的差异化卖点（写在简历底部或 cover letter）

| 卖点 | 对应实力证明 |
|---|---|
| **统计严谨**（区别于绝大多数报单点数字的工程项目）| paired bootstrap + Holm-Bonferroni，每个数字都有 95% CI |
| **数据卫生 CI 强制**（区别于"自评测训练集"的项目）| `tests/test_leakage.py` 5 项断言 in CI |
| **诚实的失败分析**（区别于"只贴正面数字"的项目）| `unsafe_selection_rate` 回归被完整 root-cause 文档化 |
| **生产级 finding 来自评测协议本身**（区别于"用 OpenAI API + RAG demo"）| 3 个 finding：tool_choice tradeoff / max_steps interaction / SFT loss masking |
| **真完整工程**（区别于"Jupyter notebook 项目"）| Docker Compose + CI + 类型化 schema + 5 个 tool + 4 个 KB doc + 32 题 benchmark + 4 个 metric |

---

## 反 BS 准备（面试 / 招生 push-back 时）

> "你这 32 题太少了，CI 都那么宽，你 96.9% 没意义。"

✅ 准备好的回答：

> 您说得对，32 题确实小（base 95% CI 是 [0.34, 0.69]，半宽 ±17 pp）。但我做的是 paired
> bootstrap 而非独立两 sample 比较 —— **paired 设计能 detect 个体级差异，统计 power
> 高得多**。我的 success_rate Δ = +43.8 pp 的 95% CI 是 [+28.1, +59.4]，**完全不跨 0**，
> 且 Holm-Bonferroni 校正后 adj. p < 0.001。规模上的局限我在报告 §8 完整承认，下一步
> 计划是 scale 到 ≈10k SFT 样本并跑 BFCL-V4 拿 leaderboard-comparable 数字。

> "unsafe_selection_rate 反而升了，你的 SFT 是不是没用？"

✅ 准备好的回答：

> 这正是我最想讲的部分（报告 §7）。这个回归是两个独立原因 superpose 的结果：(a) 我加严
> 了 unsafe detector，从只看 final tool 改成看 turn 内所有 executed tools —— 同款模型
> 在旧 detector 下 unsafe 是 28.6%，跟 baseline 一样；(b) 240 样本 × 4 epoch 的 SFT 在
> 240 样本上确实过拟合（final loss 0.003），模型对 refund/modify 词条件反射。**重要的
> 是 policy_violation_rate 全程是 0**，因为 deterministic gate 保证没有 forbidden action
> 真的到用户面前。这就是 defense-in-depth：模型有 disposition 错没关系，gate 兜底。
> 真正解决 (b) 需要 scale SFT data，不是 hyperparameter tuning（我做了 3 个 reg 实验，
> 简单 reg 会同时让 success 和 unsafe 都变差，文档 §7.4）。

> "你这跟 ToolACE / xLAM 比怎么样？"

✅ 准备好的回答：

> 不能直接比，我也不会假装能比。ToolACE-2-8B 在 BFCL-V4 上是 68.7%，但它训了 30k+
> trajectories，我训了 240。我的自建中文 service-desk benchmark 跟 BFCL（英文、通用域）
> 也不是同一 task。我项目的真实卖点不是 leaderboard 数字，而是 **policy-aware
> evaluation framework + 统计严谨 + 失败模式分析 + 全栈工程**——这些是 BFCL 测不出来
> 的维度，也是产品上线时真正要解决的问题。如果 leaderboard 数字是这个岗位/PhD 项目最关
> 心的事，我可以在 1 周内跑完 BFCL-V4 + τ²-bench retail，提供真实的 cross-benchmark
> 数字。
