# Observability

**CORE (Phase 4):** self-hosted **Langfuse** (via `docker-compose.yml`) tracing every LLM call,
tool call, and retrieval, plus prompt versioning and stored eval scores. Configure with
`LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` (from `.env`).

**STRETCH (Phase 8):** Prometheus + Grafana dashboards.

Instrumentation code is added in Phase 4.
