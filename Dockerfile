# Multi-stage build: uv resolves deps into a venv, slim runtime copies it in.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY README.md ./
RUN uv sync --frozen --no-dev

FROM python:3.12-slim-bookworm
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY src ./src
ENV PATH="/app/.venv/bin:$PATH"
# 12-factor: ALL config via env (ASKDESK_*, SENTRY_DSN, provider keys).
# Stateless service — scale horizontally behind any load balancer.
EXPOSE 8000
CMD ["uvicorn", "askdesk.api:app", "--host", "0.0.0.0", "--port", "8000"]
