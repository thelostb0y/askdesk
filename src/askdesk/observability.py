"""Observability: Sentry (errors + traces) and structured per-request logging.

Sentry initializes only when SENTRY_DSN is set, so local dev and CI run clean.
Every /ask response already carries its own usage block (tokens, cost, calls);
log_request mirrors it to stdout as JSON for log-based metrics.
"""
from __future__ import annotations

import json
import logging
import os
import time

logger = logging.getLogger("askdesk")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def init_sentry() -> bool:
    dsn = os.environ.get("SENTRY_DSN")
    if not dsn:
        return False
    import sentry_sdk

    sentry_sdk.init(dsn=dsn, traces_sample_rate=0.2, profiles_sample_rate=0.1)
    return True


def log_request(question: str, result: dict, started: float) -> None:
    logger.info(
        json.dumps(
            {
                "event": "ask",
                "question_chars": len(question),
                "grounded": result["grounded"],
                "attempts": result["attempts"],
                "latency_ms": round((time.time() - started) * 1000),
                **result["usage"],
            }
        )
    )
