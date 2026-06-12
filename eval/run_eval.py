#!/usr/bin/env python3
"""AskDesk eval harness — the platform's quality gate.

Grades the full pipeline (ingest -> hybrid retrieval -> agent graph) against a
golden Q&A set on three metrics, and FAILS (exit 1) below thresholds, so it
runs as a CI gate:

  retrieval_hit_rate  expected source doc appears in the cited evidence
  groundedness_rate   Critic verdict: answer fully supported by evidence
  keyword_recall      expected answer keywords present in the answer text

Modes:
  --offline (default)  FakeGateway + MemoryStore: deterministic, no keys, no
                       network. Verifies the platform's mechanics — retrieval
                       ranking, citation flow, critic loop — on every commit.
  --live               Real models + pgvector (env-configured). Run before a
                       release or after a model/prompt change to measure true
                       answer quality. Same golden set, same thresholds.

Usage:
  uv run python eval/run_eval.py
  uv run python eval/run_eval.py --live
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from askdesk.agents import ask
from askdesk.chunking import chunk_document
from askdesk.retrieval import Retriever

GOLDEN = Path(__file__).parent / "golden.json"
CORPUS = Path(__file__).parent.parent / "corpus" / "sample"


def build_offline_runtime():
    from askdesk.store import MemoryStore
    from askdesk.testing import FakeGateway

    gateway = FakeGateway()
    store = MemoryStore()
    chunks = []
    for path in sorted(CORPUS.glob("*.md")):
        chunks.extend(chunk_document(path.name, path.read_text()))
    store.upsert(chunks, gateway.embed([c.text for c in chunks]))
    return Retriever(store, gateway, top_k=4), gateway


def build_live_runtime():
    from askdesk.config import get_settings
    from askdesk.gateway import Gateway
    from askdesk.store import PgVectorStore

    s = get_settings()
    gateway = Gateway()
    store = PgVectorStore(s.database_url, db_schema=s.db_schema)  # corpus already ingested
    return Retriever(store, gateway, top_k=s.top_k), gateway


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="real models + pgvector")
    args = parser.parse_args()

    spec = json.loads(GOLDEN.read_text())
    cases, thresholds = spec["cases"], spec["thresholds"]
    retriever, gateway = build_live_runtime() if args.live else build_offline_runtime()

    rows, hit = [], {"retrieval": 0, "grounded": 0, "keywords": 0}
    for case in cases:
        result = ask(case["question"], retriever=retriever, gateway=gateway)
        cited = {s["source"] for s in result["sources"]}
        retrieval_ok = case["expected_source"] is None or case["expected_source"] in cited
        keywords_ok = all(k.lower() in result["answer"].lower() for k in case["expected_keywords"])
        hit["retrieval"] += retrieval_ok
        hit["grounded"] += result["grounded"]
        hit["keywords"] += keywords_ok
        rows.append((case["id"], retrieval_ok, result["grounded"], keywords_ok))

    n = len(cases)
    metrics = {
        "retrieval_hit_rate": hit["retrieval"] / n,
        "groundedness_rate": hit["grounded"] / n,
        "keyword_recall": hit["keywords"] / n,
    }

    print(f"{'case':22} {'retrieval':>9} {'grounded':>8} {'keywords':>8}")
    for case_id, r, g, k in rows:
        print(f"{case_id:22} {'PASS' if r else 'FAIL':>9} {'PASS' if g else 'FAIL':>8} "
              f"{'PASS' if k else 'FAIL':>8}")
    print()
    failed = False
    for name, value in metrics.items():
        floor = thresholds[name]
        ok = value >= floor
        failed |= not ok
        print(f"{name:20} {value:.0%}  (floor {floor:.0%})  {'OK' if ok else 'BELOW FLOOR'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
