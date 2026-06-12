#!/usr/bin/env python3
"""Ingest a directory of .md/.txt documents into the vector store.

Usage:
  uv run python scripts/ingest_corpus.py corpus/sample
  uv run python scripts/ingest_corpus.py /path/to/your/docs --max-chars 1200
"""
from __future__ import annotations

import argparse
from pathlib import Path

from askdesk.chunking import chunk_document
from askdesk.config import get_settings
from askdesk.gateway import Gateway, Usage
from askdesk.store import PgVectorStore

EMBED_BATCH = 64


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus_dir", type=Path)
    ap.add_argument("--max-chars", type=int, default=1200)
    args = ap.parse_args()

    files = sorted(p for p in args.corpus_dir.rglob("*") if p.suffix in {".md", ".txt"})
    if not files:
        raise SystemExit(f"no .md/.txt files under {args.corpus_dir}")

    chunks = []
    for path in files:
        source = str(path.relative_to(args.corpus_dir))
        chunks.extend(chunk_document(source, path.read_text(), max_chars=args.max_chars))
    print(f"{len(files)} documents -> {len(chunks)} chunks")

    settings = get_settings()
    gateway = Gateway()
    store = PgVectorStore(settings.database_url)
    usage = Usage()
    total = 0
    for i in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[i : i + EMBED_BATCH]
        embeddings = gateway.embed([c.text for c in batch], usage=usage)
        total += store.upsert(batch, embeddings)
        print(f"  upserted {total}/{len(chunks)}")
    print(f"done. embedding usage: {usage.as_dict()}")


if __name__ == "__main__":
    main()
