from __future__ import annotations

from pathlib import Path

import pytest

from askdesk.chunking import chunk_document
from askdesk.retrieval import Retriever
from askdesk.store import MemoryStore
from askdesk.testing import FakeGateway

CORPUS = Path(__file__).parent.parent / "corpus" / "sample"


@pytest.fixture
def gateway() -> FakeGateway:
    return FakeGateway()


@pytest.fixture
def store(gateway: FakeGateway) -> MemoryStore:
    """MemoryStore pre-loaded with the sample corpus via the real chunker."""
    memory = MemoryStore()
    chunks = []
    for path in sorted(CORPUS.glob("*.md")):
        chunks.extend(chunk_document(path.name, path.read_text()))
    memory.upsert(chunks, gateway.embed([c.text for c in chunks]))
    return memory


@pytest.fixture
def retriever(store: MemoryStore, gateway: FakeGateway) -> Retriever:
    return Retriever(store, gateway, top_k=4)
