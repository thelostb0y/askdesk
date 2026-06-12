"""Deterministic corpus chunking.

Paragraph-aware sliding window: split on blank lines, pack paragraphs into
chunks up to `max_chars`, overlap by carrying the last paragraph forward so
answers that straddle a boundary stay retrievable. Pure function — fully
unit-testable with no model in the loop.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    source: str  # document identifier (e.g. relative file path)
    ordinal: int  # position of the chunk within its document
    text: str


def chunk_document(source: str, text: str, max_chars: int = 1200) -> list[Chunk]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[Chunk] = []
    buf: list[str] = []
    size = 0
    for para in paragraphs:
        # paragraph alone exceeding the window gets hard-split rather than dropped
        while len(para) > max_chars:
            head, para = para[:max_chars], para[max_chars:]
            if buf:
                chunks.append(Chunk(source, len(chunks), "\n\n".join(buf)))
                buf, size = [], 0
            chunks.append(Chunk(source, len(chunks), head))
        if size + len(para) > max_chars and buf:
            chunks.append(Chunk(source, len(chunks), "\n\n".join(buf)))
            buf, size = [buf[-1]], len(buf[-1])  # overlap: carry last paragraph
        buf.append(para)
        size += len(para)
    if buf:
        chunks.append(Chunk(source, len(chunks), "\n\n".join(buf)))
    return chunks
