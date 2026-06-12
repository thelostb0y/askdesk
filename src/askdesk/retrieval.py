"""Hybrid retrieval: vector + keyword search fused with Reciprocal Rank Fusion.

Vector search catches paraphrases; keyword search catches exact terms (ids,
product names) that embeddings blur. RRF merges the two rankings without
needing the scores to be on comparable scales — rank positions only — which is
what makes it robust across stores (pgvector cosine vs ts_rank).
"""
from __future__ import annotations

from .gateway import Gateway, Usage
from .store import Hit, VectorStore

RRF_K = 60  # standard damping constant; rank 0 contributes 1/60, rank 9 ~1/69


def reciprocal_rank_fusion(rankings: list[list[Hit]], k: int) -> list[Hit]:
    scores: dict[tuple[str, int], float] = {}
    best: dict[tuple[str, int], Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking):
            key = (hit.source, hit.ordinal)
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            best.setdefault(key, hit)
    fused = sorted(scores, key=scores.get, reverse=True)[:k]
    return [Hit(best[key].source, best[key].ordinal, best[key].text, scores[key]) for key in fused]


class Retriever:
    def __init__(self, store: VectorStore, gateway: Gateway, top_k: int = 6):
        self.store = store
        self.gateway = gateway
        self.top_k = top_k

    def retrieve(self, question: str, usage: Usage | None = None) -> list[Hit]:
        embedding = self.gateway.embed([question], usage=usage)[0]
        vector_hits = self.store.vector_search(embedding, self.top_k * 2)
        keyword_hits = self.store.keyword_search(question, self.top_k * 2)
        return reciprocal_rank_fusion([vector_hits, keyword_hits], self.top_k)
