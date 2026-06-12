from askdesk.retrieval import reciprocal_rank_fusion
from askdesk.store import Hit


def _hit(source: str, ordinal: int = 0, score: float = 1.0) -> Hit:
    return Hit(source, ordinal, f"text of {source}#{ordinal}", score)


def test_rrf_prefers_items_ranked_high_in_both_lists():
    vector = [_hit("a"), _hit("b"), _hit("c")]
    keyword = [_hit("b"), _hit("c"), _hit("d")]
    fused = reciprocal_rank_fusion([vector, keyword], k=4)
    assert fused[0].source == "b"  # rank 1 + rank 0 beats everything single-list
    assert {h.source for h in fused} == {"a", "b", "c", "d"}


def test_rrf_deduplicates_by_source_and_ordinal():
    fused = reciprocal_rank_fusion([[_hit("a", 1)], [_hit("a", 1)]], k=5)
    assert len(fused) == 1


def test_hybrid_retrieval_finds_the_right_document(retriever):
    hits = retriever.retrieve("How long do customers have to request a full refund?")
    assert hits, "retrieval returned nothing"
    assert hits[0].source == "refund-policy.md"


def test_keyword_side_catches_exact_terms(retriever):
    hits = retriever.retrieve("data-processing ticket PII exports")
    assert any(h.source == "security-policy.md" for h in hits)
