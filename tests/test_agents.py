import json

from askdesk.agents import _parse_verdict, ask
from askdesk.testing import FakeGateway


def test_happy_path_returns_grounded_cited_answer(retriever, gateway):
    result = ask("How long do customers have to request a full refund?", retriever, gateway)
    assert result["grounded"] is True
    assert result["attempts"] == 1
    assert "[1]" in result["answer"]
    assert any(s["source"] == "refund-policy.md" for s in result["sources"])
    assert result["usage"]["llm_calls"] >= 2  # analyst + critic at minimum


def test_critic_rejection_triggers_one_retry(retriever):
    bad_verdict = json.dumps({"grounded": False, "unsupported_claims": ["made-up claim"]})
    # script: analyst draft, critic rejects, analyst rewrite (default), critic approves (default)
    gateway = FakeGateway(chat_script=["Bad draft with no support.", bad_verdict, None, None])
    result = ask("How long do customers have to request a full refund?", retriever, gateway)
    assert result["attempts"] == 2
    assert result["grounded"] is True


def test_retries_are_bounded_and_flagged_ungrounded(retriever):
    bad = json.dumps({"grounded": False, "unsupported_claims": ["still wrong"]})
    gateway = FakeGateway(chat_script=[None, bad, None, bad])
    result = ask("How long do refunds take?", retriever, gateway, max_retries=1)
    assert result["attempts"] == 2  # initial + 1 retry, then stop
    assert result["grounded"] is False


def test_unparseable_critic_verdict_fails_closed():
    grounded, claims = _parse_verdict("I think it looks fine to me!")
    assert grounded is False
    assert claims


def test_verdict_parsing_tolerates_code_fences():
    raw = "```json\n{\"grounded\": true, \"unsupported_claims\": []}\n```"
    grounded, claims = _parse_verdict(raw)
    assert grounded is True
    assert claims == []
