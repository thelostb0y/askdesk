"""Deterministic fakes shipped with the platform.

Internal teams (and our own CI) can run the entire stack — retrieval, agent
graph, API — with zero network and zero API keys. FakeGateway is scriptable so
tests can force specific model behaviors (e.g. an ungrounded draft) and assert
how the platform reacts.
"""
from __future__ import annotations

import hashlib
import json
import re

from .gateway import Usage

_WORD = re.compile(r"[a-z0-9]+")
FAKE_DIM = 64


def fake_embedding(text: str) -> list[float]:
    """Hashed bag-of-words: deterministic, and texts sharing vocabulary get
    similar vectors — so cosine retrieval behaves meaningfully in tests."""
    vec = [0.0] * FAKE_DIM
    for token in _WORD.findall(text.lower()):
        digest = hashlib.md5(token.encode()).digest()
        vec[digest[0] % FAKE_DIM] += 1.0
    norm = sum(v * v for v in vec) ** 0.5
    return [v / norm for v in vec] if norm else vec


_STOPWORDS = frozenset(
    "a an and are at be by do does for from had has have how in is it its of on or "
    "should that the their they this to was were what when where which who why with".split()
)

_EVIDENCE_BLOCK = re.compile(r"\[(\d+)\] \([^)]*\)\n(.*?)(?=\n\n\[\d+\] \(|\Z)", re.DOTALL)

NO_ANSWER = "The corpus does not answer this."


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS}


class FakeGateway:
    """Drop-in Gateway. Default analyst behavior is deterministic *extractive*
    answering: pick the evidence sentence with the highest content-word overlap
    with the question and cite its block; below 2 overlapping words, declare
    the corpus doesn't answer (so out-of-corpus cases behave realistically).
    The default critic approves. Pass `chat_script` to override responses in
    order (None = fall through to default behavior for that call)."""

    def __init__(self, chat_script: list[str | None] | None = None):
        self.chat_script = list(chat_script or [])
        self.chat_calls: list[tuple[str, str]] = []

    def embed(self, texts: list[str], usage: Usage | None = None) -> list[list[float]]:
        if usage is not None:
            usage.add(sum(len(t.split()) for t in texts), 0, 0.0)
        return [fake_embedding(t) for t in texts]

    def chat(self, system: str, user: str, usage: Usage | None = None) -> str:
        self.chat_calls.append((system, user))
        if usage is not None:
            usage.add(len(user.split()), 24, 0.0001)
        if self.chat_script:
            scripted = self.chat_script.pop(0)
            if scripted is not None:
                return scripted
        if "adversarial reviewer" in system:
            return json.dumps({"grounded": True, "unsupported_claims": []})
        return self._extractive_answer(user)

    @staticmethod
    def _extractive_answer(user: str) -> str:
        evidence_part, _, question_part = user.partition("\n\nQuestion:")
        question_words = _content_words(question_part.split("\n")[0])
        best: tuple[int, str, str] | None = None  # (overlap, sentence, block_n)
        for block_n, body in _EVIDENCE_BLOCK.findall(evidence_part):
            # drop markdown heading lines, then sentence-split the prose
            prose = " ".join(
                line.strip() for line in body.splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            )
            for sentence in re.split(r"(?<=[.!?])\s+", prose):
                sentence = sentence.strip()
                if not sentence:
                    continue
                overlap = len(question_words & _content_words(sentence))
                if best is None or overlap > best[0]:
                    best = (overlap, sentence, block_n)
        if best is None or best[0] < 2:
            return NO_ANSWER
        return f"{best[1]} [{best[2]}]"
