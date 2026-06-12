"""Multi-agent answer graph (LangGraph): Researcher → Analyst → Critic.

  Researcher  retrieves evidence (hybrid RAG) for the question.
  Analyst     drafts an answer using ONLY the retrieved evidence, with [n]
              citations pointing at specific chunks.
  Critic      adversarially verifies the draft: every claim must be supported
              by the cited evidence. Unsupported → bounce back to the Analyst
              with the critique (bounded retries), else mark ungrounded.

The Critic loop is the platform's groundedness guarantee: an answer never
leaves the graph claiming support it doesn't have. `grounded=False` responses
are still returned (callers may want them) but are explicitly flagged.
"""
from __future__ import annotations

import json
import re
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from .gateway import Gateway, Usage
from .retrieval import Retriever
from .store import Hit

ANALYST_SYSTEM = """\
You are an internal knowledge-base analyst. Answer the user's question using ONLY
the numbered evidence excerpts provided. Cite evidence inline as [1], [2], etc.
If the evidence does not contain the answer, say exactly: "The corpus does not
answer this." Do not use outside knowledge. Be concise."""

CRITIC_SYSTEM = """\
You are an adversarial reviewer. Given numbered evidence excerpts and a drafted
answer, decide whether EVERY factual claim in the answer is directly supported
by the cited evidence. Respond with JSON only:
{"grounded": true|false, "unsupported_claims": ["..."]}
An answer of "The corpus does not answer this." counts as grounded."""


class AgentState(TypedDict, total=False):
    question: str
    hits: list[Hit]
    answer: str
    critique: list[str]
    grounded: bool
    attempts: int


def _evidence_block(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i + 1}] ({h.source}#{h.ordinal})\n{h.text}" for i, h in enumerate(hits))


def _parse_verdict(raw: str) -> tuple[bool, list[str]]:
    """Parse the Critic's JSON, tolerating code fences / surrounding prose.
    Unparseable verdicts fail CLOSED (ungrounded) — never silently pass."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return False, ["critic returned no parseable verdict"]
    try:
        data = json.loads(match.group(0))
        return bool(data.get("grounded")), [str(c) for c in data.get("unsupported_claims", [])]
    except (json.JSONDecodeError, AttributeError, TypeError):
        return False, ["critic verdict was not valid JSON"]


def build_graph(retriever: Retriever, gateway: Gateway, usage: Usage, max_retries: int = 1):
    def researcher(state: AgentState) -> AgentState:
        return {"hits": retriever.retrieve(state["question"], usage=usage)}

    def analyst(state: AgentState) -> AgentState:
        prompt = f"Evidence:\n{_evidence_block(state['hits'])}\n\nQuestion: {state['question']}"
        if state.get("critique"):
            prompt += (
                "\n\nA reviewer rejected your previous draft for these unsupported claims:\n- "
                + "\n- ".join(state["critique"])
                + "\nRewrite the answer using only claims the evidence supports."
            )
        answer = gateway.chat(ANALYST_SYSTEM, prompt, usage=usage)
        return {"answer": answer, "attempts": state.get("attempts", 0) + 1}

    def critic(state: AgentState) -> AgentState:
        prompt = (
            f"Evidence:\n{_evidence_block(state['hits'])}\n\n"
            f"Drafted answer:\n{state['answer']}"
        )
        grounded, claims = _parse_verdict(gateway.chat(CRITIC_SYSTEM, prompt, usage=usage))
        return {"grounded": grounded, "critique": claims}

    def after_critic(state: AgentState) -> Literal["analyst", "__end__"]:
        if not state.get("grounded") and state.get("attempts", 0) <= max_retries:
            return "analyst"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("researcher", researcher)
    graph.add_node("analyst", analyst)
    graph.add_node("critic", critic)
    graph.set_entry_point("researcher")
    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst", "critic")
    graph.add_conditional_edges("critic", after_critic)
    return graph.compile()


def ask(question: str, retriever: Retriever, gateway: Gateway, max_retries: int = 1) -> dict:
    """Run the graph for one question. Returns answer + evidence + usage."""
    usage = Usage()
    app = build_graph(retriever, gateway, usage, max_retries=max_retries)
    final: AgentState = app.invoke({"question": question, "attempts": 0})
    return {
        "question": question,
        "answer": final.get("answer", ""),
        "grounded": bool(final.get("grounded")),
        "attempts": final.get("attempts", 0),
        "sources": [
            {"source": h.source, "ordinal": h.ordinal, "score": round(h.score, 5)}
            for h in final.get("hits", [])
        ],
        "usage": usage.as_dict(),
    }
