import pytest
from fastapi.testclient import TestClient

from askdesk.api import Runtime, app, get_runtime
from askdesk.config import get_settings


@pytest.fixture
def client(retriever, gateway):
    app.dependency_overrides[get_runtime] = lambda: Runtime(
        retriever=retriever, gateway=gateway, max_critic_retries=1
    )
    yield TestClient(app)
    app.dependency_overrides.clear()


AUTH = {"Authorization": f"Bearer {get_settings().api_key}"}


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_ask_requires_bearer_token(client):
    assert client.post("/ask", json={"question": "anything here"}).status_code == 401


def test_ask_returns_full_contract(client):
    resp = client.post(
        "/ask", json={"question": "Where do service secrets live?"}, headers=AUTH
    )
    assert resp.status_code == 200
    body = resp.json()
    assert {"question", "answer", "grounded", "attempts", "sources", "usage"} <= body.keys()
    assert any(s["source"] == "security-policy.md" for s in body["sources"])
    assert body["usage"]["llm_calls"] >= 2


def test_ask_validates_input(client):
    assert client.post("/ask", json={"question": "hi"}, headers=AUTH).status_code == 422
