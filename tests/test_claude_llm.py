"""ClaudeLLM against a fake Anthropic API: request shape, streaming, refusals, chat."""
import json

import anthropic
import httpx2
import pytest
from anthropic import DefaultHttpxClient
from llama_index.core import Document, MockEmbedding, VectorStoreIndex

from src import pipeline
from src.claude_llm import FALLBACK_BETA, REFUSAL_NOTE, ClaudeLLM


def _sse(chunks, stop_reason):
    events = [{"type": "message_start", "message": {
        "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
        "content": [], "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 1}}}]
    if chunks:
        events.append({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
        events += [{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": c}} for c in chunks]
        events.append({"type": "content_block_stop", "index": 0})
    events += [
        {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None}, "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
    ]
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


class FakeAnthropic:
    """Serves a canned streaming response and records every request."""

    def __init__(self):
        self.requests = []
        self.chunks = ["Hello"]
        self.stop_reason = "end_turn"

    def handler(self, request):
        self.requests.append(request)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"},
                               text=_sse(self.chunks, self.stop_reason))

    @property
    def last_body(self):
        return json.loads(self.requests[-1].content)


@pytest.fixture
def fake_api():
    return FakeAnthropic()


@pytest.fixture
def llm(fake_api):
    llm = ClaudeLLM(api_key="test-key")
    llm._client = anthropic.Anthropic(
        api_key="test-key",
        http_client=DefaultHttpxClient(transport=httpx2.MockTransport(fake_api.handler)),
    )
    return llm


def test_request_uses_low_effort_and_fallback_without_temperature(llm, fake_api):
    assert llm.complete("Hi").text == "Hello"
    body = fake_api.last_body
    assert body["model"] == "claude-opus-5-5"
    assert body["output_config"] == {"effort": "low"}
    assert body["fallbacks"] == "default"
    assert "temperature" not in body
    assert FALLBACK_BETA in fake_api.requests[-1].headers["anthropic-beta"]


def test_stream_complete_yields_text_as_it_arrives(llm, fake_api):
    fake_api.chunks = ["Docu", "Mind"]
    assert [r.delta for r in llm.stream_complete("Hi")] == ["Docu", "Mind"]


def test_refusal_is_flagged_instead_of_returned_as_an_answer(llm, fake_api):
    fake_api.chunks, fake_api.stop_reason = [], "refusal"
    assert [r.delta for r in llm.stream_complete("Hi")] == [REFUSAL_NOTE]
    assert llm.complete("Hi").text == REFUSAL_NOTE.strip()


def test_chat_keeps_roles_and_sends_context_as_system(llm, fake_api):
    index = VectorStoreIndex.from_documents(
        [Document(text="Questions about the meal allowance go to finance@lumenfield.example.")],
        embed_model=MockEmbedding(embed_dim=8),
    )
    history = [
        {"role": "user", "content": "What is the meal allowance?"},
        {"role": "assistant", "content": "$75 per day."},
    ]
    response = pipeline.stream_answer(index, llm, "Who do I ask about it?", history)
    assert "".join(response.response_gen) == "Hello"

    body = fake_api.last_body
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    assert body["messages"][-1]["content"] == "Who do I ask about it?"
    assert "finance@lumenfield.example" in body["system"]
