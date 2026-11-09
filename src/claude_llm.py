"""LlamaIndex LLM that answers with Claude through the official Anthropic SDK."""
from typing import Any, Iterator, Sequence

import anthropic
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.llms import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    CustomLLM,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback

# If Claude's safety classifiers decline a request, retry it server-side on the
# model Anthropic recommends for that refusal category.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
REFUSAL_NOTE = "\n\n_Claude declined to answer this question, so any text above is incomplete._"


class ClaudeLLM(CustomLLM):
    model: str = "claude-opus-5-5"
    # Low effort keeps thinking short, which matters most for response time.
    # Answering from a few retrieved chunks rarely benefits from more.
    effort: str = "low"
    max_tokens: int = 64000
    context_window: int = 1_000_000

    _client: anthropic.Anthropic = PrivateAttr()

    def __init__(self, api_key: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client = anthropic.Anthropic(api_key=api_key)

    @classmethod
    def class_name(cls) -> str:
        return "ClaudeLLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens,
            model_name=self.model,
            is_chat_model=True,
        )

    def _stream(self, messages: Sequence[ChatMessage]):
        # System messages (instructions plus retrieved context) go in `system`;
        # the conversation goes in `messages`.
        system = "\n\n".join(
            m.content for m in messages if m.role == MessageRole.SYSTEM and m.content
        )
        turns = [
            {"role": "assistant" if m.role == MessageRole.ASSISTANT else "user", "content": m.content}
            for m in messages
            if m.role != MessageRole.SYSTEM and m.content
        ]
        extra = {"system": system} if system else {}
        return self._client.beta.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            output_config={"effort": self.effort},
            betas=[FALLBACK_BETA],
            fallbacks="default",
            messages=turns,
            **extra,
        )

    def _chat(self, messages: Sequence[ChatMessage]) -> ChatResponse:
        with self._stream(messages) as stream:
            message = stream.get_final_message()
        if message.stop_reason == "refusal":
            text = REFUSAL_NOTE.strip()
        else:
            text = "".join(block.text for block in message.content if block.type == "text")
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=text), raw=message)

    def _stream_chat(self, messages: Sequence[ChatMessage]) -> Iterator[ChatResponse]:
        text = ""
        with self._stream(messages) as stream:
            for delta in stream.text_stream:
                text += delta
                yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=text), delta=delta)
            message = stream.get_final_message()
        # Checked after streaming: a refusal can arrive before any text or
        # part-way through, and the fallback model may also decline.
        if message.stop_reason == "refusal":
            text += REFUSAL_NOTE
            yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=text), delta=REFUSAL_NOTE)

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return self._chat(messages)

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        return self._stream_chat(messages)

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        response = self._chat([ChatMessage(role=MessageRole.USER, content=prompt)])
        return CompletionResponse(text=response.message.content, raw=response.raw)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        def gen() -> Iterator[CompletionResponse]:
            for response in self._stream_chat([ChatMessage(role=MessageRole.USER, content=prompt)]):
                yield CompletionResponse(text=response.message.content, delta=response.delta)

        return gen()
