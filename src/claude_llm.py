"""LlamaIndex LLM that answers with Claude through the official Anthropic SDK."""
from typing import Any, Iterator

import anthropic
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.llms import (
    CompletionResponse,
    CompletionResponseGen,
    CustomLLM,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback

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
        )

    def _stream(self, prompt: str):
        return self._client.beta.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            output_config={"effort": self.effort},
            betas=[FALLBACK_BETA],
            fallbacks="default",
            messages=[{"role": "user", "content": prompt}],
        )

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        with self._stream(prompt) as stream:
            message = stream.get_final_message()
        if message.stop_reason == "refusal":
            return CompletionResponse(text=REFUSAL_NOTE.strip(), raw=message)
        text = "".join(block.text for block in message.content if block.type == "text")
        return CompletionResponse(text=text, raw=message)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        def gen() -> Iterator[CompletionResponse]:
            text = ""
            with self._stream(prompt) as stream:
                for delta in stream.text_stream:
                    text += delta
                    yield CompletionResponse(text=text, delta=delta)
                message = stream.get_final_message()
            # Checked after streaming: a refusal can arrive before any text or
            # part-way through, and the fallback model may also decline.
            if message.stop_reason == "refusal":
                yield CompletionResponse(text=text + REFUSAL_NOTE, delta=REFUSAL_NOTE)

        return gen()
