"""Anthropic LLM backend."""

from anthropic import Anthropic

from identity_bias.config import LLMConfig
from identity_bias.llm.base import BaseLLM, LLMResponse


class AnthropicLLM(BaseLLM):
    """Anthropic API backend (Claude Haiku, Sonnet, etc.)."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = Anthropic(api_key=config.api_key)

    def generate(self, messages: list[dict[str, str]], **kwargs) -> LLMResponse:
        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

        # Anthropic expects system message separately
        system_msg = None
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                chat_messages.append(msg)

        create_kwargs = dict(
            model=self.config.model_name,
            messages=chat_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if system_msg:
            create_kwargs["system"] = system_msg

        response = self.client.messages.create(**create_kwargs)

        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text

        return LLMResponse(
            text=text,
            logprobs=[],  # Anthropic doesn't provide logprobs
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
        )
