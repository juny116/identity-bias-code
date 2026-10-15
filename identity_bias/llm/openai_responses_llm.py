"""OpenAI Responses API backend (gpt-5-mini and newer models)."""

from openai import OpenAI

from identity_bias.config import LLMConfig
from identity_bias.llm.base import BaseLLM, LLMResponse


class OpenAIResponsesLLM(BaseLLM):
    """OpenAI Responses API backend for newer models like gpt-5-mini.

    Uses client.responses.create instead of client.chat.completions.create.
    Converts system role to 'developer' role as required by the new API.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = OpenAI(api_key=config.api_key)

    def generate(self, messages: list[dict[str, str]], **kwargs) -> LLMResponse:
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

        # Convert messages: system → developer role, rest pass through
        input_messages = []
        for msg in messages:
            role = "developer" if msg["role"] == "system" else msg["role"]
            input_messages.append({"role": role, "content": msg["content"]})

        response = self.client.responses.create(
            model=self.config.model_name,
            input=input_messages,
            max_output_tokens=max_tokens,
        )

        text = response.output_text or ""

        prompt_tokens = 0
        completion_tokens = 0
        if hasattr(response, "usage") and response.usage:
            prompt_tokens = getattr(response.usage, "input_tokens", 0) or 0
            completion_tokens = getattr(response.usage, "output_tokens", 0) or 0

        return LLMResponse(
            text=text,
            logprobs=[],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
