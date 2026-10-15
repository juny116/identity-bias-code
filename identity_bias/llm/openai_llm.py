"""OpenAI LLM backend."""

from openai import OpenAI

from identity_bias.config import LLMConfig
from identity_bias.llm.base import BaseLLM, LLMResponse, TokenLogprob


class OpenAILLM(BaseLLM):
    """OpenAI API backend (GPT-4o, GPT-4o-mini, etc.)."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = OpenAI(api_key=config.api_key)

    def generate(self, messages: list[dict[str, str]], **kwargs) -> LLMResponse:
        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        top_logprobs = kwargs.get("top_logprobs", self.config.top_logprobs)

        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=top_logprobs,
        )

        choice = response.choices[0]
        text = choice.message.content or ""

        logprobs = []
        if choice.logprobs and choice.logprobs.content:
            for token_info in choice.logprobs.content:
                position_logprobs = []
                if token_info.top_logprobs:
                    for top in token_info.top_logprobs:
                        position_logprobs.append(
                            TokenLogprob(token=top.token, logprob=top.logprob)
                        )
                logprobs.append(position_logprobs)

        usage = response.usage
        return LLMResponse(
            text=text,
            logprobs=logprobs,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )
