"""vLLM backend via OpenAI-compatible API."""

import tiktoken

from openai import OpenAI

from identity_bias.config import LLMConfig
from identity_bias.llm.base import BaseLLM, LLMResponse, TokenLogprob


class VLLMLlm(BaseLLM):
    """vLLM backend using OpenAI-compatible API."""

    def __init__(self, config: LLMConfig, max_context: int = 32768):
        self.config = config
        self.max_context = max_context
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=config.base_url or "http://localhost:8000/v1",
            timeout=1800.0,
        )
        try:
            self._enc = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            self._enc = tiktoken.get_encoding("cl100k_base")

    def _estimate_prompt_tokens(self, messages: list[dict[str, str]]) -> int:
        return sum(len(self._enc.encode(m.get("content", ""))) + 4 for m in messages)

    def generate(self, messages: list[dict[str, str]], **kwargs) -> LLMResponse:
        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        top_logprobs = kwargs.get("top_logprobs", self.config.top_logprobs)

        # Clamp max_tokens to fit within context window
        prompt_tokens_est = self._estimate_prompt_tokens(messages)
        available = self.max_context - prompt_tokens_est - 64  # small margin
        if available < 256:
            available = 256
        max_tokens = min(max_tokens, available)

        extra_body = dict(kwargs.get("extra_body", {}))
        if not self.config.enable_thinking:
            extra_body.setdefault("chat_template_kwargs", {})["enable_thinking"] = False

        create_kwargs = dict(
            model=self.config.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=top_logprobs,
        )
        if self.config.top_p is not None:
            create_kwargs["top_p"] = self.config.top_p
        if self.config.top_k is not None:
            extra_body.setdefault("top_k", self.config.top_k)
        if self.config.reasoning_effort is not None:
            extra_body["reasoning_effort"] = self.config.reasoning_effort
        if extra_body:
            create_kwargs["extra_body"] = extra_body

        response = self.client.chat.completions.create(**create_kwargs)

        choice = response.choices[0]
        # gpt-oss thinking mode: content is the answer, reasoning is the thinking
        text = choice.message.content or ""
        # Prepend reasoning as <think> block so strip_thinking() handles it uniformly
        reasoning = getattr(choice.message, "reasoning", None)
        if reasoning:
            text = f"<think>{reasoning}</think>\n{text}"

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
