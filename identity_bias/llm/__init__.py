"""LLM abstraction layer with multi-provider support."""

from identity_bias.config import LLMConfig, LLMProvider
from identity_bias.llm.base import BaseLLM, LLMResponse, TokenLogprob


def create_llm(config: LLMConfig) -> BaseLLM:
    """Factory function to create an LLM backend from config."""
    if config.provider == LLMProvider.OPENAI:
        from identity_bias.llm.openai_llm import OpenAILLM
        return OpenAILLM(config)
    elif config.provider == LLMProvider.ANTHROPIC:
        from identity_bias.llm.anthropic_llm import AnthropicLLM
        return AnthropicLLM(config)
    elif config.provider == LLMProvider.GOOGLE:
        from identity_bias.llm.google_llm import GoogleLLM
        return GoogleLLM(config)
    elif config.provider == LLMProvider.OPENAI_RESPONSES:
        from identity_bias.llm.openai_responses_llm import OpenAIResponsesLLM
        return OpenAIResponsesLLM(config)
    elif config.provider == LLMProvider.VLLM:
        from identity_bias.llm.vllm_llm import VLLMLlm
        return VLLMLlm(config)
    else:
        raise ValueError(f"Unknown LLM provider: {config.provider}")


__all__ = ["BaseLLM", "LLMResponse", "TokenLogprob", "create_llm"]
