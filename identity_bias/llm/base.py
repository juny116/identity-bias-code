"""Base LLM interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class TokenLogprob:
    """Log probability information for a single token."""
    token: str
    logprob: float


@dataclass
class LLMResponse:
    """Response from an LLM including optional logprob info."""
    text: str
    logprobs: list[list[TokenLogprob]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class BaseLLM(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def generate(self, messages: list[dict[str, str]], **kwargs) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            **kwargs: Provider-specific overrides (temperature, max_tokens, etc.)

        Returns:
            LLMResponse with text and optional logprobs.
        """
        ...
