"""Code-level dataset, authorship and model interface types."""

from dataclasses import dataclass
from enum import Enum


class IdentityCondition(Enum):
    SELF = "self"
    OTHER_MODEL = "other_model"
    WEAK_MODEL = "weak_model"
    STRONG_MODEL = "strong_model"
    ANONYMOUS = "anonymous"
    HUMAN = "human"


class ContextCondition(Enum):
    SAME_SESSION = "same_session"
    NEW_SESSION = "new_session"
    PARAPHRASED = "paraphrased"


class Dataset(Enum):
    MATH = "math"
    GPQA = "gpqa"
    BBH = "bbh"
    AIME = "aime"
    MMLU_PRO = "mmlu_pro"
    OLYMPIAD = "olympiad"
    MINERVA = "minerva"


class LLMProvider(Enum):
    OPENAI = "openai"
    OPENAI_RESPONSES = "openai_responses"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    VLLM = "vllm"


@dataclass
class LLMConfig:
    provider: LLMProvider
    model_name: str
    max_tokens: int
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    top_logprobs: int = 5
    top_p: float | None = None
    top_k: int | None = None
    enable_thinking: bool = True
    reasoning_effort: str | None = None
