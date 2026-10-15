"""Google Gemini LLM backend."""

from google import genai
from google.genai import types

from identity_bias.config import LLMConfig
from identity_bias.llm.base import BaseLLM, LLMResponse


class GoogleLLM(BaseLLM):
    """Google Gemini API backend (Gemini 2.5 Flash, Pro, etc.)."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = genai.Client(api_key=config.api_key)

    def generate(self, messages: list[dict[str, str]], **kwargs) -> LLMResponse:
        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

        # Convert to Gemini format: system instruction + contents
        system_instruction = None
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=msg["content"])],
                ))
            elif msg["role"] == "assistant":
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=msg["content"])],
                ))

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if system_instruction:
            config.system_instruction = system_instruction

        response = self.client.models.generate_content(
            model=self.config.model_name,
            contents=contents,
            config=config,
        )

        text = response.text or ""

        prompt_tokens = 0
        completion_tokens = 0
        if response.usage_metadata:
            prompt_tokens = response.usage_metadata.prompt_token_count or 0
            completion_tokens = response.usage_metadata.candidates_token_count or 0

        return LLMResponse(
            text=text,
            logprobs=[],  # Gemini doesn't expose logprobs in standard API
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
