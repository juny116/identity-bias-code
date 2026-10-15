from typing import Any

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from .api_model import APIModel


class GoogleGemini(APIModel):
    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        api_key: str = None,
        max_retry: int = 5,
        timeout: int = 300,
        base_url: str = "https://api.openai.com/v1",
        organization: str = None,
        project: str = None,
    ):
        self.model = model
        self.api_key = api_key
        self.max_retry = max_retry
        self.timeout = timeout
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(model_name=self.model)
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.generate(*args, **kwds)

    def generate(
        self,
        messages: list,
        max_tokens: int = 100,
        temperature: float = 1,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: list = None,
        n: int = 1,
    ):
        if frequency_penalty != 0.0 or presence_penalty != 0.0:
            raise ValueError(
                "Google Gemini does not support frequency_penalty or presence_penalty"
            )
        gen_config = genai.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            candidate_count=n,
            stop_sequences=stop,
        )
        converted_messages = self._convert_message(messages)
        # catch error and retry
        for i in range(self.max_retry):
            try:
                response = self.client.generate_content(
                    contents=converted_messages,
                    generation_config=gen_config,
                    safety_settings=self.safety_settings,
                )
                return self._return_format(messages, response)
                return response.text
            except Exception as e:
                if i == self.max_retry - 1:
                    raise e

    async def agenerate(
        self,
        messages: list,
        max_tokens: int = 100,
        temperature: float = 1,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: list = None,
        n: int = 1,
    ):
        if frequency_penalty != 0.0 or presence_penalty != 0.0:
            raise ValueError(
                "Google Gemini does not support frequency_penalty or presence_penalty"
            )
        gen_config = genai.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            candidate_count=n,
            stop_sequences=stop,
        )
        messages = self._convert_message(messages)
        for i in range(self.max_retry):
            try:
                response = await self.client.generate_content_async(
                    contents=messages,
                    generation_config=gen_config,
                    safety_settings=self.safety_settings,
                )
                return response.text
            except Exception as e:
                if i == self.max_retry - 1:
                    raise e

    def _convert_message(self, message: str) -> str:
        converted_message = []
        # system -> model / assistant -> model
        role_dict = {"user": "user", "system": "model", "assistant": "model"}
        for i, m in enumerate(message):
            converted_message.append(
                {"role": role_dict.get(m["role"]), "parts": m["content"]}
            )
        return converted_message

    def _return_format(self, messages: list, response: dict) -> dict:
        # only consider case where n is 1
        messages.append({"role": "assistant", "content": response.text})
        response = {
            "messages": messages,
            "tokens": {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
            },
            "error": None,
        }
        return response

    def get_token_length(self, message: str) -> int:
        message = self._convert_message(message)
        return self.client.count_tokens(message)
