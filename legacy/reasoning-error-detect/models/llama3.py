from typing import Any
from openai import OpenAI, AsyncOpenAI
from .api_model import APIModel


class VLLMModel(APIModel):
    def __init__(
        self,
        model: str = "gpt-4",
        api_key: str = None,
        max_retry: int = 5,
        timeout: int = 120,
        base_url: str = "https://api.openai.com/v1",
    ):
        self.model = model
        self.api_key = api_key
        self.max_retry = max_retry
        self.timeout = timeout
        self.base_url = base_url
        self.client = OpenAI(
            api_key=api_key,
            max_retries=self.max_retry,
            timeout=self.timeout,
            base_url=self.base_url,
        )
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            max_retries=self.max_retry,
            timeout=self.timeout,
            base_url=self.base_url,
        )

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
        messages = self._convert_message(messages)
        # catch error and retry
        for i in range(self.max_retry):
            try:
                response = self.client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=stop,
                    n=n,
                )
                return self._return_format(messages, response)
            except Exception as e:
                # TODO: log error
                if i == self.max_retry - 1:
                    # return error message
                    return {"error": str(e)}

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
        messages = self._convert_message(messages)
        # TODO: catch error and retry
        response = await self.async_client.chat.completions.create(
            messages=messages,
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            n=n,
        )
        return self._return_format(messages, response)

    def _convert_message(self, message: str) -> str:
        return message

    def _return_format(self, messages: list, response: dict) -> dict:
        # only consider case where n is 1
        messages.append(
            {"role": "assistant", "content": response.choices[0].message.content}
        )
        response = {
            "messages": messages,
            "tokens": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
            "error": None,
        }
        return response
