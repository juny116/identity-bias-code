from abc import ABC, abstractmethod, ABCMeta


class APIModel(metaclass=ABCMeta):
    @abstractmethod
    def generate(
        self,
        messages,
        max_tokens,
        temperature,
        top_p,
        frequency_penalty,
        presence_penalty,
        stop,
        n,
    ):
        pass

    @abstractmethod
    async def agenerate(
        self,
        messages,
        max_tokens,
        temperature,
        top_p,
        frequency_penalty,
        presence_penalty,
        stop,
        n,
    ):
        pass

    @abstractmethod
    def _convert_message(
        self,
        messages,
    ):
        pass
