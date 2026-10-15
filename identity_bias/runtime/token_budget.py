"""Lazy token estimation for structured judge requests."""

from functools import lru_cache


@lru_cache(maxsize=1)
def encoder():
    import tiktoken

    try:
        return tiktoken.encoding_for_model("gpt-4o")
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def available_tokens(messages, context_window=32768, margin=128):
    prompt_tokens = int(
        sum(len(encoder().encode(m["content"])) + 4 for m in messages) * 1.25
    )
    return context_window - prompt_tokens - margin
