from .openai_gpt import OpenAIGPT
from .google_gemini import GoogleGemini


def AutoModelForAPI(model_type: str, **kwds):
    if model_type == "openai":
        return OpenAIGPT(**kwds)
    elif model_type == "google":
        return GoogleGemini(**kwds)
