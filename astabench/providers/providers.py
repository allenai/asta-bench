from inspect_ai.model._model import ModelAPI
from inspect_ai.model._providers.providers import validate_openai_client
from inspect_ai.model._registry import modelapi


@modelapi(name="openai_json")
def openai_json() -> type[ModelAPI]:
    # validate
    validate_openai_client("OpenAI API")
    # in the clear
    from .openai_json import OpenAIJSONAPI

    return OpenAIJSONAPI
