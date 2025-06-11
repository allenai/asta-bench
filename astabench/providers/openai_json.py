from inspect_ai.model._providers.openai import OpenAIAPI


class OpenAIJSONAPI(OpenAIAPI):
    """
    An OpenAI model provider that returns completions as a json object.
    Inherits from OpenAIAPI and injects response_format into the request.
    """

    def completion_params(self, config, tools):
        params = super().completion_params(config, tools)
        params["response_format"] = {"type": "json_object"}
        return params
