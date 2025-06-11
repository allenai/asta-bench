import json

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    ResponseSchema,
    get_model,
)
from inspect_ai.solver import Solver, solver
from inspect_ai.util import json_schema

from astabench.types.sqa import SQAResponse


@solver
def format_solver(gpt_version: str, require_snippets=True) -> Solver:
    model = get_model(f"openai/{gpt_version}")
    format_prompt = "The following is an AI generated report which you will format as a JSON object. Return your result as valid JSON with a single key `sections` which is a list of sections, each having keys `title`, `text`, and `citations`. Do not omit the inline citation from `text`. Each entry in `citations` should have {snippets_prompt} an `id`, each of which appears exactly in the text. Each `id` should be an inline citation as it appears in the text (with wrapping parentheses or square brackets if appropriate). Each citation should have a `title` if one is available. Any additional information about the citation should go under `metadata`. Do not modify any of the text in the report. Do not omit any of the body text in the report. Do not create a References section."
    snippets_prompt = (
        (
            "a JSON list of `snippets` extracted from the reference document ONLY if they exist in the report and "
        )
        if require_snippets
        else ""
    )
    format_prompt = format_prompt.format(snippets_prompt=snippets_prompt)

    async def solve(state, generate):
        m = state.messages[-1]
        if not isinstance(m, ChatMessageAssistant):
            raise ValueError("The last message must be from the assistant.")
        message_to_format = state.messages[-1].text
        messages_for_formatter = [
            ChatMessageSystem(content=format_prompt),
            ChatMessageUser(content=message_to_format),
        ]
        formatter_output = await model.generate(
            messages_for_formatter,
            config=GenerateConfig(
                response_schema=ResponseSchema(
                    name="sqa_response",
                    json_schema=json_schema(SQAResponse),
                )
            ),
        )
        formatted = formatter_output.choices[0].message.content
        assert isinstance(formatted, str)
        response = json.loads(formatted)
        if not require_snippets:
            for section in response["sections"]:
                for citation in section["citations"]:
                    citation["snippets"] = []
                    citation["metadata"] = {}

        result = json.dumps(
            {"report": message_to_format, "response": response}, indent=2
        )
        state.messages.append(ChatMessageAssistant(content=result))
        state.output.completion = result
        return state

    return solve
