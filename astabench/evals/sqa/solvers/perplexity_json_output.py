import json
import os
from typing import Dict, List, Optional
import logging

import aiohttp
from inspect_ai.model import ChatMessageAssistant
from inspect_ai.solver import Generate, Solver, TaskState, solver
from pydantic import create_model
from pydantic.fields import PydanticUndefined

from astabench.types.sqa import SQASection, SQAResponse, Citation

logger = logging.getLogger(__name__)


def extract_fields(model):
    fields = model.model_fields
    return {
        k: (v.annotation, PydanticUndefined if v.is_required() else v.default)
        for k, v in fields.items()
    }


async def query_perplexity(
    model,
    question: str,
    api_key: str,
    use_structured_decoding: bool,
    temperature: float,
):
    strict_metadata_citation_fields = extract_fields(Citation)
    strict_metadata_citation_fields["metadata"] = (Optional[Dict[str, str]], None)
    strict_metadata_citation = create_model(
        "StrictMetadataCitation",
        **strict_metadata_citation_fields,
    )
    strict_metadata_section_fields = extract_fields(SQASection)
    strict_metadata_section_fields["citations"] = (List[strict_metadata_citation], ...)  # type: ignore
    strict_metadata_section = create_model(
        "StrictMetadataSection",
        **strict_metadata_section_fields,
    )
    strict_metadata_response_fields = extract_fields(SQAResponse)
    strict_metadata_response_fields["sections"] = (List[strict_metadata_section], ...)  # type: ignore
    strict_metadata_response = create_model(
        "StrictMetadataResponse",
        **strict_metadata_response_fields,
    )

    schema = strict_metadata_response.model_json_schema()

    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Be precise and concise."},
            {
                "role": "user",
                "content": 'Please write a report responding to the following user query. Return your result as valid JSON with a single key `sections` which is a JSON list of sections, each having keys `title`, `text`, and `citations`. Each claim in the text should be supported by a citation. Each entry in `citations` should have a JSON list of `snippets` extracted from the reference document and an `id`, each of which appears exactly in the text. Each `id` should be of the format "[3]". Any additional information about the citation should go under `metadata`.\n\n'
                + question,
            },
        ],
        "temperature": temperature,
    }
    if use_structured_decoding:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"schema": schema},
        }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.perplexity.ai/chat/completions", headers=headers, json=payload
        ) as response:
            response.raise_for_status()
            # Extract raw content and clean JSON
            res = await response.json()
    references = res["citations"] if res.get("citations") else []
    _response = res["choices"][0]["message"]["content"]
    response_dict = json.loads(
        _response[_response.index("{") : _response.rindex("}") + 1]
    )
    return SQAResponse(**response_dict), references


@solver
def perplexity_solver(
    model: str | None = "sonar",
    use_structured_decoding: bool = True,
    temperature: float = 0.0,
) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        question = state.metadata["initial_prompt"]
        response, references = await query_perplexity(
            model,
            question,
            os.environ["PERPLEXITY_API_KEY"],
            use_structured_decoding,
            temperature,
        )
        content = json.dumps({"response": response.model_dump(mode="json")}, indent=2)
        state.messages.append(ChatMessageAssistant(content=content))
        state.output.completion = content
        return state

    return solve
