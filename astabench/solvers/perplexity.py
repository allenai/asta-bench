import logging
import os

import aiohttp
from inspect_ai.model import ChatMessageAssistant
from inspect_ai.solver import Generate, Solver, TaskState, solver

logger = logging.getLogger(__name__)


async def query_perplexity(
    model,
    question: str,
    api_key: str,
    temperature: float,
    system_prompt: str | None = None,
    search_context_size: str | None = None,
):
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": question},
        ],
        "temperature": temperature,
    }
    if search_context_size:
        payload["web_search_options"] = {"search_context_size": search_context_size}
    if system_prompt:
        payload["messages"].insert(0, {"role": "system", "content": system_prompt})

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.perplexity.ai/chat/completions", headers=headers, json=payload
        ) as response:
            response.raise_for_status()
            # Extract raw content and clean JSON
            res = await response.json()
    references = res["citations"] if res.get("citations") else []
    return res["choices"][0]["message"]["content"], references


@solver
def perplexity_solver(
    system_prompt: (
        str | None
    ) = "You are a precise scientific research assistant. Please conduct as much research as you can and answer the following scientific question thoroughly.",
    model: str | None = "sonar",
    temperature: float = 0.0,
    search_context_size: str | None = None,
) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        last_message = state.messages[-1].text
        try:
            response, references = await query_perplexity(
                model,
                last_message,
                os.environ["PERPLEXITY_API_KEY"],
                temperature,
                system_prompt=system_prompt,
            )
        except Exception as e:
            logger.exception(f"Error: {e}")
            return state
        state.messages.append(ChatMessageAssistant(content=response))
        state.output.completion = response
        return state

    return solve
