import logging
import os

from inspect_ai.model import ChatMessageAssistant
from inspect_ai.solver import Generate, Solver, TaskState, chain, solver

from astabench.evals.sqa.solvers.format_solver import format_solver
from astabench.solvers.perplexity import query_perplexity

logger = logging.getLogger(__name__)


@solver
def perplexity_solver(
    system_prompt: str | None = None,
    model: str | None = "sonar",
    temperature: float = 0.0,
) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        prompt = state.metadata["prompt_without_formatting_instructions"]
        try:
            response, references = await query_perplexity(
                model,
                prompt,
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


@solver
def formatted_solver(
    system_prompt: str | None = None,
    model: str | None = "sonar",
    search_context_size: str | None = None,
    temperature: float = 0.0,
) -> Solver:
    chainlist = [
        perplexity_solver(
            model=model,
            temperature=temperature,
            system_prompt=system_prompt,
            search_context_size=search_context_size,
        ),
        format_solver("gpt-4.1"),
    ]
    return chain(chainlist)
