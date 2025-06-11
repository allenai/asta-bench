import logging

from inspect_ai.scorer import (
    Score,
    Scorer,
    Target,
    accuracy,
    model_graded_qa,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState
from pydantic import BaseModel

LOGGER = logging.getLogger(__name__)


class LlmOutputConstraintConfig(BaseModel):
    model_name: str | None = None
    constraint: str


@scorer(metrics=[accuracy(), stderr()])
def llm_output_constraint() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        config = LlmOutputConstraintConfig.model_validate_json(target.text)
        inspect_scorer = model_graded_qa(
            model=config.model_name,
        )
        actual_target = Target(target=config.constraint)
        return await inspect_scorer(state, actual_target)

    return score
