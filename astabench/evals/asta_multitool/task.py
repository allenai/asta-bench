"""Multitool-challenge task from the original asta-eval."""

import json
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.scorer import (
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
    stderr,
    value_to_float,
)
from inspect_ai.solver import TaskState
from pydantic import BaseModel, Field

from astabench.evals.asta_multitool.json_score import json_close_values, json_recall
from astabench.evals.asta_multitool.llm_constraint import llm_output_constraint
from astabench.evals.utils import not_implemented_solver


class AstaEvalCase(BaseModel):
    case_id: str
    initial_prompt: str = Field(
        description="The initial prompt to send to the agent to start the conversation."
    )
    output_metric_configs: list[dict[str, Any]] = Field(default_factory=list)


def json_to_sample(obj: dict) -> Sample:
    instance = AstaEvalCase.model_validate(obj)

    return Sample(
        id=instance.case_id,
        input=instance.initial_prompt,
        target=json.dumps(instance.output_metric_configs),
    )


@scorer(metrics=[accuracy(), stderr()])
def score_astaeval() -> Scorer:
    """Score using the sample output config with metrics defined as in
    asta-eval.  Note: llm metrics may be slightly different due to using
    Inspect model grading instead of the LangChain evaluator."""

    async def score(state: TaskState, target: Target) -> Score:
        output_metric_configs = json.loads(target.text)
        subscores = []
        for output_metric_config in output_metric_configs:
            if output_metric_config["name"] == "llm_output_constraint":
                subscores.append(
                    await llm_output_constraint()(
                        state, Target(target=json.dumps(output_metric_config["config"]))
                    )
                )
            elif output_metric_config["name"] == "json_recall":
                subscores.append(
                    await json_recall()(
                        state, Target(target=json.dumps(output_metric_config["config"]))
                    )
                )
            elif output_metric_config["name"] == "json_close_values":
                subscores.append(
                    await json_close_values()(
                        state, Target(target=json.dumps(output_metric_config["config"]))
                    )
                )
            else:
                raise ValueError(
                    f"Unknown output metric: {output_metric_config['name']}"
                )

        # Need to convert any Inspect score symbols (e.g. 'C', 'I') to floats
        v2f = value_to_float()
        return Score(
            value=sum(v2f(subscore.value) for subscore in subscores) / len(subscores),
            answer=state.output.completion,
            explanation=json.dumps(
                [subscore.explanation for subscore in subscores], indent=2
            ),
        )

    return score


@task
def asta_multitool_challenge(seed: int = 0):
    """Task definition for the multitool-challenge from the original asta-eval."""
    dataset = json_dataset(
        str(Path(__file__).parent / "data.json"),
        sample_fields=json_to_sample,
    )

    return Task(
        dataset=dataset,
        solver=not_implemented_solver(),
        scorer=score_astaeval(),
    )
