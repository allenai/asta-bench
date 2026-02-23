from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

from inspect_ai.model import ChatMessageUser, ModelName, ModelOutput
from inspect_ai.scorer import CORRECT, INCORRECT, Target
from inspect_ai.solver import TaskState


def _load_arithmetic_task_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "ci" / "arithmetic_task.py"
    spec = importlib.util.spec_from_file_location("arithmetic_task", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


arithmetic_task = _load_arithmetic_task_module()


def _make_state(completion: str) -> TaskState:
    model_name = ModelName("mockllm/model")
    return TaskState(
        model=model_name,
        sample_id="1",
        epoch=1,
        input="Compute 4.6 + 2.1*2",
        messages=[ChatMessageUser(content="Compute 4.6 + 2.1*2")],
        target=Target(target="8.8"),
        choices=None,
        output=ModelOutput.from_content(model=str(model_name), content=completion),
        metadata={},
    )


def test_check_arithmetic_scores_correct_answer() -> None:
    score_fn = arithmetic_task.check_arithmetic()
    state = _make_state('{"answer": 8.8}')

    score = asyncio.run(score_fn(state, Target(target="8.8")))
    assert score.value == CORRECT


def test_check_arithmetic_scores_wrong_answer() -> None:
    score_fn = arithmetic_task.check_arithmetic()
    state = _make_state('{"answer": 5}')

    score = asyncio.run(score_fn(state, Target(target="8.8")))
    assert score.value == INCORRECT


def test_check_arithmetic_scores_malformed_output_as_incorrect() -> None:
    score_fn = arithmetic_task.check_arithmetic()
    state = _make_state("not-json")

    score = asyncio.run(score_fn(state, Target(target="8.8")))
    assert score.value == INCORRECT


def test_check_arithmetic_parses_negative_answers() -> None:
    score_fn = arithmetic_task.check_arithmetic()
    state = _make_state('{"answer": -3}')

    score = asyncio.run(score_fn(state, Target(target="-3")))
    assert score.value == CORRECT
