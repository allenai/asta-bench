import json
import logging
from typing import Any, Iterable

from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState
from pydantic import BaseModel

LOGGER = logging.getLogger(__name__)


def extract_json_from_response(response: str) -> dict[str, Any] | None:
    json_start = response.find("{")
    json_end = response.rfind("}") + 1
    if json_start == -1 or json_end == -1:
        return None

    try:
        return json.loads(response[json_start:json_end])
    except json.JSONDecodeError:
        LOGGER.warning(
            f"Could not decode JSON from response: {response[json_start:json_end]}"
        )
        return None


class JsonRecallConfig(BaseModel):
    list_key: str
    expected_values: Iterable[Any]
    allow_extra_keys: bool = False
    ignore_case: bool = False


@scorer(metrics=[accuracy(), stderr()])
def json_recall() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        config = JsonRecallConfig.model_validate_json(target.text)
        response = state.output.completion
        extracted_json = extract_json_from_response(response)
        if not extracted_json:
            return Score(
                value=0.0, answer="", explanation="No JSON extracted from response"
            )

        extracted_values = set(extracted_json[config.list_key])
        expected_values = set(config.expected_values)

        if config.ignore_case:
            extracted_values = {v.lower() for v in extracted_values}
            expected_values = {v.lower() for v in expected_values}

        if (not config.allow_extra_keys) and len(
            extracted_values - expected_values
        ) > 0:
            return Score(
                value=0.0,
                answer=json.dumps(sorted(extracted_values)),
                explanation="Extra keys found and allow_extra_keys is False",
            )

        recall = len(extracted_values & expected_values) / len(expected_values)

        return Score(
            value=recall,
            answer=json.dumps(sorted(extracted_values)),
            explanation=f"Recall: {recall}",
        )

    return score


class CloseValueConfig(BaseModel):
    value: Any
    tolerance: float | None = None


class JsonCloseValuesConfig(BaseModel):
    expected_values: dict[str, CloseValueConfig]


@scorer(metrics=[accuracy(), stderr()])
def json_close_values() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        config = JsonCloseValuesConfig.model_validate_json(target.text)
        response = state.output.completion
        extracted_json = extract_json_from_response(response)
        if not extracted_json:
            return Score(
                value=0.0, answer="", explanation="No JSON extracted from response"
            )

        n_vals = len(config.expected_values)
        n_matched = 0

        for key, expected in config.expected_values.items():
            if extracted_json.get(key) is None:
                if key in extracted_json and expected.value is None:
                    n_matched += 1
                continue
            if expected.tolerance is None:
                if extracted_json[key] == expected.value:
                    n_matched += 1
            else:
                if not isinstance(expected.value, (int, float)):
                    raise ValueError(
                        f"Expected value for key {key} is not a number: {extracted_json[key]}"
                    )
                if abs(extracted_json[key] - expected.value) <= expected.tolerance:
                    n_matched += 1

        return Score(
            value=n_matched / n_vals,
            answer=json.dumps(extracted_json),
            explanation=f"Matched {n_matched} out of {n_vals} values",
        )

    return score
