from types import SimpleNamespace

import pytest

from astabench.evals.sqa.retry_utils import generate_with_retry
from astabench.evals.sqa.rubric import RubricCorpusQaGenericMetric


class FakeModel:
    def __init__(self, completions):
        self._completions = iter(completions)

    async def generate(self, prompt_or_messages, config):
        return SimpleNamespace(completion=next(self._completions))


@pytest.mark.asyncio
async def test_generate_with_retry_retries_on_parsed_validator_failure():
    model = FakeModel(
        [
            '{"scores": [{"criteria_idx": 1}]}',
            '{"scores": [{"criteria_idx": 1}, {"criteria_idx": 2}]}',
        ]
    )

    def validate(parsed):
        indices = [score["criteria_idx"] for score in parsed["scores"]]
        if indices != [1, 2]:
            raise ValueError(f"incomplete criteria indices: {indices}")

    _, parsed, num_retries = await generate_with_retry(
        model=model,
        prompt_or_messages=[],
        config=SimpleNamespace(),
        max_retries=1,
        base_delay=0,
        parsed_validator=validate,
    )

    assert [score["criteria_idx"] for score in parsed["scores"]] == [1, 2]
    assert num_retries == 1


@pytest.mark.asyncio
async def test_joint_rubric_assessment_retries_until_all_criteria_are_scored():
    model = FakeModel(
        [
            """{
                "scores": [
                    {
                        "criteria": "criterion 1",
                        "criteria_idx": 1,
                        "reasoning": "partial response",
                        "score": 2,
                        "evidence": "evidence 1"
                    }
                ]
            }""",
            """{
                "scores": [
                    {
                        "criteria": "criterion 1",
                        "criteria_idx": 1,
                        "reasoning": "covers criterion 1",
                        "score": 2,
                        "evidence": "evidence 1"
                    },
                    {
                        "criteria": "criterion 2",
                        "criteria_idx": 2,
                        "reasoning": "covers criterion 2",
                        "score": 1,
                        "evidence": "evidence 2"
                    }
                ]
            }""",
        ]
    )
    metric = RubricCorpusQaGenericMetric(
        config={
            "question": "Test question",
            "ingredients": [
                {
                    "name": "criterion_a",
                    "criterion": "Assess criterion A",
                    "weight": 0.5,
                    "examples": ["example A"],
                },
                {
                    "name": "criterion_b",
                    "criterion": "Assess criterion B",
                    "weight": 0.5,
                    "examples": ["example B"],
                },
            ],
        },
        model=model,
    )

    score_components, prompt_logs = await metric._assess_properties_jointly(
        "candidate response", metric.config.ingredients
    )

    assert score_components == {"criterion_a": 1.0, "criterion_b": 0.5}
    assert prompt_logs["num_retries"][0]["data"]["num_retries"] == 1
