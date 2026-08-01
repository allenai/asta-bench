import os

os.environ.setdefault("OPENAI_API_KEY", "test")

import astabench.evals._registry  # noqa: E402, F401
from inspect_ai.scorer import SampleScore, Score  # noqa: E402

from astabench.evals.paper_finder.task import (  # noqa: E402
    score_paper_finder_with_all_name,
)


def test_litqa2_primary_metric_does_not_collide_with_aggregate_label():
    scorer = score_paper_finder_with_all_name("recall_at_30")()
    grouped_mean = scorer.__registry_info__.metadata["metrics"][0]

    result = grouped_mean(
        [
            SampleScore(
                score=Score(value=1.0),
                sample_id="litqa2_1",
                sample_metadata={"score_type": "recall_at_30"},
            )
        ]
    )

    assert result == {"recall_at_30": 1.0}
