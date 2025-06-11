import logging
from enum import Enum
from typing import Literal

from astabench.evals.paper_finder.paper_finder_utils import (
    init_references,
    update_references,
)
from astabench.evals.paper_finder.relevance import (
    load_relevance_judgement,
    Document,
    Relevance,
)
from astabench.evals.paper_finder.datamodel import (
    BroadScorerCriteria,
    CorpusId,
    ExpectedAgentOutput,
    SpecificPaperScorerCriteria,
)

LOGGER = logging.getLogger(__name__)

MAX_NORMALIZATION_MULTIPLIER_FACTOR = 10
MIN_NORMALIZATION_MULTIPLIER_FACTOR = 2

# we only want this to happen once during initialization phase
normalizer_reference, detailed_reference = init_references(
    MAX_NORMALIZATION_MULTIPLIER_FACTOR, MIN_NORMALIZATION_MULTIPLIER_FACTOR
)


class KTypes(Enum):
    KNOWN = "known"
    ESTIMATED = "estimated"


async def get_llm_relevance(
    output: ExpectedAgentOutput, metric: BroadScorerCriteria
) -> dict[CorpusId, str]:
    judgements = dict()
    to_judge = []
    qid = output.query_id
    for result in output.results:
        if result.paper_id in metric.known_to_be_good:
            judgements[result.paper_id] = Relevance.PERFECT.value
        elif (
            detailed_reference
            and detailed_reference.get(qid)
            and detailed_reference[qid].get(result.paper_id)
        ):
            judgements[result.paper_id] = detailed_reference[qid][result.paper_id]
        else:
            to_judge.append(
                Document(corpus_id=result.paper_id, markdown=result.markdown_evidence)
            )
    new_judgements = dict()
    if to_judge:
        new_judgements = await load_relevance_judgement(
            to_judge, metric.relevance_criteria
        )
        await update_references(qid, new_judgements)
        judgements.update(new_judgements)

    new_perfects = len(
        [j for j in new_judgements.values() if j == Relevance.PERFECT.value]
    )
    LOGGER.info(f"New judgements: {new_perfects}")

    return judgements


def count_relevant(
    judgements: dict[CorpusId, str], top_k_results: list[str]
) -> tuple[int, int]:
    relevant_predictions_at_full = len(
        [
            paper_id
            for paper_id, judgement in judgements.items()
            if Relevance.PERFECT.value == judgement
        ]
    )

    relevant_predictions_at_k = len(
        [
            paper_id
            for paper_id, judgement in judgements.items()
            if Relevance.PERFECT.value == judgement and paper_id in top_k_results
        ]
    )

    return relevant_predictions_at_k, relevant_predictions_at_full


def get_predefined_judgements(
    output: ExpectedAgentOutput, metric: SpecificPaperScorerCriteria
) -> dict[str, str]:
    return {
        result.paper_id: (
            Relevance.PERFECT
            if result.paper_id in metric.corpus_ids
            else Relevance.NOT_RELEVANT
        )
        for result in output.results
    }


async def calc_recall_at_k(
    output: ExpectedAgentOutput,
    query_id: str,
    judgements: dict[CorpusId, str],
    k_type: KTypes,
) -> dict[str, float]:
    k = normalizer_reference.get(query_id)

    # NOTE: the judge might have failures on its own fault - so no reason to punish the caller. that's why we:
    #   (1) judge all and then take top k.
    #   (2) treat llm_judgements and not output.results as the full set

    top_k_results = [
        result.paper_id
        for result in output.results
        if result.paper_id in judgements.keys()
    ][:k]
    relevant_predictions_at_k, relevant_predictions_at_full = count_relevant(
        judgements, top_k_results
    )

    return {
        f"recall_at_{k_type.value}_k": relevant_predictions_at_k / k,
        f"relevant_predictions_at_{k_type.value}_full": relevant_predictions_at_full,
    }
