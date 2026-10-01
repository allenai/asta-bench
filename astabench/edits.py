from agenteval.leaderboard.models import LeaderboardSubmission
from agenteval.interventions import Intervention, LbSubmissionWithDetails, CONVERSION_INTERVENTION_KIND, EDIT_INTERVENTION_KIND


EXPECTED_SUBMITTERS = {"Ai2", "Elicit", "SciSpace"}


KNOWN_AI2_SUBMITTERS = {
    "danyhai2",
    "miked-ai",
    "aakanksha19",
    "aryeh_tiktinsky_ai2",
    "varshak1",
    "aps6992",
    "pclark425",
}


ACCEPTABLE_OPENNESS = {
    "Open source & open weights",
    "Open source & closed weights",
    "Closed source & API available",
    "Closed source & UI only",
}


OPENNESS_MAPPING = {
    "Open Source + Open Weights": "Open source & open weights",
    "Open Source": "Open source & closed weights",
    "API Available": "Closed source & API available",
    "Closed": "Closed source & UI only",
}


def has_openness(
    submission_with_details: LbSubmissionWithDetails, opennesses: set[str]
) -> bool:
    return submission_with_details.lb_submission.submission.openness in opennesses


def has_submitter(
    submission_with_details: LbSubmissionWithDetails, usernames: set[str]
) -> bool:
    return submission_with_details.lb_submission.submission.username in usernames


def has_any_non_null_costs(
    submission_with_details: LbSubmissionWithDetails,
) -> bool:
    return any(
        [r.model_costs is not None for r in submission_with_details.lb_submission.results]
    )


def has_any_non_null_model_usages(
    submission_with_details: LbSubmissionWithDetails,
) -> bool:
    return any(
        [r.model_usages is not None for r in submission_with_details.lb_submission.results]
    )


def set_submitter(lb_submission: LeaderboardSubmission, new_submitter: str) -> bool:
    """Return true if something changed."""
    current_submitter = lb_submission.submission.username
    lb_submission.submission.username = new_submitter
    return current_submitter != lb_submission.submission.username


def normalize_submitter(lb_submission: LeaderboardSubmission) -> bool:
    """Return true if something changed."""
    current_submitter = lb_submission.submission.username
    agent_name = lb_submission.submission.agent_name
    if current_submitter in KNOWN_AI2_SUBMITTERS:
        if agent_name in {"Elicit", "SciSpace"}:
            to_return = set_submitter(lb_submission, agent_name)
        else:
            to_return = set_submitter(lb_submission, "Ai2")
    return to_return


def normalize_openness(lb_submission: LeaderboardSubmission):
    """Return true if something changed."""
    current_openness = lb_submission.submission.openness
    replace_with = OPENNESS_MAPPING.get(current_openness, current_openness)
    lb_submission.submission.openness = replace_with
    return current_openness != lb_submission.submission.openness


normalize_submitter_intervention = Intervention(
    eligible=lambda x: not has_submitter(x, EXPECTED_SUBMITTERS),
    transform=normalize_submitter,
)

normalize_openness_intervention = Intervention(
    eligible=lambda x: not has_openness(x, ACCEPTABLE_OPENNESS),
    transform=normalize_openness,
)
