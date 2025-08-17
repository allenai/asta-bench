from agenteval.leaderboard.models import LeaderboardSubmission
from agenteval.repairs import Intervention, LbSubmissionWithDetails, CONVERSION_INTERVENTION_KIND, EDIT_INTERVENTION_KIND


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


def has_submitter(
    submission_with_details: LbSubmissionWithDetails, usernames: set[str]
) -> bool:
    return submission_with_details.lb_submission.submission.username in usernames


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


normalize_submitter_intervention = Intervention(
    eligible=lambda x: not has_submitter(x, EXPECTED_SUBMITTERS),
    transform=normalize_submitter,
)


# intervention kind -> config name -> intervention name -> Intervention
INTERVENTIONS: dict[str, dict[str, dict[str, Intervention]]] = {
    EDIT_INTERVENTION_KIND: {
        "1.0.0": {
            "normalize_submitter": normalize_submitter_intervention
         },
        "1.0.0-dev1": {
            "normalize_submitter": normalize_submitter_intervention
         },
    },
    CONVERSION_INTERVENTION_KIND: {},
}
