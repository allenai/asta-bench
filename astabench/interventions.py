from agenteval.leaderboard.models import LeaderboardSubmission
from agenteval.interventions import Intervention, LbSubmissionWithDetails, CONVERSION_INTERVENTION_KIND, EDIT_INTERVENTION_KIND
from astabench.conversions import convert_1_0_0_dev1_to_1_0_0
from astabench.edits import normalize_submitter_intervention, normalize_openness_intervention


# intervention kind -> config name -> intervention name -> Intervention
INTERVENTIONS: dict[str, dict[str, dict[str, Intervention]]] = {
    EDIT_INTERVENTION_KIND: {
        "1.0.0": {
            "normalize_submitter": normalize_submitter_intervention,
            "normalize_openness": normalize_openness_intervention,
         },
        "1.0.0-dev1": {
            "normalize_submitter": normalize_submitter_intervention,
            "normalize_openness": normalize_openness_intervention,
         },
    },
    CONVERSION_INTERVENTION_KIND: {
        "1.0.0-dev1": {
            "convert_1_0_0_dev1_to_1_0_0": convert_1_0_0_dev1_to_1_0_0,
        }
    },
}
