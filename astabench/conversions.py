from agenteval.config import Task, load_suite_config
from agenteval.interventions import Intervention
from agenteval.leaderboard.models import LeaderboardSubmission, Readme
from agenteval.score import TaskResult
from astabench.cli import get_config_path


# src HF config -> target HF config -> split -> original name -> target name
TASK_NAME_ALIASES = {
    "1.0.0-dev1": {
        "1.0.0": {
            "test": {
                "paper_finder_test": "PaperFindingBench_test",
                "paper_finder_litqa2_test": "LitQA2_FullText_Search_test",
                "sqa_test": "ScholarQA_CS2_test",
                "arxivdigestables_test": "ArxivDIGESTables_Clean_test",
                "litqa2_test": "LitQA2_FullText_test",
                "discoverybench_test": "DiscoveryBench_test",
                "core_bench_test": "CORE_Bench_Hard_test",
                "ds1000_test": "DS_1000_test",
                "e2e_discovery_test": "E2E_Bench_test",
                "e2e_discovery_hard_test": "E2E_Bench_Hard_test",
                "super_test": "SUPER_Expert_test",
            },
            "validation": {
                "arxivdigestables_validation": "ArxivDIGESTables_Clean_validation",
                "sqa_dev": "ScholarQA_CS2_validation",
                "litqa2_validation": "LitQA2_FullText_validation",
                "paper_finder_validation": "PaperFindingBench_validation",
                "paper_finder_litqa2_validation": "LitQA2_FullText_Search_validation",
                "discoverybench_validation": "DiscoveryBench_validation",
                "core_bench_validation": "CORE_Bench_Hard_validation",
                "ds1000_validation": "DS_1000_validation",
                "e2e_discovery_validation": "E2E_Bench_validation",
                "e2e_discovery_hard_validation": "E2E_Bench_Hard_validation",
                "super_validation": "SUPER_Expert_validation",
            },
        }
    }
}


def convert_one_task_result(
    result: TaskResult,
    split: str,
    src_hf_config: str,
    target_hf_config: str,
    target_tasks_by_name: dict[str, Task],
):
    changed_something = False

    original_task_name = result.task_name
    if original_task_name in TASK_NAME_ALIASES.get(src_hf_config, {}).get(
        target_hf_config, {}
    ).get(split, {}):
        new_task_name = TASK_NAME_ALIASES[src_hf_config][target_hf_config][split][
            original_task_name
        ]
        result.task_name = new_task_name

    final_task_name = result.task_name
    if original_task_name != final_task_name:
        changed_something = True

    if final_task_name not in target_tasks_by_name:
        print(f"Unknown final task name {final_task_name}")
    else:
        expected_primary_metric_name = target_tasks_by_name[
            final_task_name
        ].primary_metric
        if expected_primary_metric_name not in result.available_metrics():
            print(
                f"Expected {expected_primary_metric_name} as the primary metric for task {final_task_name}, but don't have it."
            )

    return changed_something


def convert_task_results(
    task_results: list[TaskResult],
    split: str,
    src_hf_config: str,
    target_hf_config: str,
    target_tasks_by_name: dict[str, Task],
):
    changed_something = False
    for result in task_results:
        # changes happen in place
        changed_this_thing = convert_one_task_result(
            result=result,
            split=split,
            src_hf_config=src_hf_config,
            target_hf_config=target_hf_config,
            target_tasks_by_name=target_tasks_by_name,
        )
        changed_something = changed_something or changed_this_thing

    return changed_something


def convert_lb_submission_to_astabench_config(lb_submission: LeaderboardSubmission, config_name: str) -> bool:
    src_suite_config = lb_submission.suite_config
    target_suite_config = load_suite_config(get_config_path(config_name))

    target_tasks_by_name = target_suite_config.get_tasks_by_name(lb_submission.split)

    # changes are made in place
    changed_something = convert_task_results(
        task_results=lb_submission.results,
        split=lb_submission.split,
        src_hf_config=src_suite_config.version,
        target_hf_config=target_suite_config.version,
        target_tasks_by_name=target_tasks_by_name,
    )
    if src_suite_config != target_suite_config:
        lb_submission.suite_config = target_suite_config
        changed_something = True

    return changed_something


convert_1_0_0_dev1_to_1_0_0 = Intervention(
    eligible=lambda x: x.lb_submission.suite_config.version == "1.0.0-dev1",
    transform=lambda x: convert_lb_submission_to_astabench_config(x, "v1.0.0"),
)
