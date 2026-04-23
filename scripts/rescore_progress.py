#!/usr/bin/env python3
"""Report progress for judge-model rescoring submissions.

The progress model is based on the submissions targeted by
`scripts/rescore_judge_model_submissions.sh`:

- target submissions are any submission directories containing ScholarQA or E2E
  `.eval` logs
- a target task is considered completed when the corresponding `.eval` file in
  the rescored tree has been rewritten in place
- a submission is considered completed when all target tasks are rewritten and
  aggregate outputs (`scores.json` and `summary_stats.json`) have also been
  rewritten
- a submission is considered in progress when the rescored output directory
  exists but the submission is not yet complete
- a submission is considered pending when no rescored output directory exists

Requires `inspect_ai` at runtime to extract score deltas from `.eval` logs.
When `inspect_ai` is unavailable, the script still reports progress counts and
aggregate completion, but omits per-task score deltas.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Must match TARGET_LOG_REGEX in scripts/rescore_judge_model_submissions.sh.
TARGET_LOG_REGEX = r"(sqa-(test|dev)|e2e-discovery(-hard)?-(test|validation))"
AGGREGATE_FILES = ("scores.json", "summary_stats.json")
RESCORE_STATE_FILE = ".rescore-state.json"
PRIMARY_METRIC_NAMES = ("mean", "accuracy")
try:
    from inspect_ai.log import read_eval_log as _read_eval_log
except ImportError:
    _read_eval_log = None


def compile_target_log_pattern(
    target_log_regex: str = TARGET_LOG_REGEX,
) -> re.Pattern[str]:
    return re.compile(target_log_regex)


def target_logs_for_submission(
    submission_dir: Path, target_log_pattern: re.Pattern[str] | None = None
) -> list[Path]:
    if target_log_pattern is None:
        target_log_pattern = compile_target_log_pattern()
    return sorted(
        path
        for path in submission_dir.glob("*.eval")
        if target_log_pattern.search(path.name)
    )


def iter_target_submissions(
    submissions_root: Path, target_log_pattern: re.Pattern[str] | None = None
) -> list[Path]:
    if target_log_pattern is None:
        target_log_pattern = compile_target_log_pattern()
    targets: list[Path] = []
    for path in submissions_root.rglob("*.eval"):
        if target_log_pattern.search(path.name):
            targets.append(path.parent)
    return sorted(set(targets))


@dataclass(frozen=True)
class TaskScoreDelta:
    task_name: str
    score_label: str
    before: float
    after: float

    @property
    def delta(self) -> float:
        return self.after - self.before


@dataclass(frozen=True)
class SubmissionProgress:
    rel_path: str
    status: str
    tasks_completed: int
    tasks_total: int
    aggregate_complete: bool
    latest_update_ns: int
    current_log: str | None = None
    failure_message: str | None = None
    score_deltas: tuple[TaskScoreDelta, ...] = ()

    @property
    def tasks_remaining(self) -> int:
        return self.tasks_total - self.tasks_completed


def resume_action(progress: SubmissionProgress) -> str:
    if progress.status == "completed":
        return "skip"
    if progress.status in {"aggregate_pending", "aggregating"}:
        return "aggregate"
    return "full"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show progress for ScholarQA/E2E rescoring submissions."
    )
    parser.add_argument(
        "--submissions-root",
        default="asta-bench-submissions",
        help="Root directory containing source submissions (default: %(default)s)",
    )
    parser.add_argument(
        "--output-root",
        default="asta-bench-submissions-rescored",
        help="Root directory containing rescored submissions (default: %(default)s)",
    )
    parser.add_argument(
        "--show",
        choices=("all", "summary"),
        default="all",
        help="Whether to print all status sections or only the summary",
    )
    parser.add_argument(
        "--target-log-regex",
        default=TARGET_LOG_REGEX,
        help="Regex used to select target submission logs (default: %(default)s)",
    )
    parser.add_argument(
        "--submission-rel",
        help="If set, print JSON status/resume action for one relative submission path and exit",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_rewritten(source_file: Path, output_file: Path) -> bool:
    if not output_file.is_file():
        return False

    source_stat = source_file.stat()
    output_stat = output_file.stat()

    if output_stat.st_mtime_ns != source_stat.st_mtime_ns:
        return True
    if output_stat.st_size != source_stat.st_size:
        return True

    return sha256(source_file) != sha256(output_file)


def latest_mtime_ns(paths: Iterable[Path]) -> int:
    latest = 0
    for path in paths:
        try:
            latest = max(latest, path.stat().st_mtime_ns)
        except FileNotFoundError:
            continue
    return latest


def aggregate_rewritten(
    source_submission_dir: Path, output_submission_dir: Path
) -> bool:
    for filename in AGGREGATE_FILES:
        source_file = source_submission_dir / filename
        output_file = output_submission_dir / filename

        if not output_file.is_file():
            return False
        if source_file.exists() and not file_rewritten(source_file, output_file):
            return False

    return True


def read_submission_state(
    output_submission_dir: Path,
) -> tuple[str | None, str | None, str | None]:
    # State-file protocol written by scripts/rescore_judge_model_submissions.sh:
    # - verifying: copied output exists; inventory check running
    # - rescoring: per-log inspect score running; current_log may be set
    # - aggregating: astabench score running
    # - failed: an error occurred; failure message may be set
    state_path = output_submission_dir / RESCORE_STATE_FILE
    if not state_path.is_file():
        return (None, None, None)

    try:
        payload = json.loads(state_path.read_text())
    except Exception:
        return (None, None, None)

    stage = payload.get("stage")
    current_log = payload.get("current_log")
    message = payload.get("message")
    return (
        stage if isinstance(stage, str) else None,
        current_log if isinstance(current_log, str) else None,
        message if isinstance(message, str) else None,
    )


def primary_score_value(score_summary: dict[str, object]) -> tuple[str, float] | None:
    score_name = score_summary.get("name")
    metrics = score_summary.get("metrics")
    if not isinstance(score_name, str) or not isinstance(metrics, dict):
        return None

    for metric_name in PRIMARY_METRIC_NAMES:
        metric = metrics.get(metric_name)
        if isinstance(metric, dict):
            value = metric.get("value")
            if isinstance(value, int | float):
                return (f"{score_name}/{metric_name}", float(value))

    for metric_name, metric in metrics.items():
        if metric_name == "stderr" or not isinstance(metric, dict):
            continue
        value = metric.get("value")
        if isinstance(value, int | float):
            return (f"{score_name}/{metric_name}", float(value))

    return None


def log_primary_score(path: Path) -> tuple[str, str, float] | None:
    if _read_eval_log is None:
        return None

    try:
        log = _read_eval_log(str(path), header_only=True)
    except Exception:
        return None

    if not log.results or not log.results.scores:
        return None

    primary = primary_score_value(log.results.scores[0].model_dump(mode="json"))
    if primary is None:
        return None

    score_label, value = primary
    task_name = log.eval.task or path.stem
    return (task_name, score_label, value)


def score_delta_for_log(source_log: Path, output_log: Path) -> TaskScoreDelta | None:
    before = log_primary_score(source_log)
    after = log_primary_score(output_log)
    if before is None or after is None:
        return None

    _, before_label, before_value = before
    task_name, after_label, after_value = after
    if before_label != after_label:
        return None

    return TaskScoreDelta(
        task_name=task_name,
        score_label=after_label,
        before=before_value,
        after=after_value,
    )


def submission_progress(
    source_submission_dir: Path,
    submissions_root: Path,
    output_root: Path,
    target_log_pattern=None,
) -> SubmissionProgress:
    if target_log_pattern is None:
        target_log_pattern = compile_target_log_pattern()
    rel_path = source_submission_dir.relative_to(submissions_root).as_posix()
    output_submission_dir = output_root / rel_path
    target_logs = target_logs_for_submission(
        source_submission_dir, target_log_pattern=target_log_pattern
    )
    tasks_total = len(target_logs)

    if not output_submission_dir.is_dir():
        return SubmissionProgress(
            rel_path=rel_path,
            status="pending",
            tasks_completed=0,
            tasks_total=tasks_total,
            aggregate_complete=False,
            latest_update_ns=0,
        )

    tasks_completed = 0
    paths_for_latest: list[Path] = []
    score_deltas: list[TaskScoreDelta] = []
    state_stage, current_log, failure_message = read_submission_state(
        output_submission_dir
    )
    for source_log in target_logs:
        output_log = output_submission_dir / source_log.name
        if output_log.exists():
            paths_for_latest.append(output_log)
        if file_rewritten(source_log, output_log):
            tasks_completed += 1
            score_delta = score_delta_for_log(source_log, output_log)
            if score_delta is not None:
                score_deltas.append(score_delta)

    aggregate_complete = aggregate_rewritten(
        source_submission_dir, output_submission_dir
    )
    for filename in AGGREGATE_FILES:
        output_file = output_submission_dir / filename
        if output_file.exists():
            paths_for_latest.append(output_file)
    state_file = output_submission_dir / RESCORE_STATE_FILE
    if state_file.exists():
        paths_for_latest.append(state_file)

    if tasks_completed == tasks_total and aggregate_complete:
        status = "completed"
        current_log = None
        failure_message = None
    elif state_stage == "failed":
        status = "failed"
    elif state_stage == "aggregating":
        status = "aggregating"
        current_log = None
    elif state_stage in {"verifying", "rescoring"}:
        status = "rescoring"
    elif tasks_completed == tasks_total and not aggregate_complete:
        status = "aggregate_pending"
        current_log = None
    else:
        status = "rescoring"
    return SubmissionProgress(
        rel_path=rel_path,
        status=status,
        tasks_completed=tasks_completed,
        tasks_total=tasks_total,
        aggregate_complete=aggregate_complete,
        latest_update_ns=latest_mtime_ns(paths_for_latest),
        current_log=current_log,
        failure_message=failure_message,
        score_deltas=tuple(score_deltas),
    )


def display_status(status: str) -> str:
    return status.replace("_", " ")


def format_progress(progress: SubmissionProgress) -> str:
    aggregate = "done" if progress.aggregate_complete else "pending"
    summary = (
        f"{progress.rel_path}  "
        f"tasks {progress.tasks_completed}/{progress.tasks_total}  "
        f"remaining {progress.tasks_remaining}  "
        f"aggregate {aggregate}"
    )
    if progress.status != "completed":
        summary += f"  status {display_status(progress.status)}"
    if progress.current_log:
        summary += f"  log {progress.current_log}"
    if progress.failure_message:
        summary += f"  error {progress.failure_message}"
    return summary


def format_score_delta(score_delta: TaskScoreDelta) -> str:
    return (
        f"{score_delta.task_name} {score_delta.score_label}  "
        f"{score_delta.before:.6f} -> {score_delta.after:.6f}  "
        f"({score_delta.delta:+.6f})"
    )


def print_section(title: str, items: list[SubmissionProgress]) -> None:
    print(f"{title} ({len(items)})")
    if not items:
        print("  none")
        return
    for progress in items:
        print(f"  {format_progress(progress)}")
        for score_delta in progress.score_deltas:
            print(f"    {format_score_delta(score_delta)}")


def main() -> None:
    args = parse_args()
    submissions_root = Path(args.submissions_root)
    output_root = Path(args.output_root)
    target_log_pattern = compile_target_log_pattern(args.target_log_regex)

    if not submissions_root.is_dir():
        raise SystemExit(f"error: submissions root not found ({submissions_root})")

    if args.submission_rel:
        source_submission_dir = submissions_root / args.submission_rel
        if not source_submission_dir.is_dir():
            raise SystemExit(
                f"error: submission path not found ({source_submission_dir})"
            )
        progress = submission_progress(
            source_submission_dir,
            submissions_root=submissions_root,
            output_root=output_root,
            target_log_pattern=target_log_pattern,
        )
        print(
            json.dumps(
                {
                    "rel_path": progress.rel_path,
                    "status": progress.status,
                    "action": resume_action(progress),
                }
            )
        )
        return

    targets = iter_target_submissions(
        submissions_root, target_log_pattern=target_log_pattern
    )
    progress = [
        submission_progress(
            path,
            submissions_root=submissions_root,
            output_root=output_root,
            target_log_pattern=target_log_pattern,
        )
        for path in targets
    ]

    completed = [item for item in progress if item.status == "completed"]
    in_progress = [
        item for item in progress if item.status in {"rescoring", "aggregating"}
    ]
    failed = [item for item in progress if item.status == "failed"]
    interrupted = [item for item in progress if item.status == "aggregate_pending"]
    pending = [item for item in progress if item.status == "pending"]
    active = max(in_progress, key=lambda item: item.latest_update_ns, default=None)

    total_tasks = sum(item.tasks_total for item in progress)
    completed_tasks = sum(item.tasks_completed for item in progress)

    print(f"Target submissions: {len(progress)}")
    print(
        f"Task progress: {completed_tasks}/{total_tasks} completed, "
        f"{total_tasks - completed_tasks} remaining"
    )
    print(
        f"Submissions: {len(completed)} completed, "
        f"{len(in_progress)} in progress, {len(failed)} failed, "
        f"{len(interrupted)} interrupted, "
        f"{len(pending)} pending"
    )
    if active is None:
        print("Current in progress: none")
    else:
        print(f"Current in progress: {format_progress(active)}")

    if args.show == "summary":
        return

    print()
    print_section("Completed", completed)
    print()
    print_section("In Progress", in_progress)
    print()
    print_section("Failed", failed)
    print()
    print_section("Interrupted", interrupted)
    print()
    print_section("Pending", pending)


if __name__ == "__main__":
    main()
