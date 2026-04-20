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
"""

from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Must stay in sync with `find_target_submissions()` in
# `scripts/rescore_judge_model_submissions.sh`.
TARGET_LOG_PATTERN = re.compile(
    r"(sqa-(test|dev)|e2e-discovery(-hard)?-(test|validation))"
)
AGGREGATE_FILES = ("scores.json", "summary_stats.json")


@dataclass(frozen=True)
class SubmissionProgress:
    rel_path: str
    status: str
    tasks_completed: int
    tasks_total: int
    aggregate_complete: bool
    latest_update_ns: int

    @property
    def tasks_remaining(self) -> int:
        return self.tasks_total - self.tasks_completed


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


def target_logs_for_submission(submission_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in submission_dir.glob("*.eval")
        if TARGET_LOG_PATTERN.search(path.name)
    )


def iter_target_submissions(submissions_root: Path) -> list[Path]:
    targets: list[Path] = []
    for path in submissions_root.rglob("*.eval"):
        if TARGET_LOG_PATTERN.search(path.name):
            targets.append(path.parent)
    return sorted(set(targets))


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


def submission_progress(
    source_submission_dir: Path, submissions_root: Path, output_root: Path
) -> SubmissionProgress:
    rel_path = source_submission_dir.relative_to(submissions_root).as_posix()
    output_submission_dir = output_root / rel_path
    target_logs = target_logs_for_submission(source_submission_dir)
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
    for source_log in target_logs:
        output_log = output_submission_dir / source_log.name
        if output_log.exists():
            paths_for_latest.append(output_log)
        if file_rewritten(source_log, output_log):
            tasks_completed += 1

    aggregate_complete = aggregate_rewritten(
        source_submission_dir, output_submission_dir
    )
    for filename in AGGREGATE_FILES:
        output_file = output_submission_dir / filename
        if output_file.exists():
            paths_for_latest.append(output_file)

    status = (
        "completed"
        if tasks_completed == tasks_total and aggregate_complete
        else "in_progress"
    )
    return SubmissionProgress(
        rel_path=rel_path,
        status=status,
        tasks_completed=tasks_completed,
        tasks_total=tasks_total,
        aggregate_complete=aggregate_complete,
        latest_update_ns=latest_mtime_ns(paths_for_latest),
    )


def format_progress(progress: SubmissionProgress) -> str:
    aggregate = "done" if progress.aggregate_complete else "pending"
    return (
        f"{progress.rel_path}  "
        f"tasks {progress.tasks_completed}/{progress.tasks_total}  "
        f"remaining {progress.tasks_remaining}  "
        f"aggregate {aggregate}"
    )


def print_section(title: str, items: list[SubmissionProgress]) -> None:
    print(f"{title} ({len(items)})")
    if not items:
        print("  none")
        return
    for progress in items:
        print(f"  {format_progress(progress)}")


def main() -> None:
    args = parse_args()
    submissions_root = Path(args.submissions_root)
    output_root = Path(args.output_root)

    if not submissions_root.is_dir():
        raise SystemExit(f"error: submissions root not found ({submissions_root})")

    targets = iter_target_submissions(submissions_root)
    progress = [
        submission_progress(
            path, submissions_root=submissions_root, output_root=output_root
        )
        for path in targets
    ]

    completed = [item for item in progress if item.status == "completed"]
    in_progress = [item for item in progress if item.status == "in_progress"]
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
        f"{len(in_progress)} in progress, {len(pending)} pending"
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
    print_section("Pending", pending)


if __name__ == "__main__":
    main()
