#!/usr/bin/env python3
"""Detect and optionally archive duplicate task logs within one submission."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from inspect_ai.log import read_eval_log


@dataclass(frozen=True)
class EvalLogMetadata:
    path: Path
    task_name: str
    created: str

    @property
    def filename(self) -> str:
        return self.path.name


@dataclass(frozen=True)
class DuplicateGroup:
    task_name: str
    kept: EvalLogMetadata
    archived: tuple[EvalLogMetadata, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-dir", required=True)
    parser.add_argument(
        "--policy",
        choices=("fail", "keep-latest"),
        default="fail",
        help="How to handle duplicate task logs (default: %(default)s)",
    )
    parser.add_argument(
        "--archive-dir",
        help="Directory to move archived duplicate logs into when using keep-latest",
    )
    return parser.parse_args()


def created_string(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str):
        return value
    return ""


def load_eval_log_metadata(submission_dir: Path) -> list[EvalLogMetadata]:
    metadata: list[EvalLogMetadata] = []
    for path in sorted(submission_dir.glob("*.eval")):
        log = read_eval_log(str(path), header_only=True)
        # Normalize to agenteval's fallback form (strip package prefix, e.g.
        # "astabench/paper_finder_test" -> "paper_finder_test"). Two logs with
        # the same normalized name are considered duplicates for dedup purposes.
        task_name = log.eval.task.split("/")[-1] if log.eval.task else path.stem
        created = created_string(getattr(log.eval, "created", None))
        metadata.append(
            EvalLogMetadata(path=path, task_name=task_name, created=created)
        )
    return metadata


def duplicate_groups(
    metadata: Iterable[EvalLogMetadata],
) -> tuple[DuplicateGroup, ...]:
    grouped: dict[str, list[EvalLogMetadata]] = {}
    for entry in metadata:
        grouped.setdefault(entry.task_name, []).append(entry)

    duplicates: list[DuplicateGroup] = []
    for task_name, entries in grouped.items():
        if len(entries) < 2:
            continue
        ordered = sorted(entries, key=lambda entry: (entry.created, entry.filename))
        duplicates.append(
            DuplicateGroup(
                task_name=task_name,
                kept=ordered[-1],
                archived=tuple(ordered[:-1]),
            )
        )
    return tuple(sorted(duplicates, key=lambda group: group.task_name))


def prune_logs_json(submission_dir: Path, kept_filenames: set[str]) -> None:
    logs_json = submission_dir / "logs.json"
    if not logs_json.is_file():
        return

    payload = json.loads(logs_json.read_text())
    if not isinstance(payload, dict):
        return

    filtered = {key: value for key, value in payload.items() if key in kept_filenames}
    logs_json.write_text(json.dumps(filtered, indent=2, sort_keys=True) + "\n")


def metadata_payload(entry: EvalLogMetadata) -> dict[str, str]:
    return {
        "path": str(entry.path),
        "filename": entry.filename,
        "task_name": entry.task_name,
        "created": entry.created,
    }


def write_manifest(archive_dir: Path, groups: tuple[DuplicateGroup, ...]) -> None:
    manifest = [
        {
            "task_name": group.task_name,
            "kept": metadata_payload(group.kept),
            "archived": [metadata_payload(entry) for entry in group.archived],
        }
        for group in groups
    ]
    (archive_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def apply_keep_latest(
    submission_dir: Path,
    archive_dir: Path,
    groups: tuple[DuplicateGroup, ...],
) -> None:
    archive_dir.mkdir(parents=True, exist_ok=True)
    for group in groups:
        for entry in group.archived:
            destination = archive_dir / entry.filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(entry.path), str(destination))
            print(
                f"== archive: {entry.path} (task {group.task_name}; keeping {group.kept.filename})"
            )

    kept_filenames = {path.name for path in submission_dir.glob("*.eval")}
    prune_logs_json(submission_dir, kept_filenames)
    write_manifest(archive_dir, groups)


def main() -> None:
    args = parse_args()
    submission_dir = Path(args.submission_dir)
    metadata = load_eval_log_metadata(submission_dir)
    groups = duplicate_groups(metadata)

    if not groups:
        return

    if args.policy == "fail":
        for group in groups:
            print(
                f"duplicate task {group.task_name}: kept candidate {group.kept.filename}; "
                f"duplicates {[entry.filename for entry in group.archived]}",
                file=sys.stderr,
            )
        raise SystemExit(3)

    if not args.archive_dir:
        raise SystemExit("--archive-dir is required for --policy keep-latest")

    apply_keep_latest(submission_dir, Path(args.archive_dir), groups)


if __name__ == "__main__":
    main()
