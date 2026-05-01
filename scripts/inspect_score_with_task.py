#!/usr/bin/env python3
"""Run `inspect score` for tasks whose judge models changed; preserve otherwise.

For tasks in `RESCORE_TASKS`, preload task-local scorer definitions before
delegating to `inspect score`. This works around `inspect score` failing to
rediscover scorers from logs that only record bare scorer names and often have
`task_file=None`.

For all other tasks, exit successfully without rerunning the scorer so the
copied `.eval` file's existing scores remain intact for later aggregation.

Validated against `inspect_ai==0.3.203`.
"""

from __future__ import annotations

import sys
import warnings
from typing import Any

import inspect_ai
from inspect_ai._cli.score import score_command
from inspect_ai._util.registry import _registry, registry_key, registry_lookup
from inspect_ai.log import read_eval_log

VALIDATED_INSPECT_VERSION = "0.3.203"
# Tasks whose LLM judge/scorer model changed in recent source updates.
# Current scope:
# - SQA scorer_model: gemini-2.5-flash -> gemini-3-flash-preview
#   Caveat: historical SQA logs serialize the nested scorer model params, so
#   preloading scorer definitions is not enough to migrate SQA by itself.
#   Use an explicit --scorer override or edit the .eval scorer config.
# - E2E rubric model: claude-3.7-sonnet -> claude-sonnet-4-6
#
# All other tasks either have unchanged judge models or are deterministic, so
# their copied `.eval` scores should be preserved as-is.
RESCORE_TASKS = frozenset(
    {
        "astabench/sqa_test",
        "astabench/sqa_dev",
        "astabench/e2e_discovery_test",
        "astabench/e2e_discovery_validation",
        "astabench/e2e_discovery_hard_test",
        "astabench/e2e_discovery_hard_validation",
    }
)
RESCORE_TASK_NAMES = frozenset(
    task_name.split("/", 1)[1] for task_name in RESCORE_TASKS
)


def nested_scorer_names(value: Any) -> list[str]:
    names: list[str] = []
    if isinstance(value, dict):
        scorer_type = value.get("type")
        scorer_name = value.get("name")
        if scorer_type == "scorer" and isinstance(scorer_name, str):
            names.append(scorer_name)
        for nested in value.values():
            names.extend(nested_scorer_names(nested))
    elif isinstance(value, list | tuple):
        for nested in value:
            names.extend(nested_scorer_names(nested))
    return names


def scorer_names(log: Any) -> list[str]:
    names: list[str] = []
    if log.eval.scorers is not None:
        for scorer in log.eval.scorers:
            names.append(scorer.name)
            names.extend(nested_scorer_names(scorer.options or {}))
    elif log.results and log.results.scores is not None:
        # Defensive fallback for older logs. Current `asta-bench-submissions`
        # logs all populate `eval.scorers`, which is the richer source because
        # it also includes nested scorer references inside scorer options.
        names.extend(score.name for score in log.results.scores)

    # Preserve the first occurrence of each name so error messages remain stable.
    return list(dict.fromkeys(names))


def unresolved_scorers(names: list[str]) -> list[str]:
    return [name for name in names if registry_lookup("scorer", name) is None]


def alternate_scorer_name(name: str) -> str | None:
    if name.startswith("astabench/"):
        return name.split("/", 1)[1]
    if "/" not in name:
        return f"astabench/{name}"
    return None


def alias_missing_scorers(names: list[str]) -> None:
    """Add bidirectional aliases between bare and `astabench/...` scorer names.

    This is load-bearing for rescoring because scorer registration namespace
    depends on how `astabench` is discovered at import time in the scoring
    environment, while historical logs contain both bare and namespaced scorer
    references.
    """
    for name in names:
        if registry_lookup("scorer", name) is not None:
            continue

        alternate = alternate_scorer_name(name)
        if alternate is None:
            continue

        scorer = registry_lookup("scorer", alternate)
        if scorer is not None:
            _registry[registry_key("scorer", name)] = scorer


def normalized_task_name(task_name: str | None) -> str | None:
    if not task_name:
        return None
    if "/" in task_name:
        return task_name.split("/", 1)[1]
    return task_name


def should_rescore(log: Any) -> bool:
    return normalized_task_name(log.eval.task) in RESCORE_TASK_NAMES


def preload_scorers(log: Any, log_file: str) -> None:
    names = scorer_names(log)
    if not names:
        return

    if not unresolved_scorers(names):
        return

    # Importing the registry registers module-level scorers and task entry points.
    import astabench.evals._registry  # noqa: F401

    alias_missing_scorers(names)

    remaining = unresolved_scorers(names)
    if remaining:
        joined = ", ".join(sorted(remaining))
        raise SystemExit(
            f"unable to resolve scorer(s) for {log_file}: {joined}. "
            "Try passing --scorer explicitly."
        )


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise SystemExit(
            "usage: inspect_score_with_task.py [inspect score options] <log_file>"
        )

    if inspect_ai.__version__ != VALIDATED_INSPECT_VERSION:
        warnings.warn(
            "inspect_score_with_task.py was validated against "
            f"inspect_ai=={VALIDATED_INSPECT_VERSION}, found "
            f"{inspect_ai.__version__}. Private API imports may need updating.",
            stacklevel=1,
        )

    explicit_scorer = any(
        arg == "--scorer" or arg.startswith("--scorer=") for arg in args
    )
    # `inspect score` takes the log file as its final positional argument.
    log_file = args[-1]
    log = read_eval_log(log_file, header_only=True)

    if not explicit_scorer and not should_rescore(log):
        task_name = log.eval.task or "<unknown task>"
        print(f"== preserve: {log_file} ({task_name})", file=sys.stderr)
        return

    if not explicit_scorer:
        preload_scorers(log, log_file)

    score_command.main(args=args, prog_name="inspect score")


if __name__ == "__main__":
    main()
