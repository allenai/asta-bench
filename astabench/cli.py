"""astabench command-line interface.

The ``eval`` (solve) command is implemented natively here and shells out to
``inspect eval-set`` — it needs only Inspect and the task registry, **not**
``agent-eval``. This lets the solve environment run a newer ``inspect_ai`` than
the scorer, which is pinned by ``agent-eval`` (see allenai/gas2own#346).

Scoring / leaderboard commands (``score``, ``lb``, ``publish-logs``, …) are
provided by ``agent-eval`` and loaded lazily, so they are only required in an
environment that installed the ``[score]`` extra.
"""

import importlib.resources
import json
import os
import signal
import subprocess
import sys
from datetime import datetime

import click

from astabench.suite_config import load_suite_config

DEFAULT_CONFIG = "v1.0.0"
SPLIT_NAMES = ["validation", "test"]
EVAL_CONFIG_FILENAME = "eval_config.json"


def get_config_path(config_name):
    """Get the path to a config file in the package's config directory."""
    with importlib.resources.path("astabench.config", f"{config_name}.yml") as path:
        return os.path.abspath(path)


def verify_git_reproducibility() -> None:
    """Abort if the working tree isn't in a reproducibly-recorded git state."""
    try:
        sha_result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        origin_result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            check=True,
        )
        sha = sha_result.stdout.strip() if sha_result.returncode == 0 else None
        origin = origin_result.stdout.strip() if origin_result.returncode == 0 else None

        git_dirty = (
            subprocess.run(
                ["git", "diff", "--quiet", "--exit-code"],
                capture_output=True,
                check=False,
            ).returncode
            != 0
        )

        untracked_result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            capture_output=True,
            text=True,
            check=True,
        )
        untracked_files = untracked_result.stdout.strip().splitlines()
        if untracked_files:
            click.echo(
                f"Warning: Untracked files present: {', '.join(untracked_files)}. "
                "For reproducibility, please add, ignore, or remove these files."
            )

        if git_dirty:
            raise click.ClickException(
                f"Git working directory contains uncommitted changes. "
                f"For reproducibility, Inspect will save: origin={origin}, sha={sha}. "
                "Please commit your changes or use --ignore-git to bypass this check (not recommended)."
            )

        if sha:
            remote_exists = subprocess.run(
                ["git", "branch", "-r", "--contains", sha],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            if not remote_exists:
                raise click.ClickException(
                    f"Commit {sha} not found on remote '{origin}'. Others won't be able to "
                    "access this code version. Please push your changes or use --ignore-git "
                    "to bypass this check (not recommended)."
                )
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        if isinstance(e, click.ClickException):
            raise
        raise click.ClickException(
            f"Unable to verify git status for reproducibility: {e}. "
            "Use --ignore-git to bypass this check if git is not available."
        )


@click.command(
    name="eval",
    help="Run inspect eval-set on the suite's tasks with the given arguments.",
    context_settings={"ignore_unknown_options": True},
)
@click.option(
    "--log-dir",
    type=str,
    help="Log directory. Defaults to INSPECT_LOG_DIR or auto-generated under ./logs.",
)
@click.option(
    "--config-path",
    "config_path",
    type=str,
    default=lambda: get_config_path(DEFAULT_CONFIG),
    show_default=True,
    hidden=True,
    help="Path to a yml config file.",
)
@click.option(
    "--split",
    type=click.Choice(SPLIT_NAMES, case_sensitive=False),
    required=True,
    help="Config data split.",
)
@click.option(
    "--ignore-git",
    is_flag=True,
    help="Ignore git reproducibility checks (not recommended).",
)
@click.option(
    "--config-only",
    is_flag=True,
    help="Print the command that would be run and save eval_config locally.",
)
@click.option(
    "--display",
    type=str,
    help="Display format. Defaults to plain.",
    default="plain",
)
@click.option(
    "--task",
    "task_filters",
    multiple=True,
    help="Filter to only run tasks whose name contains this string (can be specified multiple times).",
)
@click.option(
    "--task-category",
    "task_category_filters",
    multiple=True,
    help="Filter to only run tasks with this tag (can be specified multiple times).",
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def eval_command(
    log_dir: str | None,
    config_path: str,
    split: str,
    ignore_git: bool,
    config_only: bool,
    display: str,
    task_filters: tuple[str, ...],
    task_category_filters: tuple[str, ...],
    args: tuple[str],
):
    """Run ``inspect eval-set`` with arguments and append the suite's tasks.

    Any unrecognized options in ``args`` are forwarded verbatim to
    ``inspect eval-set`` (e.g. ``--solver``, ``--model``, ``--no-score``,
    ``--log-format``, ``--limit``).
    """
    suite_config = load_suite_config(config_path)
    tasks = suite_config.get_tasks(split)

    # Apply task filtering
    if task_filters or task_category_filters:
        original_count = len(tasks)
        filtered_tasks = []
        for task in tasks:
            if task_filters:
                name_match = any(f in task.name for f in task_filters)
                if not name_match:
                    continue

            if task_category_filters:
                task_tags = task.get_tag_names()
                category_match = any(cat in task_tags for cat in task_category_filters)
                if not category_match:
                    continue

            filtered_tasks.append(task)

        tasks = filtered_tasks
        click.echo(f"Filtered to {len(tasks)} of {original_count} tasks")

        if not tasks:
            raise click.ClickException(
                "No tasks match the specified filters. "
                f"Task filters: {task_filters}, Category filters: {task_category_filters}"
            )

    if not ignore_git:
        verify_git_reproducibility()

    if not log_dir:
        log_dir = os.environ.get("INSPECT_LOG_DIR")
        if not log_dir:
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            log_dir = os.path.join(
                ".",
                "logs",
                f"{suite_config.name}_{suite_config.version}_{split}_{timestamp}",
            )
            click.echo(f"No log dir was manually set; using {log_dir}")
    logd_args = ["--log-dir", log_dir]
    display_args = ["--display", display]

    os.makedirs(log_dir, exist_ok=True)

    inspect_command = (
        ["inspect", "eval-set"]
        + list(args)
        + logd_args
        + display_args
        + [x.path for x in tasks]
    )

    # eval_config.json schema matches agenteval.models.EvalConfig so the scorer
    # (which validates it via EvalConfig.model_validate_json) round-trips it.
    eval_config = {
        "suite_config": suite_config.model_dump(mode="json"),
        "split": split,
        "inspect_command": inspect_command,
    }

    eval_config_path = os.path.join(log_dir, EVAL_CONFIG_FILENAME)
    if not os.path.exists(eval_config_path):
        with open(eval_config_path, "w", encoding="utf-8") as f:
            json.dump(eval_config, f, indent=2)
    else:
        with open(eval_config_path, "r", encoding="utf-8") as f:
            existing_config = json.load(f)
        if existing_config != eval_config:
            click.echo(
                f"Suite config does not match pre-existing config in {EVAL_CONFIG_FILENAME}. Rerun in an empty directory"
            )
            sys.exit(1)

    if config_only:
        click.echo(f"Dry run: would run command: {' '.join(inspect_command)}")
        return

    click.echo(f"Running {' '.join(inspect_command)}")
    # Popen + our own Ctrl-C handling; `subprocess.run` would automatically
    # SIGKILL the child and skip inspect's sandbox cleanup.
    proc = subprocess.Popen(inspect_command)
    interrupted = False
    try:
        returncode = proc.wait()
    except KeyboardInterrupt:
        interrupted = True
        click.echo(
            f"\nInterrupt received; waiting for inspect (pid {proc.pid}) to "
            f"finish sandbox cleanup.\n"
            f"  - If Ctrl-C went to the whole process group (the usual case "
            f"when running `astabench eval` directly in a terminal), inspect "
            f"already received SIGINT and is cleaning up -- just wait.\n"
            f"  - If only this process was signalled (e.g. a wrapper script "
            f"forwarded SIGINT by PID rather than to the process group), "
            f"inspect did NOT receive SIGINT and will not exit on its own; "
            f"in another terminal run `kill -INT {proc.pid}` to trigger its "
            f"cleanup.\n"
            f"  - Press Ctrl-C again to give up and SIGKILL inspect (may "
            f"leave orphan sandbox containers).",
            err=True,
        )
        try:
            returncode = proc.wait()
        except KeyboardInterrupt:
            click.echo("Aborting: sending SIGKILL to inspect.", err=True)
            proc.kill()
            returncode = proc.wait()

    if interrupted and returncode in (-signal.SIGINT, 128 + signal.SIGINT):
        raise click.Abort()

    if returncode != 0:
        raise click.ClickException(
            f"inspect eval-set failed while running {config_path}"
        )

    ctx = click.get_current_context()
    click.echo(
        f"You can now run '{ctx.parent.info_name if ctx.parent else 'cli'} score {log_dir}' to score the results"
    )


class _AstabenchCLI(click.Group):
    """CLI group that serves ``eval`` natively and defers scoring/leaderboard
    commands to ``agent-eval`` (lazily imported; only needed when the
    ``[score]`` extra is installed)."""

    def _agenteval_cli(self):
        from agenteval.cli import cli as ae_cli

        return ae_cli

    def list_commands(self, ctx):
        names = {"eval"}
        try:
            names |= set(self._agenteval_cli().list_commands(ctx))
        except ModuleNotFoundError:
            pass
        return sorted(names)

    def get_command(self, ctx, name):
        if name == "eval":
            return eval_command
        try:
            ae_cli = self._agenteval_cli()
        except ModuleNotFoundError as e:
            raise click.ClickException(
                f"The '{name}' command requires the scoring dependencies. "
                "Install them with: pip install 'astabench[score]'"
            ) from e
        return ae_cli.get_command(ctx, name)


cli = _AstabenchCLI(
    name="astabench",
    help="astabench: run (`eval`) and score (`score`) agent evaluations.",
)
cli.add_command(eval_command)


if __name__ == "__main__":
    cli()
