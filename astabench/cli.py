import importlib.resources
import os

import click
from agenteval.cli import cli as ae_cli
from agenteval.cli import eval_command, score_command, edit_command, check_command, convert_command

DEFAULT_CONFIG = "v1.0.0"
SPLIT_NAMES = ["validation", "test"]


def get_config_path(config_name):
    """Get the path to a config file in the package's config directory."""
    with importlib.resources.path("astabench.config", f"{config_name}.yml") as path:
        return os.path.abspath(path)


def add_astabench_interventions_registry(ctx, param, value):
    return tuple(set(value).union({"astabench:astabench.interventions"}))


for cmd in (eval_command, score_command):
    for param in cmd.params:
        if isinstance(param, click.Option) and param.name == "config_path":
            param.default = get_config_path(DEFAULT_CONFIG)
            param.show_default = True
            param.hidden = True
        elif isinstance(param, click.Option) and param.name == "split":
            param.type = click.Choice(SPLIT_NAMES, case_sensitive=False)


for cmd in (edit_command, convert_command, check_command):
    for param in cmd.params:
        if isinstance(param, click.Option) and param.name == "registry":
            param.callback = add_astabench_interventions_registry


# Export the CLI
cli = ae_cli
