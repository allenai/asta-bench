import re

from timestamped_run import ExecCmd, run_exp_cmds

# Number of output tokens to generate in a single call
MAX_OUTPUT_TOKENS = 4096

# 64000 max generated tokens (thinking+output) minus max output tokens
SONNET_MAX_REASONING = 64000 - MAX_OUTPUT_TOKENS - 1
OPUS_MAX_REASONING = 32000 - MAX_OUTPUT_TOKENS - 1

DEFAULT_SPLIT = "test"


def prepare_norabench_cmd(
    exp_args: list[str],
    single_task: bool = False,
    split: str | None = None,
    name: str | None = None,
) -> ExecCmd:
    main_cli = "inspect" if single_task else "astabench"
    if not single_task and split is None:
        raise ValueError("split must be specified for multi-task runs")
    extra_args = (
        []
        if single_task
        else [
            "--split",
            split,
            # We should be able to use git now that we're on opensource
            # "--ignore-git",
            "--retry-attempts",
            "1",
        ]
    )

    cmd = ["unbuffer", "uv", "run", main_cli, "eval"]

    cmd.extend(extra_args)

    # Add the custom args last; I think later args override earlier ones though
    # I'm not actually sure.  In any case, we don't expect any conflicts in
    # this simple script
    cmd.extend(exp_args)

    return ExecCmd(cmd=[str(x) for x in cmd], skip=False, name_prefix=name)


def get_general_args(parallelism: int = 16):
    return [
        "--sandbox",
        "docker:/astabench/astabench/solvers/sandbox_util/sandbox_compose.yaml",
        "--max-retries",
        "5",
        "--max-samples",
        str(parallelism),
        "--max-connections",
        str(parallelism),
        "--max-sandboxes",
        str(parallelism),
        "--max-subprocesses",
        # 2 slots per sample (one for mcp server and one for sandbox.exec)
        str(2 * parallelism),
        "--retry-on-error=3",
        "--display",
        "plain",
        "--log-level",
        "info",
        "--fail-on-error",
        "0.3",
        "--log-dir=PLACEHOLDER_LOG_DIR",
    ]


def conf_to_solver_args(conf: dict[str, str | int | bool]) -> list[str]:
    sargs = []
    for k, v in conf.items():
        if isinstance(v, bool):
            v = int(v)
        v = str(v)
        sargs.extend(["-S", f"{k}={v}"])
    return sargs


def model_args_slug(modelargs: list[str]) -> str:
    modelname = "unknown"
    otherargs_slug = ""
    skipnext = False
    for idx, arg in enumerate(modelargs):
        if skipnext:
            skipnext = False
            continue
        arg = str(arg)
        if arg == "--model":
            modelname = modelargs[idx + 1]
            skipnext = True
            continue
        m = re.match(r"--model=(?P<modelname>.*)$", arg)
        if m:
            modelname = m.group("modelname")
            continue

        otherargs_slug += arg.replace("--", "").replace("/", "_")

    if "/" in modelname:
        modelname = modelname.rpartition("/")[2]
    return f"{modelname}_{otherargs_slug}"


def main():
    """Run all experiments with the specified parameter combinations.

    Currently, everything here will run on the TEST set.
    """
    exp_cmds = []

    # With SUPER included, 8-parallel seems to be the best we can do; on a 64
    # GB instance we go down to about 10 GB free.  If we had more mem, we could
    # potentially ramp to around 16-parallel (at which point I've seen other
    # instability/rate-limit issues)
    general_args = get_general_args(parallelism=8)

    named_solverargs = {
        "react": [
            "--solver",
            "astabench/solvers/react/basic_agent.py@instantiated_basic_agent",
        ],
        "smolagents": [
            "--solver",
            "astabench/solvers/smolagents/agent.py@smolagents_coder",
        ],
        # TODO: Orchestrator, maybe magentic one solvers at some point
    }
    # ReACT is the primary since it generally did better than smolagents in our
    # previous exps
    MAIN_SOLVER_NAME = "react"
    SECONDARY_SOLVER_NAME = "smolagents"

    # Kyle says maybe 100 needed for some SUPER tasks, though empirically it
    # seems like general solvers usually don't hit even 50, either because they
    # don't need it or they aren't good enough to go longer
    react_steps = 100

    solver_tool_config = {
        "max_steps": react_steps,
        # Tasks will add when needed; prelims suggested that it probably isn't
        # helpful (possibly harmful) to add the search tools when the task
        # isn't search-related
        "with_search_tools": False,
        # Not enough support in val exps to justify enabling outside of the
        # coding tasks
        "with_stateful_python": False,
        # Defaulting to no editors because the old prelim exps suggested they aren't helpful
        "with_table_editor": False,
        "with_report_editor": False,
        # Defaulting to no thinking tool since there isn't enough evidence that
        # it helps; we could still do more ablations since there was some support
        # for it in some prelim exps, but findings are inconsistent
        "with_thinking_tool": False,
    }

    ###################################
    # Model sweep
    ###################################

    # Validation experiments with o4 suggested that "medium" had a performance
    # boost above "low" with barely higher cost, but "high" cost substantially
    # more without a huge boost (3% improvement for 3x the cost).
    openai_reasoning_level = "medium"

    # Validation experiments with sonnet-4 suggested that increasing the
    # reasoning cap beyond 4096 had a nearly 0 impact on cost but marginally
    # higher results.  So, no harm in maxing it out by default.
    anthropic_reasoning_tokens = SONNET_MAX_REASONING

    all_model_configs = [
        # Note: grok-3 had an inactivity timeout in its initial run; not clear
        # if it was the model or a spurious eval issue
        ["--model", "azureai/grok-3"],
        ["--model", "azureai/Mistral-Large-2411"],
        ["--model", "together/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"],
        ["--model", "together/deepseek-ai/DeepSeek-V3"],
        # Deepseek-R1 has known formatting-following issues and costs a lot,
        # maybe not a great one to use
        # ["--model", "together/deepseek-ai/DeepSeek-R1"],
        ["--model", "openai/gpt-4.1-nano-2025-04-14"],
        ["--model", "openai/gpt-4.1-2025-04-14"],
        ["--model", "anthropic/claude-3-5-haiku-20241022"],
        [
            "--model",
            "anthropic/claude-sonnet-4-20250514",
            "--reasoning-tokens",
            anthropic_reasoning_tokens,
        ],
        # Inspect doesn't seem to have settings for configuring reasoning on
        # Gemini; the defaults are pretty dynamic so probably fine
        ["--model", "google/gemini-2.5-pro-preview-06-05"],
        ["--model", "google/gemini-2.5-flash-preview-05-20"],
        # Depending on final results from reasoning ablations, we may need to revisit these; hopefully not though
        [
            "--model",
            "anthropic/claude-opus-4-20250514",
            "--reasoning-tokens",
            min(anthropic_reasoning_tokens, OPUS_MAX_REASONING),
        ],
        [
            "--model",
            "openai/o3-2025-04-16",
            "--reasoning-effort",
            openai_reasoning_level,
        ],
        # If something must be skipped, skip o3-pro; 11x the cost for 2%
        # improvement, based on validation results
        [
            "--model",
            "openai/o3-pro-2025-06-10",
            "--reasoning-effort",
            openai_reasoning_level,
        ],
        [
            "--model",
            "openai/o4-mini-2025-04-16",
            "--reasoning-effort",
            openai_reasoning_level,
        ],
        # Weak, but cheap and gives a good "low cost low perf" datapoint
        ["--model", "together/meta-llama/Llama-4-Scout-17B-16E-Instruct"],
        # TODO: Consider trying to bring qwen back, but we'd probably need to
        # think about history compression; not quite plug-and-play
    ]

    for modelargs in all_model_configs:
        exp_cmds.append(
            prepare_norabench_cmd(
                [
                    *general_args,
                    *modelargs,
                    *named_solverargs[MAIN_SOLVER_NAME],
                    *conf_to_solver_args(solver_tool_config),
                ],
                name=f"widesweep_{model_args_slug(modelargs)}",
                split=DEFAULT_SPLIT,
            )
        )

    # Smaller sweep for the secondary solver; just get a couple points to cover
    # the various providers; since the secondary is smolagents, we can't use
    # `o[34]` models; those break it (TODO: exclude dynamically)
    core_model_configs = [
        ["--model", "together/meta-llama/Llama-4-Scout-17B-16E-Instruct"],
        ["--model", "anthropic/claude-3-5-haiku-20241022"],
        ["--model", "openai/gpt-4.1-2025-04-14"],
        ["--model", "google/gemini-2.5-flash-preview-05-20"],
        [
            "--model",
            "anthropic/claude-sonnet-4-20250514",
            "--reasoning-tokens",
            anthropic_reasoning_tokens,
        ],
        [
            "--model",
            "anthropic/claude-opus-4-20250514",
            "--reasoning-tokens",
            min(anthropic_reasoning_tokens, OPUS_MAX_REASONING),
        ],
    ]

    for modelargs in core_model_configs:
        exp_cmds.append(
            prepare_norabench_cmd(
                [
                    *general_args,
                    *modelargs,
                    *named_solverargs[SECONDARY_SOLVER_NAME],
                    *conf_to_solver_args(solver_tool_config),
                ],
                name=f"smallsweep_smolagents_{model_args_slug(modelargs)}",
                split=DEFAULT_SPLIT,
            )
        )

    ###################################
    # Single-task solvers
    ###################################

    # Several aren't ready to run yet; let's just skip all of them for now
    _skip_from_idx = len(exp_cmds)

    # DV model isn't currently configurable
    exp_cmds.append(
        prepare_norabench_cmd(
            [
                *general_args,
                "--model",
                "mockllm/model",
                "--solver",
                "astabench/solvers/datavoyager/agent.py@datavoyager_solver",
                "astabench/discoverybench_test",
            ],
            single_task=True,
            name=f"dv_dbench_test_{model_args_slug(modelargs)}",
        )
    )

    for modelargs in core_model_configs + [["--model", "openai/gpt-4o-2024-08-06"]]:
        exp_cmds.append(
            prepare_norabench_cmd(
                [
                    *general_args,
                    *modelargs,
                    "-S",
                    f"max_tries={react_steps}",
                    "-S",
                    "json_output=1",
                    "--solver",
                    "astabench/solvers/code_agent/agent.py@code_agent",
                    "astabench/super_test",
                ],
                single_task=True,
                name=f"codeagent_super_test_{model_args_slug(modelargs)}",
            )
        )

    # Cannot do final run until
    # https://github.com/allenai/ai2-scholarqa-lib/pull/33 merges
    exp_cmds.append(
        prepare_norabench_cmd(
            [
                *general_args,
                "--model",
                "mockllm/model",
                "--solver",
                "astabench/solvers/arxivdigestables/asta_table_agent.py@tables_solver",
                "astabench/arxivdigestables_test",
            ],
            single_task=True,
            name=f"tablesolver_arxivdigestables_test_{model_args_slug(modelargs)}",
        )
    )

    for conf in exp_cmds[_skip_from_idx:]:
        conf.skip = True

    # We're on a tight schedule.  Let's skip everything except the crucial
    # models to start.
    urgent_model_names = [
        "together/meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "openai/gpt-4.1-2025-04-14",
        "google/gemini-2.5-flash-preview-05-20",
        "anthropic/claude-3-5-haiku-20241022",
        "anthropic/claude-sonnet-4-20250514",
        "openai/o3-2025-04-16",
    ]

    for conf in exp_cmds:
        if not any(model in arg for model in urgent_model_names for arg in conf.cmd):
            conf.skip = True

    run_exp_cmds(exp_cmds)


if __name__ == "__main__":
    main()
