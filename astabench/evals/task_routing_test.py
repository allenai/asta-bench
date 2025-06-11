import json
from typing import Any

import numpy as np
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset
from inspect_ai.log import ToolEvent, transcript
from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)
from inspect_ai.scorer import INCORRECT, Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState, use_tools
from inspect_ai.tool import Tool, ToolDef, tool

from astabench.evals.arxivdigestables.task import arxivdigestables
from astabench.evals.sqa.task import sqa
from astabench.evals.discoverybench.task import discoverybench
from astabench.evals.inspect_eval_wrappers import core_bench_test, ds1000, gpqa_diamond
from astabench.evals.labbench.litqa2 import litqa2
from astabench.evals.paper_finder.task import paper_finder
from astabench.evals.utils import not_implemented_solver

TOOL_MOCK_MODEL_NAME = "openai/gpt-4o-mini"


async def generate_str(model, *args, **kwargs) -> str:
    resp = await model.generate(*args, **kwargs)
    if not isinstance(resp.choices[0].message.content, str):
        raise ValueError("Expected string response from model")
    return resp.choices[0].message.content


@tool
def sqa_tool() -> Tool:
    async def run_sqa(query: str) -> str:
        """Answer a question based on scientific literature.

        Args:
            query: The question to answer.
        """
        model = get_model(TOOL_MOCK_MODEL_NAME)
        return await generate_str(
            model,
            [
                ChatMessageSystem(
                    content="You are an expert scientific report writer. Write a short answer to the user query and include citations to scientific papers."
                ),
                ChatMessageUser(content=query),
            ],
            config=GenerateConfig(max_tokens=1000),
        )

    return run_sqa


@tool
def tables_tool() -> Tool:
    async def run_tables(query: str) -> str:
        """Generate a paper comparison table based on a query.

        Args:
            query: The question to answer.
        """
        model = get_model(TOOL_MOCK_MODEL_NAME)
        return await generate_str(
            model,
            [
                ChatMessageSystem(
                    content="You are simulating an expert table generator.  Generate a table comparing the papers related to the user query.  You may make up the table content if you do not know about the papers."
                ),
                ChatMessageUser(content=query),
            ],
            config=GenerateConfig(max_tokens=1000),
        )

    return run_tables


@tool
def paper_finder_tool() -> Tool:
    async def run_paper_finder(query: str) -> str:
        """Find scientific papers based on a query.

        Args:
            query: The query to search for.
        """
        model = get_model(TOOL_MOCK_MODEL_NAME)
        return await generate_str(
            model,
            [
                ChatMessageSystem(
                    content="You are pretending to be a paper search engine. Give a list of 1-10 papers related to the user query."
                ),
                ChatMessageUser(content=query),
            ],
            config=GenerateConfig(max_tokens=1000),
        )

    return run_paper_finder


@tool
def data_voyager_tool() -> Tool:
    async def run_dv(query: str) -> str:
        """Do data anaylsis based on a query.

        Args:
            query: The data analysis task description.
        """
        model = get_model(TOOL_MOCK_MODEL_NAME)
        return await generate_str(
            model,
            [
                ChatMessageSystem(
                    content="You are pretending to be an expert data analyst. Make up an answer to the given data analysis task."
                ),
                ChatMessageUser(content=query),
            ],
            config=GenerateConfig(max_tokens=1000),
        )

    return run_dv


@tool
def experiment_runner_tool() -> Tool:
    async def run_execagent(query: str) -> str:
        """Run an experiment based on a query.

        Args:
            query: The experiment description.
        """
        model = get_model(TOOL_MOCK_MODEL_NAME)
        return await generate_str(
            model,
            [
                ChatMessageSystem(
                    content="You are pretending to be an experiment runner. Write a short description of the experiment you would run based on the user query and make up some output."
                ),
                ChatMessageUser(content=query),
            ],
            config=GenerateConfig(max_tokens=1000),
        )

    return run_execagent


@scorer(metrics=[accuracy(), stderr()])
def check_tool_calls(required_tools: list[Tool]) -> Scorer:
    """Scorer to check if the solution is correct and used the right tool."""

    async def score(state: TaskState, target: Target) -> Score:
        # Check if necessary tools are present; if not, is suggests that the
        # solver overwrote the tool list rather than appending
        actual_calls = []
        for req_tool in required_tools:
            tool_name = ToolDef(sqa_tool()).name
            if not any(ToolDef(tool).name == tool_name for tool in state.tools):
                return Score(
                    value=INCORRECT,
                    answer="",
                    explanation=f"No {tool_name} tool found in solver state",
                )

        # Get calls to the tools
        for event in transcript().events:
            if not isinstance(event, ToolEvent):
                continue
            actual_calls.append(
                {"function": event.function, "arguments": event.arguments}
            )

        expected_calls = json.loads(target.text)["expected_calls"]

        n_recall = 0
        for expected_call in expected_calls:
            found = False
            for actual_call in actual_calls:
                if actual_call["function"] == expected_call["function"] and all(
                    actual_call["arguments"][k] == v  # type: ignore
                    for k, v in expected_call["arguments"].items()
                ):
                    found = True
                    break
            if found:
                n_recall += 1

        recall = n_recall / len(expected_calls)

        return Score(
            value=recall,
            answer=f"Answer: {state.output.completion}\n\Tool calls: {actual_calls}",
            explanation=f"Expected calls: {expected_calls}",
        )

    return score


@task
def tool_dispatch_task():

    tools = []
    combined_samples = []

    # These don't fit in cleanly right now; might be split between categories
    # (e.g., litqa2 splits between paperfinder and sqa) or not have a valid
    # category
    excluded_tasks = [  # noqa
        ds1000,
        litqa2,
        gpqa_diamond,
    ]

    tasks: dict[str, dict[str, Any]] = {
        "sqa": {"tasks": [sqa()], "tool": sqa_tool()},
        "tables": {"tasks": [arxivdigestables()], "tool": tables_tool()},
        "paper_finder": {"tasks": [paper_finder()], "tool": paper_finder_tool()},
        "data_voyager": {"tasks": [discoverybench()], "tool": data_voyager_tool()},
        "experiment_runner": {
            "tasks": [core_bench_test()],
            "tool": experiment_runner_tool(),
        },
    }

    samples_per_task = 10
    rng = np.random.default_rng(0)

    for task_name, tasks_obj in tasks.items():
        tools.append(tasks_obj["tool"])
        for task_obj in tasks_obj["tasks"]:
            dset_obj = task_obj.dataset
            for sample in rng.choice(
                dset_obj.samples,
                min(samples_per_task, len(dset_obj.samples)),
                replace=False,
            ):
                sample.target = json.dumps(
                    {
                        "expected_calls": [
                            {
                                "function": ToolDef(tasks_obj["tool"]).name,
                                "arguments": {},
                            }
                        ]
                    }
                )
                combined_samples.append(sample)

    return Task(
        dataset=MemoryDataset(samples=combined_samples),
        setup=use_tools(*tools),
        # Solver that just tells the user to specify a real solver via cli
        solver=not_implemented_solver(),
        scorer=check_tool_calls(tools),
    )
