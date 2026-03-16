from __future__ import annotations

from unittest.mock import patch

import pytest
from inspect_ai.model import ChatMessageUser, ModelName, ModelOutput
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState
from inspect_ai.tool import ToolDef

from astabench.evals.arxivdigestables.task import setup_snippet_tool


def _make_state() -> TaskState:
    model_name = ModelName("mockllm/model")
    return TaskState(
        model=model_name,
        sample_id="1",
        epoch=1,
        input="test input",
        messages=[ChatMessageUser(content="test input")],
        target=Target(target=""),
        choices=None,
        output=ModelOutput.from_content(model=str(model_name), content=""),
        metadata={"corpus_ids": ["1", "2"]},
    )


async def _fake_generate(*args, **kwargs):
    raise AssertionError("generate should not be called in this test")


@pytest.mark.asyncio
async def test_setup_snippet_tool_calls_snippet_search_with_keyword_args():
    call_log: dict[str, object] = {}

    async def fake_snippet_tool(
        query: str, limit: int, paper_ids: list[str] | None = None
    ):
        call_log["args"] = ()
        call_log["kwargs"] = {
            "query": query,
            "limit": limit,
            "paper_ids": paper_ids,
        }
        return {"ok": True}

    snippet_tool = ToolDef(
        name="snippet_search",
        description="snippet_search",
        tool=fake_snippet_tool,
        parameters={
            "query": "query",
            "limit": "limit",
            "paper_ids": "paper_ids",
        },
    ).as_tool()

    def fake_use_tools(tool):
        async def runner(state, generate):
            await tool(query="needle", limit=3, paper_ids="CorpusId:1,CorpusId:999")
            return state

        return runner

    with patch(
        "astabench.evals.arxivdigestables.task.make_asta_mcp_tools",
        return_value=[snippet_tool],
    ), patch(
        "astabench.evals.arxivdigestables.task.use_tools",
        side_effect=fake_use_tools,
    ):
        solver = setup_snippet_tool()
        await solver(_make_state(), _fake_generate)

    assert call_log["args"] == ()
    assert call_log["kwargs"] == {
        "query": "needle",
        "limit": 3,
        "paper_ids": ["CorpusId:1"],
    }
