"""SEA-AD Agent solver using Claude Agent SDK.

Wraps claude_agent_sdk.query() as an InspectAI solver, connecting to Asta
MCP tools for academic paper search and retrieval.
"""

import logging
import os

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)
from inspect_ai.model import ModelOutput, ModelUsage
from inspect_ai.solver import Generate, Solver, TaskState, solver

from astabench.util.model import record_model_usage_with_inspect

logger = logging.getLogger(__name__)

ASTA_MCP_URL = "https://asta-tools.apps.allenai.org/mcp/v1"

ALLOWED_TOOLS = [
    "mcp__asta__search_papers_by_relevance",
    "mcp__asta__search_paper_by_title",
    "mcp__asta__snippet_search",
    "mcp__asta__get_paper",
    "mcp__asta__get_paper_batch",
    "mcp__asta__get_citations",
    "mcp__asta__get_author_papers",
    "mcp__asta__search_authors_by_name",
]

SYSTEM_PROMPT = """\
You are a research assistant with access to Semantic Scholar tools via MCP. \
Your job is to find relevant academic papers and synthesize well-cited answers \
to research questions.

## Search strategy
- Decompose complex questions into targeted searches.
- Try multiple search strategies with different keyword variants.
- If initial results are sparse, broaden or rephrase your queries.
- Use snippet_search for finding specific claims within papers.
- Use get_paper to fetch full details (abstract, authors, year) for promising hits.

## Response format
Follow the formatting instructions in the user's prompt exactly. \
Return only the requested output — no commentary, no markdown fences."""


def _asta_mcp_config() -> dict:
    headers: dict[str, str] = {}
    api_key = os.environ.get("ASTA_TOOL_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    return {
        "type": "http",
        "url": ASTA_MCP_URL,
        "headers": headers,
    }


@solver
def sea_ad_agent(
    model: str = "claude-sonnet-4-20250514",
    max_turns: int = 15,
    max_budget_usd: float = 5.0,
) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        system_prompt = (
            state.metadata.get("system_prompt") if state.metadata else None
        ) or SYSTEM_PROMPT

        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            model=model,
            max_turns=max_turns,
            max_budget_usd=max_budget_usd,
            mcp_servers={"asta": _asta_mcp_config()},
            allowed_tools=ALLOWED_TOOLS,
            permission_mode="bypassPermissions",
        )

        parts: list[str] = []
        result_message: ResultMessage | None = None

        async for message in query(prompt=state.input_text, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        parts.append(block.text)
            elif isinstance(message, ResultMessage):
                result_message = message
                if message.is_error:
                    logger.error(
                        "Agent error (subtype=%s): %s",
                        message.subtype,
                        message.result,
                    )

        state.output = ModelOutput.from_content(
            model=model, content="".join(parts)
        )

        if result_message:
            if result_message.usage:
                usage = result_message.usage
                try:
                    model_usage = ModelUsage(
                        input_tokens=usage.get("input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0),
                        total_tokens=usage.get("input_tokens", 0)
                        + usage.get("output_tokens", 0),
                    )
                    record_model_usage_with_inspect(
                        f"anthropic/{model}", model_usage, normalize_name=False
                    )
                except (ValueError, TypeError):
                    logger.warning(
                        "Failed to record model usage: %s", usage, exc_info=True
                    )
            logger.info(
                "Agent completed: turns=%d cost=$%s duration=%dms",
                result_message.num_turns,
                result_message.total_cost_usd,
                result_message.duration_ms,
            )

        return state

    return solve
