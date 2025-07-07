import json
import logging
import os

import pytest
from inspect_ai.tool import ContentText, ToolDef, ToolError

from astabench.tools.s2_api_tools import make_s2api_toolsource

logger = logging.getLogger(__name__)


def parse_tool_result(result):
    """Parse tool result from list[ContentText] to dict/list."""
    if isinstance(result, list) and result and isinstance(result[0], ContentText):
        if len(result) == 1:
            # Single ContentText, parse its JSON
            return json.loads(result[0].text)
        else:
            # Multiple ContentText items, parse each one
            return [json.loads(item.text) for item in result]
    return result


def find_tool_error(exc_group):
    """Find ToolError in nested ExceptionGroup."""
    for exc in exc_group.exceptions:
        if isinstance(exc, ToolError):
            return exc
        elif isinstance(exc, ExceptionGroup):
            found = find_tool_error(exc)
            if found:
                return found
    return None


@pytest.mark.skipif(not os.getenv("S2_API_KEY"), reason="S2_API_KEY is missing.")
@pytest.mark.asyncio
async def test_get_paper_with_arxiv_id():
    """Test get_paper with arXiv ID format."""
    tool_source = make_s2api_toolsource()
    tools = await tool_source.tools()
    tools_dict = {ToolDef(tool).name: tool for tool in tools}

    get_paper = tools_dict["get_paper"]
    result = await get_paper(
        paper_id="arxiv:2409.11363", fields="title,authors,abstract"
    )

    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], ContentText)
    data = parse_tool_result(result)
    # Single paper endpoints return the paper data directly
    assert "title" in data
    assert "authors" in data
    assert "abstract" in data
    logger.info(f"get_paper with arxiv ID result: {data}")


@pytest.mark.skipif(not os.getenv("S2_API_KEY"), reason="S2_API_KEY is missing.")
@pytest.mark.asyncio
async def test_get_paper_with_corpus_id():
    """Test get_paper with Semantic Scholar corpus ID."""
    tool_source = make_s2api_toolsource()
    tools = await tool_source.tools()
    tools_dict = {ToolDef(tool).name: tool for tool in tools}

    get_paper = tools_dict["get_paper"]
    result = await get_paper(paper_id="4c913d59d150fe7581386b87dfd9f90448a9adee")

    assert result is not None
    assert isinstance(result, list)
    assert len(result) >= 1  # May return multiple ContentText items
    assert isinstance(result[0], ContentText)
    data = parse_tool_result(result)

    assert "paperId" in data or "title" in data

    logger.info(f"get_paper with corpus ID result: {data}")


@pytest.mark.skipif(not os.getenv("S2_API_KEY"), reason="S2_API_KEY is missing.")
@pytest.mark.asyncio
async def test_get_paper_batch():
    """Test get_paper_batch with multiple paper IDs."""
    tool_source = make_s2api_toolsource()
    tools = await tool_source.tools()
    tools_dict = {ToolDef(tool).name: tool for tool in tools}

    get_paper_batch = tools_dict["get_paper_batch"]
    result = await get_paper_batch(
        ids=["arxiv:2409.11363", "arxiv:2409.11364"], fields="title"
    )

    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 2  # Should return 2 ContentText items for 2 IDs
    assert isinstance(result[0], ContentText)
    data = parse_tool_result(result)

    assert isinstance(data, list)
    assert len(data) == 2
    for paper in data:
        assert "title" in paper

    logger.info(f"get_paper_batch result: {data}")


@pytest.mark.skipif(not os.getenv("S2_API_KEY"), reason="S2_API_KEY is missing.")
@pytest.mark.asyncio
async def test_get_citations():
    """Test get_citations for a paper."""
    tool_source = make_s2api_toolsource()
    tools = await tool_source.tools()
    tools_dict = {ToolDef(tool).name: tool for tool in tools}

    get_citations = tools_dict["get_citations"]
    result = await get_citations(paper_id="arxiv:2409.11363", fields="title")

    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], ContentText)
    data = parse_tool_result(result)
    # Citations endpoint returns {'data': [...], 'offset': ..., 'total': ...}
    assert "data" in data
    assert isinstance(data["data"], list)
    logger.info(f"get_citations result count: {len(data['data'])}")


@pytest.mark.skipif(not os.getenv("S2_API_KEY"), reason="S2_API_KEY is missing.")
@pytest.mark.asyncio
async def test_search_authors_by_name():
    """Test search_authors_by_name."""
    tool_source = make_s2api_toolsource()
    tools = await tool_source.tools()
    tools_dict = {ToolDef(tool).name: tool for tool in tools}

    search_authors = tools_dict["search_authors_by_name"]
    result = await search_authors(
        name="Jonathan Bragg", fields="name,authorId,paperCount"
    )

    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], ContentText)
    data = parse_tool_result(result)
    # Search endpoints return {'data': [...]}
    assert "data" in data
    assert isinstance(data["data"], list)
    assert len(data["data"]) > 0
    for author in data["data"]:
        assert "name" in author
        assert "authorId" in author
        assert "paperCount" in author
    logger.info(f"search_authors_by_name result: {data}")


@pytest.mark.skipif(not os.getenv("S2_API_KEY"), reason="S2_API_KEY is missing.")
@pytest.mark.asyncio
async def test_get_author_papers():
    """Test get_author_papers."""
    tool_source = make_s2api_toolsource()
    tools = await tool_source.tools()
    tools_dict = {ToolDef(tool).name: tool for tool in tools}

    get_author_papers = tools_dict["get_author_papers"]
    result = await get_author_papers(
        author_id="2274102473",
        fields="title",  # Only request title field to match original test
    )

    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], ContentText)
    data = parse_tool_result(result)
    # Author papers endpoint returns {'data': [...], 'offset': ..., 'total': ...}
    assert "data" in data
    assert isinstance(data["data"], list)
    assert len(data["data"]) > 0
    for paper in data["data"]:
        assert "title" in paper
    logger.info(f"get_author_papers result count: {len(data['data'])}")


@pytest.mark.skipif(not os.getenv("S2_API_KEY"), reason="S2_API_KEY is missing.")
@pytest.mark.asyncio
async def test_search_papers_by_relevance():
    """Test search_papers_by_relevance."""
    tool_source = make_s2api_toolsource()
    tools = await tool_source.tools()
    tools_dict = {ToolDef(tool).name: tool for tool in tools}

    search_papers = tools_dict["search_papers_by_relevance"]
    result = await search_papers(
        keyword="deep learning", fields="title,authors,abstract"
    )

    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], ContentText)
    data = parse_tool_result(result)
    # Search endpoints return {'data': [...], 'offset': ..., 'total': ...}
    assert "data" in data
    assert isinstance(data["data"], list)
    assert len(data["data"]) > 0
    for paper in data["data"]:
        assert "title" in paper
        assert "authors" in paper
        assert "abstract" in paper
    logger.info(f"search_papers_by_relevance result count: {len(data['data'])}")


@pytest.mark.skipif(not os.getenv("S2_API_KEY"), reason="S2_API_KEY is missing.")
@pytest.mark.asyncio
async def test_search_paper_by_title():
    """Test search_paper_by_title."""
    tool_source = make_s2api_toolsource()
    tools = await tool_source.tools()
    tools_dict = {ToolDef(tool).name: tool for tool in tools}

    search_by_title = tools_dict["search_paper_by_title"]
    result = await search_by_title(
        title="CORE-Bench: Fostering the Credibility of Published Research Through a Computational Reproducibility Agent Benchmark",
        fields="isOpenAccess,title,fieldsOfStudy,url,venue,year,tldr",
    )

    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], ContentText)
    data = parse_tool_result(result)
    # Search endpoints return {'data': [...]}
    assert "data" in data
    assert isinstance(data["data"], list)
    assert len(data["data"]) > 0
    for paper in data["data"]:
        assert "title" in paper
    logger.info(f"search_paper_by_title result: {data}")


@pytest.mark.skipif(not os.getenv("S2_API_KEY"), reason="S2_API_KEY is missing.")
@pytest.mark.asyncio
async def test_snippet_search():
    """Test snippet_search."""
    tool_source = make_s2api_toolsource()
    tools = await tool_source.tools()
    tools_dict = {ToolDef(tool).name: tool for tool in tools}

    snippet_search = tools_dict["snippet_search"]
    result = await snippet_search(
        query="Tasks require interpreting both text and image output from code. The vision-based questions require extracting results from attributes of figures, graphs, plots, or PDF tables.",
        fields="title",
    )

    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], ContentText)
    data = parse_tool_result(result)
    # Snippet search returns {'data': [...], 'score': ...}
    assert "data" in data
    assert isinstance(data["data"], list)
    logger.info(f"snippet_search result count: {len(data['data'])}")


@pytest.mark.skipif(not os.getenv("S2_API_KEY"), reason="S2_API_KEY is missing.")
@pytest.mark.asyncio
async def test_error_invalid_fields_in_get_paper_batch():
    """Test get_paper_batch with invalid fields format (trailing comma)."""
    tool_source = make_s2api_toolsource()
    tools = await tool_source.tools()
    tools_dict = {ToolDef(tool).name: tool for tool in tools}

    get_paper_batch = tools_dict["get_paper_batch"]
    with pytest.raises(ExceptionGroup) as exc_info:
        await get_paper_batch(
            ids=["arxiv:2409.11363", "arxiv:2409.11364"],
            fields="title,",  # Invalid - trailing comma
        )

    tool_error = find_tool_error(exc_info.value)
    assert tool_error is not None, "Expected ToolError in ExceptionGroup"
    logger.info(f"Expected error for invalid fields: {tool_error}")


@pytest.mark.skipif(not os.getenv("S2_API_KEY"), reason="S2_API_KEY is missing.")
@pytest.mark.asyncio
async def test_error_invalid_fields_in_search_authors():
    """Test search_authors_by_name with invalid field names."""
    tool_source = make_s2api_toolsource()
    tools = await tool_source.tools()
    tools_dict = {ToolDef(tool).name: tool for tool in tools}

    search_authors = tools_dict["search_authors_by_name"]
    with pytest.raises(ExceptionGroup) as exc_info:
        await search_authors(
            name="Jonathan Bragg",
            fields="name,author_id,paper_count",  # Invalid field names
        )

    tool_error = find_tool_error(exc_info.value)
    assert tool_error is not None, "Expected ToolError in ExceptionGroup"
    logger.info(f"Expected error for invalid fields: {tool_error}")


@pytest.mark.skipif(not os.getenv("S2_API_KEY"), reason="S2_API_KEY is missing.")
@pytest.mark.asyncio
async def test_error_wrong_parameter_name_in_get_paper():
    """Test get_paper with wrong parameter name."""
    tool_source = make_s2api_toolsource()
    tools = await tool_source.tools()
    tools_dict = {ToolDef(tool).name: tool for tool in tools}

    get_paper = tools_dict["get_paper"]
    with pytest.raises(ExceptionGroup) as exc_info:
        await get_paper(
            paperId="4c913d59d150fe7581386b87dfd9f90448a9adee"  # Wrong param name
        )

    tool_error = find_tool_error(exc_info.value)
    assert tool_error is not None, "Expected ToolError in ExceptionGroup"
    assert "paperId" in str(tool_error)
    logger.info(f"Expected error for wrong parameter: {tool_error}")


@pytest.mark.skipif(not os.getenv("S2_API_KEY"), reason="S2_API_KEY is missing.")
@pytest.mark.asyncio
async def test_error_invalid_paper_id_type():
    """Test get_paper with invalid paper ID type (integer instead of string)."""
    tool_source = make_s2api_toolsource()
    tools = await tool_source.tools()
    tools_dict = {ToolDef(tool).name: tool for tool in tools}

    get_paper = tools_dict["get_paper"]
    with pytest.raises(ExceptionGroup) as exc_info:
        await get_paper(paper_id=7857660)  # Invalid - should be string

    tool_error = find_tool_error(exc_info.value)
    assert tool_error is not None, "Expected ToolError in ExceptionGroup"
    logger.info(f"Expected error for invalid ID type: {tool_error}")


@pytest.mark.skipif(not os.getenv("S2_API_KEY"), reason="S2_API_KEY is missing.")
@pytest.mark.asyncio
async def test_make_s2api_toolsource():
    """Test the make_s2api_toolsource function."""
    tool_source = make_s2api_toolsource()
    tools = await tool_source.tools()

    assert tools is not None
    assert len(tools) > 0

    # Check that expected tools are present
    tool_names = {ToolDef(tool).name for tool in tools}
    expected_tools = {
        "get_paper",
        "get_paper_batch",
        "get_citations",
        "search_authors_by_name",
        "get_author_papers",
        "search_papers_by_relevance",
        "search_paper_by_title",
        "snippet_search",
    }

    for expected in expected_tools:
        assert expected in tool_names, f"Expected tool '{expected}' not found"

    logger.info(f"Found {len(tools)} tools: {tool_names}")


@pytest.mark.skipif(not os.getenv("S2_API_KEY"), reason="S2_API_KEY is missing.")
@pytest.mark.asyncio
async def test_make_s2api_toolsource_without_api_key():
    """Test make_s2api_toolsource raises error when API key is not provided."""
    # Temporarily remove the API key from environment
    original_key = os.environ.pop("S2_API_KEY", None)

    try:
        with pytest.raises(ValueError) as exc_info:
            make_s2api_toolsource(api_key=None)
        assert "S2_API_KEY is not set" in str(exc_info.value)
    finally:
        # Restore the API key if it existed
        if original_key:
            os.environ["S2_API_KEY"] = original_key
