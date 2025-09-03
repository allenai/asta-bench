import os

import pytest


def pytest_collection_modifyitems(config, items):
    """Automatically skip tests marked with asta_tool_api_request when ASTA_TOOL_KEY is not set."""
    if not os.getenv("ASTA_TOOL_KEY"):
        skip_asta = pytest.mark.skip(reason="ASTA_TOOL_KEY is not set")
        for item in items:
            if "asta_tool_api_request" in item.keywords:
                item.add_marker(skip_asta)
