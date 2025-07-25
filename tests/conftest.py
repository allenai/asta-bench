import os

import pytest


def pytest_collection_modifyitems(config, items):
    """Automatically skip tests marked with s2_api_request when S2_API_KEY is not set."""
    if not os.getenv("S2_API_KEY"):
        skip_s2 = pytest.mark.skip(reason="S2_API_KEY is not set")
        for item in items:
            if "s2_api_request" in item.keywords:
                item.add_marker(skip_s2)
