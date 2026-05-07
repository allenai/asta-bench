import os
import sys
import importlib


def test_registry_import_without_model_keys(monkeypatch):
    # Remove any environment variables that look like credentials
    for key in list(os.environ):
        if any(token in key for token in ["KEY", "TOKEN", "SECRET"]):
            monkeypatch.delenv(key, raising=False)

    # Ensure a fresh import
    sys.modules.pop("astabench.evals._registry", None)

    import astabench.evals._registry  # noqa: F401
