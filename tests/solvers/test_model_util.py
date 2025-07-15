"""Tests for the model_util module."""

import pytest
from inspect_ai.log import ModelEvent, transcript
from inspect_ai.model import ModelUsage
from inspect_ai.model._model import init_model_usage, model_usage

from astabench.solvers.model_util import (
    normalize_model_name,
    record_model_usage_with_inspect,
)


def test_already_normalized():
    """Test that already normalized names are returned as-is."""
    # Known model with provider
    assert "openai/gpt-4o" == normalize_model_name("openai/gpt-4o")

    # Provider with two slashes
    assert (
        "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct"
        == normalize_model_name("together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct")
    )

    # Provider with two slashes with Inspect form instead of litellm form
    assert (
        normalize_model_name("together/meta-llama/Llama-4-Scout-17B-16E-Instruct")
        == "together/meta-llama/Llama-4-Scout-17B-16E-Instruct"
    )

    # Unknown model with provider-like prefix
    assert "acme/unknown-model" == normalize_model_name("acme/unknown-model")


def test_known_models():
    """Test normalization of common models recognized by litellm."""
    # Test OpenAI model
    assert normalize_model_name("gpt-4o") == "openai/gpt-4o"

    # Test Anthropic model
    assert (
        normalize_model_name("claude-3-opus-20240229")
        == "anthropic/claude-3-opus-20240229"
    )


def test_unknown_model():
    """Test handling of unrecognized models."""
    # A completely made-up model that shouldn't match any provider
    result = normalize_model_name("completely-fictional-model-12345")
    # The function should return the original name for unrecognized models
    assert result == "completely-fictional-model-12345"


def test_invalid_name():
    invalid_names = [
        "/model",
        "provider/",
        "",
        " ",
        " / ",
        "provider//model",
    ]

    for name in invalid_names:
        with pytest.raises(ValueError):
            normalize_model_name(name)


def test_record_model_usage_with_inspect():
    init_model_usage()

    usage = ModelUsage(
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
    )
    record_model_usage_with_inspect("gpt-4o", usage, normalize_name=True)

    assert model_usage() == {"openai/gpt-4o": usage}

    ev = transcript().events[-1]
    assert isinstance(ev, ModelEvent)
    assert ev.model == "openai/gpt-4o"
