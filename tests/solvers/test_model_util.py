"""Tests for the model_util module."""

import pytest
from inspect_ai.log import ModelEvent, transcript
from inspect_ai.model import ModelUsage
from inspect_ai.model._model import init_model_usage, model_usage

from astabench.solvers.model_util import _is_valid_model_usage
from astabench.solvers.model_util import logger as model_util_logger
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


def test_is_valid_model_usage():
    """Test validation of ModelUsage objects."""
    # Valid usage
    is_valid, error = _is_valid_model_usage(
        ModelUsage(
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
        )
    )
    assert is_valid is True
    assert error is None

    # Missing required fields
    is_valid, error = _is_valid_model_usage(
        ModelUsage(output_tokens=20, total_tokens=30)
    )
    assert is_valid is False

    is_valid, error = _is_valid_model_usage(
        ModelUsage(input_tokens=10, total_tokens=30)
    )
    assert is_valid is False

    is_valid, error = _is_valid_model_usage(
        ModelUsage(input_tokens=10, output_tokens=20)
    )
    assert is_valid is False

    # Negative token counts
    is_valid, error = _is_valid_model_usage(
        ModelUsage(
            input_tokens=-10,
            output_tokens=20,
            total_tokens=30,
        )
    )
    assert is_valid is False
    assert "negative" in error.lower()

    is_valid, error = _is_valid_model_usage(
        ModelUsage(
            input_tokens=10,
            output_tokens=-20,
            total_tokens=30,
        )
    )
    assert is_valid is False
    assert "negative" in error.lower()

    is_valid, error = _is_valid_model_usage(
        ModelUsage(
            input_tokens=10,
            output_tokens=20,
            total_tokens=-30,
        )
    )
    assert is_valid is False
    assert "negative" in error.lower()

    # Total tokens greater than sum of components
    is_valid, error = _is_valid_model_usage(
        ModelUsage(
            input_tokens=10,
            output_tokens=20,
            total_tokens=50,  # Greater than 10 + 20
        )
    )
    assert is_valid is False

    # Valid usage with additional fields (e.g., reasoning_tokens)
    is_valid, error = _is_valid_model_usage(
        ModelUsage(
            input_tokens=10,
            output_tokens=20,
            total_tokens=35,
            reasoning_tokens=5,
        )
    )
    assert is_valid is True
    assert error is None

    # Edge case: all tokens are 1 (valid)
    is_valid, error = _is_valid_model_usage(
        ModelUsage(
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
        )
    )
    assert is_valid is True
    assert error is None

    # Different libs name things differently so it's easy to use wrong field
    # names (e.g. prompt_tokens/completion_tokens instead of
    # input_tokens/output_tokens)
    try:
        wrong_field_names = ModelUsage(
            prompt_tokens=10,  # Wrong field name
            completion_tokens=20,  # Wrong field name
            total_tokens=30,
        )
    except (TypeError, ValueError):
        # If ModelUsage construction itself fails due to wrong field names, that's also acceptable
        pass

    # If construction succeeds, validation should catch the missing fields
    is_valid, error = _is_valid_model_usage(wrong_field_names)
    assert is_valid is False


def test_record_model_usage_with_inspect_validation():
    """Test validation in record_model_usage_with_inspect."""
    init_model_usage()

    # Valid usage should work
    record_model_usage_with_inspect(
        "gpt-4o",
        ModelUsage(
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
        ),
        normalize_name=True,
    )

    # Invalid usage should raise ValueError by default
    invalid_usage = ModelUsage(
        input_tokens=10,
        output_tokens=20,
        total_tokens=50,  # Invalid: greater than sum
    )
    with pytest.raises(ValueError) as excinfo:
        record_model_usage_with_inspect("gpt-4o", invalid_usage)
        assert "invalid modelusage" in str(excinfo.value).lower()

    # With allow_invalid=True, should log warning instead of raising
    # We'll capture the log output to verify the warning is logged
    from unittest.mock import patch

    with patch.object(model_util_logger, "warning") as mock_warning:
        record_model_usage_with_inspect("gpt-4o", invalid_usage, allow_invalid=True)
        mock_warning.assert_called_once()
        warning_message = mock_warning.call_args[0][0]
        assert "invalid modelusage" in warning_message.lower()
