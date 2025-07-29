"""Tests for ToolsetConfig."""

import pytest
from inspect_ai.tool import ToolDef
from pydantic import ValidationError

from astabench.tools import ToolsetConfig


class TestToolsetConfig:
    """Test ToolsetConfig validation and functionality."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ToolsetConfig()
        assert config.with_search_tools is False
        assert config.with_stateful_python is False
        assert config.with_report_editor is False
        assert config.with_table_editor is False
        assert config.with_thinking_tool is False
        assert config.with_editor_submit is False

    def test_basic_config_creation(self):
        """Test creating config with various settings."""
        config = ToolsetConfig(
            with_search_tools=False,
            with_stateful_python=True,
            with_report_editor=True,
        )
        assert config.with_search_tools is False
        assert config.with_stateful_python is True
        assert config.with_report_editor is True
        assert config.with_table_editor is False

    def test_editor_submit_validation_success(self):
        """Test that with_editor_submit works when at least one editor is enabled."""
        # Should work with report editor
        config = ToolsetConfig(with_report_editor=True, with_editor_submit=True)
        assert config.with_editor_submit is True

        # Should work with table editor
        config = ToolsetConfig(with_table_editor=True, with_editor_submit=True)
        assert config.with_editor_submit is True

        # Should work with both editors
        config = ToolsetConfig(
            with_report_editor=True, with_table_editor=True, with_editor_submit=True
        )
        assert config.with_editor_submit is True

    def test_editor_submit_validation_failure(self):
        """Test that with_editor_submit fails when no editors are enabled."""
        with pytest.raises(ValueError, match="requires at least one editor tool"):
            ToolsetConfig(with_editor_submit=True)

        with pytest.raises(ValueError, match="requires at least one editor tool"):
            ToolsetConfig(
                with_search_tools=True,
                with_stateful_python=True,
                with_thinking_tool=True,
                with_editor_submit=True,
            )

    def test_extra_fields_forbidden(self):
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError):
            ToolsetConfig(invalid_field=True)

        with pytest.raises(ValidationError):
            ToolsetConfig.model_validate(
                {"with_search_tools": True, "typo_field": True}
            )

    def test_model_validate(self):
        """Test model_validate method with dict input."""
        config_dict = {
            "with_search_tools": False,
            "with_stateful_python": True,
            "with_report_editor": True,
            "with_editor_submit": True,
        }
        config = ToolsetConfig.model_validate(config_dict)
        assert config.with_search_tools is False
        assert config.with_stateful_python is True
        assert config.with_report_editor is True
        assert config.with_editor_submit is True

    def test_create_tools_basic(self):
        """Test basic tool creation (without actually calling the tool functions)."""
        # Test with minimal config
        config = ToolsetConfig(
            with_search_tools=False,
            with_stateful_python=False,
            with_report_editor=False,
            with_table_editor=False,
            with_thinking_tool=False,
        )
        tools = config.create_tools()
        assert len(tools) == 0

    def test_create_tools_search_only(self):
        """Test tool creation with search tools only."""
        config = ToolsetConfig(
            with_search_tools=True,
            with_stateful_python=False,
            with_report_editor=False,
            with_table_editor=False,
            with_thinking_tool=False,
        )
        tools = config.create_tools()
        # Should have paper_search and snippet_search
        assert len(tools) == 2
        tool_names = [ToolDef(tool).name for tool in tools]
        assert "paper_search" in tool_names
        assert "snippet_search" in tool_names

    def test_field_descriptions(self):
        """Test that Field descriptions are properly set."""
        fields = ToolsetConfig.model_fields

        # Check that documented fields have descriptions
        assert (
            fields["with_search_tools"].description
            == "Include paper_search and snippet_search tools"
        )
        assert (
            "Enable submission via editor tools"
            in fields["with_editor_submit"].description
        )
