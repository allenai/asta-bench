from typing import Any

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Inline citation details."""

    id: str = Field(
        ..., description="Inline citation marker, e.g., [1], (Smith 2020), etc."
    )
    snippets: list[str] = Field(
        default_factory=list,
        description="Supporting text snippets from within the cited paper.",
    )
    title: str | None = Field(
        None, description="Title of the cited paper (if available)."
    )
    metadata: dict[str, Any] | None = Field(
        None, description="Optional extra citation details (json-serializable)."
    )


class SQASection(BaseModel):
    """A section of a report."""

    title: str | None = Field(None, description="Section title")
    # For report editor, cleaner to allow text=None but keeping required str so as not to change sqa task behavior
    text: str = Field(
        "",
        description=(
            "Section text with inline citation markers. "
            "Multiple citations should appear separately, e.g., [1] [2] or (Smith 2020) (Jones 2021)."
        ),
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Citation details for each inline citation in the text (no repeated citation ids even if the marker appears multiple times).",
    )


class SQAResponse(BaseModel):
    """Top-level report object."""

    sections: list[SQASection] = Field(
        default_factory=list, description="List of report sections."
    )
