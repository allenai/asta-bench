"""Pydantic models for citation evaluation data."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class Citation(BaseModel):
    """A citation with ID and snippets."""
    id: str
    snippets: str


class Claim(BaseModel):
    """A claim with supporting/non-supporting citations."""
    text: str
    supporting: List[str]
    non_supporting: List[str]
    is_fully_supported: bool


class EvaluationRow(BaseModel):
    """A single row from the citation evaluation CSV."""
    question: str
    text: str
    citations: List[Citation]
    claims: List[Claim]
    # Include other fields that might be useful
    eval_component: Optional[str] = None
    response: Optional[str] = None
    answer: Optional[str] = None
    citation_recall_score: Optional[float] = None
    citation_precision_score: Optional[float] = None


class CitationAnnotation(BaseModel):
    """User annotation for a citation."""
    citation_id: str
    label: str  # "supporting", "non_supporting", "irrelevant"


class ClaimAnnotation(BaseModel):
    """User annotation for a claim."""
    claim_index: int
    claim_text: str
    citation_annotations: List[CitationAnnotation]
    missing_citations: Optional[str] = None
    notes: Optional[str] = None
    is_fully_supported_annotation: Optional[str] = None  # "agree", "disagree", or None


class FileAnnotation(BaseModel):
    """Annotations for an entire file."""
    filename: str
    timestamp: str
    row_annotations: Dict[int, List[Optional[ClaimAnnotation]]]  # row_index -> claim annotations
