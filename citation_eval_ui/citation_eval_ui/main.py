"""FastAPI application for citation evaluation UI."""

import json
import uvicorn
import markdown
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .data_loader import DataLoader
from .models import FileAnnotation, ClaimAnnotation, CitationAnnotation


def normalize_citation_id(citation_id: str) -> str:
    """Normalize citation ID by removing surrounding parentheses and brackets."""
    if not citation_id:
        return citation_id
    
    # Remove surrounding parentheses and brackets
    citation_id = citation_id.strip()
    if (citation_id.startswith('(') and citation_id.endswith(')')) or \
       (citation_id.startswith('[') and citation_id.endswith(']')):
        citation_id = citation_id[1:-1]
    
    return citation_id.strip()


def citation_ids_match(id1: str, id2: str) -> bool:
    """Check if two citation IDs match after normalization."""
    return normalize_citation_id(id1) == normalize_citation_id(id2)


def markdown_filter(text: str) -> str:
    """Convert markdown text to HTML."""
    if not text:
        return ""
    
    # Configure markdown with extensions for better rendering
    md = markdown.Markdown(
        extensions=['extra', 'codehilite', 'toc'],
        extension_configs={
            'codehilite': {
                'css_class': 'highlight'
            }
        }
    )
    
    return md.convert(text)


app = FastAPI(title="Citation Evaluation UI", version="0.1.0")

# Get the package directory
package_dir = Path(__file__).parent

# Mount static files
app.mount("/static", StaticFiles(directory=package_dir / "static"), name="static")

# Setup templates
templates = Jinja2Templates(directory=package_dir / "templates")

# Add custom functions to Jinja2 environment
templates.env.globals['normalize_citation_id'] = normalize_citation_id
templates.env.globals['citation_ids_match'] = citation_ids_match
templates.env.filters['normalize_citation_id'] = normalize_citation_id
templates.env.filters['markdown'] = markdown_filter

# Initialize data loader
data_loader = DataLoader()

# Annotations directory
annotations_dir = package_dir / "annotations"
annotations_dir.mkdir(exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Home page with file selection."""
    try:
        files = data_loader.list_csv_files()
        file_stats = []
        for filename in files:
            try:
                stats = data_loader.get_file_stats(filename)
                file_stats.append(stats)
            except Exception as e:
                print(f"Error getting stats for {filename}: {e}")
                file_stats.append({
                    "filename": filename,
                    "error": str(e)
                })
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "files": file_stats
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "files": [],
            "error": str(e)
        })


@app.get("/annotate/{filename:path}", response_class=HTMLResponse)
async def annotate(request: Request, filename: str, row: int = 0, claim: int = 0):
    """Main annotation interface."""
    try:
        rows = data_loader.load_file(filename)
        if not rows:
            raise HTTPException(status_code=404, detail="No data found in file")
        
        if row >= len(rows):
            row = 0
        
        current_row = rows[row]
        if claim >= len(current_row.claims):
            claim = 0
        
        current_claim = current_row.claims[claim]
        
        # Sort citations: supporting first, then non-supporting, then others
        sorted_citations = []
        other_citations = []
        
        for citation in current_row.citations:
            normalized_citation_id = normalize_citation_id(citation.id)
            is_supporting = any(citation_ids_match(citation.id, supp) for supp in current_claim.supporting)
            is_non_supporting = any(citation_ids_match(citation.id, non_supp) for non_supp in current_claim.non_supporting)
            
            if is_supporting:
                sorted_citations.insert(0, citation)  # Supporting at the beginning
            elif is_non_supporting:
                # Insert non-supporting after supporting but before others
                supporting_count = sum(1 for c in sorted_citations 
                                     if any(citation_ids_match(c.id, supp) for supp in current_claim.supporting))
                sorted_citations.insert(supporting_count, citation)
            else:
                other_citations.append(citation)
        
        # Add other citations at the end
        sorted_citations.extend(other_citations)
        
        # Replace the citations list with sorted version
        current_row.citations = sorted_citations
        
        # Load existing annotations if available
        annotations = load_annotations(filename)
        current_annotations = annotations.row_annotations.get(row, [])
        claim_annotation = None
        if claim < len(current_annotations):
            claim_annotation = current_annotations[claim]
        
        return templates.TemplateResponse("annotate.html", {
            "request": request,
            "filename": filename,
            "row": row,
            "claim": claim,
            "current_row": current_row,
            "current_claim": current_claim,
            "claim_annotation": claim_annotation,
            "total_rows": len(rows),
            "total_claims": len(current_row.claims),
            "has_prev_claim": claim > 0,
            "has_next_claim": claim < len(current_row.claims) - 1,
            "has_prev_row": row > 0,
            "has_next_row": row < len(rows) - 1
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/save-annotation")
async def save_annotation(
    filename: str = Form(...),
    row: int = Form(...),
    claim: int = Form(...),
    citation_labels: str = Form(...),
    missing_citations: str = Form(""),
    notes: str = Form(""),
    is_fully_supported_annotation: str = Form("")
):
    """Save annotation for a claim."""
    try:
        # Parse citation labels
        citation_annotations = []
        labels_data = json.loads(citation_labels)
        
        for citation_id, label in labels_data.items():
            if label:  # Only save non-empty labels
                citation_annotations.append(CitationAnnotation(
                    citation_id=normalize_citation_id(citation_id),
                    label=label
                ))
        
        # Get current claim text
        rows = data_loader.load_file(filename)
        claim_text = rows[row].claims[claim].text
        
        # Create claim annotation
        claim_annotation = ClaimAnnotation(
            claim_index=claim,
            claim_text=claim_text,
            citation_annotations=citation_annotations,
            missing_citations=missing_citations if missing_citations else None,
            notes=notes if notes else None,
            is_fully_supported_annotation=is_fully_supported_annotation if is_fully_supported_annotation else None
        )
        
        # Load existing annotations
        annotations = load_annotations(filename)
        
        # Update annotations
        if row not in annotations.row_annotations:
            annotations.row_annotations[row] = []
        
        # Ensure the list is long enough
        while len(annotations.row_annotations[row]) <= claim:
            annotations.row_annotations[row].append(None)
        
        annotations.row_annotations[row][claim] = claim_annotation
        annotations.timestamp = datetime.now().isoformat()
        
        # Save annotations
        save_annotations(annotations)
        
        return JSONResponse({"status": "success"})
        
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/api/files")
async def get_files():
    """Get list of available CSV files."""
    try:
        files = data_loader.list_csv_files()
        return JSONResponse(files)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


def load_annotations(filename: str) -> FileAnnotation:
    """Load annotations for a file."""
    annotation_file = annotations_dir / f"{filename}.json"
    
    if annotation_file.exists():
        with open(annotation_file, 'r') as f:
            data = json.load(f)
            return FileAnnotation(**data)
    else:
        return FileAnnotation(
            filename=filename,
            timestamp=datetime.now().isoformat(),
            row_annotations={}
        )


def save_annotations(annotations: FileAnnotation):
    """Save annotations for a file."""
    annotation_file = annotations_dir / f"{annotations.filename}.json"
    
    with open(annotation_file, 'w') as f:
        json.dump(annotations.model_dump(), f, indent=2)


def main():
    """Main entry point for the CLI."""
    print("Starting Citation Evaluation UI...")
    print("Open http://localhost:8000 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()