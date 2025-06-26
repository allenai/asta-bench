"""Data loader for citation evaluation CSV files."""

import ast
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from .models import Citation, Claim, EvaluationRow


class DataLoader:
    """Loads and parses citation evaluation CSV files."""
    
    def __init__(self, data_dir: str = "../test_dvc_logs/debug_logs"):
        self.data_dir = Path(data_dir)
        self._cache = {}  # Simple in-memory cache
    
    def list_csv_files(self) -> List[str]:
        """List all citation_eval.csv files in the data directory and subdirectories."""
        pattern = "**/*citation_eval.csv"
        files = list(self.data_dir.glob(pattern))
        # Return relative paths from the data directory
        return [str(f.relative_to(self.data_dir)) for f in sorted(files)]
    
    def load_file(self, filename: str) -> List[EvaluationRow]:
        """Load and parse a citation evaluation CSV file."""
        file_path = self.data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check cache first
        cache_key = f"{filename}_{file_path.stat().st_mtime}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        df = pd.read_csv(file_path)
        rows = []
        
        for _, row in df.iterrows():
            try:
                # Parse citations - handle both string and list formats
                citations_raw = row['citations']
                if isinstance(citations_raw, str):
                    citations_data = ast.literal_eval(citations_raw)
                else:
                    citations_data = citations_raw
                
                citations = [Citation(**cit) for cit in citations_data]
                
                # Parse claims - handle string format
                claims_raw = row['claims']
                if isinstance(claims_raw, str):
                    claims_data = ast.literal_eval(claims_raw)
                else:
                    claims_data = claims_raw
                
                claims = [Claim(**claim) for claim in claims_data]
                
                # Create evaluation row
                eval_row = EvaluationRow(
                    question=row['question'],
                    text=row['text'],
                    citations=citations,
                    claims=claims,
                    eval_component=row.get('eval_component'),
                    response=row.get('response'),
                    answer=row.get('answer'),
                    citation_recall_score=row.get('citation_recall_score'),
                    citation_precision_score=row.get('citation_precision_score')
                )
                
                rows.append(eval_row)
                
            except Exception as e:
                print(f"Error parsing row {len(rows)}: {e}")
                continue
        
        # Cache the result
        self._cache[cache_key] = rows
        
        return rows
    
    def get_file_stats(self, filename: str) -> Dict[str, Any]:
        """Get statistics about a CSV file."""
        rows = self.load_file(filename)
        total_claims = sum(len(row.claims) for row in rows)
        total_citations = sum(len(row.citations) for row in rows)
        
        return {
            "filename": filename,
            "total_rows": len(rows),
            "total_claims": total_claims,
            "total_citations": total_citations,
            "avg_claims_per_row": total_claims / len(rows) if rows else 0,
            "avg_citations_per_row": total_citations / len(rows) if rows else 0
        }