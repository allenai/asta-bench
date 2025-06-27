#!/usr/bin/env python3
"""
Script to process citations_summary.csv and use an LLM to match half_credit citations with to_check citations.
"""

import csv
import json
import ast
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

import litellm
from pydantic import BaseModel


class CitationMatch(BaseModel):
    """Structured output for citation matching results."""
    matched_citations: List[str]


@dataclass
class ProcessedRow:
    """Container for processed CSV row data."""
    question: str
    csv_path: str
    all_citations: List[str]
    half_credit: List[str]
    to_check: List[str]
    num_half_credit_missing: int
    matched_citations: List[str] = None


def parse_citation_list(citation_str: str) -> List[str]:
    """Parse citation string into list of citations."""
    try:
        return ast.literal_eval(citation_str)
    except (ValueError, SyntaxError):
        return []


def match_citations_with_llm(half_credit: List[str], to_check: List[str]) -> List[str]:
    """Use LLM to match half_credit citations with to_check citations."""
    if not half_credit or not to_check:
        return []
    
    prompt = f"""
You are tasked with matching citations. Given a list of "half_credit" citations and a list of "to_check" citations, determine which half_credit citations match any of the citations in the to_check list.

Consider citations as matching if they refer to the same paper, even with formatting differences (e.g., "Smith et al., 2020" vs "Smith et al. 2020" or '_2_' vs '2' or bigger formatting differences).

Half credit citations: {half_credit}
Citations to check: {to_check}

Return only the half_credit citations that match any citation in the to_check list. Return your result in JSON format with a key "matched_citations": List[str] containing a list of matched citations.
"""
    
    try:
        response = litellm.completion(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            response_format=CitationMatch,
            temperature=0.1
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get("matched_citations", [])
    
    except Exception as e:
        print(f"Error processing with LLM: {e}")
        return []


def process_csv(input_path: str, output_path: str) -> None:
    """Process the CSV file and write results."""
    processed_rows = []
    
    with open(input_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            # Parse citation lists
            all_citations = parse_citation_list(row['all_citations'])
            half_credit = parse_citation_list(row['half_credit'])
            to_check = parse_citation_list(row['to_check'])
            
            # Create processed row
            processed_row = ProcessedRow(
                question=row['question'],
                csv_path=row['csv_path'],
                all_citations=all_citations,
                half_credit=half_credit,
                to_check=to_check,
                num_half_credit_missing=int(row['num_half_credit_missing'])
            )
            
            # Use LLM to match citations
            print(f"Processing: {processed_row.question[:50]}...")
            matched_citations = match_citations_with_llm(half_credit, to_check)
            processed_row.matched_citations = matched_citations
            
            processed_rows.append(processed_row)
    
    # Write results to new CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'question', 'csv_path', 'all_citations', 'half_credit', 
            'to_check', 'num_half_credit_missing', 'matched_citations'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in processed_rows:
            writer.writerow({
                'question': row.question,
                'csv_path': row.csv_path,
                'all_citations': str(row.all_citations),
                'half_credit': str(row.half_credit),
                'to_check': str(row.to_check),
                'num_half_credit_missing': row.num_half_credit_missing,
                'matched_citations': str(row.matched_citations)
            })
    
    print(f"Results written to {output_path}")


def main():
    """Main function."""
    input_path = "/data/new_astabench/citations_summary.csv"
    output_path = "/data/new_astabench/citation_matches_results.csv"
    
    if not Path(input_path).exists():
        print(f"Error: Input file {input_path} does not exist")
        return
    
    process_csv(input_path, output_path)


if __name__ == "__main__":
    main()
