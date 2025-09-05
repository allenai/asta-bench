#!/usr/bin/env python3
"""
Script 1: Extract non-supporting citations from citation evaluation logs.
Extracts all citations marked as non_supporting along with the claims they failed to support.
"""

import json
import pandas as pd
import ast
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


def extract_non_supporting_citations(csv_path: str) -> Dict[str, Any]:
    """
    Extract non-supporting citations from a citation evaluation CSV file.
    
    Args:
        csv_path: Path to the citation evaluation CSV file
        
    Returns:
        Dictionary containing non-supporting citations and metadata
    """
    df = pd.read_csv(csv_path)
    
    non_supporting_citations = []
    total_claims = 0
    total_citations = 0
    total_rows = len(df)
    
    for idx, row in df.iterrows():
        if pd.notna(row['claims']):
            try:
                claims = ast.literal_eval(row['claims'])
                citations = ast.literal_eval(row['citations']) if pd.notna(row['citations']) else []
                
                # Create citation lookup dictionary
                citation_dict = {}
                if isinstance(citations, list):
                    for cit in citations:
                        if isinstance(cit, dict) and 'id' in cit:
                            snippet = cit.get('snippets', '')
                            # Store with original ID
                            citation_dict[cit['id']] = snippet
                            # Also store with normalized ID (brackets and parentheses removed)
                            normalized_id = cit['id'].replace('[', '').replace(']', '').replace('(', '').replace(')', '')
                            if normalized_id != cit['id']:
                                citation_dict[normalized_id] = snippet
                
                # Process each claim
                for claim in claims:
                    total_claims += 1
                    
                    # Extract non-supporting citations
                    non_supporting_refs = claim.get('non_supporting', [])
                    
                    for ref in non_supporting_refs:
                        total_citations += 1
                        # Normalize the reference by removing brackets and parentheses
                        normalized_ref = ref.replace('[', '').replace(']', '').replace('(', '').replace(')', '')
                        
                        # Try to find the citation snippet
                        snippet = ''
                        if normalized_ref in citation_dict:
                            snippet = citation_dict[normalized_ref]
                        elif ref in citation_dict:
                            snippet = citation_dict[ref]
                        else:
                            snippet = 'No snippet found.'
                        
                        non_supporting_entry = {
                            'question': row['question'],
                            'claim_text': claim.get('text', ''),
                            'citation_id': ref,
                            'citation_snippet': snippet,
                            'is_fully_supported': claim.get('is_fully_supported', False),
                            'supporting_refs': claim.get('supporting', []),
                            'all_non_supporting_refs': non_supporting_refs
                        }
                        
                        non_supporting_citations.append(non_supporting_entry)
                        
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
    
    return {
        'total_rows': total_rows,
        'total_claims': total_claims,
        'non_supporting_citations_count': len(non_supporting_citations),
        'unique_citations': total_citations,
        'non_supporting_citations': non_supporting_citations,
        'non_supporting_percentage': (len(non_supporting_citations) / total_citations * 100) if total_citations > 0 else 0
    }


def main():
    """Main function to process all systems."""
    
    # Define systems to analyze - using relative paths from scripts directory
    systems = {
        'elicit': '../../test_dvc_logs/debug_logs/task_sqa_solver_elicit_citation_eval.csv',
        'openai_deep_research': '../../test_dvc_logs/debug_logs/task_sqa_solver_openai_deep_research_citation_eval.csv',
        'perplexity_dr': '../../test_dvc_logs/debug_logs/task_sqa_solver_perplexity_dr_citation_eval.csv',
        'claude_4.0': '../../test_dvc_logs/debug_logs/task_sqa_solver_sqa_claude-4.0_citation_eval.csv'
    }
    
    all_results = {}
    
    print("Extracting non-supporting citations from citation evaluation logs...")
    print("=" * 80)
    
    for system_name, csv_path in systems.items():
        print(f"\nProcessing {system_name}...")
        
        if not Path(csv_path).exists():
            print(f"  Warning: File not found - {csv_path}")
            continue
            
        try:
            results = extract_non_supporting_citations(csv_path)
            all_results[system_name] = results
            
            # Print statistics
            print(f"  Total rows: {results['total_rows']}")
            print(f"  Total claims: {results['total_claims']}")
            print(f"  Non-supporting citations: {results['non_supporting_citations_count']}")
            if results['unique_citations'] > 0:
                print(f"  Percentage non-supporting: {results['non_supporting_percentage']:.2f}%")
            
        except Exception as e:
            print(f"  Error processing {system_name}: {e}")
            continue
    
    # Save results to JSON - in data/extracted directory
    output_file = '../data/extracted/non_supporting_citations_extracted.json'
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({
            'extraction_date': datetime.now().isoformat(),
            'systems': all_results
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    total_non_supporting = 0
    for system_name, results in all_results.items():
        count = results.get('non_supporting_citations_count', 0)
        total_non_supporting += count
        print(f"{system_name:25} {count:6} non-supporting citations")
    
    print(f"{'TOTAL':25} {total_non_supporting:6} non-supporting citations")
    
    # Create a combined list of all non-supporting citations for sampling
    all_non_supporting = []
    for system_name, results in all_results.items():
        for citation in results.get('non_supporting_citations', []):
            citation['system'] = system_name
            all_non_supporting.append(citation)
    
    # Save combined non-supporting citations for easy access by next scripts
    combined_file = '../data/extracted/all_non_supporting_citations.json'
    with open(combined_file, 'w') as f:
        json.dump(all_non_supporting, f, indent=2)
    
    print(f"\nCombined non-supporting citations saved to {combined_file}")
    print(f"Total non-supporting citations across all systems: {len(all_non_supporting)}")


if __name__ == "__main__":
    main()