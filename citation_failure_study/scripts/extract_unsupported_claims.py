#!/usr/bin/env python3
"""
Script 1: Extract unsupported claims from citation evaluation logs.
Extracts all claims where is_fully_supported = False along with their cited text snippets.
"""

import json
import pandas as pd
import ast
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


def extract_unsupported_claims(csv_path: str) -> Dict[str, Any]:
    """
    Extract unsupported claims from a citation evaluation CSV file.
    
    Args:
        csv_path: Path to the citation evaluation CSV file
        
    Returns:
        Dictionary containing unsupported claims and metadata
    """
    df = pd.read_csv(csv_path)
    
    unsupported_claims = []
    total_claims = 0
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
                    
                    if not claim.get('is_fully_supported', True):
                        # Extract unsupported claim with its citations
                        unsupported_entry = {
                            'question': row['question'],
                            'claim_text': claim.get('text', ''),
                            'supporting_refs': claim.get('supporting', []),
                            'non_supporting_refs': claim.get('non_supporting', []),
                            'cited_snippets': {}
                        }
                        
                        # Get snippets for all referenced citations
                        all_refs = claim.get('supporting', []) + claim.get('non_supporting', [])
                        for ref in all_refs:
                            # Normalize the reference by removing brackets and parentheses
                            normalized_ref = ref.replace('[', '').replace(']', '').replace('(', '').replace(')', '')
                            
                            # Try to find the citation with normalized keys
                            if normalized_ref in citation_dict:
                                unsupported_entry['cited_snippets'][ref] = citation_dict[normalized_ref]
                            elif ref in citation_dict:
                                # Fallback to original ref if it exists
                                unsupported_entry['cited_snippets'][ref] = citation_dict[ref]
                            else:
                                unsupported_entry['cited_snippets'][ref] = 'No snippet found.'
                        
                        unsupported_claims.append(unsupported_entry)
                        
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
    
    return {
        'total_rows': total_rows,
        'total_claims': total_claims,
        'unsupported_claims_count': len(unsupported_claims),
        'unsupported_claims': unsupported_claims,
        'unsupported_percentage': (len(unsupported_claims) / total_claims * 100) if total_claims > 0 else 0
    }


def main():
    """Main function to process all three systems."""
    
    # Define systems to analyze - using relative paths from scripts directory
    systems = {
        'elicit': '../../test_dvc_logs/debug_logs/task_sqa_solver_elicit_citation_eval.csv',
        'openai_deep_research': '../../test_dvc_logs/debug_logs/task_sqa_solver_openai_deep_research_citation_eval.csv',
        'perplexity_dr': '../../test_dvc_logs/debug_logs/task_sqa_solver_perplexity_dr_citation_eval.csv',
        'claude_4.0': '../../test_dvc_logs/debug_logs/task_sqa_solver_sqa_claude-4.0_citation_eval.csv'
    }
    
    all_results = {}
    
    print("Extracting unsupported claims from citation evaluation logs...")
    print("=" * 80)
    
    for system_name, csv_path in systems.items():
        print(f"\nProcessing {system_name}...")
        
        if not Path(csv_path).exists():
            print(f"  Warning: File not found - {csv_path}")
            continue
            
        try:
            results = extract_unsupported_claims(csv_path)
            all_results[system_name] = results
            
            # Print statistics
            print(f"  Total rows: {results['total_rows']}")
            print(f"  Total claims: {results['total_claims']}")
            print(f"  Unsupported claims: {results['unsupported_claims_count']}")
            print(f"  Percentage unsupported: {results['unsupported_percentage']:.2f}%")
            
        except Exception as e:
            print(f"  Error processing {system_name}: {e}")
            continue
    
    # Save results to JSON - in data/extracted directory
    output_file = '../data/extracted/unsupported_claims_extracted.json'
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
    
    total_unsupported = 0
    for system_name, results in all_results.items():
        count = results.get('unsupported_claims_count', 0)
        total_unsupported += count
        print(f"{system_name:25} {count:6} unsupported claims")
    
    print(f"{'TOTAL':25} {total_unsupported:6} unsupported claims")
    
    # Create a combined list of all unsupported claims for sampling
    all_unsupported = []
    for system_name, results in all_results.items():
        for claim in results.get('unsupported_claims', []):
            claim['system'] = system_name
            all_unsupported.append(claim)
    
    # Save combined unsupported claims for easy access by next scripts
    combined_file = '../data/extracted/all_unsupported_claims.json'
    with open(combined_file, 'w') as f:
        json.dump(all_unsupported, f, indent=2)
    
    print(f"\nCombined unsupported claims saved to {combined_file}")
    print(f"Total unsupported claims across all systems: {len(all_unsupported)}")


if __name__ == "__main__":
    main()
