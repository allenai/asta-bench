#!/usr/bin/env python3
"""
Extract total counts of claims and citations from citation evaluation logs.
This provides the denominators needed for calculating true failure rates.
Uses pandas vectorized operations for better performance.
"""

import json
import pandas as pd
import ast
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


def safe_eval(x):
    """Safely evaluate a string representation of a Python literal."""
    if pd.isna(x):
        return []
    try:
        return ast.literal_eval(x)
    except:
        return []


def count_claims(claims_list):
    """Count the number of claims in a list."""
    if isinstance(claims_list, list):
        return len(claims_list)
    return 0


def count_failed_claims(claims_list):
    """Count the number of failed claims in a list."""
    if not isinstance(claims_list, list):
        return 0
    return sum(1 for claim in claims_list if not claim.get('is_fully_supported', True))


def count_failed_citations(claims_list):
    """Count the total number of non-supporting citations across all claims."""
    if not isinstance(claims_list, list):
        return 0
    total = 0
    for claim in claims_list:
        total += len(claim.get('non_supporting', []))
    return total


def count_citations(claims_list):
    """Count the number of citations in a list."""
    if not isinstance(claims_list, list):
        return 0
    total = 0
    for claim in claims_list:
        total += len(claim.get('non_supporting', []))
        total += len(claim.get('supporting', []))
    return total


def extract_totals_from_csv(csv_path: str) -> Dict[str, int]:
    """
    Extract total counts from a citation evaluation CSV file using pandas operations.
    
    Args:
        csv_path: Path to the citation evaluation CSV file
        
    Returns:
        Dictionary containing total counts
    """
    df = pd.read_csv(csv_path)
    
    # Parse claims and citations columns
    df['claims_parsed'] = df['claims'].apply(safe_eval)

    # Count totals using vectorized operations
    df['num_claims'] = df['claims_parsed'].apply(count_claims)
    df['num_citations'] = df['claims_parsed'].apply(count_citations)
    df['num_failed_claims'] = df['claims_parsed'].apply(count_failed_claims)
    df['num_failed_citations'] = df['claims_parsed'].apply(count_failed_citations)
    
    # Aggregate totals
    total_rows = len(df)
    total_claims = df['num_claims'].sum()
    total_citations = df['num_citations'].sum()
    failed_claims = df['num_failed_claims'].sum()
    failed_citations = df['num_failed_citations'].sum()
    
    return {
        'total_rows': int(total_rows),
        'total_claims': int(total_claims),
        'total_citations': int(total_citations),
        'failed_claims': int(failed_claims),
        'failed_citations': int(failed_citations),
        'claim_failure_rate': (failed_claims / total_claims * 100) if total_claims > 0 else 0,
        'citation_failure_rate': (failed_citations / total_citations * 100) if total_citations > 0 else 0
    }


def main():
    """Main function to process all systems and extract totals."""
    
    # Define systems to analyze - using relative paths from scripts directory
    systems = {
        'elicit': '../../test_dvc_logs/debug_logs/task_sqa_solver_elicit_citation_eval.csv',
        'openai_deep_research': '../../test_dvc_logs/debug_logs/task_sqa_solver_openai_deep_research_citation_eval.csv',
        'perplexity_dr': '../../test_dvc_logs/debug_logs/task_sqa_solver_perplexity_dr_citation_eval.csv',
        'claude_4.0': '../../test_dvc_logs/debug_logs/task_sqa_solver_sqa_claude-4.0_citation_eval.csv'
    }
    
    all_results = {}
    
    print("Extracting total counts from citation evaluation logs...")
    print("=" * 80)
    
    grand_totals = {
        'total_claims': 0,
        'total_citations': 0,
        'failed_claims': 0,
        'failed_citations': 0
    }
    
    for system_name, csv_path in systems.items():
        print(f"\nProcessing {system_name}...")
        
        if not Path(csv_path).exists():
            print(f"  Warning: File not found - {csv_path}")
            continue
            
        try:
            results = extract_totals_from_csv(csv_path)
            all_results[system_name] = results
            
            # Update grand totals
            grand_totals['total_claims'] += results['total_claims']
            grand_totals['total_citations'] += results['total_citations']
            grand_totals['failed_claims'] += results['failed_claims']
            grand_totals['failed_citations'] += results['failed_citations']
            
            # Print statistics
            print(f"  Total rows: {results['total_rows']:,}")
            print(f"  Total claims: {results['total_claims']:,}")
            print(f"  Total citations: {results['total_citations']:,}")
            print(f"  Failed claims: {results['failed_claims']:,} ({results['claim_failure_rate']:.2f}%)")
            print(f"  Failed citations: {results['failed_citations']:,} ({results['citation_failure_rate']:.2f}%)")
            
        except Exception as e:
            print(f"  Error processing {system_name}: {e}")
            continue
    
    # Calculate grand total rates
    grand_totals['claim_failure_rate'] = (
        (grand_totals['failed_claims'] / grand_totals['total_claims'] * 100) 
        if grand_totals['total_claims'] > 0 else 0
    )
    grand_totals['citation_failure_rate'] = (
        (grand_totals['failed_citations'] / grand_totals['total_citations'] * 100) 
        if grand_totals['total_citations'] > 0 else 0
    )
    
    # Save results to JSON - in data/statistics directory
    output_file = '../data/statistics/totals.json'
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'extraction_date': datetime.now().isoformat(),
        'systems': all_results,
        'grand_totals': grand_totals
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    print(f"\nGrand Totals:")
    print(f"  Total claims across all systems: {grand_totals['total_claims']:,}")
    print(f"  Total citations across all systems: {grand_totals['total_citations']:,}")
    print(f"  Failed claims: {grand_totals['failed_claims']:,} ({grand_totals['claim_failure_rate']:.2f}%)")
    print(f"  Failed citations: {grand_totals['failed_citations']:,} ({grand_totals['citation_failure_rate']:.2f}%)")
    
    print(f"\nPer-System Breakdown:")
    for system_name, results in all_results.items():
        print(f"\n{system_name}:")
        print(f"  Claims: {results['total_claims']:,} (failure rate: {results['claim_failure_rate']:.2f}%)")
        print(f"  Citations: {results['total_citations']:,} (failure rate: {results['citation_failure_rate']:.2f}%)")


if __name__ == "__main__":
    main()
