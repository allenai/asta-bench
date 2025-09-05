#!/usr/bin/env python3
"""
Script 3: Assign citation failure mode categories to all non-supporting citations.
Uses the categories identified in Script 2 to classify all non-supporting citations.
"""

import json
import os
from typing import List, Dict, Any
from pathlib import Path
from litellm import completion
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

# Load environment variables
load_dotenv()


def load_citation_failure_modes(file_path: str = '../data/categories/citation_failure_mode_categories.json') -> Dict:
    """Load citation failure mode categories from JSON file."""
    if not Path(file_path).exists():
        # Try the template file as fallback
        template_path = '../data/categories/citation_failure_mode_categories_template.json'
        if Path(template_path).exists():
            print(f"Using template file: {template_path}")
            file_path = template_path
        else:
            raise FileNotFoundError(f"Please run categorize_citation_failure_modes.py first to generate {file_path}")
    
    with open(file_path, 'r') as f:
        return json.load(f)


def load_non_supporting_citations(file_path: str = '../data/extracted/all_non_supporting_citations.json') -> List[Dict]:
    """Load all non-supporting citations from JSON file."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Please run extract_non_supporting_citations.py first to generate {file_path}")
    
    with open(file_path, 'r') as f:
        return json.load(f)


def classify_single_citation(citation: Dict, failure_modes: List[Dict]) -> Dict:
    """
    Classify a single non-supporting citation using LLM via litellm.
    
    Args:
        citation: Single citation to classify
        failure_modes: List of citation failure mode definitions
        
    Returns:
        Classified citation with added failure_mode and reasoning fields
    """
    # Prepare the prompt
    prompt = """You are classifying a non-supporting citation based on predefined failure mode categories.

Here are the citation failure mode categories to use:
"""
    
    # Add failure mode definitions
    for mode in failure_modes:
        prompt += f"\n{mode['code']}:"
        prompt += f"\n  Name: {mode['name']}"
        prompt += f"\n  Description: {mode['description']}"
        if 'characteristics' in mode:
            prompt += f"\n  Characteristics: {', '.join(mode['characteristics'])}"
        prompt += "\n"
    
    prompt += "\nDetermine which failure mode category best applies to this citation-claim pair.\n"
    prompt += "Analyze why the citation doesn't support the claim.\n\n"
    
    # Add the citation-claim pair to classify
    prompt += "Citation-Claim Pair:\n"
    prompt += f"  Claim Text: {citation['claim_text']}\n"  # Truncate long claims
    prompt += f"  Citation ID: {citation.get('citation_id', 'Unknown')}\n"
    prompt += f"  Citation Snippet: {citation.get('citation_snippet', '')}\n"  # Truncate long snippets
    
    # Add context about other citations if available
    if citation.get('supporting_refs'):
        prompt += f"  Note: This claim also has supporting citations: {citation['supporting_refs'][:3]}\n"
    
    prompt += """
Output your classification as a JSON object with this structure:
{
  "assigned_category": "category_code",
  "reasoning": "Brief explanation of why this category was chosen"
}

Choose the category code from the provided failure modes above."""
    
    # Use structured output for reliable parsing
    from pydantic import BaseModel, Field
    
    class CitationClassification(BaseModel):
        assigned_category: str = Field(description="The failure mode category code")
        reasoning: str = Field(description="Brief explanation of why this category was chosen")
    
    try:
        response = completion(
            model="gemini/gemini-2.5-pro",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format=CitationClassification,
        )
        
        # Parse the structured response
        result = response.choices[0].message.content

        if isinstance(result, str):
            result = json.loads(result)

        if hasattr(result, 'model_dump'):
            result = result.model_dump()
        elif hasattr(result, 'dict'):
            result = result.dict()
        
        # Add classification to citation
        citation['assigned_category'] = result['assigned_category']
        citation['classification_reasoning'] = result['reasoning']
        
    except Exception as e:
        # More detailed error logging
        import traceback
        error_detail = f"{str(e)}\nTraceback:\n{traceback.format_exc()}"
        print(f"Error classifying citation: {error_detail[:500]}")
        citation['assigned_category'] = 'classification_error'
        citation['classification_reasoning'] = str(e)[:200]  # Truncate long error messages
    
    return citation


def classify_citations_batch(citations: List[Dict], failure_modes: List[Dict], n_jobs: int = -1) -> List[Dict]:
    """
    Classify multiple citations in parallel.
    
    Args:
        citations: List of citations to classify
        failure_modes: List of failure mode definitions
        n_jobs: Number of parallel jobs (-1 uses all cores)
        
    Returns:
        List of classified citations
    """
    print(f"Classifying {len(citations)} citations using {n_jobs} parallel jobs...")
    
    # Use joblib for parallel processing with progress bar
    classified = Parallel(n_jobs=n_jobs)(
        delayed(classify_single_citation)(citation, failure_modes)
        for citation in tqdm(citations, desc="Classifying citations")
    )
    
    return classified


def generate_analysis_report(classified_citations: List[Dict], failure_modes: List[Dict]) -> Dict[str, Any]:
    """
    Generate analysis report from classified citations.
    
    Args:
        classified_citations: List of classified citations
        failure_modes: List of failure mode definitions
        
    Returns:
        Analysis report dictionary
    """
    from collections import Counter, defaultdict
    
    # Count citations by category
    category_counts = Counter(c['assigned_category'] for c in classified_citations)
    
    # Group citations by system
    system_counts = defaultdict(lambda: defaultdict(int))
    for citation in classified_citations:
        system = citation.get('system', 'unknown')
        category = citation['assigned_category']
        system_counts[system][category] += 1
    
    # Calculate statistics
    total_citations = len(classified_citations)
    
    # Build category statistics
    category_stats = []
    for mode in failure_modes:
        code = mode['code']
        count = category_counts.get(code, 0)
        percentage = (count / total_citations * 100) if total_citations > 0 else 0
        
        category_stats.append({
            'code': code,
            'name': mode['name'],
            'count': count,
            'percentage': round(percentage, 2),
            'description': mode['description']
        })
    
    # Add error category if present
    error_count = category_counts.get('classification_error', 0)
    if error_count > 0:
        category_stats.append({
            'code': 'classification_error',
            'name': 'Classification Error',
            'count': error_count,
            'percentage': round(error_count / total_citations * 100, 2),
            'description': 'Citations that could not be classified'
        })
    
    # Sort by count
    category_stats.sort(key=lambda x: x['count'], reverse=True)
    
    # Build system statistics
    system_stats = {}
    for system, categories in system_counts.items():
        system_total = sum(categories.values())
        system_stats[system] = {
            'total': system_total,
            'categories': dict(categories),
            'top_category': max(categories, key=categories.get) if categories else None
        }
    
    return {
        'total_citations_analyzed': total_citations,
        'category_statistics': category_stats,
        'system_statistics': system_stats,
        'analysis_date': datetime.now().isoformat()
    }


def main():
    """Main function to classify all non-supporting citations."""
    import argparse
    import random
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Classify non-supporting citations into failure modes')
    parser.add_argument('--sample', type=int, default=None,
                        help='Sample size for classification (default: classify all)')
    parser.add_argument('--stratified', action='store_true',
                        help='Use stratified sampling with equal split across systems')
    args = parser.parse_args()
    
    print("Classifying non-supporting citations into failure mode categories...")
    print("=" * 80)
    
    # Load failure modes
    try:
        failure_mode_data = load_citation_failure_modes()
        failure_modes = failure_mode_data.get('citation_failure_modes', [])
        print(f"Loaded {len(failure_modes)} citation failure mode categories")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Load all non-supporting citations
    try:
        all_citations = load_non_supporting_citations()
        print(f"Loaded {len(all_citations)} non-supporting citations")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Sample if requested
    if args.sample:
        if args.stratified:
            from collections import defaultdict
            
            # Group citations by system
            citations_by_system = defaultdict(list)
            for citation in all_citations:
                citations_by_system[citation.get('system', 'unknown')].append(citation)
            
            # Calculate samples per system
            num_systems = len(citations_by_system)
            samples_per_system = args.sample // num_systems
            remainder = args.sample % num_systems
            
            sampled_citations = []
            for i, (system, system_citations) in enumerate(citations_by_system.items()):
                n_samples = samples_per_system + (1 if i < remainder else 0)
                n_samples = min(n_samples, len(system_citations))
                sampled = random.sample(system_citations, n_samples)
                sampled_citations.extend(sampled)
            
            print(f"\nStratified sampling: {len(sampled_citations)} citations across {num_systems} systems")
            citations_to_classify = sampled_citations
            
        else:
            sample_size = min(args.sample, len(all_citations))
            citations_to_classify = random.sample(all_citations, sample_size)
            print(f"Random sampling: {len(citations_to_classify)} citations")
    else:
        citations_to_classify = all_citations
        print(f"Classifying all {len(citations_to_classify)} citations")
    
    # Get n_jobs from params.yaml if it exists
    n_jobs = -1  # Default value
    params_path = '../params.yaml'
    if Path(params_path).exists():
        import yaml
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
            n_jobs = params.get('citation_n_jobs', n_jobs)
    
    # Classify citations
    citations_to_classify = [c for c in citations_to_classify if c.get('citation_snippet', '') != 'No snippet found.']
    classified_citations = classify_citations_batch(
        citations_to_classify,
        failure_modes,
        n_jobs=n_jobs
    )
    
    # Generate analysis report
    report = generate_analysis_report(classified_citations, failure_modes)
    
    output_file = '../data/classified/classified_non_supporting_citations.json'
    report_file = '../reports/citation_failure_mode_analysis_report.json'
    summary_file = '../reports/classified_citations_summary.csv'
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(classified_citations, f, indent=2)
    
    print(f"\nClassified citations saved to {output_file}")
    
    # Save analysis report
    Path(report_file).parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Analysis report saved to {report_file}")
    
    # Create summary CSV
    df_data = []
    for stat in report['category_statistics']:
        df_data.append({
            'Category Code': stat['code'],
            'Category Name': stat['name'],
            'Count': stat['count'],
            'Percentage': stat['percentage'],
            'Description': stat['description']
        })
    
    df = pd.DataFrame(df_data)
    df.to_csv(summary_file, index=False)
    print(f"Summary CSV saved to {summary_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("CLASSIFICATION RESULTS")
    print("=" * 80)
    
    print(f"\nTotal citations classified: {report['total_citations_analyzed']}")
    
    print("\nCategory Distribution:")
    for stat in report['category_statistics']:
        print(f"  {stat['name']:30} {stat['count']:6} ({stat['percentage']:5.1f}%)")
    
    print("\nSystem Breakdown:")
    for system, stats in report['system_statistics'].items():
        print(f"  {system:25} {stats['total']:6} citations")
        if stats['top_category']:
            print(f"    Top category: {stats['top_category']}")


if __name__ == "__main__":
    main()
