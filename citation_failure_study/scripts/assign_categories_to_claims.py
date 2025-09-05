#!/usr/bin/env python3
"""
Script 3: Assign failure mode categories to all unsupported claims.
Uses the categories identified in Script 2 to classify all unsupported claims.
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


def load_failure_modes(file_path: str = '../data/categories/failure_mode_categories.json') -> Dict:
    """Load failure mode categories from JSON file."""
    if not Path(file_path).exists():
        # Try the template file as fallback
        template_path = '../data/categories/failure_mode_categories_template.json'
        if Path(template_path).exists():
            print(f"Using template file: {template_path}")
            file_path = template_path
        else:
            raise FileNotFoundError(f"Please run categorize_failure_modes.py first to generate {file_path}")
    
    with open(file_path, 'r') as f:
        return json.load(f)


def load_unsupported_claims(file_path: str = '../data/extracted/all_unsupported_claims.json') -> List[Dict]:
    """Load all unsupported claims from JSON file."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Please run extract_unsupported_claims.py first to generate {file_path}")
    
    with open(file_path, 'r') as f:
        return json.load(f)


def classify_single_claim(claim: Dict, failure_modes: List[Dict]) -> Dict:
    """
    Classify a single claim using LLM via litellm.
    
    Args:
        claim: Single claim to classify
        failure_modes: List of failure mode definitions
        
    Returns:
        Classified claim with added failure_mode and reasoning fields
    """
    # Prepare the prompt
    prompt = """You are classifying an unsupported claim based on predefined failure mode categories.

Here are the failure mode categories to use:
"""
    
    # Add failure mode definitions
    for mode in failure_modes:
        prompt += f"\n{mode['code']}:"
        prompt += f"\n  Name: {mode['name']}"
        prompt += f"\n  Description: {mode['description']}"
        if 'characteristics' in mode:
            prompt += f"\n  Characteristics: {', '.join(mode['characteristics'])}"
        prompt += "\n"
    
    prompt += "\nDetermine which failure mode category best applies to this claim.\n"
    prompt += "Consider the claim text, whether citations are provided, and if provided, whether they support the claim.\n\n"
    
    # Add the claim to classify
    prompt += "Claim:\n"
    prompt += f"  Text: {claim['claim_text']}\n"  # Truncate long claims
    
    has_citations = len(claim.get('supporting_refs', [])) > 0 or len(claim.get('non_supporting_refs', [])) > 0

    if has_citations:
        if claim.get('supporting_refs'):
            prompt += f"  Supporting refs: {claim['supporting_refs']}\n"
        if claim.get('non_supporting_refs'):
            prompt += f"  Non-supporting refs: {claim['non_supporting_refs']}\n"
        
        # Include snippet sample if available
        snippets = claim.get('cited_snippets', {})
        if snippets:
            snippet_content = '\n\n'.join([f"- {snip}" for snip in snippets.values()])
            prompt += f"  Snippets: {snippet_content}...\n"
    
    prompt += """
Output your classification as a JSON object with this structure:
{
  "assigned_category": "category_code",
  "reasoning": "Brief explanation of why this category was chosen"
}

Respond with ONLY the JSON object, no other text."""
    
    # Get response from LLM via litellm with structured output
    try:
        from pydantic import BaseModel, Field
        
        class SingleClaimClassification(BaseModel):
            assigned_category: str = Field(description="The failure mode category code")
            reasoning: str = Field(description="Brief explanation of why this category was chosen")
        
        response = completion(
            model="gemini/gemini-2.5-pro",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format=SingleClaimClassification
        )
        
        # Parse the structured response
        result = response.choices[0].message.content
        
        if isinstance(result, str):
            result = json.loads(result)
        
        if hasattr(result, 'model_dump'):
            result = result.model_dump()
        elif hasattr(result, 'dict'):
            result = result.dict()
        
        # Add classification to claim
        claim['failure_mode'] = result.get('assigned_category', 'unknown')
        claim['classification_reasoning'] = result.get('reasoning', '')
                
    except Exception as e:
        # Fallback: mark as unclassified
        claim['failure_mode'] = 'error'
        claim['classification_reasoning'] = str(e)[:100]
    
    return claim


def classify_all_claims(claims: List[Dict], failure_modes: List[Dict], 
                       n_jobs: int = 4) -> List[Dict]:
    """
    Classify all claims using parallel processing.
    
    Args:
        claims: All unsupported claims to classify
        failure_modes: List of failure mode definitions
        n_jobs: Number of parallel jobs to run (default: 4)
        
    Returns:
        List of classified claims
    """
    print(f"Processing {len(claims)} claims using {n_jobs if n_jobs > 0 else 'all available'} parallel workers...")
    
    # Create a wrapper function that includes failure_modes
    def classify_wrapper(claim):
        return classify_single_claim(claim, failure_modes)
    
    # Process claims in parallel with progress bar
    classified_claims = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(classify_wrapper)(claim) 
        for claim in tqdm(claims, desc="Classifying claims")
    )
    
    return classified_claims


def generate_analysis_report(classified_claims: List[Dict], failure_modes: List[Dict]) -> Dict:
    """
    Generate comprehensive analysis report of classified claims.
    
    Args:
        classified_claims: Claims with assigned failure modes
        failure_modes: List of failure mode definitions
        
    Returns:
        Dictionary containing analysis results
    """
    # Create mode lookup
    mode_lookup = {mode['code']: mode['name'] for mode in failure_modes}
    
    # Overall statistics
    total_claims = len(classified_claims)
    
    # Distribution by failure mode
    mode_distribution = {}
    for claim in classified_claims:
        mode = claim.get('failure_mode', 'unknown')
        mode_distribution[mode] = mode_distribution.get(mode, 0) + 1
    
    # Distribution by system
    system_distribution = {}
    system_mode_distribution = {}
    
    for claim in classified_claims:
        system = claim.get('system', 'unknown')
        mode = claim.get('failure_mode', 'unknown')
        
        # Overall system count
        if system not in system_distribution:
            system_distribution[system] = {'total': 0, 'by_mode': {}}
        system_distribution[system]['total'] += 1
        
        # Mode distribution per system
        if mode not in system_distribution[system]['by_mode']:
            system_distribution[system]['by_mode'][mode] = 0
        system_distribution[system]['by_mode'][mode] += 1
    
    report = {
        'total_claims_classified': total_claims,
        'failure_mode_distribution': {
            mode: {
                'count': count,
                'percentage': (count / total_claims * 100) if total_claims > 0 else 0,
                'name': mode_lookup.get(mode, mode)
            }
            for mode, count in mode_distribution.items()
        },
        'system_analysis': system_distribution,
        'top_failure_modes': sorted(
            mode_distribution.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
    }
    
    return report


def main():
    """Main function to assign categories to all claims."""
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Assign failure mode categories to unsupported claims')
    parser.add_argument('--sample', type=int, default=None,
                        help='Number of claims to randomly sample and classify (default: classify all)')
    parser.add_argument('--stratified', action='store_true',
                        help='Use stratified sampling with equal split across systems')
    args = parser.parse_args()
    
    print("Assigning failure mode categories to unsupported claims...")
    print("=" * 80)
    
    # Load failure modes
    try:
        failure_mode_data = load_failure_modes()
        failure_modes = failure_mode_data.get('failure_modes', [])
        print(f"Loaded {len(failure_modes)} failure mode categories")
        
        for mode in failure_modes:
            print(f"  - {mode['name']} ({mode['code']})")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Load all unsupported claims
    try:
        all_claims = load_unsupported_claims()
        print(f"\nLoaded {len(all_claims)} unsupported claims")
        
        # Sample if requested
        if args.sample:
            import random
            sample_size = min(args.sample, len(all_claims))
            
            if args.stratified:
                # Stratified sampling across systems (equal split)
                from collections import defaultdict
                import math
                
                # Group claims by system
                claims_by_system = defaultdict(list)
                for claim in all_claims:
                    claims_by_system[claim.get('system', 'unknown')].append(claim)
                
                # Calculate equal samples per system
                num_systems = len(claims_by_system)
                samples_per_system = sample_size // num_systems
                remainder = sample_size % num_systems
                
                claims_to_classify = []
                system_counts = {}
                
                for i, (system, system_claims) in enumerate(claims_by_system.items()):
                    # Add one extra sample to first systems if there's a remainder
                    n_samples = samples_per_system + (1 if i < remainder else 0)
                    
                    # Ensure we don't sample more than available
                    n_samples = min(n_samples, len(system_claims))
                    
                    # Sample from this system
                    sampled = random.sample(system_claims, n_samples)
                    claims_to_classify.extend(sampled)
                    system_counts[system] = n_samples
                
                # Shuffle the combined list to mix systems
                random.shuffle(claims_to_classify)
                
                print(f"Stratified sampling (equal split): {len(claims_to_classify)} claims sampled")
                for system, count in system_counts.items():
                    print(f"  - {system}: {count} claims")
            else:
                # Simple random sampling
                claims_to_classify = random.sample(all_claims, sample_size)
                print(f"Randomly sampled {sample_size} claims to classify")
        else:
            claims_to_classify = all_claims
            print(f"Will classify all {len(claims_to_classify)} claims")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    print("\nClassifying claims using Gemini via litellm...")

    claims_to_classify = [c for c in claims_to_classify if not all(s == 'No snippet found.' for s in c['cited_snippets'].values())]
    try:
        # Classify claims
        classified_claims = classify_all_claims(claims_to_classify, failure_modes, n_jobs=-1)
        
        # Generate analysis report
        report = generate_analysis_report(classified_claims, failure_modes)
        
        # Save classified claims
        output_file = '../data/classified/classified_unsupported_claims.json'
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump({
                'classification_date': datetime.now().isoformat(),
                'total_claims': len(classified_claims),
                'is_sample': args.sample is not None,
                'sample_size': args.sample if args.sample else len(all_claims),
                'stratified_sampling': args.stratified if args.sample else False,
                'total_available_claims': len(all_claims),
                'failure_modes_used': failure_modes,
                'claims': classified_claims,
                'analysis_report': report
            }, f, indent=2)
        
        print(f"\nClassification complete! Results saved to {output_file}")
        
        # Save analysis report separately
        report_file = '../reports/failure_mode_analysis_report.json'
        Path(report_file).parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, 'w') as f:
            json.dump({
                'report_date': datetime.now().isoformat(),
                'is_sample': args.sample is not None,
                'sample_size': args.sample if args.sample else len(all_claims),
                'stratified_sampling': args.stratified if args.sample else False,
                **report
            }, f, indent=2)
        
        print(f"Analysis report saved to {report_file}")
        
        # Create CSV for easier analysis
        df_data = []
        for claim in classified_claims:
            df_data.append({
                'system': claim.get('system', ''),
                'claim_text': claim.get('claim_text', '')[:200],  # Truncate for CSV
                'failure_mode': claim.get('failure_mode', ''),
                'reasoning': claim.get('classification_reasoning', '')[:100],
                'has_supporting_refs': len(claim.get('supporting_refs', [])) > 0,
                'has_non_supporting_refs': len(claim.get('non_supporting_refs', [])) > 0
            })
        
        df = pd.DataFrame(df_data)
        csv_file = '../reports/classified_claims_summary.csv'
        Path(csv_file).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_file, index=False)
        print(f"Summary CSV saved to {csv_file}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("CLASSIFICATION SUMMARY")
        print("=" * 80)
        
        print(f"\nTotal claims classified: {report['total_claims_classified']}")
        
        print("\nTop 5 Failure Modes:")
        for mode_code, count in report['top_failure_modes']:
            mode_name = next((m['name'] for m in failure_modes if m['code'] == mode_code), mode_code)
            percentage = (count / report['total_claims_classified'] * 100) if report['total_claims_classified'] > 0 else 0
            print(f"  {mode_name:30} {count:6} ({percentage:.1f}%)")
        
        print("\nBreakdown by System:")
        for system, data in report['system_analysis'].items():
            print(f"\n  {system}:")
            print(f"    Total unsupported claims: {data['total']}")
            top_modes = sorted(data['by_mode'].items(), key=lambda x: x[1], reverse=True)[:3]
            for mode, count in top_modes:
                mode_name = next((m['name'] for m in failure_modes if m['code'] == mode), mode)
                print(f"    - {mode_name}: {count}")
        
            
    except Exception as e:
        print(f"Error during classification: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
