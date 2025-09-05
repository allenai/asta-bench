#!/usr/bin/env python3
"""
Script 2: Use Gemini-2.5-pro to analyze a random sample of unsupported claims
and identify failure mode categories.
"""

import json
import random
import os
from typing import List, Dict, Any
from pathlib import Path
from litellm import completion
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()


def load_unsupported_claims(file_path: str = '../data/extracted/all_unsupported_claims.json') -> List[Dict]:
    """Load unsupported claims from JSON file."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Please run extract_unsupported_claims.py first to generate {file_path}")
    
    with open(file_path, 'r') as f:
        return json.load(f)


def prepare_sample_for_analysis(claims: List[Dict], sample_size: int = 100, stratified: bool = False) -> List[Dict]:
    """
    Prepare a sample of claims for analysis.
    
    Args:
        claims: List of all unsupported claims
        sample_size: Number of claims to sample
        stratified: If True, use equal stratified sampling across systems
        
    Returns:
        List of sampled claims
    """
    if stratified:
        from collections import defaultdict
        
        # Group claims by system
        claims_by_system = defaultdict(list)
        for claim in claims:
            claims_by_system[claim.get('system', 'unknown')].append(claim)
        
        # Calculate equal samples per system
        num_systems = len(claims_by_system)
        samples_per_system = sample_size // num_systems
        remainder = sample_size % num_systems
        
        sampled_claims = []
        for i, (system, system_claims) in enumerate(claims_by_system.items()):
            # Add one extra sample to first systems if there's a remainder
            n_samples = samples_per_system + (1 if i < remainder else 0)
            
            # Ensure we don't sample more than available
            n_samples = min(n_samples, len(system_claims))
            
            # Sample from this system
            sampled = random.sample(system_claims, n_samples)
            sampled_claims.extend(sampled)
        
        # Shuffle to mix systems
        random.shuffle(sampled_claims)
        
        print(f"\nStratified sampling (equal split across {num_systems} systems):")
        for system in claims_by_system:
            count = sum(1 for c in sampled_claims if c.get('system') == system)
            print(f"  - {system}: {count} claims")
            
    else:
        # Ensure we don't sample more than available
        actual_sample_size = min(sample_size, len(claims))
        
        # Random sample
        sampled_claims = random.sample(claims, actual_sample_size)
    
    # Prepare for analysis - simplify the structure
    prepared_sample = []
    for claim in sampled_claims:
        prepared_item = {
            'system': claim.get('system', 'unknown'),
            'claim': claim.get('claim_text', ''),
            'has_supporting_refs': len(claim.get('supporting_refs', [])) > 0,
            'has_non_supporting_refs': len(claim.get('non_supporting_refs', [])) > 0,
            'supporting_refs': claim.get('supporting_refs', []),
            'non_supporting_refs': claim.get('non_supporting_refs', []),
            'cited_snippets_sample': ''
        }
        
        # Add a sample of cited snippets
        snippets = claim.get('cited_snippets', {})
        if snippets:
            first_snippet = list(snippets.values())[0] if snippets else ''
            if first_snippet == 'No snippet found.': continue
            prepared_item['cited_snippets_sample'] = first_snippet if first_snippet else ''
        
        prepared_sample.append(prepared_item)
    
    return prepared_sample


def analyze_failure_modes_with_llm(sample: List[Dict]) -> Dict[str, Any]:
    """
    Use LLM to analyze the sample and identify failure mode categories.
    
    Args:
        sample: List of sampled unsupported claims
        
    Returns:
        Dictionary containing identified failure modes and analysis
    """
    
    # Prepare the prompt
    prompt = """You are analyzing unsupported claims from scientific question-answering systems. 
These claims were marked as "not fully supported" by their citations.

Your task is to identify common failure modes that explain why these claims are not fully supported.

Here are randomly sampled unsupported claims from different systems (elicit, openai_deep_research, perplexity_dr, claude_4.0):

"""
    
    # Add each claim to the prompt
    for i, item in enumerate(sample, 1):
        prompt += f"\n--- Claim {i} (System: {item['system']}) ---\n"
        prompt += f"Claim: {item['claim']}\n"
        
        if item['has_supporting_refs'] or item['has_non_supporting_refs']:
            prompt += "Citations provided: Yes\n"
            if item['supporting_refs']:
                prompt += f"  Supporting refs: {item['supporting_refs']}\n"
            if item['non_supporting_refs']:
                prompt += f"  Non-supporting refs: {item['non_supporting_refs']}\n"
            if item['cited_snippets_sample']:
                prompt += f"  Sample snippet: {item['cited_snippets_sample']}...\n"
        else:
            prompt += "Citations provided: No\n"
    
    prompt += """

Based on these examples, identify and describe the main failure mode categories. For each category:
1. Give it a short, descriptive name (e.g., "no_citation", "irrelevant_citation", "partial_support")
2. Provide a clear description of what characterizes this failure mode
3. Estimate what percentage of the sample falls into this category

Output your analysis as a JSON object with the following structure:
{
  "failure_modes": [
    {
      "code": "short_code_name",
      "name": "Human readable name",
      "description": "Clear description of this failure mode",
      "characteristics": ["list", "of", "key", "characteristics"],
      "estimated_percentage": 15
    }
  ],
  "additional_observations": "Any other important observations about the failures"
}

Focus on identifying 5-10 distinct failure modes that cover most of the cases."""
    
    # Get response from LLM using litellm with structured output
    from pydantic import BaseModel, Field
    from typing import List
    
    class FailureMode(BaseModel):
        code: str = Field(description="Short code name for the failure mode")
        name: str = Field(description="Human readable name")
        description: str = Field(description="Clear description of this failure mode")
        characteristics: List[str] = Field(description="List of key characteristics")
        estimated_percentage: float = Field(description="Estimated percentage of sample")
    
    class FailureModeAnalysis(BaseModel):
        failure_modes: List[FailureMode]
        additional_observations: str = Field(description="Any other important observations about the failures")
    
    response = completion(
        model="gemini/gemini-2.5-pro",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format=FailureModeAnalysis
    )
    
    # Parse the structured response
    try:
        # The response should already be structured as FailureModeAnalysis
        result = response.choices[0].message.content
        
        # If it's a string, parse it as JSON
        if isinstance(result, str):
            result = json.loads(result)
        
        # Convert Pydantic models to dict if needed
        if hasattr(result, 'model_dump'):
            result = result.model_dump()
        elif hasattr(result, 'dict'):
            result = result.dict()
            
    except Exception as e:
        print(f"Error parsing response: {e}")
        
        # Create a fallback structure
        result = {
            "failure_modes": [
                {
                    "code": "no_citation",
                    "name": "No Citation Provided",
                    "description": "Claim lacks any citation or reference",
                    "characteristics": ["No references listed", "Unsupported assertion"],
                    "estimated_percentage": 20
                },
                {
                    "code": "irrelevant_citation",
                    "name": "Irrelevant Citation",
                    "description": "Citation provided but doesn't support the claim",
                    "characteristics": ["Citation exists but off-topic", "Mismatched content"],
                    "estimated_percentage": 30
                }
            ],
            "additional_observations": "Failed to get proper analysis from Gemini",
            "error": str(e)
        }
    
    return result


def main():
    """Main function to categorize failure modes."""
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Categorize failure modes from unsupported claims')
    parser.add_argument('--stratified', action='store_true',
                        help='Use stratified sampling with equal split across systems')
    parser.add_argument('--sample-size', type=int, default=100,
                        help='Number of claims to sample for analysis (default: 100)')
    args = parser.parse_args()
    
    print("Categorizing failure modes for unsupported claims...")
    print("=" * 80)
    
    # Load unsupported claims
    try:
        all_claims = load_unsupported_claims()
        print(f"Loaded {len(all_claims)} unsupported claims")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Prepare sample
    sample = prepare_sample_for_analysis(all_claims, sample_size=args.sample_size, stratified=args.stratified)
    print(f"Prepared sample of {len(sample)} claims for analysis")
    
    print("\nAnalyzing with Gemini-2.5-pro via litellm...")
    
    try:
        # Analyze failure modes
        analysis_result = analyze_failure_modes_with_llm(sample)
        
        # Add metadata
        final_result = {
            'analysis_date': datetime.now().isoformat(),
            'sample_size': len(sample),
            'stratified_sampling': args.stratified,
            'total_claims_analyzed': len(all_claims),
            'failure_modes': analysis_result.get('failure_modes', []),
            'additional_observations': analysis_result.get('additional_observations', ''),
            'sampled_claims': sample  # Include the sample for reference
        }
        
        # Save results
        output_file = '../data/categories/failure_mode_categories.json'
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(final_result, f, indent=2)
        
        print(f"\nAnalysis complete! Results saved to {output_file}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("IDENTIFIED FAILURE MODES:")
        print("=" * 80)
        
        for mode in final_result['failure_modes']:
            print(f"\n{mode['name']} ({mode['code']})")
            print(f"  Description: {mode['description']}")
            print(f"  Estimated percentage: {mode.get('estimated_percentage', 'N/A')}%")
            if 'characteristics' in mode:
                print(f"  Characteristics: {', '.join(mode['characteristics'])}")
        
        if final_result.get('additional_observations'):
            print(f"\nAdditional observations: {final_result['additional_observations']}")
            
    except Exception as e:
        print(f"Error during analysis: {e}")
        
        # Save a basic template for manual editing if needed
        template = {
            'analysis_date': datetime.now().isoformat(),
            'sample_size': len(sample),
            'total_claims_analyzed': len(all_claims),
            'failure_modes': [
                {
                    "code": "no_citation",
                    "name": "No Citation Provided",
                    "description": "The claim lacks any citation or reference to support it",
                    "characteristics": ["No references listed", "Unsupported assertion"],
                    "estimated_percentage": 15
                },
                {
                    "code": "irrelevant_citation", 
                    "name": "Irrelevant Citation",
                    "description": "Citation is provided but content doesn't support the claim",
                    "characteristics": ["Citation exists but off-topic", "Mismatched content"],
                    "estimated_percentage": 25
                },
                {
                    "code": "partial_support",
                    "name": "Partial Support Only",
                    "description": "Citation supports only part of the claim, not all assertions",
                    "characteristics": ["Some aspects supported", "Other aspects unsupported"],
                    "estimated_percentage": 30
                },
                {
                    "code": "overstated_claim",
                    "name": "Overstated Claim",
                    "description": "Claim makes stronger assertion than what citation supports",
                    "characteristics": ["Exaggerated conclusion", "Goes beyond evidence"],
                    "estimated_percentage": 20
                },
                {
                    "code": "wrong_citation_format",
                    "name": "Wrong Citation Format",
                    "description": "Citation exists but is improperly referenced or formatted",
                    "characteristics": ["Malformed reference", "Citation parsing error"],
                    "estimated_percentage": 10
                }
            ],
            'additional_observations': "Template generated due to API error",
            'sampled_claims': sample[:10]  # Include first 10 for reference
        }
        
        template_file = '../data/categories/failure_mode_categories_template.json'
        Path(template_file).parent.mkdir(parents=True, exist_ok=True)
        with open(template_file, 'w') as f:
            json.dump(template, f, indent=2)
        
        print(f"\nTemplate saved to {template_file} for manual review")


if __name__ == "__main__":
    main()
