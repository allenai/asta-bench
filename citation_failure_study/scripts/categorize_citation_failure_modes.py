#!/usr/bin/env python3
"""
Script 2: Use Gemini-2.5-pro to analyze a random sample of non-supporting citations
and identify citation failure mode categories.
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


def load_non_supporting_citations(file_path: str = '../data/extracted/all_non_supporting_citations.json') -> List[Dict]:
    """Load non-supporting citations from JSON file."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Please run extract_non_supporting_citations.py first to generate {file_path}")
    
    with open(file_path, 'r') as f:
        return json.load(f)


def prepare_sample_for_analysis(citations: List[Dict], sample_size: int = 100, stratified: bool = False) -> List[Dict]:
    """
    Prepare a sample of non-supporting citations for analysis.
    
    Args:
        citations: List of all non-supporting citations
        sample_size: Number of citations to sample
        stratified: If True, use equal stratified sampling across systems
        
    Returns:
        List of sampled citations
    """
    if stratified:
        from collections import defaultdict
        
        # Group citations by system
        citations_by_system = defaultdict(list)
        for citation in citations:
            citations_by_system[citation.get('system', 'unknown')].append(citation)
        
        # Calculate equal samples per system
        num_systems = len(citations_by_system)
        samples_per_system = sample_size // num_systems
        remainder = sample_size % num_systems
        
        sampled_citations = []
        for i, (system, system_citations) in enumerate(citations_by_system.items()):
            # Add one extra sample to first systems if there's a remainder
            n_samples = samples_per_system + (1 if i < remainder else 0)
            
            # Ensure we don't sample more than available
            n_samples = min(n_samples, len(system_citations))
            
            # Sample from this system
            sampled = random.sample(system_citations, n_samples)
            sampled_citations.extend(sampled)
        
        # Shuffle to mix systems
        random.shuffle(sampled_citations)
        
        print(f"\nStratified sampling (equal split across {num_systems} systems):")
        for system in citations_by_system:
            count = sum(1 for c in sampled_citations if c.get('system') == system)
            print(f"  - {system}: {count} citations")
            
    else:
        # Ensure we don't sample more than available
        actual_sample_size = min(sample_size, len(citations))
        
        # Random sample
        sampled_citations = random.sample(citations, actual_sample_size)
    
    # Prepare for analysis - simplify the structure
    prepared_sample = []
    for citation in sampled_citations:
        if citation.get('citation_snippet', '') == 'No snippet found.': continue
        prepared_item = {
            'system': citation.get('system', 'unknown'),
            'claim': citation.get('claim_text', ''),
            'citation_id': citation.get('citation_id', ''),
            'citation_snippet': citation.get('citation_snippet', ''),
            'is_fully_supported': citation.get('is_fully_supported', False),
            'has_supporting_refs': len(citation.get('supporting_refs', [])) > 0,
            'other_non_supporting_refs': len(citation.get('all_non_supporting_refs', [])) > 1
        }
        
        prepared_sample.append(prepared_item)
    
    return prepared_sample


def analyze_citation_failure_modes_with_llm(sample: List[Dict]) -> Dict[str, Any]:
    """
    Use LLM to analyze the sample and identify citation failure mode categories.
    
    Args:
        sample: List of sampled non-supporting citations
        
    Returns:
        Dictionary containing identified failure modes and analysis
    """
    
    # Prepare the prompt
    prompt = """You are analyzing non-supporting citations from scientific question-answering systems. 
These are citations that were marked as "non-supporting" for specific claims.

Your task is to identify common failure modes that explain why these citations don't support (at least partially) their associated claims.

Here are randomly sampled non-supporting citations from different systems (elicit, openai_deep_research, perplexity_dr, claude_4.0):

"""
    
    # Add each citation-claim pair to the prompt
    for i, item in enumerate(sample, 1):
        prompt += f"\n--- Example {i} (System: {item['system']}) ---\n"
        prompt += f"Claim: {item['claim']}\n"
        prompt += f"Citation ID: {item['citation_id']}\n"
        prompt += f"Citation Snippet: {item['citation_snippet']}...\n"
        prompt += f"Claim has other supporting refs: {item['has_supporting_refs']}\n"
        prompt += f"Multiple non-supporting refs: {item['other_non_supporting_refs']}\n"
    
    prompt += """

Based on these examples, identify and describe the main failure mode categories for why citations don't support (even partially) claims.
Consider the relationship between claim content and citation content. For each category:
1. Give it a short, descriptive name (e.g., "topic_mismatch", "insufficient_evidence", "contradictory_evidence")
2. Provide a clear description of what characterizes this failure mode
3. Estimate what percentage of the sample falls into this category

Output your analysis as a JSON object with the following structure:
{
  "citation_failure_modes": [
    {
      "code": "short_code_name",
      "name": "Human readable name",
      "description": "Clear description of why citations fail in this mode",
      "characteristics": ["list", "of", "key", "characteristics"],
      "estimated_percentage": 15
    }
  ],
  "additional_observations": "Any other important observations about citation failures"
}

Focus on identifying 5-10 distinct failure modes that cover most of the cases."""
    
    # Get response from LLM using litellm with structured output
    from pydantic import BaseModel, Field
    from typing import List
    
    class CitationFailureMode(BaseModel):
        code: str = Field(description="Short code name for the citation failure mode")
        name: str = Field(description="Human readable name")
        description: str = Field(description="Clear description of why citations fail in this mode")
        characteristics: List[str] = Field(description="List of key characteristics")
        estimated_percentage: float = Field(description="Estimated percentage of sample")
    
    class CitationFailureModeAnalysis(BaseModel):
        citation_failure_modes: List[CitationFailureMode]
        additional_observations: str = Field(description="Any other important observations about citation failures")
    
    response = completion(
        model="gemini/gemini-2.5-pro",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format=CitationFailureModeAnalysis
    )
    
    # Parse the structured response
    try:
        # The response should already be structured as CitationFailureModeAnalysis
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
            "citation_failure_modes": [
                {
                    "code": "topic_mismatch",
                    "name": "Topic Mismatch",
                    "description": "Citation discusses different topic than claim",
                    "characteristics": ["Different subject matter", "Unrelated content"],
                    "estimated_percentage": 25
                },
                {
                    "code": "insufficient_evidence",
                    "name": "Insufficient Evidence",
                    "description": "Citation doesn't provide enough evidence to support claim",
                    "characteristics": ["Partial information only", "Missing key details"],
                    "estimated_percentage": 30
                },
                {
                    "code": "contradictory_evidence",
                    "name": "Contradictory Evidence",
                    "description": "Citation actually contradicts or opposes the claim",
                    "characteristics": ["Opposite findings", "Conflicting information"],
                    "estimated_percentage": 15
                },
                {
                    "code": "scope_mismatch",
                    "name": "Scope Mismatch",
                    "description": "Citation has different scope than claim (e.g., general vs specific)",
                    "characteristics": ["Different granularity", "Mismatched context"],
                    "estimated_percentage": 20
                },
                {
                    "code": "temporal_mismatch",
                    "name": "Temporal Mismatch",
                    "description": "Citation from different time period or outdated",
                    "characteristics": ["Outdated information", "Different time context"],
                    "estimated_percentage": 10
                }
            ],
            "additional_observations": "Failed to get proper analysis from Gemini",
            "error": str(e)
        }
    
    return result


def main():
    """Main function to categorize citation failure modes."""
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Categorize citation failure modes from non-supporting citations')
    parser.add_argument('--stratified', action='store_true',
                        help='Use stratified sampling with equal split across systems')
    parser.add_argument('--sample-size', type=int, default=100,
                        help='Number of citations to sample for analysis (default: 100)')
    args = parser.parse_args()
    
    print("Categorizing citation failure modes for non-supporting citations...")
    print("=" * 80)
    
    # Load non-supporting citations
    try:
        all_citations = load_non_supporting_citations()
        print(f"Loaded {len(all_citations)} non-supporting citations")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Prepare sample
    sample = prepare_sample_for_analysis(all_citations, sample_size=args.sample_size, stratified=args.stratified)
    print(f"Prepared sample of {len(sample)} citations for analysis")
    
    print("\nAnalyzing with Gemini-2.5-pro via litellm...")
    
    try:
        # Analyze failure modes
        analysis_result = analyze_citation_failure_modes_with_llm(sample)
        
        # Add metadata
        final_result = {
            'analysis_date': datetime.now().isoformat(),
            'sample_size': len(sample),
            'stratified_sampling': args.stratified,
            'total_citations_analyzed': len(all_citations),
            'citation_failure_modes': analysis_result.get('citation_failure_modes', []),
            'additional_observations': analysis_result.get('additional_observations', ''),
            'sampled_citations': sample  # Include the sample for reference
        }
        
        # Save results
        output_file = '../data/categories/citation_failure_mode_categories.json'
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(final_result, f, indent=2)
        
        print(f"\nAnalysis complete! Results saved to {output_file}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("IDENTIFIED CITATION FAILURE MODES:")
        print("=" * 80)
        
        for mode in final_result['citation_failure_modes']:
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
            'total_citations_analyzed': len(all_citations),
            'citation_failure_modes': [
                {
                    "code": "topic_mismatch",
                    "name": "Topic Mismatch",
                    "description": "Citation discusses a different topic than what the claim asserts",
                    "characteristics": ["Different subject matter", "Unrelated content", "Wrong domain"],
                    "estimated_percentage": 25
                },
                {
                    "code": "insufficient_evidence",
                    "name": "Insufficient Evidence",
                    "description": "Citation doesn't provide enough evidence to fully support the claim",
                    "characteristics": ["Partial information only", "Missing key details", "Incomplete support"],
                    "estimated_percentage": 30
                },
                {
                    "code": "contradictory_evidence",
                    "name": "Contradictory Evidence",
                    "description": "Citation actually contradicts or opposes what the claim states",
                    "characteristics": ["Opposite findings", "Conflicting information", "Disagreement"],
                    "estimated_percentage": 15
                },
                {
                    "code": "scope_mismatch",
                    "name": "Scope Mismatch",
                    "description": "Citation has different scope or granularity than the claim",
                    "characteristics": ["Too general", "Too specific", "Different context"],
                    "estimated_percentage": 20
                },
                {
                    "code": "weak_relevance",
                    "name": "Weak Relevance",
                    "description": "Citation is tangentially related but not directly relevant",
                    "characteristics": ["Indirect connection", "Weak link", "Peripheral relevance"],
                    "estimated_percentage": 10
                }
            ],
            'additional_observations': "Template generated due to API error",
            'sampled_citations': sample[:10]  # Include first 10 for reference
        }
        
        template_file = '../data/categories/citation_failure_mode_categories_template.json'
        Path(template_file).parent.mkdir(parents=True, exist_ok=True)
        with open(template_file, 'w') as f:
            json.dump(template, f, indent=2)
        
        print(f"\nTemplate saved to {template_file} for manual review")


if __name__ == "__main__":
    main()
