#!/usr/bin/env python3
"""
Extract example failures from OpenAI Deep Research system and format as markdown report.
"""

import json
import random
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def load_classified_data():
    """Load classified claims and citations data."""
    # Load classified claims (sample)
    with open('../data/classified/classified_unsupported_claims.json', 'r') as f:
        claims_data = json.load(f)
    
    # Load all unsupported claims for more examples
    with open('../data/extracted/all_unsupported_claims.json', 'r') as f:
        all_claims = json.load(f)
    
    # Load classified citations
    with open('../data/classified/classified_non_supporting_citations.json', 'r') as f:
        citations_data = json.load(f)
    
    # Load all citations for more examples
    with open('../data/extracted/all_non_supporting_citations.json', 'r') as f:
        all_citations = json.load(f)
    
    return claims_data, all_claims, citations_data, all_citations


def get_openai_examples(claims_data, all_claims, citations_data, all_citations):
    """Extract OpenAI Deep Research examples from the data."""
    
    # Get classified OpenAI claims
    openai_claims_classified = [
        c for c in claims_data.get('claims', []) 
        if c['system'] == 'openai_deep_research'
    ]
    
    # Get all OpenAI claims for additional examples
    openai_claims_all = [
        c for c in all_claims 
        if c['system'] == 'openai_deep_research'
    ]
    
    # Get classified OpenAI citations
    openai_citations_classified = [
        c for c in citations_data 
        if c['system'] == 'openai_deep_research'
    ]
    
    # Get all OpenAI citations for additional examples
    openai_citations_all = [
        c for c in all_citations 
        if c['system'] == 'openai_deep_research'
    ]
    
    return (openai_claims_classified, openai_claims_all, 
            openai_citations_classified, openai_citations_all)


def select_claim_examples(classified_claims, all_claims, num_per_category=3):
    """Select diverse examples of claim failures."""
    
    # Group classified claims by failure mode
    by_mode = defaultdict(list)
    for claim in classified_claims:
        mode = claim.get('failure_mode', 'unknown')
        if mode and mode != 'claim_is_fully_supported':
            by_mode[mode].append(claim)
    
    # If we need more examples, sample from all claims
    # (these won't have classification reasoning but show the pattern)
    unclassified_by_pattern = defaultdict(list)
    for claim in all_claims:
        # Categorize by presence of references
        if not claim.get('non_supporting_refs'):
            if 'without_any_citations' not in by_mode or len(by_mode['without_any_citations']) < num_per_category:
                unclassified_by_pattern['without_any_citations'].append(claim)
        elif any('unavailable' in str(ref).lower() for ref in claim.get('non_supporting_refs', [])):
            if 'citation_unavailable' not in by_mode or len(by_mode['citation_unavailable']) < num_per_category:
                unclassified_by_pattern['citation_unavailable'].append(claim)
    
    # Select examples
    selected = {}
    for mode, claims in by_mode.items():
        # Take up to num_per_category classified examples
        selected[mode] = claims[:num_per_category]
        
        # If we don't have enough, add unclassified examples
        if len(selected[mode]) < num_per_category and mode in unclassified_by_pattern:
            remaining = num_per_category - len(selected[mode])
            additional = random.sample(
                unclassified_by_pattern[mode], 
                min(remaining, len(unclassified_by_pattern[mode]))
            )
            selected[mode].extend(additional)
    
    return selected


def select_citation_examples(classified_citations, all_citations, num_per_category=3):
    """Select diverse examples of citation failures."""
    
    # Group by category
    by_category = defaultdict(list)
    for citation in classified_citations:
        category = citation.get('assigned_category', 'unknown')
        if category:
            by_category[category].append(citation)
    
    # Select examples
    selected = {}
    for category, citations in by_category.items():
        # Prioritize examples with good reasoning
        citations_with_reasoning = [c for c in citations if c.get('classification_reasoning')]
        citations_without = [c for c in citations if not c.get('classification_reasoning')]
        
        # Take classified examples first
        selected[category] = citations_with_reasoning[:num_per_category]
        
        # Add more if needed
        if len(selected[category]) < num_per_category:
            remaining = num_per_category - len(selected[category])
            selected[category].extend(citations_without[:remaining])
    
    return selected


def format_claim_example(claim, index):
    """Format a single claim example as markdown."""
    lines = []
    
    lines.append(f"#### Example {index}")
    lines.append("")
    lines.append(f"**Question:** {claim.get('question', 'N/A')}")
    lines.append("")
    lines.append(f"**Claim:** {claim.get('claim_text', 'N/A')}")
    lines.append("")
    
    # Add references if present
    non_supporting = claim.get('non_supporting_refs', [])
    if non_supporting:
        lines.append(f"**References Provided:** {', '.join(str(r) for r in non_supporting)}")
    else:
        lines.append("**References Provided:** None")
    lines.append("")
    
    # Add snippets if available - check both 'cited_snippets' and 'non_supporting_snippets'
    snippets = claim.get('cited_snippets') or claim.get('non_supporting_snippets')
    if snippets:
        lines.append("**Citation Snippets:**")
        lines.append("")
        for ref, snippet in snippets.items():
            # Format snippet with proper truncation
            if snippet:
                snippet_text = snippet.strip()
                if len(snippet_text) > 500:
                    snippet_text = snippet_text[:500] + "..."
                lines.append(f"**[{ref}]:** {snippet_text}")
                lines.append("")
    
    # Add classification reasoning if available
    if claim.get('classification_reasoning'):
        lines.append(f"**Analysis:** {claim['classification_reasoning']}")
        lines.append("")
    
    return '\n'.join(lines)


def format_citation_example(citation, index):
    """Format a single citation example as markdown."""
    lines = []
    
    lines.append(f"#### Example {index}")
    lines.append("")
    lines.append(f"**Question:** {citation.get('question', 'N/A')}")
    lines.append("")
    lines.append(f"**Claim:** {citation.get('claim_text', 'N/A')}")
    lines.append("")
    lines.append(f"**Citation ID:** {citation.get('citation_id', 'N/A')}")
    lines.append("")
    
    # Add snippet
    snippet = citation.get('citation_snippet', '')
    if snippet:
        # Truncate if too long
        if len(snippet) > 500:
            snippet = snippet[:500] + "..."
        lines.append(f"**Citation Content:** {snippet}")
        lines.append("")
    
    # Add classification reasoning
    if citation.get('classification_reasoning'):
        lines.append(f"**Analysis:** {citation['classification_reasoning']}")
        lines.append("")
    
    return '\n'.join(lines)


def generate_markdown_report(claim_examples, citation_examples):
    """Generate the complete markdown report."""
    lines = []
    
    # Header
    lines.append("# OpenAI Deep Research Failure Examples")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("This report provides concrete examples of citation and claim failures from the OpenAI Deep Research system, organized by failure mode.")
    lines.append("")
    
    # Claim failures section
    lines.append("## Claim Failures")
    lines.append("")
    lines.append("These are claims that were identified as unsupported or partially supported by their citations.")
    lines.append("")
    
    # Define failure mode descriptions
    failure_mode_descriptions = {
        'partial_support': 'Claims where citations are relevant but don\'t fully substantiate all aspects',
        'without_any_citations': 'Claims presented as facts but lacking any citation',
        'irrelevant_citation': 'Claims with citations unrelated to the subject matter',
        'citation_unavailable': 'Claims where citation content could not be retrieved',
        'no_citation': 'Claims explicitly lacking citations'
    }
    
    # Add claim examples by category
    for mode in ['without_any_citations', 'partial_support', 'irrelevant_citation', 'citation_unavailable']:
        if mode in claim_examples and claim_examples[mode]:
            lines.append(f"### {mode.replace('_', ' ').title()}")
            lines.append("")
            
            # Add description
            if mode in failure_mode_descriptions:
                lines.append(f"*{failure_mode_descriptions[mode]}*")
                lines.append("")
            
            # Add examples
            for i, claim in enumerate(claim_examples[mode], 1):
                lines.append(format_claim_example(claim, i))
            
            lines.append("---")
            lines.append("")
    
    # Citation failures section
    lines.append("## Citation Failures")
    lines.append("")
    lines.append("These are individual citations that don't support their associated claims.")
    lines.append("")
    
    # Define category descriptions
    category_descriptions = {
        'insufficient_detail': 'Citations providing general information but lacking specific evidence needed',
        'topic_mismatch': 'Citations from same domain but focusing on different aspects',
        'factual_mismatch': 'Citations with different facts than claimed',
        'source_unavailable': 'Citations where content could not be retrieved',
        'metadata_only': 'Citations containing only bibliographic information',
        'contradiction': 'Citations that contradict the claim',
        'flawed_claim_premise': 'Claims based on false premises about citations'
    }
    
    # Add citation examples by category
    for category in ['insufficient_detail', 'topic_mismatch', 'source_unavailable', 'metadata_only']:
        if category in citation_examples and citation_examples[category]:
            lines.append(f"### {category.replace('_', ' ').title()}")
            lines.append("")
            
            # Add description
            if category in category_descriptions:
                lines.append(f"*{category_descriptions[category]}*")
                lines.append("")
            
            # Add examples
            for i, citation in enumerate(citation_examples[category], 1):
                lines.append(format_citation_example(citation, i))
            
            lines.append("---")
            lines.append("")
    
    # Footer
    lines.append("## Summary")
    lines.append("")
    lines.append("These examples illustrate the various ways OpenAI Deep Research fails to properly support its claims with citations:")
    lines.append("")
    lines.append("1. **Missing Citations:** Many claims lack any citations despite being factual assertions")
    lines.append("2. **Partial Support:** Citations are on-topic but miss crucial details or evidence")
    lines.append("3. **Topic Mismatch:** Citations discuss related but different aspects than claimed")
    lines.append("4. **Unavailable Sources:** System cites sources it cannot actually access or verify")
    lines.append("")
    
    return '\n'.join(lines)


def main():
    """Main function."""
    print("Loading classified data...")
    claims_data, all_claims, citations_data, all_citations = load_classified_data()
    
    print("Extracting OpenAI Deep Research examples...")
    (openai_claims_classified, openai_claims_all, 
     openai_citations_classified, openai_citations_all) = get_openai_examples(
        claims_data, all_claims, citations_data, all_citations
    )
    
    print(f"Found {len(openai_claims_classified)} classified claims")
    print(f"Found {len(openai_claims_all)} total claims")
    print(f"Found {len(openai_citations_classified)} classified citations")
    print(f"Found {len(openai_citations_all)} total citations")
    
    print("Selecting claim examples...")
    claim_examples = select_claim_examples(openai_claims_classified, openai_claims_all)
    
    print("Selecting citation examples...")
    citation_examples = select_citation_examples(openai_citations_classified, openai_citations_all)
    
    print("Generating markdown report...")
    report = generate_markdown_report(claim_examples, citation_examples)
    
    # Save report
    output_path = Path('../reports/openai_failure_examples.md')
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Report saved to: {output_path}")
    
    # Print summary
    print("\nExample counts by category:")
    print("\nClaim failures:")
    for mode, examples in claim_examples.items():
        print(f"  {mode}: {len(examples)} examples")
    
    print("\nCitation failures:")
    for category, examples in citation_examples.items():
        print(f"  {category}: {len(examples)} examples")


if __name__ == "__main__":
    main()