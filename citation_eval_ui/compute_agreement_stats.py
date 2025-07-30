#!/usr/bin/env python3
"""
Compute summary statistics about annotation agreement in citation evaluation data.
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


def load_annotations(json_path: Path) -> Dict:
    """Load annotations from a JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def compute_citation_agreement(annotations: Dict) -> Dict:
    """Compute agreement statistics for citation labels."""
    stats = {
        'total_claims': 0,
        'total_citations': 0,
        'citation_label_distribution': defaultdict(int),
        'agreement_with_initial': {
            'agree': 0,
            'disagree': 0,
            'total': 0
        },
        'is_fully_supported_agreement': {
            'agree': 0,
            'disagree': 0,
            'null': 0,
            'total': 0
        },
        'is_fully_supported_confusion': {
            'initial_true_annotated_agree': 0,
            'initial_true_annotated_disagree': 0,
            'initial_false_annotated_agree': 0,
            'initial_false_annotated_disagree': 0
        }
    }
    
    for row_id, row_data in annotations['row_annotations'].items():
        for claim_id, claim_data in row_data.items():
            stats['total_claims'] += 1
            
            # Count citation labels
            if 'citation_annotations' in claim_data:
                for citation in claim_data['citation_annotations']:
                    stats['total_citations'] += 1
                    label = citation.get('label', 'unknown')
                    stats['citation_label_distribution'][label] += 1
            
            # Check agreement with initial labels
            initial_supporting = set(claim_data.get('initial_supporting', []))
            initial_non_supporting = set(claim_data.get('initial_non_supporting', []))
            
            if 'citation_annotations' in claim_data:
                for citation in claim_data['citation_annotations']:
                    citation_id = citation.get('citation_id', '')
                    label = citation.get('label', '')
                    
                    # Compare with initial classification
                    if citation_id in initial_supporting:
                        if label == 'supporting':
                            stats['agreement_with_initial']['agree'] += 1
                        else:
                            stats['agreement_with_initial']['disagree'] += 1
                        stats['agreement_with_initial']['total'] += 1
                    elif citation_id in initial_non_supporting:
                        if label in ['non_supporting', 'not_supporting']:
                            stats['agreement_with_initial']['agree'] += 1
                        else:
                            stats['agreement_with_initial']['disagree'] += 1
                        stats['agreement_with_initial']['total'] += 1
            
            # Check is_fully_supported agreement
            initial_fully_supported = claim_data.get('initial_is_fully_supported', None)
            annotation_fully_supported = claim_data.get('is_fully_supported_annotation', None)
            
            if annotation_fully_supported is not None:
                if annotation_fully_supported == 'disagree':
                    stats['is_fully_supported_agreement']['disagree'] += 1
                elif annotation_fully_supported == 'agree':
                    stats['is_fully_supported_agreement']['agree'] += 1
                else:
                    stats['is_fully_supported_agreement']['null'] += 1
                stats['is_fully_supported_agreement']['total'] += 1
                
                # Build confusion matrix data
                if initial_fully_supported is not None and annotation_fully_supported in ['agree', 'disagree']:
                    if initial_fully_supported == True:
                        if annotation_fully_supported == 'agree':
                            stats['is_fully_supported_confusion']['initial_true_annotated_agree'] += 1
                        else:  # disagree
                            stats['is_fully_supported_confusion']['initial_true_annotated_disagree'] += 1
                    else:  # initial_fully_supported == False
                        if annotation_fully_supported == 'agree':
                            stats['is_fully_supported_confusion']['initial_false_annotated_agree'] += 1
                        else:  # disagree
                            stats['is_fully_supported_confusion']['initial_false_annotated_disagree'] += 1
    
    return stats


def print_statistics(stats: Dict, filename: str):
    """Print formatted statistics."""
    print(f"\n{'='*60}")
    print(f"Agreement Statistics for: {filename}")
    print(f"{'='*60}")
    
    print(f"\nTotal claims annotated: {stats['total_claims']}")
    print(f"Total citations evaluated: {stats['total_citations']}")
    
    print("\nCitation Label Distribution:")
    for label, count in sorted(stats['citation_label_distribution'].items()):
        percentage = (count / stats['total_citations'] * 100) if stats['total_citations'] > 0 else 0
        print(f"  {label}: {count} ({percentage:.1f}%)")
    
    print("\nAgreement with Initial Citation Labels:")
    agree_stats = stats['agreement_with_initial']
    if agree_stats['total'] > 0:
        agree_pct = agree_stats['agree'] / agree_stats['total'] * 100
        disagree_pct = agree_stats['disagree'] / agree_stats['total'] * 100
        print(f"  Agree: {agree_stats['agree']} ({agree_pct:.1f}%)")
        print(f"  Disagree: {agree_stats['disagree']} ({disagree_pct:.1f}%)")
        print(f"  Total evaluated: {agree_stats['total']}")
    else:
        print("  No initial labels to compare")
    
    print("\nIs Fully Supported Annotation Distribution:")
    fully_supported_stats = stats['is_fully_supported_agreement']
    if fully_supported_stats['total'] > 0:
        for label in ['agree', 'disagree', 'null']:
            count = fully_supported_stats[label]
            percentage = count / fully_supported_stats['total'] * 100
            print(f"  {label}: {count} ({percentage:.1f}%)")
        print(f"  Total: {fully_supported_stats['total']}")
    else:
        print("  No is_fully_supported annotations found")
    
    # Print confusion matrix
    confusion = stats['is_fully_supported_confusion']
    total_confusion = sum(confusion.values())
    if total_confusion > 0:
        print("\nIs Fully Supported Confusion Matrix:")
        print("                       Annotator Assessment")
        print("                       Agree    Disagree")
        print("Initial    Supported   {:>5}    {:>8}".format(
            confusion['initial_true_annotated_agree'],
            confusion['initial_true_annotated_disagree']
        ))
        print("Assessment Not Supp.   {:>5}    {:>8}".format(
            confusion['initial_false_annotated_agree'],
            confusion['initial_false_annotated_disagree']
        ))


def main():
    """Main function to compute statistics for all annotation files."""
    annotation_dir = Path("/data/new_astabench/citation_eval_ui/citation_eval_ui/annotations")
    exported_dir = Path("/data/new_astabench/citation_eval_ui/citation_eval_ui/exported_annotations")
    
    all_stats = {}
    
    # Process all JSON files in annotations directory
    for json_file in annotation_dir.glob("*.json"):
        annotations = load_annotations(json_file)
        stats = compute_citation_agreement(annotations)
        all_stats[json_file.name] = stats
        print_statistics(stats, json_file.name)
    
    # Process exported annotations
    for json_file in exported_dir.glob("*.json"):
        annotations = load_annotations(json_file)
        stats = compute_citation_agreement(annotations)
        all_stats[f"exported/{json_file.name}"] = stats
        print_statistics(stats, f"exported/{json_file.name}")
    
    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    print(f"Total files analyzed: {len(all_stats)}")
    
    # Aggregate statistics
    total_claims = sum(s['total_claims'] for s in all_stats.values())
    total_citations = sum(s['total_citations'] for s in all_stats.values())
    total_agree = sum(s['agreement_with_initial']['agree'] for s in all_stats.values())
    total_disagree = sum(s['agreement_with_initial']['disagree'] for s in all_stats.values())
    total_comparisons = sum(s['agreement_with_initial']['total'] for s in all_stats.values())
    
    print(f"\nTotal claims across all files: {total_claims}")
    print(f"Total citations across all files: {total_citations}")
    
    if total_comparisons > 0:
        overall_agreement_pct = total_agree / total_comparisons * 100
        print(f"\nOverall agreement with initial labels: {overall_agreement_pct:.1f}%")
        print(f"  Total agreements: {total_agree}")
        print(f"  Total disagreements: {total_disagree}")
        print(f"  Total comparisons: {total_comparisons}")
    
    # Aggregate confusion matrix
    total_confusion = {
        'initial_true_annotated_agree': sum(s['is_fully_supported_confusion']['initial_true_annotated_agree'] for s in all_stats.values()),
        'initial_true_annotated_disagree': sum(s['is_fully_supported_confusion']['initial_true_annotated_disagree'] for s in all_stats.values()),
        'initial_false_annotated_agree': sum(s['is_fully_supported_confusion']['initial_false_annotated_agree'] for s in all_stats.values()),
        'initial_false_annotated_disagree': sum(s['is_fully_supported_confusion']['initial_false_annotated_disagree'] for s in all_stats.values())
    }
    
    total_confusion_count = sum(total_confusion.values())
    if total_confusion_count > 0:
        print("\nOverall Is Fully Supported Confusion Matrix:")
        print("                       Annotator Assessment")
        print("                       Agree    Disagree    Total")
        initial_true_total = total_confusion['initial_true_annotated_agree'] + total_confusion['initial_true_annotated_disagree']
        initial_false_total = total_confusion['initial_false_annotated_agree'] + total_confusion['initial_false_annotated_disagree']
        print("Initial    Supported   {:>5}    {:>8}    {:>5}".format(
            total_confusion['initial_true_annotated_agree'],
            total_confusion['initial_true_annotated_disagree'],
            initial_true_total
        ))
        print("Assessment Not Supp.   {:>5}    {:>8}    {:>5}".format(
            total_confusion['initial_false_annotated_agree'],
            total_confusion['initial_false_annotated_disagree'],
            initial_false_total
        ))
        print("           Total       {:>5}    {:>8}    {:>5}".format(
            total_confusion['initial_true_annotated_agree'] + total_confusion['initial_false_annotated_agree'],
            total_confusion['initial_true_annotated_disagree'] + total_confusion['initial_false_annotated_disagree'],
            total_confusion_count
        ))
        
        # Calculate accuracy
        accuracy = (total_confusion['initial_true_annotated_agree'] + total_confusion['initial_false_annotated_disagree']) / total_confusion_count * 100
        print(f"\nAccuracy: {accuracy:.1f}%")


if __name__ == "__main__":
    main()