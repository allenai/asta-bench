#!/usr/bin/env python3
"""
Generate bar charts for claim and citation failure rates from the same data sources
used by generate_markdown_report.py
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import the functions from the report generator
import sys
sys.path.append('scripts')
from generate_markdown_report import (
    load_claims_data, 
    load_citations_data, 
    load_totals_data,
    create_claims_failure_table,
    create_citations_failure_table
)

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14

def create_bar_charts():
    """Generate bar charts for both tables."""
    
    # Load data using the same functions as the report generator
    claims_df = load_claims_data('reports/classified_claims_summary.csv')
    citations_df = load_citations_data('reports/classified_citations_summary.csv')
    totals_data = load_totals_data('data/statistics/totals.json')
    
    # Load report metadata for sample size info
    with open('reports/failure_mode_analysis_report.json', 'r') as f:
        report_data = json.load(f)
    
    # Create the same tables as in the report
    claims_table = create_claims_failure_table(claims_df, totals_data, report_data)
    citations_table = create_citations_failure_table(citations_df, claims_df, totals_data)
    
    # Rename claude_4.0 to SQA in both tables
    if 'claude_4.0' in claims_table.columns:
        claims_table = claims_table.rename(columns={'claude_4.0': 'SQA'})
    if 'claude_4.0' in citations_table.columns:
        citations_table = citations_table.rename(columns={'claude_4.0': 'SQA'})
    
    # Define consistent color palette for systems (using cleaned names)
    system_colors = {
        'SQA': '#1f77b4',  # blue
        'elicit': '#ff7f0e',  # orange
        'OpenAI Deep Research': '#2ca02c',  # green
        'Perplexity DR': '#d62728',  # red
        'Total': '#9467bd'  # purple
    }
    
    # Table 1: Claim Failure Rates
    # Exclude 'no_citation' and 'claim_is_fully_supported' (not a failure mode)
    claims_filtered = claims_table.drop(['no_citation', 'claim_is_fully_supported'], errors='ignore')
    
    # Prepare data for plotting
    claims_melted = claims_filtered.reset_index().melt(
        id_vars=['failure_mode'], 
        var_name='System', 
        value_name='Failure Rate (%)'
    )
    
    # Remove underscores from failure modes and system names
    claims_melted['failure_mode'] = claims_melted['failure_mode'].str.replace('_', ' ').str.title()
    claims_melted['System'] = claims_melted['System'].str.replace('_', ' ').str.replace('openai deep research', 'OpenAI Deep Research').str.replace('perplexity dr', 'Perplexity DR')
    
    # Create bar chart for Table 1
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    
    # Get unique systems in the order they appear in the data
    system_order = claims_melted['System'].unique()
    palette = [system_colors.get(sys, '#808080') for sys in system_order]
    
    # Define the order of failure modes (starting with Without Any Citations)
    failure_mode_order = ['Without Any Citations', 'Partial Support', 'Irrelevant Citation', 'Citation Unavailable']
    # Filter to only include modes that exist in the data
    failure_mode_order = [mode for mode in failure_mode_order if mode in claims_melted['failure_mode'].unique()]
    
    sns.barplot(
        data=claims_melted, 
        x='failure_mode', 
        y='Failure Rate (%)', 
        hue='System', 
        hue_order=system_order,
        order=failure_mode_order,
        palette=palette,
        ax=ax1
    )
    ax1.set_title('Claim Failure Rates by System', fontweight='bold')
    ax1.set_xlabel('Failure Mode')
    ax1.set_ylabel('Failure Rate (%)')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha='right')
    ax1.legend(title='System', loc='upper right')
    plt.tight_layout()
    plt.savefig('reports/claim_failure_rates_chart.png', dpi=150, bbox_inches='tight')
    print("Saved: reports/claim_failure_rates_chart.png")
    plt.show()
    
    # Table 2: Citation Failure Rates
    # Prepare data for plotting
    citations_melted = citations_table.reset_index().melt(
        id_vars=['Category'], 
        var_name='System', 
        value_name='Failure Rate (%)'
    )
    
    # Remove underscores from categories and system names
    citations_melted['Category'] = citations_melted['Category'].str.replace('_', ' ').str.title()
    citations_melted['System'] = citations_melted['System'].str.replace('_', ' ').str.replace('openai deep research', 'OpenAI Deep Research').str.replace('perplexity dr', 'Perplexity DR')
    
    # Create bar chart for Table 2
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    # Use the same system order and colors as in the first plot
    system_order_citations = citations_melted['System'].unique()
    palette_citations = [system_colors.get(sys, '#808080') for sys in system_order_citations]
    
    sns.barplot(
        data=citations_melted, 
        x='Category', 
        y='Failure Rate (%)', 
        hue='System', 
        hue_order=system_order_citations,
        palette=palette_citations,
        ax=ax2
    )
    ax2.set_title('Citation Failure Rates by Category', fontweight='bold')
    ax2.set_xlabel('Category')
    ax2.set_ylabel('Failure Rate (%)')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha='right')
    ax2.legend(title='System', loc='upper right')
    plt.tight_layout()
    plt.savefig('reports/citation_failure_rates_chart.png', dpi=150, bbox_inches='tight')
    print("Saved: reports/citation_failure_rates_chart.png")
    plt.show()

if __name__ == "__main__":
    create_bar_charts()
