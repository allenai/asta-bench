# coding: utf-8

import pandas as pd
import glob
import os
import re
from astabench.evals.sqa.citation_eval import clean_citation

# Find all citation_eval.csv files recursively
csv_files = glob.glob('test_dvc_logs/debug_logs/**/*citation_eval.csv', recursive=True)

# Read and process all CSV files
all_data = []

for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    df['csv_path'] = '/'.join(csv_path.split('/')[2:])
    df['claims'] = df.claims.apply(eval)
    df['half_credit'] = df.half_credit.apply(eval)
    df = df.explode('claims').dropna()
    df['all_citations'] = df.claims.apply(lambda x: x.get('supporting', []) + x.get('non_supporting', []))
    all_data.append(df)

# Combine all dataframes
combined_df = pd.concat(all_data, ignore_index=True)

# Group by (question, csv_path) and aggregate citations
result = combined_df.groupby(['csv_path', 'question']).agg({
    'all_citations': lambda x: list(set([clean_citation(cite) for sublist in x for cite in sublist])),
    'half_credit': lambda x: [clean_citation(hc) for hc in x.iloc[0]]
}).reset_index()

result['to_check'] = result.apply(lambda row: list(set(row['all_citations']) - set(row['half_credit'])), axis=1)
result['num_half_credit_missing'] = result.apply(lambda row: len(row['all_citations']) - len(set(row['half_credit']).intersection(set(row['all_citations']))), axis=1)
print('num questions with at least 1 missing citation:')
print(result.groupby('csv_path')['num_half_credit_missing'].apply(lambda x: (x > 0).sum()))
print('distribution of num missing citations when at least 1 is missing:')
print(result[result['num_half_credit_missing'] > 0].groupby('csv_path')['num_half_credit_missing'].describe())
result = result[result.half_credit.apply(len) > 0]
# Set question as index
result = result.set_index('question')

# Write result to CSV
result.to_csv('citations_summary.csv')
