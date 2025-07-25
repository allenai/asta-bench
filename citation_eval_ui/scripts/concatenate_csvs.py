#!/usr/bin/env python3
"""
Script to concatenate multiple CSV files and add a column with the source filename.

Usage:
    python concatenate_csvs.py input1.csv input2.csv ... -o output.csv
    python concatenate_csvs.py *.csv -o combined.csv
"""

import argparse
import pandas as pd
from pathlib import Path
import sys


def concatenate_csvs(input_files, output_file, source_column_name='source_file', sample_size=None, random_seed=None):
    """
    Concatenate multiple CSV files and add a column with the source filename.
    
    Args:
        input_files: List of input CSV file paths
        output_file: Output CSV file path
        source_column_name: Name of the column to add with source filename
        sample_size: Number of rows to randomly sample from each file (None for all rows)
        random_seed: Random seed for reproducible sampling
    """
    all_dataframes = []
    
    for file_path in input_files:
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            original_size = len(df)
            
            # Sample rows if requested
            if sample_size is not None and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=random_seed)
                print(f"✓ Loaded {file_path} (sampled {sample_size} from {original_size} rows)")
            else:
                print(f"✓ Loaded {file_path} ({len(df)} rows)")
            
            # Add source filename column
            df[source_column_name] = Path(file_path).name
            
            all_dataframes.append(df)
            
        except Exception as e:
            print(f"✗ Error loading {file_path}: {e}", file=sys.stderr)
            continue
    
    if not all_dataframes:
        print("No CSV files were successfully loaded.", file=sys.stderr)
        return False
    
    # Concatenate all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Move source_file column to the first position
    cols = combined_df.columns.tolist()
    cols.remove(source_column_name)
    cols = [source_column_name] + cols
    combined_df = combined_df[cols]
    
    # Save to output file
    combined_df.to_csv(output_file, index=False)
    
    print(f"\n✓ Combined {len(all_dataframes)} files into {output_file}")
    print(f"  Total rows: {len(combined_df)}")
    
    # Show summary of rows per file
    print("\nRows per source file:")
    for source, count in combined_df[source_column_name].value_counts().items():
        print(f"  {source}: {count}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Concatenate multiple CSV files and add source filename column',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Concatenate specific files
    python concatenate_csvs.py file1.csv file2.csv file3.csv -o combined.csv
    
    # Concatenate all CSV files in directory
    python concatenate_csvs.py *.csv -o all_data.csv
    
    # Use custom source column name
    python concatenate_csvs.py *.csv -o combined.csv --source-column filename
    
    # Sample 100 random rows from each file
    python concatenate_csvs.py *.csv -o sampled.csv --sample 100
    
    # Sample with a specific random seed for reproducibility
    python concatenate_csvs.py *.csv -o sampled.csv --sample 100 --seed 42
        """
    )
    
    parser.add_argument(
        'input_files',
        nargs='+',
        help='Input CSV files to concatenate'
    )
    
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output CSV file path'
    )
    
    parser.add_argument(
        '--source-column',
        default='source_file',
        help='Name of the column to add with source filename (default: source_file)'
    )
    
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Number of rows to randomly sample from each file (default: use all rows)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducible sampling'
    )
    
    args = parser.parse_args()
    
    # Convert input files to Path objects and check they exist
    input_paths = []
    for file_pattern in args.input_files:
        # Handle glob patterns
        if '*' in file_pattern:
            paths = list(Path('.').glob(file_pattern))
            if not paths:
                print(f"Warning: No files match pattern '{file_pattern}'", file=sys.stderr)
            input_paths.extend(paths)
        else:
            path = Path(file_pattern)
            if not path.exists():
                print(f"Error: File '{file_pattern}' does not exist", file=sys.stderr)
                sys.exit(1)
            input_paths.append(path)
    
    if not input_paths:
        print("Error: No input files found", file=sys.stderr)
        sys.exit(1)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_paths = []
    for path in input_paths:
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)
    
    print(f"Found {len(unique_paths)} CSV files to concatenate")
    if args.sample:
        print(f"Will sample {args.sample} rows from each file")
    
    # Concatenate the files
    success = concatenate_csvs(
        unique_paths, 
        args.output, 
        args.source_column,
        sample_size=args.sample,
        random_seed=args.seed
    )
    
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()