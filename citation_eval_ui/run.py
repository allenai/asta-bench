#!/usr/bin/env python3
"""Launch script for Citation Evaluation UI."""

import os
import sys
from pathlib import Path

# Add the package to Python path
package_dir = Path(__file__).parent
sys.path.insert(0, str(package_dir))

# Set environment variable for data directory
data_dir = package_dir.parent / "test_dvc_logs" / "debug_logs"
if not data_dir.exists():
    print(f"Warning: Data directory not found at {data_dir}")
    print("Make sure citation_eval.csv files are available in test_dvc_logs/debug_logs/")

# Import and run
from citation_eval_ui.main import main

if __name__ == "__main__":
    main()