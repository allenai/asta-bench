#!/usr/bin/env python3
"""Generate LaTeX table rows for ReAct and Smolagents results in the ASTA paper."""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

# Base directory for results
RESULTS_DIR = Path("tmp/final_result_dirs")

# Decimal places for formatting
SCORE_DECIMALS = 4  # Decimal places for score columns
COST_DECIMALS = 4   # Decimal places for cost columns

# Mappings from agent names to LaTeX macros
AGENT_MACROS = {
    "ReAct": r"\agentReAct",
    "Smolagents": r"\agentSmolagents",
}

# Mappings from model names to LaTeX macros
MODEL_MACROS = {
    "GPT-4.1": r"\modelGPTFourPointOneShort",
    "GPT-4o": r"\modelGPTFourOShort",
    "Claude-3.5-Haiku": r"\modelClaudeThreeFiveHaikuShort",
    "Claude-Sonnet-4": r"\modelClaudeSonnetFourShort",
    "Gemini-2.5-Flash": r"\modelGeminiTwoPointFiveFlashShort",
    "Llama-4-Scout": r"\modelLlamaFourScoutShort",
    "o3": r"\modelOThreeShort",
}

# Mappings for openness symbols
OPENNESS_SYMBOLS = {
    "Open Source": r"\opennessSymbolOpenClosedWeight",
    "Open Source + Open Weights": r"\opennessSymbolOpenOpenWeight",
}

# Mappings for tooling symbols
TOOLING_SYMBOLS = {
    "Standard": r"\toolingSymbolStandard",
    "Custom with Standard Search": r"\toolingSymbolEquivalent",
}


def load_agent_data(dir_path: Path) -> Tuple[Dict, Dict]:
    """Load submission and summary stats from a result directory."""
    submission_path = dir_path / "submission.json"
    stats_path = dir_path / "summary_stats.json"
    
    if not submission_path.exists():
        raise FileNotFoundError(f"Missing submission.json in {dir_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing summary_stats.json in {dir_path}")
    
    with open(submission_path) as f:
        submission = json.load(f)
    with open(stats_path) as f:
        stats = json.load(f)
    
    return submission, stats


def format_value(value: Optional[float], stderr: Optional[float], decimals: int) -> str:
    """Format a value with optional stderr.
    
    Note: stderr is multiplied by 1.96 to get 95% confidence interval.
    """
    if value is None:
        return r"{\red{?}}"
    
    formatted_val = f"{value:.{decimals}f}"
    
    if stderr is not None:
        # Convert stderr to 95% CI by multiplying by 1.96
        ci_95 = stderr * 1.96
        formatted_ci = f"{ci_95:.{decimals}f}"
        return f"{formatted_val} +- {formatted_ci}"
    else:
        return formatted_val


def get_agent_model_from_name(agent_name: str) -> Tuple[str, str]:
    """Extract agent type and model from agent name like 'ReAct-GPT-4.1'."""
    parts = agent_name.split("-", 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot parse agent name: {agent_name}")
    return parts[0], parts[1]


def generate_overall_table_row(submission: Dict, stats: Dict) -> str:
    """Generate a row for the overall results table."""
    agent_type, model = get_agent_model_from_name(submission["agent_name"])
    
    openness = OPENNESS_SYMBOLS.get(submission["openness"])
    tooling = TOOLING_SYMBOLS.get(submission["tool_usage"])
    agent = AGENT_MACROS.get(agent_type)
    model_macro = MODEL_MACROS.get(model)
    
    if not all([openness, tooling, agent, model_macro]):
        raise ValueError(f"Missing mapping for {submission['agent_name']}")
    
    # Get aggregated scores (no stderr in overall table)
    overall = stats["stats"].get("overall", {})
    lit = stats["stats"].get("tag/lit", {})
    code = stats["stats"].get("tag/code", {})
    data = stats["stats"].get("tag/data", {})
    discovery = stats["stats"].get("tag/discovery", {})
    
    row = f"    {openness} & {tooling} & {agent} & {model_macro}"
    
    # Overall
    row += f" & {overall.get('score', 0) * 100:.{SCORE_DECIMALS}f}"
    row += f" & {overall.get('cost', 0):.{COST_DECIMALS}f}"
    
    # Literature Understanding
    row += f" & {lit.get('score', 0) * 100:.{SCORE_DECIMALS}f}"
    row += f" & {lit.get('cost', 0):.{COST_DECIMALS}f}"
    
    # Code & Execution
    row += f" & {code.get('score', 0) * 100:.{SCORE_DECIMALS}f}"
    row += f" & {code.get('cost', 0):.{COST_DECIMALS}f}"
    
    # Data Analysis
    row += f" & {data.get('score', 0) * 100:.{SCORE_DECIMALS}f}"
    row += f" & {data.get('cost', 0):.{COST_DECIMALS}f}"
    
    # End-to-End Discovery
    row += f" & {discovery.get('score', 0) * 100:.{SCORE_DECIMALS}f}"
    row += f" & {discovery.get('cost', 0):.{COST_DECIMALS}f}"
    
    row += r" \\"
    return row


def generate_lit_search_row(submission: Dict, stats: Dict) -> str:
    """Generate a row for the literature search table."""
    agent_type, model = get_agent_model_from_name(submission["agent_name"])
    
    openness = OPENNESS_SYMBOLS.get(submission["openness"])
    tooling = TOOLING_SYMBOLS.get(submission["tool_usage"])
    agent = AGENT_MACROS.get(agent_type)
    model_macro = MODEL_MACROS.get(model)
    
    if not all([openness, tooling, agent, model_macro]):
        raise ValueError(f"Missing mapping for {submission['agent_name']}")
    
    paper_finder = stats["stats"].get("task/paper_finder_test", {})
    litqa_search = stats["stats"].get("task/paper_finder_litqa2_test", {})
    
    row = f"    {openness} & {tooling} & {agent} & {model_macro}"
    
    # PaperFindingBench
    score = paper_finder.get("score", 0) * 100
    stderr = paper_finder.get("score_stderr", 0) * 100 if paper_finder.get("score_stderr") else None
    row += f" & {format_value(score, stderr, SCORE_DECIMALS)}"
    
    cost = paper_finder.get("cost", 0)
    cost_stderr = paper_finder.get("cost_stderr", 0) if paper_finder.get("cost_stderr") else None
    row += f" & {format_value(cost, cost_stderr, COST_DECIMALS)}"
    
    # LitQA2-FT Search
    score = litqa_search.get("score", 0) * 100
    stderr = litqa_search.get("score_stderr", 0) * 100 if litqa_search.get("score_stderr") else None
    row += f" & {format_value(score, stderr, SCORE_DECIMALS)}"
    
    cost = litqa_search.get("cost", 0)
    cost_stderr = litqa_search.get("cost_stderr", 0) if litqa_search.get("cost_stderr") else None
    row += f" & {format_value(cost, cost_stderr, COST_DECIMALS)}"
    
    row += r" \\"
    return row


def generate_lit_qa_row(submission: Dict, stats: Dict) -> str:
    """Generate a row for the literature QA table."""
    agent_type, model = get_agent_model_from_name(submission["agent_name"])
    
    openness = OPENNESS_SYMBOLS.get(submission["openness"])
    tooling = TOOLING_SYMBOLS.get(submission["tool_usage"])
    agent = AGENT_MACROS.get(agent_type)
    model_macro = MODEL_MACROS.get(model)
    
    if not all([openness, tooling, agent, model_macro]):
        raise ValueError(f"Missing mapping for {submission['agent_name']}")
    
    sqa = stats["stats"].get("task/sqa_test", {})
    litqa = stats["stats"].get("task/litqa2_test", {})
    
    row = f"    {openness} & {tooling} & {agent} & {model_macro}"
    
    # ScholarQABench2
    score = sqa.get("score", 0) * 100
    stderr = sqa.get("score_stderr", 0) * 100 if sqa.get("score_stderr") else None
    row += f" & {format_value(score, stderr, SCORE_DECIMALS)}"
    
    cost = sqa.get("cost", 0)
    cost_stderr = sqa.get("cost_stderr", 0) if sqa.get("cost_stderr") else None
    row += f" & {format_value(cost, cost_stderr, COST_DECIMALS)}"
    
    # LitQA2-FT
    score = litqa.get("score", 0) * 100
    stderr = litqa.get("score_stderr", 0) * 100 if litqa.get("score_stderr") else None
    row += f" & {format_value(score, stderr, SCORE_DECIMALS)}"
    
    cost = litqa.get("cost", 0)
    cost_stderr = litqa.get("cost_stderr", 0) if litqa.get("cost_stderr") else None
    row += f" & {format_value(cost, cost_stderr, COST_DECIMALS)}"
    
    row += r" \\"
    return row


def generate_lit_table_row(submission: Dict, stats: Dict) -> str:
    """Generate a row for the literature tables table."""
    agent_type, model = get_agent_model_from_name(submission["agent_name"])
    
    openness = OPENNESS_SYMBOLS.get(submission["openness"])
    tooling = TOOLING_SYMBOLS.get(submission["tool_usage"])
    agent = AGENT_MACROS.get(agent_type)
    model_macro = MODEL_MACROS.get(model)
    
    if not all([openness, tooling, agent, model_macro]):
        raise ValueError(f"Missing mapping for {submission['agent_name']}")
    
    tables = stats["stats"].get("task/arxivdigestables_test", {})
    
    row = f"    {openness} & {tooling} & {agent} & {model_macro}"
    
    # ArxivDIGESTables-Clean
    score = tables.get("score", 0) * 100
    stderr = tables.get("score_stderr", 0) * 100 if tables.get("score_stderr") else None
    row += f" & {format_value(score, stderr, SCORE_DECIMALS)}"
    
    cost = tables.get("cost", 0)
    cost_stderr = tables.get("cost_stderr", 0) if tables.get("cost_stderr") else None
    row += f" & {format_value(cost, cost_stderr, COST_DECIMALS)}"
    
    row += r" \\"
    return row


def generate_coding_row(submission: Dict, stats: Dict) -> str:
    """Generate a row for the coding table."""
    agent_type, model = get_agent_model_from_name(submission["agent_name"])
    
    openness = OPENNESS_SYMBOLS.get(submission["openness"])
    tooling = TOOLING_SYMBOLS.get(submission["tool_usage"])
    agent = AGENT_MACROS.get(agent_type)
    model_macro = MODEL_MACROS.get(model)
    
    if not all([openness, tooling, agent, model_macro]):
        raise ValueError(f"Missing mapping for {submission['agent_name']}")
    
    super_test = stats["stats"].get("task/super_test", {})
    core_bench = stats["stats"].get("task/core_bench_test", {})
    ds1000 = stats["stats"].get("task/ds1000_test", {})
    
    row = f"    {openness} & {tooling} & {agent} & {model_macro}"
    
    # Super
    score = super_test.get("score", 0) * 100
    stderr = super_test.get("score_stderr", 0) * 100 if super_test.get("score_stderr") else None
    row += f" & {format_value(score, stderr, SCORE_DECIMALS)}"
    
    cost = super_test.get("cost", 0)
    cost_stderr = super_test.get("cost_stderr", 0) if super_test.get("cost_stderr") else None
    row += f" & {format_value(cost, cost_stderr, COST_DECIMALS)}"
    
    # CoreBench
    score = core_bench.get("score", 0) * 100
    stderr = core_bench.get("score_stderr", 0) * 100 if core_bench.get("score_stderr") else None
    row += f" & {format_value(score, stderr, SCORE_DECIMALS)}"
    
    cost = core_bench.get("cost", 0)
    cost_stderr = core_bench.get("cost_stderr", 0) if core_bench.get("cost_stderr") else None
    row += f" & {format_value(cost, cost_stderr, COST_DECIMALS)}"
    
    # DataSci (DS1000)
    score = ds1000.get("score", 0) * 100
    stderr = ds1000.get("score_stderr", 0) * 100 if ds1000.get("score_stderr") else None
    row += f" & {format_value(score, stderr, SCORE_DECIMALS)}"
    
    cost = ds1000.get("cost", 0)
    cost_stderr = ds1000.get("cost_stderr", 0) if ds1000.get("cost_stderr") else None
    row += f" & {format_value(cost, cost_stderr, COST_DECIMALS)}"
    
    row += r" \\"
    return row


def generate_endtoend_row(submission: Dict, stats: Dict) -> str:
    """Generate a row for the end-to-end table."""
    agent_type, model = get_agent_model_from_name(submission["agent_name"])
    
    openness = OPENNESS_SYMBOLS.get(submission["openness"])
    tooling = TOOLING_SYMBOLS.get(submission["tool_usage"])
    agent = AGENT_MACROS.get(agent_type)
    model_macro = MODEL_MACROS.get(model)
    
    if not all([openness, tooling, agent, model_macro]):
        raise ValueError(f"Missing mapping for {submission['agent_name']}")
    
    e2e = stats["stats"].get("task/e2e_discovery_test", {})
    e2e_hard = stats["stats"].get("task/e2e_discovery_hard_test", {})
    
    row = f"    {openness} & {tooling} & {agent} & {model_macro}"
    
    # End-to-End Discovery
    score = e2e.get("score", 0) * 100
    stderr = e2e.get("score_stderr", 0) * 100 if e2e.get("score_stderr") else None
    row += f" & {format_value(score, stderr, SCORE_DECIMALS)}"
    
    cost = e2e.get("cost", 0)
    cost_stderr = e2e.get("cost_stderr", 0) if e2e.get("cost_stderr") else None
    row += f" & {format_value(cost, cost_stderr, COST_DECIMALS)}"
    
    # End-to-End Discovery Hard
    score = e2e_hard.get("score", 0) * 100
    stderr = e2e_hard.get("score_stderr", 0) * 100 if e2e_hard.get("score_stderr") else None
    row += f" & {format_value(score, stderr, SCORE_DECIMALS)}"
    
    cost = e2e_hard.get("cost", 0)
    cost_stderr = e2e_hard.get("cost_stderr", 0) if e2e_hard.get("cost_stderr") else None
    row += f" & {format_value(cost, cost_stderr, COST_DECIMALS)}"
    
    row += r" \\"
    return row


def main():
    """Generate LaTeX rows for all tables."""
    # Find all ReAct and Smolagents directories
    agent_dirs = []
    for dir_path in RESULTS_DIR.iterdir():
        if dir_path.is_dir() and ("ReAct" in dir_path.name or "Smolagents" in dir_path.name):
            agent_dirs.append(dir_path)
    
    # Sort for consistent ordering
    agent_dirs.sort(key=lambda x: x.name)
    
    # Load all data
    all_data = []
    for dir_path in agent_dirs:
        try:
            submission, stats = load_agent_data(dir_path)
            all_data.append((dir_path.name, submission, stats))
        except Exception as e:
            print(f"Error loading {dir_path.name}: {e}")
            raise
    
    # Generate rows for each table
    print("=== Table 1: Overall Results ===")
    for name, submission, stats in all_data:
        try:
            print(generate_overall_table_row(submission, stats))
        except Exception as e:
            print(f"Error generating overall row for {name}: {e}")
    
    print("\n=== Table 2: Literature Search ===")
    for name, submission, stats in all_data:
        try:
            print(generate_lit_search_row(submission, stats))
        except Exception as e:
            print(f"Error generating lit search row for {name}: {e}")
    
    print("\n=== Table 3: Literature QA ===")
    for name, submission, stats in all_data:
        try:
            print(generate_lit_qa_row(submission, stats))
        except Exception as e:
            print(f"Error generating lit QA row for {name}: {e}")
    
    print("\n=== Table 4: Literature Tables ===")
    for name, submission, stats in all_data:
        try:
            print(generate_lit_table_row(submission, stats))
        except Exception as e:
            print(f"Error generating lit table row for {name}: {e}")
    
    print("\n=== Table 5: Coding ===")
    for name, submission, stats in all_data:
        try:
            print(generate_coding_row(submission, stats))
        except Exception as e:
            print(f"Error generating coding row for {name}: {e}")
    
    print("\n=== Table 6: End-to-End ===")
    for name, submission, stats in all_data:
        try:
            print(generate_endtoend_row(submission, stats))
        except Exception as e:
            print(f"Error generating end-to-end row for {name}: {e}")


if __name__ == "__main__":
    main()
