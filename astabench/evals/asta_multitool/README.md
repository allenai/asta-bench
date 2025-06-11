# Multitool-challenge Task

This directory contains the InspectAI wrapper for the multitool-challenge task from the old asta-eval.  Only the `output_metric_configs` in the dataset are used here, but the original `tool_metric_configs` are preserved in the JSON files for reference.

## Setup

From repo root, run:
```sh
pip install -e "."
```

## Task Description

- Queries about scientific tasks that require multiple primitive actions to solve
    - "multiple primitive actions" is with respect to the asta toolset; it may be simpler in astabench depending on what tools are enabled
    - Typical actions include, e.g.:
        - "search for a paper"
        - "extract information from a paper" (paperqa)
        - "analyze a dataset" / "run an experiment/code"
    - Example question: `"for the bert, roberta, and codeact (wang et al) papers, find the url in the paper for the github repository and return them to me as JSON (paper -> url).  E.g. {\"bert\": \"http...\", \"roberta\": \"http...\", \"codeact\": \"http...\"}`
        - Requires (for each paper): [paper search] -> [paper qa to find github url]
- Questions are hard, in the sense that they failed in the original asta system, despite the tools having the capability to solve them
- Two kinds of metrics, depending on question: either the model is instructed to output JSON which is judged exactly, or the model is allowed to output free text which is judged against rubric(s) with an llm
    - All produce a score between 0 and 1

## Usage

You can run the eval using InspectAI's CLI:

```bash
inspect eval astabench/evals/asta_multitool \
    --model openai/gpt-4o-mini \
    --solver astabench/solvers/llm.py@llm_with_prompt
```

