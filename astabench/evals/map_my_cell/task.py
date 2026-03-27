"""MapMyCell benchmark: cell type literature recall.

Given a cell type name (e.g., "Pvalb-MET-3"), find the papers from the
known literature about that cell type. Scored by recall — what percentage
of known ground-truth papers did the agent find?

Ground truth is seeded from Gouwens & Tasic's MET-type classification
(Gouwens et al. 2020, Cell), which lists previous_work citations for each
morpho-electric transcriptomic (MET) type.
"""

import importlib.resources
import json
import logging

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import (
    Score,
    Scorer,
    Target,
    mean,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState, use_tools

from astabench.evals.utils import extract_json_from_response, not_implemented_solver
from astabench.tools import make_asta_mcp_tools

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a neuroscience literature search assistant. You will be given a \
cell type name from the Allen Brain Cell Atlas or related mouse cortex \
taxonomy. Find academic papers that have studied or characterized this \
cell type.

Return your answer as JSON with a single key "papers" containing a list \
of objects, each with:
- "paper_id": the Semantic Scholar Corpus ID (numeric string)
- "title": the paper title
- "authors_year": first author et al., year (e.g., "Gouwens et al., 2019")
"""


def _load_data() -> dict:
    data_path = importlib.resources.files("astabench.evals.map_my_cell") / "data.json"
    with importlib.resources.as_file(data_path) as p:
        return json.loads(p.read_text())


def _build_samples(examples: list[dict]) -> list[Sample]:
    samples = []
    for ex in examples:
        prompt = (
            f"Find papers that have previously studied or characterized "
            f"the cell type: {ex['cell_type']}"
        )
        target_data = {
            "ground_truth_paper_ids": ex["ground_truth_paper_ids"],
            "ground_truth_papers": ex["ground_truth_papers"],
            "n_expected": ex["n_expected"],
        }
        samples.append(
            Sample(
                id=ex["id"],
                input=prompt,
                target=json.dumps(target_data),
                metadata={
                    "cell_type": ex["cell_type"],
                    "n_expected": ex["n_expected"],
                    "system_prompt": SYSTEM_PROMPT,
                },
            )
        )
    return samples


def _extract_paper_ids(response: str) -> set[str]:
    """Extract paper IDs from agent response, trying JSON then fallback."""
    ids = set()
    parsed = extract_json_from_response(response)
    if parsed and "papers" in parsed:
        for paper in parsed["papers"]:
            pid = str(paper.get("paper_id", "")).strip()
            if pid:
                ids.add(pid)
    # Also scan raw response for corpus ID patterns
    import re
    for match in re.finditer(r"(?:CorpusId:|corpus.?id[:\s]*)(\d+)", response, re.IGNORECASE):
        ids.add(match.group(1))
    return ids


@scorer(metrics=[mean(), stderr()])
def score_recall() -> Scorer:
    """Score by recall: what fraction of ground-truth papers did the agent find?"""

    async def score(state: TaskState, target: Target) -> Score:
        target_data = json.loads(target.text)
        n_expected = target_data["n_expected"]
        response_lower = state.output.completion.lower()

        found_ids = _extract_paper_ids(state.output.completion)
        expected_ids = set(target_data["ground_truth_paper_ids"])

        matched = set()
        match_details = []

        # 1. Match by S2 paper ID
        for pid in expected_ids & found_ids:
            matched.add(pid)
            match_details.append(f"id:{pid}")

        # 2. Match by author+year citation string (e.g., "Gouwens et al., 2019")
        for gt in target_data["ground_truth_papers"]:
            if gt["paperId"] in matched:
                continue
            cite = gt["citation"]
            # Extract surname and year for flexible matching
            import re
            m = re.match(r"(\w+)", cite)
            surname = m.group(1).lower() if m else ""
            year = str(gt["year"])
            if surname and year and surname in response_lower and year in response_lower:
                matched.add(gt["paperId"])
                match_details.append(f"cite:{cite}")

        # 3. Match by title keywords (first 4 non-trivial words)
        for gt in target_data["ground_truth_papers"]:
            if gt["paperId"] in matched:
                continue
            title_lower = gt["title"].lower()
            stop = {"of", "the", "in", "and", "a", "an", "for", "to", "by"}
            key_words = [w for w in title_lower.split() if w not in stop][:4]
            if len(key_words) >= 3 and all(w in response_lower for w in key_words):
                matched.add(gt["paperId"])
                match_details.append(f"title:{gt['citation']}")

        recall = len(matched) / n_expected if n_expected > 0 else 0.0

        return Score(
            value=recall,
            answer=f"{len(matched)}/{n_expected} papers found",
            explanation=(
                f"Matches: {match_details}. "
                f"Expected: {[p['citation'] for p in target_data['ground_truth_papers']]}."
            ),
        )

    return score


@task
def map_my_cell(
    with_search_tools: bool = True,
    limit: int | None = None,
) -> Task:
    """MapMyCell cell type literature recall benchmark.

    Args:
        with_search_tools: Whether to provide Asta MCP search tools.
        limit: Max samples (None = all 28).
    """
    data = _load_data()
    examples = data["examples"]
    if limit:
        examples = examples[:limit]

    tool_setups = []
    if with_search_tools:
        try:
            tool_setups.append(use_tools(make_asta_mcp_tools()))
        except Exception:
            logger.error(
                "Could not create Asta MCP tools (ASTA_TOOL_KEY not set?). "
                "Running without search tools."
            )

    return Task(
        dataset=MemoryDataset(samples=_build_samples(examples)),
        setup=tool_setups or None,
        solver=not_implemented_solver(),
        scorer=score_recall(),
        fail_on_error=False,
    )
