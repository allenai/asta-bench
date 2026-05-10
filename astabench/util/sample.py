"""Sample-level helpers for asta-bench tasks."""

from inspect_ai.dataset import Dataset


def set_insertion_date(dataset: Dataset, insertion_date: str) -> Dataset:
    """Set ``metadata["insertion_date"]`` on every sample in ``dataset``
    (mutates in place; existing values are preserved).

    Lets non-MCP agent surfaces (e.g. CLI-based agents that read host env
    vars before launching their sandbox) mirror the same cutoff astabench's
    MCP wrapper applies via ``make_asta_mcp_tools(insertion_date=...)``.
    """
    for sample in dataset:
        meta = dict(sample.metadata or {})
        meta.setdefault("insertion_date", insertion_date)
        sample.metadata = meta
    return dataset
