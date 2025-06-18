"""Shared types/data-structures between sandbox and main app.

Use dataclasses, not pydantic, since it's less likely to have incompatiblities
due to differences in the environment/packages.

Consider replacing this with a more structured approach in the future, such as
using a shared library installed in the sandbox.
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class CellHistoryItem:
    """A single code cell in the Jupyter history."""

    code: str

    output: str | None = None

    # For reliable elapsed measurements we really should be using
    # `perf_counter`; could do it with some fancy transformations but for
    # simplicity we just trust that clocks don't sync while running
    start_timestamp_rfc3339: str
    end_timestamp_rfc3339: str | None = None

    was_interrupted: bool = False
    """Whether the cell execution was interrupted by the user or timed out."""

    @staticmethod
    def get_rfc3339_timestamp() -> str:
        """Get the current time in RFC3339 format."""
        return datetime.now().isoformat()

    def end(self, was_interrupted: bool = False) -> None:
        """End the cell execution and set the end timestamp."""
        if self.end_timestamp_rfc3339 is not None:
            raise ValueError("Cell execution already finalized.")
        self.end_timestamp_rfc3339 = self.get_rfc3339_timestamp()
        self.was_interrupted = was_interrupted

    def is_finalized(self) -> bool:
        return self.end_timestamp_rfc3339 is not None
