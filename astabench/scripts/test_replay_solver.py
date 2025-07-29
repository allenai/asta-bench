"""Test script for the replay solver."""

import asyncio
import concurrent.futures
import gzip
import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from queue import PriorityQueue
from typing import Any, Dict, List, Optional, Tuple

import httpx
from inspect_ai.tool import ToolDef

from astabench.tools import async_make_asta_s2_mcp_tools

logger = logging.getLogger(__name__)


class ToolCallReplay:
    """Manages replay of tool calls from a log file."""

    def __init__(self, log_path: str = "logs/tool_call_log.json.gz"):
        self.log_path = log_path
        self.tool_calls: List[Dict[str, Any]] = []
        self.tools_by_name: Dict[str, Any] = {}
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self.start_time: Optional[float] = None

        # Error tracking
        self.error_counts = defaultdict(int)
        self.http_error_counts = defaultdict(int)
        self.rate_limit_errors = 0

    def load_log(self) -> None:
        """Load and parse the tool call log."""
        logger.info(f"Loading tool call log from {self.log_path}")

        with gzip.open(self.log_path, "rt") as f:
            self.tool_calls = json.load(f)

        logger.info(f"Loaded {len(self.tool_calls)} tool calls")

        # Parse timestamps and calculate start time
        for call in self.tool_calls:
            call["timestamp_dt"] = datetime.fromisoformat(
                call["timestamp"].replace("+00:00", "Z")
            )

        # Sort by timestamp (should already be sorted)
        self.tool_calls.sort(key=lambda x: x["timestamp_dt"])

        # Calculate relative timings from the first call
        first_timestamp = self.tool_calls[0]["timestamp_dt"]
        for call in self.tool_calls:
            delta = (call["timestamp_dt"] - first_timestamp).total_seconds()
            call["relative_time"] = delta

    async def setup_tools(self, insertion_date: Optional[str] = None) -> None:
        """Initialize the tools that will be used for replay."""
        logger.info("Setting up tools for replay")

        # Get the tools (matching the original solver)
        tools = await async_make_asta_s2_mcp_tools(insertion_date=insertion_date)

        # Create mapping from tool names to functions
        self.tools_by_name = {ToolDef(t).name: t for t in tools}

        logger.info(f"Available tools: {list(self.tools_by_name.keys())}")

    async def execute_tool_call(self, call: Dict[str, Any]) -> Tuple[str, Any]:
        """Execute a single tool call and return its ID and result."""
        tool_name = call["function"]
        arguments = call["arguments"]
        call_id = call["id"]

        logger.info(
            f"Executing tool call {call_id} for {tool_name} (originally {call['timestamp']})"
        )

        try:
            if tool_name not in self.tools_by_name:
                logger.warning(f"Tool {tool_name} not found in available tools")
                return call_id, {"error": f"Tool {tool_name} not found"}

            tool = self.tools_by_name[tool_name]

            # Execute the tool call
            logger.debug(f"Executing {tool_name} with args: {arguments}")
            result = await tool(**arguments)

            return call_id, result

        except Exception as e:
            # Handle ExceptionGroups by examining their sub-exceptions
            http_error = None
            if isinstance(e, BaseExceptionGroup):
                # Look for HTTPStatusError in the exception group
                for sub_exc in e.exceptions:
                    if isinstance(sub_exc, httpx.HTTPStatusError):
                        http_error = sub_exc
                        break
            elif isinstance(e, httpx.HTTPStatusError):
                http_error = e

            # Track different types of errors
            error_type = type(e).__name__
            self.error_counts[error_type] += 1

            # Special handling for HTTP errors
            if http_error:
                status_code = http_error.response.status_code
                self.http_error_counts[status_code] += 1

                if status_code == 429:
                    self.rate_limit_errors += 1
                    logger.warning(
                        f"Rate limit hit for {tool_name} (call {call_id}): {http_error}"
                    )
                else:
                    logger.warning(
                        f"HTTP {status_code} error for {tool_name} (call {call_id}): {http_error}"
                    )
            else:
                logger.exception(f"Error executing tool {tool_name} (call {call_id})")

            return call_id, {"error": str(e), "error_type": error_type}

    async def replay_calls(self) -> Dict[str, Any]:
        """Replay all tool calls with proper timing."""
        logger.info("Starting tool call replay")

        # Track results
        results = {}
        pending_futures = []

        # Start time for replay
        self.start_time = time.time()

        # Create a priority queue for scheduled calls
        call_queue = PriorityQueue()

        # Add all calls to the queue with their scheduled times
        for call in self.tool_calls:
            scheduled_time = self.start_time + call["relative_time"]
            call_queue.put((scheduled_time, call))

        # Process calls as their time comes
        while not call_queue.empty() or pending_futures:
            # Get the next scheduled call
            if not call_queue.empty():
                scheduled_time, call = call_queue.get()

                # Wait until it's time to execute this call
                current_time = time.time()
                if scheduled_time > current_time:
                    wait_time = scheduled_time - current_time
                    logger.debug(f"Waiting {wait_time:.3f}s for next call")
                    await asyncio.sleep(wait_time)

                # Submit the call for execution
                future = asyncio.create_task(self.execute_tool_call(call))
                pending_futures.append((call["id"], future))

            # Check for completed futures
            if pending_futures:
                # Process completed futures without blocking
                done_indices = []
                for i, (call_id, future) in enumerate(pending_futures):
                    if future.done():
                        try:
                            _, result = await future
                            results[call_id] = result
                            done_indices.append(i)
                        except Exception as e:
                            logger.exception(f"Error getting result for {call_id}")
                            results[call_id] = {"error": str(e)}
                            done_indices.append(i)

                # Remove completed futures
                for i in reversed(done_indices):
                    pending_futures.pop(i)

            # Brief sleep to avoid busy waiting
            if not call_queue.empty() or pending_futures:
                await asyncio.sleep(0.01)

        # Wait for any remaining futures
        for call_id, future in pending_futures:
            try:
                _, result = await future
                results[call_id] = result
            except Exception as e:
                logger.exception(f"Error getting final result for {call_id}")
                results[call_id] = {"error": str(e)}

        elapsed_time = time.time() - self.start_time
        logger.info(f"Replay completed in {elapsed_time:.2f} seconds")

        return results


async def run_replay(
    log_path: str = "logs/tool_call_log.json.gz",
    insertion_date: Optional[str] = None,
    speedup_factor: float = 1.0,
) -> None:
    logger.info(f"Starting replay solver with log: {log_path}")

    # Create replay manager
    replay = ToolCallReplay(log_path)

    # Load the log
    replay.load_log()

    # Apply speedup factor if specified
    if speedup_factor != 1.0:
        logger.info(f"Applying speedup factor of {speedup_factor}x")
        for call in replay.tool_calls:
            call["relative_time"] /= speedup_factor

    # Setup tools
    await replay.setup_tools(insertion_date)

    # Get basic statistics
    total_calls = len(replay.tool_calls)
    duration = replay.tool_calls[-1]["relative_time"] if total_calls > 0 else 0

    logger.info(f"Replaying {total_calls} tool calls over {duration:.2f} seconds")

    # Replay the calls
    results = await replay.replay_calls()

    # Store results in state
    metadata = dict()
    metadata["replay_results"] = results
    metadata["replay_stats"] = {
        "total_calls": total_calls,
        "successful_calls": sum(1 for r in results.values() if "error" not in r),
        "failed_calls": sum(1 for r in results.values() if "error" in r),
        "duration": duration,
        "speedup_factor": speedup_factor,
        "error_counts": dict(replay.error_counts),
        "http_error_counts": dict(replay.http_error_counts),
        "rate_limit_errors": replay.rate_limit_errors,
    }

    # Log summary
    stats = metadata["replay_stats"]
    logger.info(
        f"Replay complete: {stats['successful_calls']}/{total_calls} successful, "
        f"{stats['failed_calls']} failed"
    )

    logger.info(f"Rate limit errors (429): {replay.rate_limit_errors}")
    if replay.http_error_counts:
        logger.info(f"HTTP error breakdown: {dict(replay.http_error_counts)}")
    if replay.error_counts:
        logger.info(f"Error type breakdown: {dict(replay.error_counts)}")

    return state


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Suppress HTTPX and MCP debug logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("mcp.client.streamable_http").setLevel(logging.WARNING)

    asyncio.run(run_replay())
