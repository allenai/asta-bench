# AstaBench Run Post-Mortem

This document tracks issues encountered while running and submitting AstaBench evaluations, and whether each issue came from our code/process or external dependencies. Follow the linked GitHub issues and PRs for details, workarounds, and fixes.

## Summary

| Issue | Description | Internal/External Cause |
| --- | --- | --- |
| Docker/Inspect sandbox startup failures ([nora-issues#2648](https://github.com/allenai/nora-issues/issues/2648), [asta-bench#135](https://github.com/allenai/asta-bench/pull/135)) | Docker Engine v29 broke compatibility with Inspect's sandbox startup, causing tasks to time out before running. | External |
| `snippet_search` MCP wrapper crash ([asta-bench#140](https://github.com/allenai/asta-bench/pull/140)) | An Inspect/MCP upgrade surfaced a positional-argument bug in our snippet-search wrapper, causing literature/table tasks to fail. | External |
| README split guidance mismatch | Full-suite final runs should use the `test` split, but quickstart guidance implied scaling the `validation` smoke test by only removing `--limit 1`. | Internal |
| Inspect/agent-eval compatibility upgrade for new model runs ([nora-issues#2720](https://github.com/allenai/nora-issues/issues/2720), [nora-issues#2733](https://github.com/allenai/nora-issues/issues/2733), [agent-eval#77](https://github.com/allenai/agent-eval/pull/77), [asta-bench#146](https://github.com/allenai/asta-bench/pull/146)) | New target models required newer Inspect reasoning support; newer Inspect logs also required leaderboard schema updates. | External |
