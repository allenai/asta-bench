This document is targeted at Ai2 internal devs who are creating new evaluations or agents for AstaBench.

### Submitting to the leaderboard

Private Ai2 submission flow (sample); you'll also need an `HF_TOKEN` for an account that is a member of the Ai2 huggingface org:
```bash
# Step 1. Publish to submissions repo.
astabench publish logs/2025-01-15-myagent-run/ \
  --agent-name "MyAgent" \
  --agent-description "v1.0" \
  --openness ow \  # ow=open-weight
  --tool-usage s \  # s=standard
  --submissions-repo-id allenai/asta-bench-internal-submissions

# Output: hf://allenai/asta-bench-internal-submissions/1.0.0-dev1/validation/username_MyAgent_2025-01-01T00-00-00
# This step generates a submission.json file with your agent name, description and details and includes the file with your logs and config in the submissions repo

# Step 2. Score the remote submission (use path from step 1).
astabench score hf://allenai/asta-bench-internal-submissions/1.0.0-dev1/validation/username_MyAgent_2025-01-01T00-00-00
# This will write files scores.json and summary_stats.json to a summaries directory in the submissions repo; the scores file is needed in this location to publish results to leaderboard

# Step 3. Publish data to display in the leaderboard UI.
astabench lb publish hf://allenai/asta-bench-internal-submissions/1.0.0-dev1/validation/username_MyAgent_2025-01-01T00-00-00 \
  --repo-id allenai/asta-bench-internal-results  # destination leaderboard results repo
# This step expects eval_config.json and submission.json to be available in your submission files and scores.json to be present in the submission summaries directory. It integrates these files for publication to the leaderboard.
```

The example above uses the **internal** repos. To submit to the **public** leaderboard, swap both: `allenai/asta-bench-submissions` for Step 1's `--submissions-repo-id` (and the `hf://` paths in Steps 2–3), and `allenai/asta-bench-results` for Step 3's `--repo-id`.

Always pass `--repo-id` explicitly in Step 3. The destination leaderboard results repo (where Step 3 writes the display data) is distinct from the submissions repo (the `hf://` source path), and if `--repo-id` is omitted it silently defaults to `allenai/asta-bench-internal-results`. Unlike Step 1's `--submissions-repo-id` (which falls back to `$SUBMISSIONS_REPO_ID`), this default is hardcoded with no env-var override, so an omitted flag publishes to the internal leaderboard regardless of intent.

**Check your agenteval lib version.** When the schema changes, older versions may produce incompatible submission schemas that fail to appear on the leaderboard.

Note to Ai2 internal users: Steps 2 and 3 may be run centrally, e.g. to re-score with up-to-date costs and make controlled updates to leaderboard display data.

For bulk runs of already scored results, it is possible to run:
`cat filenames_to_publish.txt | xargs astabench lb publish --repo-id allenai/asta-bench-internal-results`

### Rescoring Existing Submissions

Sometimes an existing submission needs to be rescored after it has already been downloaded, for example when an LLM-based judge model expires or becomes unavailable, or when the configured judge/scorer model for a task is updated in the source code. In that case, keep the original submissions tree untouched and write rescored output to a separate tree.

This workflow uses the same `solvers/scorer` uv project described in [Decoupled Solve->Score](README.md#decoupled-solvescore-cross-version).

Prerequisites:
- submissions already downloaded under `<target-dir>` (default: `asta-bench-submissions/`)

#### Downloading Existing Submissions

If you need to pull an existing submissions tree into `<target-dir>` (default: `asta-bench-submissions/`), use a Hugging Face dataset snapshot. This workflow assumes you have access to a Hugging Face dataset repo that stores submissions. For Ai2 internal usage, the submissions repo is `allenai/asta-bench-internal-submissions`; see [Submitting to the leaderboard](#submitting-to-the-leaderboard) for the internal flow. Public leaderboard uploads may not exist in a shared downloadable submissions tree.

To download an entire submissions tree:

```bash
export HF_TOKEN=<your-huggingface-token>

uv run python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="<submissions-repo-id>",
    repo_type="dataset",
    local_dir="<target-dir>",
    token=True,
)
PY
```

To download only specific submission directories, restrict the snapshot with `allow_patterns`:

```bash
export HF_TOKEN=<your-huggingface-token>

uv run python - <<'PY'
from huggingface_hub import snapshot_download

SUBMISSIONS = [
    "1.0.0-dev1/test/username_MyAgent_2025-01-01T00-00-00",
    "1.0.0-dev1/validation/username_MyOtherAgent_2025-01-02T00-00-00",
]

snapshot_download(
    repo_id="<submissions-repo-id>",
    repo_type="dataset",
    local_dir="<target-dir>",
    allow_patterns=[f"{path}/*" for path in SUBMISSIONS],
    token=True,
)
PY
```

`scripts/rescore_judge_model_submissions.sh` is intentionally guarded as a template. By default it exits with:

```text
This script is a template and needs to be adapted to your use case.  Edit and try again
```

Before running it, edit the script for the current rescore pass: confirm the target detection logic, update `TARGET_LOG_REGEX` or replace the regex-based selection if needed, and then remove or adapt the template guard.

To rescore the currently supported LLM-judged tasks (ScholarQA and E2E-Bench), run:

Using the default directories:

```bash
./scripts/rescore_judge_model_submissions.sh \
  --duplicate-task-policy keep-latest
```

Or with explicit directories:

```bash
./scripts/rescore_judge_model_submissions.sh \
  --submissions-root <target-dir> \
  --output-root <target-rescored-dir> \
  --duplicate-task-policy keep-latest
```

Defaults: `<target-dir>` -> `asta-bench-submissions/`, `<target-rescored-dir>` -> `asta-bench-submissions-rescored/`.

Note: `<target-rescored-dir>` must not overlap with `<target-dir>`.

This script:
1. Finds submissions under `<target-dir>` that contain targeted ScholarQA or E2E-Bench logs
2. Copies only those submissions into `<target-rescored-dir>`
3. Re-runs per-log `inspect score --overwrite` only for the targeted tasks
4. Re-runs `astabench score` for the copied submission directory

Useful companion commands:

```bash
python scripts/rescore_progress.py \
  --submissions-root <target-dir> \
  --output-root <target-rescored-dir>
python scripts/rescore_progress.py \
  --submissions-root <target-dir> \
  --output-root <target-rescored-dir> \
  --show summary
./scripts/rescore_judge_model_submissions.sh \
  --submissions-root <target-dir> \
  --output-root <target-rescored-dir> \
  --resume \
  --duplicate-task-policy keep-latest
```

Notes:
- `scripts/rescore_progress.py` shows which submissions are pending, in progress, failed, interrupted, or completed, and prints before/after score deltas for rewritten target logs.
- `--resume` enables safe retry support:
  - completed submissions are skipped
  - submissions interrupted after per-log scoring but before aggregation are resumed from the aggregation step
  - everything else is rerun from scratch for that submission
- `--duplicate-task-policy keep-latest` is required for submissions that contain multiple `.eval` logs for the same normalized task; otherwise the default `fail` policy stops the batch at the first such submission. Older duplicates are archived under `<target-rescored-dir>/_duplicate_task_logs_archive/<submission-rel>/`, alongside a `manifest.json` describing the kept vs archived files.
- SQA needs special handling when changing its judge model. Historical SQA `.eval` logs serialize the nested scorer model params, so preloading current scorer definitions does not automatically replace `gemini-2.5-flash` with the new SQA default. Use an explicit `inspect score --scorer ...` override or edit the `.eval` scorer config for SQA model migrations.
- This workflow only targets the supported LLM-judged task types (ScholarQA and E2E-Bench); other tasks in the same submissions keep their original scores. The targeted log set is controlled by `TARGET_LOG_REGEX` in `scripts/rescore_judge_model_submissions.sh` and can be overridden with `--target-log-regex <pattern>`.
- Tasks whose scorers need live Docker sandbox state (for example CORE-Bench, DS-1000, and SUPER) are not rescored from existing logs, because the sandbox state is not preserved after the original eval run.

### Adding a Benchmark

InspectAI has [extensive documentation](https://inspect.aisi.org.uk/tasks.html) on how to implement task code; this section covers additional considerations when adding a task to AstaBench to ensure compatibility with suite expectations.

To add a new task to AstaBench, you should implement it as an InspectAI task.  In general, the new task should go in a new directory in [`astabench/evals`](/astabench/evals), unless it is a direct extension or variation of an existing task (e.g., a new set of annotated data with the same format/scoring as the original task).

After creating the task, **be sure to register it** in [`astabench/evals/_registry.py`](/astabench/evals/_registry.py) so that it can be discovered by InspectAI as `astabench/[YOUR_TASK_NAME]`; this should be done for all tasks, regardless of whether they will be part of the official set of AstaBench tasks run via `astabench eval`.

Tasks in the official `astabench eval` set will additionally need to be added to the latest `astabench/config/v*.yml` file; this requires establishing a single "primary" metric for the task, and creating separate task definitions for `validation` and `test` splits (necessary because there is no way to pass task-arguments to the underlying `inspect eval-set`).

For an example of a custom eval, see `astabench/evals/demo/arithmetic` in this repo or other InspectAI evals in [`inspect_evals`](https://github.com/UKGovernmentBEIS/inspect_evals).
The directory includes examples of a simple eval (`astabench/evals/demo/arithmetic/task.py`), an eval that checks tool calls (`astabench/evals/demo/arithmetic/task_with_tools.py`), and an eval that checks a stylistic rubric (`astabench/evals/demo/arithmetic/task_with_rubric.py`).

For more complex tasks that require a custom scorer, check the [Inspect documentation](https://inspect.ai-safety-institute.org.uk/scorers.html).

#### Input/output format

To ensure that general agents can run on the tasks, task samples should always be solvable based on chat input + tools, where a "chat" is text/image/audio that can be fed directly to a (possibly multimodal) LLM and "tools" are `inspect_ai.Tool`s passed via `state.tools`.  That is:

- the user input chat should contain all necessary task instructions/answer formatting instructions
    - corrolary for solver design: general solvers should not have task-specific instructions or parameters, though they may have generic problem-solving instructions (e.g. "think step-by-step")
- it should be possible to achieve a perfect score by reading the input, performing some actions with `state.tools` and observing results, and setting `state.output.completion`.
    - it is ideal if the task can be fully judged based on `state.output.completion`; this is not always feasible, so some tasks may judge intermediate states of tool calls or sandbox files if they provide the necessary tools for manipulating those
        - e.g. see how [super scores code trajectory](https://github.com/allenai/asta-bench/blob/main/astabench/evals/super/task.py#L70)
        - a solver may internally let the model use some kind of `submit()` tool to submit the answer; that tool can set the `state.output.completion` chat to the answer
- If the task depends on some large structured data source (e.g. a database), the task should provide a tool (e.g. via `use_tools(...)` in the task `setup=` parameter) that the agent can use to access the resource
- it should not be necessary for solvers to read/write task-specific structured data like `state.choices`/`state.store`/`state.metadata`.
    - E.g., a task should not expect to read the answer from `state.store["answer"]`, unless it provides the agent with a tool that can set that field
- the input/output chats may be structured (e.g. JSON) as long as they are consumable by a chat LLM

For multiple choice tasks, there is a [`format_multichoice_as_textin`](https://github.com/allenai/astabench/blob/691327e55fd9b43fe2686f28ddee9cfd5fff7ad5/astabench/evals/utils.py) utility that can be used to convert Inspect's `sample.choices` into a pure text format as required (used by e.g. [litqa2](https://github.com/allenai/asta-bench/blob/f507043c159fa6b236a87473b1acdf8ab8b016f1/astabench/evals/labbench/litqa2/task.py#L277)).
