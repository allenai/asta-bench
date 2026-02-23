# Scorer (frozen scoring environment)

This is a dedicated uv sub-project for scoring Inspect logs with a frozen
`inspect_ai` version. It is not a solver implementation.

## Dependencies

This environment is configured in `solvers/scorer/pyproject.toml`.

- It installs local `astabench` from this repo.
- It pins `inspect_ai==0.3.179`.
- It uses a uv override so scoring stays pinned even though local `astabench`
  depends on `inspect_ai==0.3.114`.

Install deps with:

```bash
./solvers/scorer/setup.sh
```

## Usage

Recommended: use the wrapper from repo root:

```bash
./scripts/eval_then_score.sh -- \
  --split validation --solver react --model openai/gpt-4.1 --limit 1
```

This handles:
1. Solve with `astabench eval --no-score --log-format json`
2. Materialize scores with `inspect score --overwrite` in `solvers/scorer`
3. Aggregate with `astabench score` in `solvers/scorer`

Manual scoring command (single log):

```bash
uv run --project "solvers/scorer" --frozen -- \
  inspect score --overwrite <log_file>
```

If Inspect cannot infer the scorer from the log:

```bash
uv run --project "solvers/scorer" --frozen -- \
  inspect score --scorer path/to/task.py@scorer_fn --overwrite <log_file>
```

Aggregate command:

```bash
LITELLM_LOCAL_MODEL_COST_MAP=True \
  uv run --project "solvers/scorer" --frozen -- astabench score <log_dir>
```
