#!/usr/bin/env bash

# Two-phase "solve then score" wrapper.
#
# Runs:
#  1) Solve in a selected uv project with `--no-score --log-format json`
#  2) Score those logs in the frozen `solvers/scorer` uv project
#
# This decouples solving dependencies from scoring dependencies (notably
# `inspect_ai`) while keeping logs as the stable artifact boundary.

set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  scripts/eval_then_score.sh [--solve-project <path>] [--scorer-project <path>] \
    [--log-dir <dir>] [--scorer <scorer_spec>] [--] <astabench eval args...>

Examples:
  scripts/eval_then_score.sh -- \
    --split validation --solver react --model openai/gpt-4.1 --limit 1

  scripts/eval_then_score.sh --solve-project /path/to/solver-env --log-dir logs/two_phase -- \
    --split validation --solver my_solver --model openai/gpt-4.1 --limit 1

Notes:
  - The solve phase enforces `--no-score --log-format json`.
  - Do not pass `--log-format` in eval args (the wrapper enforces JSON logs).
  - Requires `jq` to read `logs.json` (uses: `jq -r 'keys[]'`).
  - For custom/non-registered scorers, pass `--scorer <path.py@scorer_name>`.
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required (expected: jq -r 'keys[]' <log_dir>/logs.json)" >&2
  exit 2
fi

solve_project="."
scorer_project="solvers/scorer"
log_dir=""
score_scorer=""

while [ $# -gt 0 ]; do
  case "$1" in
    --solve-project)
      if [ $# -lt 2 ]; then
        echo "error: --solve-project requires a value" >&2
        exit 2
      fi
      solve_project="$2"
      shift 2
      ;;
    --solve-project=*)
      solve_project="${1#--solve-project=}"
      shift
      ;;
    --scorer-project)
      if [ $# -lt 2 ]; then
        echo "error: --scorer-project requires a value" >&2
        exit 2
      fi
      scorer_project="$2"
      shift 2
      ;;
    --scorer-project=*)
      scorer_project="${1#--scorer-project=}"
      shift
      ;;
    --log-dir)
      if [ $# -lt 2 ]; then
        echo "error: --log-dir requires a value" >&2
        exit 2
      fi
      log_dir="$2"
      shift 2
      ;;
    --log-dir=*)
      log_dir="${1#--log-dir=}"
      shift
      ;;
    --scorer)
      if [ $# -lt 2 ]; then
        echo "error: --scorer requires a value" >&2
        exit 2
      fi
      score_scorer="$2"
      shift 2
      ;;
    --scorer=*)
      score_scorer="${1#--scorer=}"
      shift
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

if [ ! -f "${solve_project}/pyproject.toml" ]; then
  echo "error: solve project must contain pyproject.toml (${solve_project})" >&2
  exit 2
fi
if [ ! -f "${scorer_project}/pyproject.toml" ]; then
  echo "error: scorer project must contain pyproject.toml (${scorer_project})" >&2
  exit 2
fi

if [ -z "${log_dir}" ]; then
  ts="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
  log_dir="logs/two_phase_${ts}"
fi

mkdir -p "${log_dir}"
eval_args=("$@")

for ((i = 0; i < ${#eval_args[@]}; i++)); do
  arg="${eval_args[$i]}"
  case "${arg}" in
    --log-format|--log-format=*|--score)
      echo "error: do not pass ${arg}; this wrapper enforces --no-score --log-format json" >&2
      exit 2
      ;;
  esac
done

echo "== solve: project=${solve_project} -> ${log_dir}" >&2
uv run --project "${solve_project}" --frozen -- astabench eval \
  --log-dir "${log_dir}" \
  --no-score \
  --log-format json \
  "${eval_args[@]}"

if [ ! -f "${log_dir}/logs.json" ]; then
  echo "error: ${log_dir}/logs.json not found" >&2
  exit 1
fi

echo "== score: inspect score (project=${scorer_project})" >&2
log_files="$(jq -r 'keys[]' "${log_dir}/logs.json")"
if [ -z "${log_files}" ]; then
  echo "error: no log files listed in ${log_dir}/logs.json" >&2
  exit 1
fi

while IFS= read -r log_file; do
  [ -z "${log_file}" ] && continue
  score_cmd=(
    uv run --project "${scorer_project}" --frozen -- inspect score
  )
  if [ -n "${score_scorer}" ]; then
    score_cmd+=(--scorer "${score_scorer}")
  fi
  score_cmd+=(--overwrite)
  score_cmd+=("${log_dir}/${log_file}")
  "${score_cmd[@]}"
done <<<"${log_files}"

echo "== score: aggregate ${log_dir} (project=${scorer_project})" >&2
# Required by astabench score: prep_litellm_cost_map() raises without this.
LITELLM_LOCAL_MODEL_COST_MAP=True uv run --project "${scorer_project}" --frozen -- astabench score "${log_dir}"

echo "== done: scored logs in ${log_dir}" >&2
