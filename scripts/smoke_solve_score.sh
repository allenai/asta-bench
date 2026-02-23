#!/usr/bin/env bash

# Cross-version solve->score smoke.
#
# This proves:
# - Solving can run in one uv project with one inspect_ai version
# - Scoring can run in a separate frozen uv project with a different version
# - The artifact boundary is stable logs (JSON + --no-score)

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

solve_project="${1:-.}"
scorer_project="solvers/scorer"

config_path="scripts/ci/two_phase_smoke.yml"
solver_spec="scripts/arithmetic_solver.py@arithmetic_solver"
scorer_spec="scripts/ci/arithmetic_task.py@check_arithmetic"
model="mockllm/model"

if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required (expected: jq -r 'keys[]' <log_dir>/logs.json)" >&2
  exit 2
fi

if [ ! -f "${solve_project}/pyproject.toml" ]; then
  echo "error: solve project must contain pyproject.toml (${solve_project})" >&2
  exit 2
fi
if [ ! -f "${scorer_project}/pyproject.toml" ]; then
  echo "error: missing scorer project (${scorer_project}/pyproject.toml)" >&2
  exit 2
fi

solve_inspect_version="$(uv run --project "${solve_project}" --frozen -- python -c 'import inspect_ai; print(inspect_ai.__version__)')"
scorer_inspect_version="$(uv run --project "${scorer_project}" --frozen -- python -c 'import inspect_ai; print(inspect_ai.__version__)')"

echo "== inspect versions: solve(${solve_inspect_version}) scorer(${scorer_inspect_version})" >&2
if [ "${solve_inspect_version}" = "${scorer_inspect_version}" ]; then
  echo "warning: solve and scorer inspect_ai versions match; decoupling still works but cross-version coverage is reduced" >&2
  if [ "${IS_CI:-}" = "true" ]; then
    echo "::warning::solve and scorer inspect_ai versions match; cross-version smoke coverage is reduced" >&2
  fi
fi

if command -v mktemp >/dev/null 2>&1; then
  # macOS: mktemp -d -t prefix ; Linux: mktemp -d
  log_dir="$(mktemp -d -t solve-score-XXXXXXXX 2>/dev/null || mktemp -d)"
else
  ts="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
  log_dir="logs/solve_score_smoke_${ts}"
  mkdir -p "${log_dir}"
fi

cleanup() {
  if [ "${log_dir:-}" != "" ] && [[ "${log_dir}" == /tmp/* || "${log_dir}" == /var/folders/* ]]; then
    rm -rf "${log_dir}"
  fi
}
trap cleanup EXIT

echo "== solve: ${solve_project} -> ${log_dir}" >&2
uv run --project "${solve_project}" --frozen -- astabench eval \
  --log-dir "${log_dir}" \
  --config-path "${config_path}" \
  --split validation \
  --ignore-git \
  --solver "${solver_spec}" \
  --model "${model}" \
  --limit 1 \
  --no-score \
  --log-format json

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
  uv run --project "${scorer_project}" --frozen -- inspect score \
    --scorer "${scorer_spec}" \
    --overwrite \
    "${log_dir}/${log_file}"
done <<<"${log_files}"

echo "== score: aggregate (astabench score)" >&2
LITELLM_LOCAL_MODEL_COST_MAP=True uv run --project "${scorer_project}" --frozen -- astabench score "${log_dir}"

scores_file="${log_dir}/scores.json"
if [ ! -f "${scores_file}" ]; then
  echo "error: missing ${scores_file}" >&2
  exit 1
fi

accuracy="$(
  jq -r '
    [
      .results[]?
      | .metrics[]?
      | select(.name == "check_arithmetic/accuracy")
      | .value
    ] | if length == 0 then empty else .[0] end
  ' "${scores_file}"
)"
if [ -z "${accuracy}" ]; then
  echo "error: missing check_arithmetic/accuracy metric in scores.json" >&2
  exit 1
fi

if ! awk -v value="${accuracy}" 'BEGIN { exit !(value > 0.999999999 && value < 1.000000001) }'; then
  echo "error: expected check_arithmetic/accuracy=1.0, got ${accuracy}" >&2
  exit 1
fi

echo "== smoke: ok (check_arithmetic/accuracy=${accuracy})" >&2
