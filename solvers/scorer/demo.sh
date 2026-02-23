#!/usr/bin/env bash
# Score a directory of logs in the frozen scorer environment.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

if [ $# -lt 1 ]; then
  echo "Usage: ./solvers/scorer/demo.sh <log_dir> [score args...]" >&2
  echo "Example: ./solvers/scorer/demo.sh logs/my-run" >&2
  exit 2
fi

log_dir="$1"
shift

LITELLM_LOCAL_MODEL_COST_MAP=True \
  uv run --project "solvers/scorer" --frozen -- astabench score "${log_dir}" "$@"
