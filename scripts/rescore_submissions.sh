#!/usr/bin/env bash

# Rescore downloaded submissions into a separate output tree.
#
# For each source submission directory under:
#   asta-bench-submissions/<version>/<split>/<submission_name>
#
# this script:
#   1) copies the submission into a separate rescored tree
#   2) re-runs `inspect score --action overwrite --overwrite` on each copied `.eval` log
#   3) re-runs `astabench score` to aggregate task metrics and costs
#
# The solve phase from scripts/eval_then_score.sh is intentionally omitted:
# these submission directories already contain the raw eval logs.
# The source submissions tree is left untouched.
# The script is idempotent over the output tree: allowlisted logs are rescored,
# and preserved logs keep their copied scores across re-runs.

set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  scripts/rescore_submissions.sh [options]

Options:
  --submissions-root <dir>   Root directory containing versioned submissions.
                             Default: asta-bench-submissions
  --output-root <dir>        Root directory for copied + rescored submissions.
                             Default: asta-bench-submissions-rescored
  --scorer-project <path>    uv project used for inspect/astabench scoring.
                             Default: solvers/scorer
  --scorer <scorer_spec>     Optional scorer override for `inspect score`.
  --refresh                  Re-copy output submissions from source before rescoring.
  -h, --help                 Show this help.

Examples:
  scripts/rescore_submissions.sh

  scripts/rescore_submissions.sh --refresh

Notes:
  - Source submissions are never mutated.
  - If an output submission directory already exists, it is reused and rescored in place.
  - Use --refresh to discard an existing output submission copy and recreate it from source.
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required" >&2
  exit 2
fi

submissions_root="asta-bench-submissions"
output_root="asta-bench-submissions-rescored"
scorer_project="solvers/scorer"
score_scorer=""
refresh_output=0

while [ $# -gt 0 ]; do
  case "$1" in
    --submissions-root)
      if [ $# -lt 2 ]; then
        echo "error: --submissions-root requires a value" >&2
        exit 2
      fi
      submissions_root="$2"
      shift 2
      ;;
    --submissions-root=*)
      submissions_root="${1#--submissions-root=}"
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
    --output-root)
      if [ $# -lt 2 ]; then
        echo "error: --output-root requires a value" >&2
        exit 2
      fi
      output_root="$2"
      shift 2
      ;;
    --output-root=*)
      output_root="${1#--output-root=}"
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
    --refresh)
      refresh_output=1
      shift
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [ ! -d "${submissions_root}" ]; then
  echo "error: submissions root not found (${submissions_root})" >&2
  exit 2
fi
if [ ! -f "${scorer_project}/pyproject.toml" ]; then
  echo "error: scorer project must contain pyproject.toml (${scorer_project})" >&2
  exit 2
fi

mkdir -p "$(dirname "${output_root}")"

canonical_path() {
  local path="$1"
  local dir
  local base

  if [ -d "${path}" ]; then
    (
      cd "${path}" && pwd -P
    )
    return
  fi

  dir="$(dirname "${path}")"
  base="$(basename "${path}")"
  dir="$(
    cd "${dir}" 2>/dev/null && pwd -P
  )" || return 1
  printf '%s/%s\n' "${dir}" "${base}"
}

source_root_canonical="$(canonical_path "${submissions_root}")" || {
  echo "error: unable to resolve submissions root (${submissions_root})" >&2
  exit 2
}
output_root_canonical="$(canonical_path "${output_root}")" || {
  echo "error: unable to resolve output root parent (${output_root})" >&2
  exit 2
}

if [ "${source_root_canonical}" = "${output_root_canonical}" ] \
  || [[ "${output_root_canonical}" == "${source_root_canonical}/"* ]] \
  || [[ "${source_root_canonical}" == "${output_root_canonical}/"* ]]; then
  echo "error: --output-root must not overlap with --submissions-root" >&2
  exit 2
fi

prepare_output_submission_dir() {
  local source_submission_dir="$1"
  local output_submission_dir="$2"

  if [ -e "${output_submission_dir}" ] && [ ! -d "${output_submission_dir}" ]; then
    echo "error: output path exists and is not a directory (${output_submission_dir})" >&2
    return 1
  fi

  if [ "${refresh_output}" -eq 1 ] && [ -d "${output_submission_dir}" ]; then
    rm -rf "${output_submission_dir}"
    echo "== refresh: ${output_submission_dir}" >&2
  fi

  if [ -d "${output_submission_dir}" ]; then
    echo "== reuse: ${output_submission_dir}" >&2
    return 0
  fi

  mkdir -p "$(dirname "${output_submission_dir}")"
  cp -a "${source_submission_dir}" "${output_submission_dir}"
  echo "== copy: ${source_submission_dir} -> ${output_submission_dir}" >&2
}

list_log_files() {
  local submission_dir="$1"

  if [ -f "${submission_dir}/logs.json" ]; then
    jq -r 'keys[]' "${submission_dir}/logs.json" | LC_ALL=C sort
    return
  fi

  list_actual_eval_files "${submission_dir}"
}

list_actual_eval_files() {
  local submission_dir="$1"
  local prefix="${submission_dir%/}/"

  find "${submission_dir}" -maxdepth 1 -type f -name '*.eval' -print | LC_ALL=C sort | while IFS= read -r path; do
    printf '%s\n' "${path#$prefix}"
  done
}

verify_log_inventory() {
  local submission_dir="$1"
  local listed_logs
  local actual_logs
  local listed_count
  local actual_count

  if [ ! -f "${submission_dir}/logs.json" ]; then
    return 0
  fi

  listed_logs="$(jq -r 'keys[]' "${submission_dir}/logs.json" | LC_ALL=C sort)"
  actual_logs="$(list_actual_eval_files "${submission_dir}")"

  if [ "${listed_logs}" != "${actual_logs}" ]; then
    listed_count="$(printf '%s\n' "${listed_logs}" | sed '/^$/d' | wc -l | tr -d ' ')"
    actual_count="$(printf '%s\n' "${actual_logs}" | sed '/^$/d' | wc -l | tr -d ' ')"
    echo "error: logs.json inventory does not match .eval files in ${submission_dir}" >&2
    echo "error: listed=${listed_count} actual=${actual_count}" >&2
    return 1
  fi
}

rescore_one_submission() {
  local version="$1"
  local split="$2"
  local source_submission_dir="$3"
  local output_submission_dir="$4"
  local submission_name="$5"
  local log_files
  local log_file

  prepare_output_submission_dir "${source_submission_dir}" "${output_submission_dir}" || return 1
  verify_log_inventory "${output_submission_dir}" || return 1

  echo "== rescore: ${version}/${split}/${submission_name}" >&2

  log_files="$(list_log_files "${output_submission_dir}")" || return 1
  if [ -z "${log_files}" ]; then
    echo "error: no eval logs found in ${output_submission_dir}" >&2
    return 1
  fi

  while IFS= read -r log_file; do
    [ -z "${log_file}" ] && continue
    if [ ! -f "${output_submission_dir}/${log_file}" ]; then
      echo "error: log file listed but missing: ${output_submission_dir}/${log_file}" >&2
      return 1
    fi

    echo "  score: ${log_file}" >&2
    score_cmd=(
      uv run --project "${scorer_project}" --frozen --
      python scripts/inspect_score_with_task.py
    )
    if [ -n "${score_scorer}" ]; then
      score_cmd+=(--scorer "${score_scorer}")
    fi
    score_cmd+=(--action overwrite --overwrite)
    score_cmd+=("${output_submission_dir}/${log_file}")
    if ! "${score_cmd[@]}"; then
      echo "error: inspect score failed for ${output_submission_dir}/${log_file}" >&2
      return 1
    fi
  done <<<"${log_files}"

  aggregate_cmd=(
    uv run --project "${scorer_project}" --frozen -- astabench score
    "${output_submission_dir}"
  )
  if ! env LITELLM_LOCAL_MODEL_COST_MAP=True "${aggregate_cmd[@]}"; then
    echo "error: aggregate scoring failed for ${output_submission_dir}" >&2
    return 1
  fi
}

scored_count=0
failed_count=0

while IFS= read -r version_dir; do
  [ -n "${version_dir}" ] || continue

  version="$(basename "${version_dir}")"
  config_path="astabench/config/v${version}.yml"
  # Skip unknown version directories that have no suite config.
  if [ ! -f "${config_path}" ]; then
    echo "== skip: no config file for version ${version} (${config_path})" >&2
    continue
  fi

  while IFS= read -r split_dir; do
    [ -n "${split_dir}" ] || continue

    split="$(basename "${split_dir}")"
    case "${split}" in
      test|validation) ;;
      *)
        echo "== skip: unrecognized split ${split_dir}" >&2
        continue
        ;;
    esac

    while IFS= read -r submission_dir; do
      [ -n "${submission_dir}" ] || continue

      submission_name="$(basename "${submission_dir}")"
      output_submission_dir="${output_root}/${version}/${split}/${submission_name}"

      if ! rescore_one_submission "${version}" "${split}" "${submission_dir}" "${output_submission_dir}" "${submission_name}"; then
        failed_count=$((failed_count + 1))
        echo "== error: failed ${version}/${split}/${submission_name}; continuing" >&2
        continue
      fi

      scored_count=$((scored_count + 1))
    done < <(find "${split_dir}" -mindepth 1 -maxdepth 1 -type d | LC_ALL=C sort)
  done < <(find "${version_dir}" -mindepth 1 -maxdepth 1 -type d | LC_ALL=C sort)
done < <(find "${submissions_root}" -mindepth 1 -maxdepth 1 -type d | LC_ALL=C sort)

if [ "$((scored_count + failed_count))" -eq 0 ]; then
  echo "warning: no submission directories found under ${submissions_root}" >&2
else
  echo "== done: rescored ${scored_count} submissions" >&2
fi

if [ "${failed_count}" -gt 0 ]; then
  echo "== done: ${failed_count} submissions failed" >&2
  exit 1
fi
