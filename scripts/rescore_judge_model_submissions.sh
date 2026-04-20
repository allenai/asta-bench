#!/usr/bin/env bash

# Rescore only submissions whose `.eval` logs match TARGET_LOG_REGEX.
#
# This script is self-contained:
#   1) finds matching submission directories under the submissions tree
#   2) deletes only the matching submission's output directory
#   3) copies that submission into the rescored tree
#   4) re-runs per-log `inspect score` on the copied `.eval` logs
#   5) re-runs aggregate `astabench score` on the copied submission
#
# The source submissions tree is never mutated.

set -euo pipefail

# Edit this regex to change which submission logs are targeted for rescoring.
TARGET_LOG_REGEX='(sqa-(test|dev)|e2e-discovery(-hard)?-(test|validation))'
RESCORE_STATE_FILE='.rescore-state.json'

usage() {
  cat <<EOF >&2
Usage:
  scripts/rescore_judge_model_submissions.sh [options]

Options:
  --submissions-root <dir>   Root directory containing versioned submissions.
                             Default: asta-bench-submissions
  --output-root <dir>        Root directory for copied + rescored submissions.
                             Default: asta-bench-submissions-rescored
  --scorer-project <path>    uv project used for inspect/astabench scoring.
                             Default: solvers/scorer
  --scorer <scorer_spec>     Optional scorer override for \`inspect score\`.
  --target-log-regex <regex> Regex used to select target submission logs.
                             Default: ${target_log_regex_default}
  --dry-run                  Print matching submissions without rescoring.
  -h, --help                 Show this help.
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  target_log_regex_default="${TARGET_LOG_REGEX}"
  usage
  exit 0
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required" >&2
  exit 2
fi

target_log_regex_default="${TARGET_LOG_REGEX}"

submissions_root="asta-bench-submissions"
output_root="asta-bench-submissions-rescored"
scorer_project="solvers/scorer"
score_scorer=""
target_log_regex="${target_log_regex_default}"
dry_run=0

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
    --target-log-regex)
      if [ $# -lt 2 ]; then
        echo "error: --target-log-regex requires a value" >&2
        exit 2
      fi
      target_log_regex="$2"
      shift 2
      ;;
    --target-log-regex=*)
      target_log_regex="${1#--target-log-regex=}"
      shift
      ;;
    --dry-run)
      dry_run=1
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

find_target_submissions() {
  # `grep -E` returns 1 when the target regex matches nothing. With
  # `set -o pipefail`, that would otherwise abort the script even though
  # "no matches" is a valid outcome for this helper.
  find "${submissions_root}" -type f -name '*.eval' -print \
    | LC_ALL=C grep -E "${target_log_regex}" \
    | sed -E 's#/[^/]+$##' \
    | LC_ALL=C sort -u || true
}

list_actual_eval_files() {
  local submission_dir="$1"

  find "${submission_dir}" -maxdepth 1 -type f -name '*.eval' -print \
    | sed -E 's#.*/##' \
    | LC_ALL=C sort
}

list_log_files() {
  local submission_dir="$1"

  if [ -f "${submission_dir}/logs.json" ]; then
    jq -r 'keys[]' "${submission_dir}/logs.json" | LC_ALL=C sort
    return
  fi

  list_actual_eval_files "${submission_dir}"
}

verify_log_inventory() {
  local submission_dir="$1"
  local listed
  local actual
  local listed_count
  local actual_count

  [ -f "${submission_dir}/logs.json" ] || return 0

  listed="$(list_log_files "${submission_dir}")" || return 1
  actual="$(list_actual_eval_files "${submission_dir}")" || return 1
  if [ "${listed}" = "${actual}" ]; then
    return 0
  fi

  listed_count="$(printf '%s\n' "${listed}" | sed '/^$/d' | wc -l | tr -d ' ')"
  actual_count="$(printf '%s\n' "${actual}" | sed '/^$/d' | wc -l | tr -d ' ')"

  echo "error: logs.json inventory does not match .eval files in ${submission_dir}" >&2
  echo "error: listed=${listed_count} actual=${actual_count}" >&2
  return 1
}

write_submission_state() {
  local output_submission_dir="$1"
  local stage="$2"
  local current_log="${3:-}"
  local message="${4:-}"

  jq -n \
    --arg stage "${stage}" \
    --arg current_log "${current_log}" \
    --arg message "${message}" \
    '{
      stage: $stage,
      current_log: (if $current_log == "" then null else $current_log end),
      message: (if $message == "" then null else $message end)
    }' > "${output_submission_dir}/${RESCORE_STATE_FILE}"
}

clear_submission_state() {
  local output_submission_dir="$1"

  rm -f "${output_submission_dir}/${RESCORE_STATE_FILE}"
}

prepare_output_submission_dir() {
  local source_submission_dir="$1"
  local output_submission_dir="$2"

  if [ -e "${output_submission_dir}" ] && [ ! -d "${output_submission_dir}" ]; then
    echo "error: output path exists and is not a directory (${output_submission_dir})" >&2
    return 1
  fi

  if [ -d "${output_submission_dir}" ]; then
    rm -rf "${output_submission_dir}"
    echo "== refresh: ${output_submission_dir}" >&2
  fi

  mkdir -p "$(dirname "${output_submission_dir}")"
  cp -a "${source_submission_dir}" "${output_submission_dir}"
  echo "== copy: ${source_submission_dir} -> ${output_submission_dir}" >&2
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
  write_submission_state "${output_submission_dir}" "verifying"
  if ! verify_log_inventory "${output_submission_dir}"; then
    write_submission_state "${output_submission_dir}" "failed" "" "logs.json inventory mismatch"
    return 1
  fi

  echo "== rescore: ${version}/${split}/${submission_name}" >&2

  log_files="$(list_log_files "${output_submission_dir}")" || {
    write_submission_state "${output_submission_dir}" "failed" "" "unable to list eval logs"
    return 1
  }
  if [ -z "${log_files}" ]; then
    write_submission_state "${output_submission_dir}" "failed" "" "no eval logs found"
    echo "error: no eval logs found in ${output_submission_dir}" >&2
    return 1
  fi

  while IFS= read -r log_file; do
    [ -z "${log_file}" ] && continue
    if [ ! -f "${output_submission_dir}/${log_file}" ]; then
      write_submission_state "${output_submission_dir}" "failed" "${log_file}" "log file listed but missing"
      echo "error: log file listed but missing: ${output_submission_dir}/${log_file}" >&2
      return 1
    fi

    write_submission_state "${output_submission_dir}" "rescoring" "${log_file}"
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
      write_submission_state "${output_submission_dir}" "failed" "${log_file}" "inspect score failed"
      echo "error: inspect score failed for ${output_submission_dir}/${log_file}" >&2
      return 1
    fi
  done <<<"${log_files}"

  write_submission_state "${output_submission_dir}" "aggregating"
  aggregate_cmd=(
    uv run --project "${scorer_project}" --frozen -- astabench score
    "${output_submission_dir}"
  )
  if ! env LITELLM_LOCAL_MODEL_COST_MAP=True "${aggregate_cmd[@]}"; then
    write_submission_state "${output_submission_dir}" "failed" "" "aggregate scoring failed"
    echo "error: aggregate scoring failed for ${output_submission_dir}" >&2
    return 1
  fi

  clear_submission_state "${output_submission_dir}"
}

matched_count=0
scored_count=0

while IFS= read -r submission_dir; do
  [ -n "${submission_dir}" ] || continue
  matched_count=$((matched_count + 1))

  rel="${submission_dir#${submissions_root}/}"
  version="$(basename "$(dirname "$(dirname "${submission_dir}")")")"
  split="$(basename "$(dirname "${submission_dir}")")"
  submission_name="$(basename "${submission_dir}")"
  output_submission_dir="${output_root}/${rel}"
  config_path="astabench/config/v${version}.yml"

  if [ ! -f "${config_path}" ]; then
    echo "== skip: no config file for version ${version} (${config_path})" >&2
    continue
  fi

  case "${split}" in
    test|validation) ;;
    *)
      echo "== skip: unrecognized split ${split} for ${rel}" >&2
      continue
      ;;
  esac

  if [ "${dry_run}" -eq 1 ]; then
    printf '%s\n' "${submission_dir}"
    continue
  fi

  echo "== queue: ${rel}" >&2
  if ! rescore_one_submission "${version}" "${split}" "${submission_dir}" "${output_submission_dir}" "${submission_name}"; then
    echo "== error: failed ${rel}" >&2
    exit 1
  fi

  scored_count=$((scored_count + 1))
done < <(find_target_submissions)

if [ "${matched_count}" -eq 0 ]; then
  echo "warning: no target submissions found under ${submissions_root} using regex ${target_log_regex}" >&2
  exit 0
fi

if [ "${dry_run}" -eq 1 ]; then
  echo "== done: matched ${matched_count} submissions" >&2
  exit 0
fi

echo "== done: rescored ${scored_count} submissions" >&2
