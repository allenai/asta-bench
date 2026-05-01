#!/usr/bin/env bash

# Rescore only submissions whose `.eval` logs match TARGET_LOG_REGEX.
#
# This script is self-contained:
#   1) finds matching submission directories under the submissions tree
#   2) safely resumes or refreshes only the matching submission's output directory
#   3) copies that submission into the rescored tree when a full rerun is needed
#   4) re-runs per-log `inspect score` on the copied `.eval` logs
#   5) re-runs aggregate `astabench score` on the copied submission
#
# The source submissions tree is never mutated.

set -euo pipefail

# Template guard: this regex-based log detection is intentionally per-rescore
# and may miss renamed logs or other LLM-judged tasks. Adapt it before use.
echo "This script is a template and needs to be adapted to your use case.  Edit and try again"; exit 1

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
  --duplicate-task-policy <policy>
                             How to handle duplicate task logs in one submission.
                             Choices: fail, keep-latest
                             Default: fail
  --resume                   Safe resume mode:
                             completed => skip
                             aggregate_pending/aggregating => aggregate only
                             everything else => rerun from scratch
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
duplicate_task_policy="fail"
target_log_regex="${target_log_regex_default}"
resume=0
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
    --duplicate-task-policy)
      if [ $# -lt 2 ]; then
        echo "error: --duplicate-task-policy requires a value" >&2
        exit 2
      fi
      duplicate_task_policy="$2"
      shift 2
      ;;
    --duplicate-task-policy=*)
      duplicate_task_policy="${1#--duplicate-task-policy=}"
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
    --resume)
      resume=1
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

if [ "${resume}" -eq 1 ]; then
  if command -v python3 >/dev/null 2>&1; then
    python_bin="python3"
  elif command -v python >/dev/null 2>&1; then
    python_bin="python"
  else
    echo "error: python3 or python is required for --resume" >&2
    exit 2
  fi
fi

if [ ! -d "${submissions_root}" ]; then
  echo "error: submissions root not found (${submissions_root})" >&2
  exit 2
fi
case "${duplicate_task_policy}" in
  fail|keep-latest) ;;
  *)
    echo "error: --duplicate-task-policy must be one of: fail, keep-latest" >&2
    exit 2
    ;;
esac
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

resume_info_for_submission() {
  local rel="$1"

  "${python_bin}" scripts/rescore_progress.py \
    --submissions-root "${submissions_root}" \
    --output-root "${output_root}" \
    --target-log-regex "${target_log_regex}" \
    --submission-rel "${rel}"
}

handle_duplicate_task_logs() {
  local rel="$1"
  local output_submission_dir="$2"
  local archive_dir="${output_root}/_duplicate_task_logs_archive/${rel}"
  local duplicate_output
  local duplicate_cmd=(
    uv run --project "${scorer_project}" --frozen --
    python scripts/dedupe_eval_logs.py
    --submission-dir "${output_submission_dir}"
    --policy "${duplicate_task_policy}"
  )

  if [ "${duplicate_task_policy}" = "keep-latest" ]; then
    duplicate_cmd+=(--archive-dir "${archive_dir}")
  fi

  duplicate_output="$("${duplicate_cmd[@]}" 2>&1)" || {
    if [ -n "${duplicate_output}" ]; then
      printf '%s\n' "${duplicate_output}" >&2
    fi
    write_submission_state "${output_submission_dir}" "failed" "" "duplicate task logs detected"
    return 1
  }

  if [ -n "${duplicate_output}" ]; then
    printf '%s\n' "${duplicate_output}" >&2
  fi
}

prepare_submission_for_scoring() {
  local rel="$1"
  local output_submission_dir="$2"

  write_submission_state "${output_submission_dir}" "verifying"
  if ! handle_duplicate_task_logs "${rel}" "${output_submission_dir}"; then
    return 1
  fi
  if ! verify_log_inventory "${output_submission_dir}"; then
    write_submission_state "${output_submission_dir}" "failed" "" "logs.json inventory mismatch"
    return 1
  fi
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

aggregate_submission() {
  local output_submission_dir="$1"

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

rescore_one_submission() {
  local rel="$1"
  local version="$2"
  local split="$3"
  local source_submission_dir="$4"
  local output_submission_dir="$5"
  local submission_name="$6"
  local log_files
  local log_file

  prepare_output_submission_dir "${source_submission_dir}" "${output_submission_dir}" || return 1
  prepare_submission_for_scoring "${rel}" "${output_submission_dir}" || return 1

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

  aggregate_submission "${output_submission_dir}" || return 1
}

aggregate_only_submission() {
  local rel="$1"
  local version="$2"
  local split="$3"
  local output_submission_dir="$4"
  local submission_name="$5"

  if [ ! -d "${output_submission_dir}" ]; then
    echo "error: output submission directory missing for aggregate-only resume (${output_submission_dir})" >&2
    return 1
  fi

  prepare_submission_for_scoring "${rel}" "${output_submission_dir}" || return 1
  echo "== aggregate: ${version}/${split}/${submission_name}" >&2
  aggregate_submission "${output_submission_dir}" || return 1
}

matched_count=0
rescored_count=0
aggregate_only_count=0
skipped_count=0

while IFS= read -r submission_dir; do
  [ -n "${submission_dir}" ] || continue
  matched_count=$((matched_count + 1))

  rel="${submission_dir#${submissions_root}/}"
  version="$(basename "$(dirname "$(dirname "${submission_dir}")")")"
  split="$(basename "$(dirname "${submission_dir}")")"
  submission_name="$(basename "${submission_dir}")"
  output_submission_dir="${output_root}/${rel}"
  config_path="astabench/config/v${version}.yml"
  resume_action="full"
  resume_status="pending"
  resume_info=""

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

  if [ "${resume}" -eq 1 ]; then
    resume_info="$(resume_info_for_submission "${rel}")" || {
      echo "error: unable to determine resume action for ${rel}" >&2
      exit 1
    }
    resume_action="$(printf '%s' "${resume_info}" | jq -r '.action')"
    resume_status="$(printf '%s' "${resume_info}" | jq -r '.status')"
    case "${resume_action}" in
      skip|aggregate|full) ;;
      *)
        echo "error: invalid resume action for ${rel}: ${resume_action}" >&2
        exit 1
        ;;
    esac
  fi

  if [ "${dry_run}" -eq 1 ]; then
    if [ "${resume}" -eq 1 ]; then
      printf '%s\t%s\t%s\n' "${resume_action}" "${resume_status}" "${submission_dir}"
    else
      printf '%s\n' "${submission_dir}"
    fi
    continue
  fi

  case "${resume_action}" in
    skip)
      echo "== skip: already complete ${rel}" >&2
      skipped_count=$((skipped_count + 1))
      continue
      ;;
    aggregate)
      echo "== resume: aggregate-only ${rel} (${resume_status})" >&2
      if ! aggregate_only_submission "${rel}" "${version}" "${split}" "${output_submission_dir}" "${submission_name}"; then
        echo "== error: failed ${rel}" >&2
        exit 1
      fi
      aggregate_only_count=$((aggregate_only_count + 1))
      continue
      ;;
    full)
      if [ "${resume}" -eq 1 ]; then
        echo "== resume: rerun ${rel} (${resume_status})" >&2
      fi
      ;;
  esac

  echo "== queue: ${rel}" >&2
  if ! rescore_one_submission "${rel}" "${version}" "${split}" "${submission_dir}" "${output_submission_dir}" "${submission_name}"; then
    echo "== error: failed ${rel}" >&2
    exit 1
  fi

  rescored_count=$((rescored_count + 1))
done < <(find_target_submissions)

if [ "${matched_count}" -eq 0 ]; then
  echo "warning: no target submissions found under ${submissions_root} using regex ${target_log_regex}" >&2
  exit 0
fi

if [ "${dry_run}" -eq 1 ]; then
  echo "== done: matched ${matched_count} submissions" >&2
  exit 0
fi

if [ "${resume}" -eq 1 ]; then
  echo "== done: rescored ${rescored_count} submissions, aggregated ${aggregate_only_count}, skipped ${skipped_count}" >&2
else
  echo "== done: rescored ${rescored_count} submissions" >&2
fi
