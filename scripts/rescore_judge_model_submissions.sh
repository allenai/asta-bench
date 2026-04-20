#!/usr/bin/env bash

# Rescore only submissions that contain ScholarQA or E2E task logs.
#
# Each matching submission is processed independently:
#   1) delete only that submission's rescored output directory, if it exists
#   2) build a temporary one-submission tree with the expected version/split layout
#   3) invoke scripts/rescore_submissions.sh on that temporary tree
#
# This keeps progress visible per submission and avoids wiping the entire
# asta-bench-submissions-rescored tree.

set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  scripts/rescore_judge_model_submissions.sh [options]

Options:
  --submissions-root <dir>   Root directory containing versioned submissions.
                             Default: asta-bench-submissions
  --output-root <dir>        Root directory for copied + rescored submissions.
                             Default: asta-bench-submissions-rescored
  --scorer-project <path>    uv project used for inspect/astabench scoring.
                             Default: solvers/scorer
  --dry-run                  Print matching submissions without rescoring.
  -h, --help                 Show this help.
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

submissions_root="asta-bench-submissions"
output_root="asta-bench-submissions-rescored"
scorer_project="solvers/scorer"
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

find_target_submissions() {
  find "${submissions_root}" -type f -name '*.eval' | \
    rg '(sqa-(test|dev)|e2e-discovery(-hard)?-(test|validation))' | \
    sed -E 's#/[^/]+$##' | \
    LC_ALL=C sort -u
}

matched_count=0
scored_count=0
failed_count=0

while IFS= read -r submission_dir; do
  [ -n "${submission_dir}" ] || continue
  matched_count=$((matched_count + 1))

  rel="${submission_dir#${submissions_root}/}"
  output_submission_dir="${output_root}/${rel}"

  if [ "${dry_run}" -eq 1 ]; then
    printf '%s\n' "${submission_dir}"
    continue
  fi

  version="$(basename "$(dirname "$(dirname "${submission_dir}")")")"
  split="$(basename "$(dirname "${submission_dir}")")"

  echo "== queue: ${rel}" >&2
  rm -rf "${output_submission_dir}"

  tmp_root="$(mktemp -d /tmp/asta-rescore-one.XXXXXX)"
  mkdir -p "${tmp_root}/${version}/${split}"
  cp -a "${submission_dir}" "${tmp_root}/${version}/${split}/"

  if scripts/rescore_submissions.sh \
    --submissions-root "${tmp_root}" \
    --output-root "${output_root}" \
    --scorer-project "${scorer_project}"; then
    scored_count=$((scored_count + 1))
  else
    failed_count=$((failed_count + 1))
    echo "== error: failed ${rel}; continuing" >&2
  fi

  rm -rf "${tmp_root}"
done < <(find_target_submissions)

if [ "${matched_count}" -eq 0 ]; then
  echo "warning: no ScholarQA/E2E submissions found under ${submissions_root}" >&2
  exit 0
fi

if [ "${dry_run}" -eq 1 ]; then
  echo "== done: matched ${matched_count} submissions" >&2
  exit 0
fi

echo "== done: rescored ${scored_count} submissions" >&2
if [ "${failed_count}" -gt 0 ]; then
  echo "== done: ${failed_count} submissions failed" >&2
  exit 1
fi
