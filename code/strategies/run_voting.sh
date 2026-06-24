#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash code/strategies/run_voting.sh --domain DOMAIN --voter-files A B C --output PATH [options]

Aggregate three existing judge prediction files by majority vote using the
fixed strategy subset. To generate fresh voter files first, run
code/run_judges/run_deepresearch_judge.sh or code/run_judges/run_agenticcoding_judge.sh multiple times with stochastic sampling.

Options:
  --domain DOMAIN           deepresearch | agenticcoding. Required.
  --voter-files A B C       Three voter prediction files. Required.
  --voter-labels A B C      Voter labels. Default: voter1 voter2 voter3
  --output PATH             Aggregated prediction output. Required.
  --gold PATH               Override gold/structure file used for aggregation.
                            Defaults to data/strategy_fixed20_subset/deepresearch_subset_labels.json for DR
                            and data/strategy_fixed20_subset/agenticcoding_subset_labels.json for AC.
  --help, -h                Show help.

Outputs:
  Aggregated majority-vote prediction JSON at --output.
EOF
}

DOMAIN=""
OUTPUT=""
GOLD=""
VOTER_FILES=()
VOTER_LABELS=("voter1" "voter2" "voter3")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --domain) DOMAIN="$2"; shift 2 ;;
    --voter-files) VOTER_FILES=("$2" "$3" "$4"); shift 4 ;;
    --voter-labels) VOTER_LABELS=("$2" "$3" "$4"); shift 4 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --gold) GOLD="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${DOMAIN}" || -z "${OUTPUT}" || ${#VOTER_FILES[@]} -ne 3 ]]; then
  echo "[ERROR] --domain, --voter-files A B C, and --output are required" >&2
  usage
  exit 1
fi
case "${DOMAIN}" in deepresearch|agenticcoding) ;; *) echo "[ERROR] --domain must be deepresearch or agenticcoding" >&2; exit 1 ;; esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
if [[ -z "${GOLD}" ]]; then
  if [[ "${DOMAIN}" == "deepresearch" ]]; then
    GOLD="${ROOT}/data/strategy_fixed20_subset/deepresearch_subset_labels.json"
  else
    GOLD="${ROOT}/data/strategy_fixed20_subset/agenticcoding_subset_labels.json"
  fi
fi

python3 code/strategies/aggregate_voting_outputs.py \
  --domain "${DOMAIN}" \
  --gold "${GOLD}" \
  --voter-files "${VOTER_FILES[0]}" "${VOTER_FILES[1]}" "${VOTER_FILES[2]}" \
  --voter-labels "${VOTER_LABELS[0]}" "${VOTER_LABELS[1]}" "${VOTER_LABELS[2]}" \
  --output "${OUTPUT}"

printf '[SUCCESS] Majority-vote output written to %s\n' "${OUTPUT}"
