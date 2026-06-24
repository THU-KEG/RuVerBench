#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash code/strategies/run_batch_size_experiment.sh --judge-model NAME --judge-base-url URL [options]

Generate fresh judge outputs for batch-size experiments using the packaged
RuVerBench judge wrappers and the fixed strategy subset.

Options:
  --judge-model NAME        Judge model name/marker. Required.
  --judge-base-url URL      OpenAI-compatible base URL. Required.
  --judge-api-key KEY       API key. Default: EMPTY
  --batch-sizes LIST        Comma-separated batch sizes. Default: 1,2,4
  --scenario NAME           both | deepresearch | agenticcoding. Default: both
  --limit N                 Limit examples. Default: 0
  --max-workers N           Parallel workers. Default: 4
  --prompt-style NAME       Prompt style. Default: standard
  --help, -h                Show help.

Outputs:
  outputs/generated_predictions/batch_size/deepresearch/batch_<N>/
  outputs/generated_predictions/batch_size/agenticcoding/batch_<N>/<model>.json
EOF
}

JUDGE_MODEL=""
JUDGE_BASE_URL=""
JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"
BATCH_SIZES="1,2,4"
SCENARIO="both"
LIMIT_VALUE="0"
MAX_WORKERS="4"
PROMPT_STYLE="standard"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --judge-model) JUDGE_MODEL="$2"; shift 2 ;;
    --judge-base-url) JUDGE_BASE_URL="$2"; shift 2 ;;
    --judge-api-key) JUDGE_API_KEY="$2"; shift 2 ;;
    --batch-sizes) BATCH_SIZES="$2"; shift 2 ;;
    --scenario) SCENARIO="$2"; shift 2 ;;
    --limit) LIMIT_VALUE="$2"; shift 2 ;;
    --max-workers) MAX_WORKERS="$2"; shift 2 ;;
    --prompt-style) PROMPT_STYLE="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${JUDGE_MODEL}" || -z "${JUDGE_BASE_URL}" ]]; then
  echo "[ERROR] --judge-model and --judge-base-url are required" >&2
  usage
  exit 1
fi
case "${SCENARIO}" in both|deepresearch|agenticcoding) ;; *) echo "[ERROR] bad --scenario" >&2; exit 1 ;; esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
IFS=',' read -r -a BATCH_ARRAY <<< "${BATCH_SIZES}"

for batch_size in "${BATCH_ARRAY[@]}"; do
  echo "[INFO] batch_size=${batch_size}"
  if [[ "${SCENARIO}" == "both" || "${SCENARIO}" == "deepresearch" ]]; then
    JUDGE_MODEL="${JUDGE_MODEL}" JUDGE_BASE_URL="${JUDGE_BASE_URL}" JUDGE_API_KEY="${JUDGE_API_KEY}" \
    MODEL_RESPONSE_FILE="${ROOT}/data/benchmark/deepresearch_responses.json" \
    RUBRICS_FILE="${ROOT}/data/strategy_fixed20_subset/deepresearch_subset_dataset.json" \
    LIMIT="${LIMIT_VALUE}" MAX_WORKERS="${MAX_WORKERS}" BATCH_SIZE="${batch_size}" PROMPT_STYLE="${PROMPT_STYLE}" \
    OUTPUT_DIR="${ROOT}/outputs/generated_predictions/batch_size/deepresearch/batch_${batch_size}" \
      bash code/run_judges/run_deepresearch_judge.sh
  fi
  if [[ "${SCENARIO}" == "both" || "${SCENARIO}" == "agenticcoding" ]]; then
    JUDGE_MODEL="${JUDGE_MODEL}" JUDGE_BASE_URL="${JUDGE_BASE_URL}" JUDGE_API_KEY="${JUDGE_API_KEY}" \
    TRAJECTORIES="${ROOT}/data/benchmark/agenticcoding_trajectories.jsonl" \
    DATA_FILE="${ROOT}/data/strategy_fixed20_subset/agenticcoding_subset_dataset.jsonl" \
    LIMIT="${LIMIT_VALUE}" MAX_WORKERS="${MAX_WORKERS}" BATCH_SIZE="${batch_size}" PROMPT_STYLE="${PROMPT_STYLE}" \
    OUTPUT_FILE="${ROOT}/outputs/generated_predictions/batch_size/agenticcoding/batch_${batch_size}/${JUDGE_MODEL}.json" \
      bash code/run_judges/run_agenticcoding_judge.sh
  fi
done

printf '[SUCCESS] Batch-size generation completed.\n'
