#!/usr/bin/env bash
set -euo pipefail

# Generate DeepResearch judge prediction files.
# Run from the RuVerBench package root or set RUVERBENCH_ROOT explicitly.

usage() {
  cat <<'EOF'
Usage:
  bash code/run_judges/run_deepresearch_judge.sh [options via env]

Generates DeepResearch judge prediction files by calling an OpenAI-compatible judge API.

Required environment:
  JUDGE_MODEL       Judge model name/marker.
  JUDGE_BASE_URL    OpenAI-compatible base URL, e.g. http://127.0.0.1:8000/v1.
  JUDGE_API_KEY     API key. Default: OPENAI_API_KEY or EMPTY.

Optional environment:
  MODEL_RESPONSE_FILE  Default: data/benchmark/deepresearch_responses.json
  RUBRICS_FILE         Default: data/benchmark/deepresearch_dataset.json
  OUTPUT_DIR           Default: outputs/generated_predictions/deepresearch
  MAX_WORKERS          Default: 4
  BATCH_SIZE           Default: 1
  PROMPT_STYLE         standard | strict | semantic | evidence_first. Default: standard
  LIMIT                Smoke-test limit. Default: 0 (all)
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

ROOT="${RUVERBENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
JUDGE_MODEL="${JUDGE_MODEL:-${MODEL:-}}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-}"
JUDGE_API_KEY="${JUDGE_API_KEY:-${OPENAI_API_KEY:-EMPTY}}"
MODEL_RESPONSE_FILE="${MODEL_RESPONSE_FILE:-${ROOT}/data/benchmark/deepresearch_responses.json}"
RUBRICS_FILE="${RUBRICS_FILE:-${ROOT}/data/benchmark/deepresearch_dataset.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/generated_predictions/deepresearch}"
MAX_WORKERS="${MAX_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
PROMPT_STYLE="${PROMPT_STYLE:-standard}"
LIMIT="${LIMIT:-0}"

if [[ -z "${JUDGE_MODEL}" ]]; then
  echo "[ERROR] Set JUDGE_MODEL or MODEL." >&2
  exit 1
fi

if [[ -z "${JUDGE_BASE_URL}" ]]; then
  echo "[ERROR] Set JUDGE_BASE_URL to an OpenAI-compatible /v1 endpoint." >&2
  exit 1
fi

cd "${ROOT}/code/run_judges/deepresearch"

python3 -u code/rubric_eval/main.py \
  --model_file "${MODEL_RESPONSE_FILE}" \
  --rubrics_file "${RUBRICS_FILE}" \
  --result_dir "${OUTPUT_DIR}" \
  --judge_model "${JUDGE_MODEL}" \
  --judge_base_url "${JUDGE_BASE_URL}" \
  --judge_api_key "${JUDGE_API_KEY}" \
  --max_workers "${MAX_WORKERS}" \
  --batch_size "${BATCH_SIZE}" \
  --prompt_style "${PROMPT_STYLE}" \
  --limit "${LIMIT}"
