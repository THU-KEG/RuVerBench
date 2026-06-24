#!/usr/bin/env bash
set -euo pipefail

# Generate AgenticCoding judge prediction files.
# Run from the RuVerBench package root or set RUVERBENCH_ROOT explicitly.

usage() {
  cat <<'EOF'
Usage:
  bash code/run_judges/run_agenticcoding_judge.sh [options via env]

Generates AgenticCoding judge prediction files by calling an OpenAI-compatible judge API.

Required environment:
  JUDGE_MODEL       Judge model name/marker.
  JUDGE_BASE_URL    OpenAI-compatible base URL, e.g. http://127.0.0.1:8000/v1.
  JUDGE_API_KEY     API key. Default: OPENAI_API_KEY or EMPTY.

Optional environment:
  TRAJECTORIES       Default: data/benchmark/agenticcoding_trajectories.jsonl
  DATA_FILE          Default: data/benchmark/agenticcoding_dataset.jsonl
  OUTPUT_FILE        Default: outputs/generated_predictions/agenticcoding/<JUDGE_MODEL>.json
  MAX_WORKERS        Default: 4
  BATCH_SIZE         Default: 1
  PROMPT_STYLE       standard | strict | semantic | evidence_first. Default: standard
  MAX_PROMPT_CHARS   Default: 0
  LIMIT              Smoke-test limit. Default: 0 (all)
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
TRAJECTORIES="${TRAJECTORIES:-${ROOT}/data/benchmark/agenticcoding_trajectories.jsonl}"
DATA_FILE="${DATA_FILE:-${ROOT}/data/benchmark/agenticcoding_dataset.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-${ROOT}/outputs/generated_predictions/agenticcoding/${JUDGE_MODEL}.json}"
MAX_WORKERS="${MAX_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
PROMPT_STYLE="${PROMPT_STYLE:-standard}"
MAX_PROMPT_CHARS="${MAX_PROMPT_CHARS:-0}"
LIMIT="${LIMIT:-0}"

if [[ -z "${JUDGE_MODEL}" ]]; then
  echo "[ERROR] Set JUDGE_MODEL or MODEL." >&2
  exit 1
fi

if [[ -z "${JUDGE_BASE_URL}" ]]; then
  echo "[ERROR] Set JUDGE_BASE_URL to an OpenAI-compatible /v1 endpoint." >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_FILE}")"
cd "${ROOT}/code/run_judges/agenticcoding"

CMD=(python3 -u eval.py
  --trajectories "${TRAJECTORIES}"
  --data "${DATA_FILE}"
  --output "${OUTPUT_FILE}"
  --model "${JUDGE_MODEL}"
  --limit "${LIMIT}"
  --workers "${MAX_WORKERS}"
  --batch-size "${BATCH_SIZE}"
  --prompt-style "${PROMPT_STYLE}"
  --max-prompt-chars "${MAX_PROMPT_CHARS}")

export DETECTOR_API_BACKEND="openai"
CMD+=(--judge-base-url "${JUDGE_BASE_URL}" --judge-api-key "${JUDGE_API_KEY}")

"${CMD[@]}"
