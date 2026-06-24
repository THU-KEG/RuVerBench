#!/usr/bin/env bash
set -euo pipefail

# Package-local AgenticCoding judge runner. Kept for users who expect a run.sh
# next to eval.py; delegates to the top-level RuVerBench wrapper.

ROOT="${RUVERBENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
exec bash "${ROOT}/code/run_judges/run_agenticcoding_judge.sh" "$@"
