#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${RELEASE_ROOT}/.venv/bin/python}"

cd "${SCRIPT_DIR}"
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_ours_015_accuracy.py" "$@"
