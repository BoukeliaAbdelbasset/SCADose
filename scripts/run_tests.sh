#!/usr/bin/env bash
# ==============================================================================
# SCADose: Automated Pytest Verification Suite
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "======================================================================"
echo " Running SCADose Automated Test Verification Suite (T1-T15)"
echo " Repository: ${REPO_ROOT}"
echo "======================================================================"

PYTEST="${PYTEST:-pytest}"

cd "${REPO_ROOT}"

"${PYTEST}" tests/test_phase1a_audit.py tests/test_phase1b_1c_pipeline.py -v

echo "======================================================================"
echo " All Unit and Mathematical Assertions Passed!"
echo "======================================================================"

