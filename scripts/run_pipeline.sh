#!/usr/bin/env bash
# ==============================================================================
# SCADose: Sex Chromosome Aneuploidy Dosage Analysis Pipeline Orchestrator
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "======================================================================"
echo " Starting SCADose Computational Genomics Pipeline Execution"
echo " Repository: ${REPO_ROOT}"
echo " Date: $(date)"
echo "======================================================================"

PYTHON="${PYTHON:-python}"

cd "${REPO_ROOT}"

echo ""
echo ">>> [1/7] Stage 01: Multi-cohort clinical data audit & standardization..."
"${PYTHON}" src/01_clinical_data_audit.py

echo ""
echo ">>> [2/7] Stage 02: Within-study moderated linear modeling (limma-trend)..."
"${PYTHON}" src/02_within_study_differential_expression.py

echo ""
echo ">>> [3/7] Stage 03: Random-effects meta-analysis & RNA-seq replication..."
"${PYTHON}" src/03_cross_study_meta_analysis.py

echo ""
echo ">>> [4/7] Stage 04: Calibrated systems biology, deconvolution, mediation & ML..."
"${PYTHON}" src/04_calibrated_systems_biology.py

echo ""
echo ">>> [5/7] Stage 05: Rendering publication-ready vector figures..."
"${PYTHON}" src/05_publication_figures.py

echo ""
echo ">>> [6/7] Stage 06: Reference audit against Crossref & PubMed APIs..."
"${PYTHON}" src/audit_references.py

echo ""
echo ">>> [7/7] Stage 07: Auto-generating manuscript macros..."
"${PYTHON}" src/generate_manuscript_macros.py

echo ""
echo "======================================================================"
echo " SCADose Pipeline Execution Completed Successfully!"
echo " Results written to: results/"
echo " Figures written to: figures/"
echo "======================================================================"

