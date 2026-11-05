#!/usr/bin/env bash
# ==============================================================================
# SCADose: Manuscript Build Script (PDF, DOCX, and Markdown)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MANUSCRIPT_DIR="${REPO_ROOT}/manuscript"
LATEX_DIR="${MANUSCRIPT_DIR}/latex"

echo "======================================================================"
echo " Building SCADose Publication Manuscript"
echo " Working directory: ${LATEX_DIR}"
echo "======================================================================"

cd "${LATEX_DIR}"

echo ">>> Pass 1: pdflatex..."
pdflatex -interaction=nonstopmode main.tex

echo ">>> Pass 2: bibtex..."
bibtex main

echo ">>> Pass 3: pdflatex (resolving citations)..."
pdflatex -interaction=nonstopmode main.tex

echo ">>> Pass 4: pdflatex (resolving references & floats)..."
pdflatex -interaction=nonstopmode main.tex

echo ">>> Compiling LaTeX Cover Letter..."
pdflatex -interaction=nonstopmode cover_letter.tex

echo ">>> Converting LaTeX to Word DOCX..."
PANDOC="${PANDOC:-pandoc}"
"${PANDOC}" main.tex -o "${MANUSCRIPT_DIR}/manuscript.docx" --citeproc --bibliography=references.bib

echo ">>> Converting LaTeX to Markdown MANUSCRIPT.md..."
"${PANDOC}" main.tex -o "${MANUSCRIPT_DIR}/MANUSCRIPT.md" --citeproc --bibliography=references.bib -t gfm+tex_math_dollars

echo ">>> Applying clean header and formatting to MANUSCRIPT.md..."
python3 "${SCRIPT_DIR}/format_manuscript_md.py"

# Sync build artifacts to TheLastRescue if available
RESCUE_DIR="${REPO_ROOT}/../TheLastRescue"
if [ -d "${RESCUE_DIR}" ]; then
    echo ">>> Syncing build artifacts to ${RESCUE_DIR}..."
    cp -f "${LATEX_DIR}/main.pdf" "${RESCUE_DIR}/manuscript_draft.pdf" 2>/dev/null || true
    cp -f "${LATEX_DIR}/main.pdf" "${RESCUE_DIR}/latex/main.pdf" 2>/dev/null || true
    cp -f "${LATEX_DIR}/cover_letter.pdf" "${RESCUE_DIR}/latex/cover_letter.pdf" 2>/dev/null || true
    cp -f "${MANUSCRIPT_DIR}/manuscript.docx" "${RESCUE_DIR}/manuscript.docx" 2>/dev/null || true
    cp -f "${MANUSCRIPT_DIR}/manuscript.docx" "${RESCUE_DIR}/manuscript_draft.docx" 2>/dev/null || true
fi

echo "======================================================================"
echo " Build Complete!"
echo " Generated PDF:          ${LATEX_DIR}/main.pdf"
echo " Generated Cover Letter: ${LATEX_DIR}/cover_letter.pdf"
echo " Generated DOCX:         ${MANUSCRIPT_DIR}/manuscript.docx"
echo " Generated MD:           ${MANUSCRIPT_DIR}/MANUSCRIPT.md"
echo "======================================================================"

