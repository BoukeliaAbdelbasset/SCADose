# ==============================================================================
# SCADose: Sex Chromosome Aneuploidy Dosage Analysis Framework Makefile
# ==============================================================================
SHELL := /bin/bash
PYTHON := /home/abdelbasset/anaconda3/envs/safa/bin/python
PYTEST := /home/abdelbasset/anaconda3/envs/safa/bin/pytest
PANDOC := /home/abdelbasset/anaconda3/bin/pandoc

export PYTHON
export PYTEST
export PANDOC

.PHONY: all pipeline test manuscript clean help

help:
	@echo "SCADose Master Build Targets:"
	@echo "  make all         - Run complete pipeline, execute test suite, build manuscript (PDF & DOCX)"
	@echo "  make pipeline    - Run end-to-end computational pipeline (Stages 01-07 via scripts/run_pipeline.sh)"
	@echo "  make test        - Run automated pytest verification suite (scripts/run_tests.sh)"
	@echo "  make manuscript  - Compile LaTeX manuscript (main.pdf), Word docx, and Markdown (scripts/build_manuscript.sh)"
	@echo "  make clean       - Remove intermediate LaTeX compilation artifacts"

all: pipeline test manuscript

pipeline:
	@echo "=== [SCADose] Executing Computational Pipeline ==="
	bash scripts/run_pipeline.sh

test:
	@echo "=== [SCADose] Executing Pytest Verification Suite ==="
	bash scripts/run_tests.sh

manuscript:
	@echo "=== [SCADose] Building Publication Manuscript ==="
	bash scripts/build_manuscript.sh

clean:
	rm -f manuscript/latex/*.aux manuscript/latex/*.bbl manuscript/latex/*.blg manuscript/latex/*.log manuscript/latex/*.out manuscript/latex/*.toc

