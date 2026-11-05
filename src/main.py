#!/usr/bin/env python3
"""
SCADose - Master Pipeline Orchestrator (Rebuilt Q1 Publication Pipeline)
Module: src.main

Description:
    Executes the rigorous end-to-end computational genomics workflow for Sex Chromosome Aneuploidies:
      1. Stage 01: Multi-cohort clinical data audit, scaling standardization, and sex concordance.
      2. Stage 02: Pre-specified within-study moderated linear modeling (limma-trend).
      3. Stage 03: Cross-study random-effects meta-analysis and independent RNA-Seq replication.
      4. Stage 04: Calibrated systems biology (quanTIseq/ABIS deconvolution, bootstrap mediation,
                   decoupleR TF activity, signed WGCNA, nested ML, and null FDR simulation).
      5. Stage 05: Publication-ready vector figures (Figs 1-12 and Fig S1).
      6. Stage 06: Citation and reference verification against Crossref/PubMed APIs.
      7. Stage 07: Manuscript macro generation directly binding computational numbers into LaTeX.

Usage:
    cd SCADose
    python -m src.main
"""

import logging
import subprocess
import sys
import time
from pathlib import Path

# Paths
SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [SCADose_Orchestrator]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("SCADose_Orchestrator")

PIPELINE_STAGES = [
    ("Stage 01: Clinical Data Audit & Standardization", SRC_DIR / "01_clinical_data_audit.py"),
    ("Stage 02: Within-Study Differential Expression", SRC_DIR / "02_within_study_differential_expression.py"),
    ("Stage 03: Cross-Study Meta-Analysis & RNA Replication", SRC_DIR / "03_cross_study_meta_analysis.py"),
    ("Stage 04: Calibrated Systems Biology & Simulation", SRC_DIR / "04_calibrated_systems_biology.py"),
    ("Stage 05: Publication Figure Generation", SRC_DIR / "05_publication_figures.py"),
    ("Stage 06: Reference & Citation Crossref Audit", SRC_DIR / "audit_references.py"),
    ("Stage 07: Auto-generated Manuscript Macros", SRC_DIR / "generate_manuscript_macros.py"),
]


def run_stage(idx: int, total_stages: int, name: str, script_path: Path) -> bool:
    """Executes a single pipeline layer in a distinct subprocess."""
    logger.info("=" * 80)
    logger.info("RUNNING STAGE %d/%d: %s", idx, total_stages, name)
    logger.info("Script: %s", script_path.name)
    logger.info("=" * 80)

    start_time = time.time()
    cmd = [sys.executable, str(script_path)]

    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=False)
    elapsed = time.time() - start_time

    if result.returncode != 0:
        logger.error("Stage %d (%s) failed with exit code %d.", idx, name, result.returncode)
        return False

    logger.info("STAGE %d COMPLETED SUCCESSFULLY in %.2f seconds.", idx, elapsed)
    return True


def main():
    """Main orchestrator function."""
    logger.info("=" * 80)
    logger.info("STARTING THELASTRESCUE RIGOROUS PUBLICATION PIPELINE")
    logger.info("Repository Root: %s", REPO_ROOT)
    logger.info("Python Runtime:  %s", sys.executable)
    logger.info("=" * 80)

    overall_start = time.time()
    total_stages = len(PIPELINE_STAGES)

    for idx, (name, path) in enumerate(PIPELINE_STAGES, start=1):
        success = run_stage(idx, total_stages, name, path)
        if not success:
            logger.critical("PIPELINE HALTED DUE TO FAILURE IN STAGE %d (%s).", idx, name)
            sys.exit(1)

    total_time = time.time() - overall_start
    logger.info("=" * 80)
    logger.info("THELASTRESCUE PIPELINE COMPLETED SUCCESSFULLY IN %.2f SECONDS.", total_time)
    logger.info("All audit logs, matrices, meta-analyses, models, figures, and macros are verified.")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
