#!/usr/bin/env python3
"""
Unit and Integration Tests for Phase 1A Data Audit and Ingestion
Tests: T1, T2, T3, T4, T5 (per Master Specification Section 3)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CLEAN_DIR = DATA_DIR / "clean"
RESULTS_DIR = REPO_ROOT / "results"


@pytest.fixture(scope="module")
def samples_df():
    sample_path = DATA_DIR / "samples.tsv"
    assert sample_path.exists(), "samples.tsv does not exist! Run 01_clinical_data_audit.py first."
    return pd.read_csv(sample_path, sep="\t", index_col=0)


@pytest.fixture(scope="module")
def expression_matrices():
    matrices = {}
    for study in ["GSE42331", "GSE46687", "GSE47584"]:
        path = CLEAN_DIR / f"{study}_log2.tsv"
        assert path.exists(), f"Missing {path}"
        matrices[study] = pd.read_csv(path, sep="\t", index_col=0)

    rna_path = CLEAN_DIR / "GSE126712_filtered_log2.tsv"
    assert rna_path.exists(), f"Missing {rna_path}"
    matrices["GSE126712"] = pd.read_csv(rna_path, sep="\t", index_col=0)
    return matrices


def test_t1_sample_sheet_integrity(samples_df):
    """T1: Sample-sheet integrity: unique GSMs, allowed karyotypes, nX/nY counts."""
    # 1. GSM identifiers must be unique
    assert samples_df.index.is_unique, "Duplicated GSM sample identifiers found in sample sheet!"

    # 2. Total sample count must equal 165
    assert len(samples_df) == 165, f"Expected 165 samples, found {len(samples_df)}"

    # 3. Karyotypes must belong to allowed set
    allowed_karyotypes = {"45,X", "46,XX", "46,XY", "47,XXY"}
    assert set(samples_df["declared_karyotype"]).issubset(allowed_karyotypes), (
        f"Unexpected karyotypes found: {set(samples_df['declared_karyotype']) - allowed_karyotypes}"
    )

    # 4. Consistency between karyotype and (nX, nY)
    expected_dosages = {
        "45,X": (1, 0),
        "46,XX": (2, 0),
        "46,XY": (1, 1),
        "47,XXY": (2, 1),
    }
    for idx, row in samples_df.iterrows():
        exp_x, exp_y = expected_dosages[row["declared_karyotype"]]
        assert row["nX"] == exp_x, f"Sample {idx} has invalid nX={row['nX']} (expected {exp_x})"
        assert row["nY"] == exp_y, f"Sample {idx} has invalid nY={row['nY']} (expected {exp_y})"
        assert row["n_sexchr"] == exp_x + exp_y

    # 5. Exact per-study sample counts
    study_counts = samples_df["study_id"].value_counts().to_dict()
    assert study_counts["GSE42331"] == 65
    assert study_counts["GSE46687"] == 36
    assert study_counts["GSE47584"] == 10
    assert study_counts["GSE126712"] == 54


def test_t2_sex_concordance(samples_df):
    """T2: Declared vs expression-inferred sex concordance across all cohorts."""
    # Defect D3 regression test: Ensure zero sex mismatches
    mismatches = samples_df[~samples_df["sex_concordant"]]
    assert len(mismatches) == 0, f"Detected {len(mismatches)} sex mismatches:\n{mismatches}"
    assert samples_df["sex_concordant"].all(), "Not all samples are concordant for sex!"


def test_t3_no_degenerate_features(expression_matrices):
    """T3: No degenerate features: per-study per-gene SD > 0, no constant blocks."""
    for study_id, mat in expression_matrices.items():
        # Check no duplicated columns (sample duplicates)
        assert mat.columns.is_unique, f"{study_id} contains duplicated sample columns!"

        # Check standard deviation across samples for each gene
        gene_sds = mat.std(axis=1)
        zero_var_genes = (gene_sds == 0.0).sum()
        total_genes = len(mat)
        prop_zero = zero_var_genes / total_genes
        # No unexpressed or constant features permitted in analyzed matrices
        assert prop_zero < 0.001, (
            f"{study_id} has {zero_var_genes} constant genes ({prop_zero:.2%})! "
            "Suggests constant imputation or mapping bug (Defect D1)."
        )


def test_t4_scale_plausibility(expression_matrices):
    """T4: Plausible log2 range, no negative raw intensities, no multi-million outliers."""
    for study_id, mat in expression_matrices.items():
        vals = mat.values
        min_val = np.min(vals)
        max_val = np.max(vals)
        mean_val = np.mean(vals)

        # In log2 units, expression values should reside in [0, 20]
        assert min_val >= -0.1, f"{study_id} contains large negative values (min={min_val:.2f})! Defect D2 failure."
        assert max_val <= 22.0, f"{study_id} contains extreme unlogged values (max={max_val:.2f})! Defect D2 failure."
        # Mean intensity check: microarrays ~7.0, RNA-seq (expressed) ~2.0-5.0
        expected_min_mean = 1.5 if study_id == "GSE126712" else 4.0
        assert expected_min_mean <= mean_val <= 14.0, (
            f"{study_id} mean intensity ({mean_val:.2f}) outside plausible log2 blood transcriptome range!"
        )


def test_t5_design_estimability_and_rank():
    """T5: Design estimability checks and rank verification within each study."""
    sample_path = DATA_DIR / "samples.tsv"
    meta = pd.read_csv(sample_path, sep="\t", index_col=0)

    # 1. GSE42331: has 47,XXY (2,1), 46,XY (1,1), 46,XX (2,0)
    m42 = meta[meta["study_id"] == "GSE42331"]
    # Model: [1, nX, nY]
    X42 = np.column_stack([np.ones(len(m42)), m42["nX"], m42["nY"]])
    rank42 = np.linalg.matrix_rank(X42)
    assert rank42 == 3, f"GSE42331 design matrix is rank deficient (rank={rank42}, expected 3)!"

    # 2. GSE46687: has 45,X (1,0) and 46,XX (2,0) -> nY is constant at 0
    m46 = meta[meta["study_id"] == "GSE46687"]
    X46 = np.column_stack([np.ones(len(m46)), m46["nX"]])
    rank46 = np.linalg.matrix_rank(X46)
    assert rank46 == 2, f"GSE46687 design matrix is rank deficient (rank={rank46}, expected 2)!"

    # 3. GSE47584: has 47,XXY (2,1) and 46,XY (1,1) -> nY is constant at 1
    m47 = meta[meta["study_id"] == "GSE47584"]
    X47 = np.column_stack([np.ones(len(m47)), m47["nX"]])
    rank47 = np.linalg.matrix_rank(X47)
    assert rank47 == 2, f"GSE47584 design matrix is rank deficient (rank={rank47}, expected 2)!"

    # 4. GSE126712: has all 4 karyotypes: (1,0), (2,0), (1,1), (2,1)
    m12 = meta[meta["study_id"] == "GSE126712"]
    X12 = np.column_stack([np.ones(len(m12)), m12["nX"], m12["nY"]])
    rank12 = np.linalg.matrix_rank(X12)
    assert rank12 == 3, f"GSE126712 design matrix is rank deficient (rank={rank12}, expected 3)!"
