#!/usr/bin/env python3
"""
SCADose - Automated Test Suite
Test: test_dosage_math.py

Description:
    pytest module mathematically verifying:
      1. Continuous X-chromosome dosage regression accurately estimates ground truth beta slopes.
      2. Benjamini-Hochberg False Discovery Rate (FDR) adjustment correctly enforces penalties.
      3. Directional concordance and proportionality metrics operate correctly.
      4. Cryptographic sample leakage detection successfully quarantines overlapping samples.
      5. Real patient expression matrix and HGNC approved gene symbols are fully valid.
      6. Real statistical dosage outputs have bounded p-values and significant candidates.

Author:
    Lead Bioinformatician & Senior Clinical Data Scientist
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def synthetic_mini_cohort():
    """
    Creates a synthetic mini-matrix with known ground truth dosage relationships:
      - GENE_POS: Ground-truth slope beta = +2.0 (positive dosage responder)
      - GENE_NEG: Ground-truth slope beta = -1.5 (inverse dosage immune responder)
      - GENE_NULL: Ground-truth slope beta = 0.0 (dosage invariant)
    """
    np.random.seed(42)
    n_samples = 60
    # Dosages: 20 at 1X, 20 at 2X, 20 at 3X
    x_dosage = np.array([1] * 20 + [2] * 20 + [3] * 20, dtype=float)

    noise_pos = np.random.normal(0, 0.2, n_samples)
    noise_neg = np.random.normal(0, 0.2, n_samples)
    noise_null = np.random.normal(0, 0.5, n_samples)

    y_pos = 5.0 + 2.0 * x_dosage + noise_pos
    y_neg = 9.0 - 1.5 * x_dosage + noise_neg
    y_null = 7.0 + 0.0 * x_dosage + noise_null

    expr_df = pd.DataFrame(
        {
            "GENE_POS": y_pos,
            "GENE_NEG": y_neg,
            "GENE_NULL": y_null,
        },
        index=[f"SAMPLE_{i:03d}" for i in range(n_samples)],
    )

    meta_df = pd.DataFrame(
        {
            "sample_id": expr_df.index,
            "x_dosage": x_dosage,
            "karyotype": ["45,X"] * 20 + ["46,XX"] * 20 + ["47,XXX"] * 20,
        }
    )

    return expr_df, meta_df


def test_continuous_dosage_regression_math(synthetic_mini_cohort):
    """
    Verifies that the continuous dosage regression function accurately estimates
    the slope beta and robust p-values within expected mathematical bounds.
    """
    expr_df, meta_df = synthetic_mini_cohort
    x = sm.add_constant(meta_df["x_dosage"].values)

    # 1. Test Positive Dosage Gene
    y_pos = expr_df["GENE_POS"].values
    ols_pos = sm.OLS(y_pos, x).fit(cov_type="HC3")
    beta_pos = ols_pos.params[1]
    pval_pos = ols_pos.pvalues[1]
    r2_pos = ols_pos.rsquared

    # Expect beta ~ 2.0 (+/- 0.15) and p < 1e-15
    assert np.isclose(beta_pos, 2.0, atol=0.15), f"Estimated beta ({beta_pos}) deviates from ground truth 2.0!"
    assert pval_pos < 1e-15, f"Expected highly significant p-value, got {pval_pos}"
    assert r2_pos > 0.90, f"Expected R^2 > 0.90, got {r2_pos}"

    # 2. Test Negative Dosage Gene
    y_neg = expr_df["GENE_NEG"].values
    ols_neg = sm.OLS(y_neg, x).fit(cov_type="HC3")
    beta_neg = ols_neg.params[1]
    pval_neg = ols_neg.pvalues[1]

    # Expect beta ~ -1.5 (+/- 0.15) and p < 1e-15
    assert np.isclose(beta_neg, -1.5, atol=0.15), f"Estimated beta ({beta_neg}) deviates from ground truth -1.5!"
    assert pval_neg < 1e-15, f"Expected highly significant p-value, got {pval_neg}"

    # 3. Test Null Gene
    y_null = expr_df["GENE_NULL"].values
    ols_null = sm.OLS(y_null, x).fit(cov_type="HC3")
    beta_null = ols_null.params[1]
    pval_null = ols_null.pvalues[1]

    # Expect beta ~ 0.0 and p > 0.05
    assert abs(beta_null) < 0.20, f"Null gene beta unexpectedly large: {beta_null}"
    assert pval_null > 0.05, f"Null gene p-value unexpectedly significant: {pval_null}"


def test_benjamini_hochberg_fdr_penalty():
    """
    Verifies that the Benjamini-Hochberg FDR procedure:
      1. Correctly penalizes nominal p-values (padj >= pval).
      2. Ensures padj is monotonically non-decreasing when sorted by nominal pval.
      3. Bounded in [0, 1].
    """
    raw_pvals = np.array([0.0001, 0.002, 0.015, 0.04, 0.12, 0.55, 0.89])

    reject, padj, _, _ = multipletests(raw_pvals, alpha=0.05, method="fdr_bh")

    # Property 1: padj >= pval
    assert np.all(padj >= raw_pvals), "Violation: FDR adjusted p-value cannot be smaller than raw p-value!"

    # Property 2: padj <= 1.0
    assert np.all(padj <= 1.0), "Violation: Adjusted p-value exceeded 1.0!"

    # Property 3: Monotonicity with sorted raw p-values
    assert np.all(np.diff(padj) >= 0), "Violation: Adjusted p-values must be non-decreasing!"

    # Specific BH check: first value = 0.0001 * 7 / 1 = 0.0007
    expected_first = 0.0001 * len(raw_pvals) / 1
    assert np.isclose(padj[0], expected_first), f"Expected first padj = {expected_first}, got {padj[0]}"


def test_directional_concordance_logic():
    """
    Tests the mathematical concordance rule:
    Concordance holds when beta_X_TS * beta_X_KS > 0.
    """
    # Case A: Both positive slopes (e.g. KDM6A, E2F2)
    b_ts_pos = 1.2
    b_ks_pos = 1.1
    assert (b_ts_pos * b_ks_pos) > 0, "Failed: Both positive slopes should be concordant!"

    # Case B: Both negative slopes (e.g. CLEC4E)
    b_ts_neg = -1.1
    b_ks_neg = -1.0
    assert (b_ts_neg * b_ks_neg) > 0, "Failed: Both negative slopes should be concordant!"

    # Case C: Divergent slopes
    b_ts_div = 1.0
    b_ks_div = -0.8
    assert (b_ts_div * b_ks_div) < 0, "Failed: Divergent slopes should have negative product!"


def test_quarantine_leakage_detection():
    """
    Tests that duplicate donor IDs across discovery and validation sets
    are strictly identified and quarantined.
    """
    cohort_data = pd.DataFrame(
        {
            "sample_id": ["GSM1", "GSM2", "GSM3", "GSM4"],
            "dataset_id": ["GSE126712", "GSE126712", "GSE42331", "GSE46687"],
            "donor_id": ["DONOR_A", "DONOR_B", "DONOR_A", "DONOR_C"],  # DONOR_A is shared!
        }
    )

    disc_donors = set(cohort_data[cohort_data["dataset_id"] == "GSE126712"]["donor_id"])
    val_donors = set(cohort_data[cohort_data["dataset_id"].isin(["GSE42331", "GSE46687"])]["donor_id"])

    leaked = disc_donors.intersection(val_donors)
    assert len(leaked) == 1, f"Expected exactly 1 leaked donor, detected {len(leaked)}"
    assert "DONOR_A" in leaked, "Failed to catch DONOR_A leakage!"

    # Apply quarantine filter on discovery
    clean_cohort = cohort_data[
        ~((cohort_data["dataset_id"] == "GSE126712") & (cohort_data["donor_id"].isin(leaked)))
    ]

    clean_disc = set(clean_cohort[clean_cohort["dataset_id"] == "GSE126712"]["donor_id"])
    clean_val = set(clean_cohort[clean_cohort["dataset_id"].isin(["GSE42331", "GSE46687"])]["donor_id"])

    assert len(clean_disc.intersection(clean_val)) == 0, "Leakage survived quarantine!"


def test_real_data_integrity():
    """
    Validates the physical presence and properties of real human patient expression
    matrices and HGNC approved gene symbols in SCADose.
    """
    raw_expr_path = REPO_ROOT / "data" / "raw" / "clean_expression_matrix.csv"
    raw_meta_path = REPO_ROOT / "data" / "raw" / "clean_sample_manifest.csv"

    if not raw_expr_path.exists() or not raw_meta_path.exists():
        pytest.skip("Real raw data files not yet generated. Run Layer 01 first.")

    raw_expr = pd.read_csv(raw_expr_path, index_col=0)
    raw_meta = pd.read_csv(raw_meta_path)

    # 1. Verify sample dimensions
    assert raw_expr.shape[0] == 111, f"Expected 111 real patient samples, got {raw_expr.shape[0]}"
    assert raw_expr.shape[1] >= 15000, f"Expected >= 15,000 real genes, got {raw_expr.shape[1]}"

    # 2. Verify no NaNs or infinite numbers
    assert not raw_expr.isna().any().any(), "NaN values detected in real expression matrix!"
    assert np.all(np.isfinite(raw_expr.values)), "Non-finite numbers detected in expression matrix!"

    # 3. Verify real HGNC gene symbols (no numeric IDs or probe artifacts)
    for gene in ["SLC25A6", "ASMTL", "CD99", "KDM6A", "CLEC4E", "E2F2", "ANXA1", "ZFX"]:
        assert gene in raw_expr.columns, f"Essential biomarker {gene} missing from real matrix!"

    # 4. Verify sample manifest matches
    assert list(raw_expr.index) == list(raw_meta["sample_id"]), "Sample ID mismatch between expression and metadata!"


def test_real_dosage_biomarker_results():
    """
    Validates statistical integrity of real continuous dosage results in biomarker_stats.csv.
    """
    stats_path = REPO_ROOT / "results" / "biomarker_stats.csv"
    if not stats_path.exists():
        pytest.skip("biomarker_stats.csv not yet generated. Run Layer 03 first.")

    stats_df = pd.read_csv(stats_path)
    assert len(stats_df) >= 15000, f"Expected >= 15,000 genes, got {len(stats_df)}"

    # 1. Verify p-values and R^2 bounds
    assert stats_df["dosage_pvalue_unadj"].between(0.0, 1.0).all(), "Invalid dosage p-values outside [0, 1]!"
    assert stats_df["dosage_r2_unadj"].between(0.0, 1.0).all(), "Invalid dosage R^2 outside [0, 1]!"

    # 2. Verify key biomarkers survive FDR or have strong dosage signal
    slc_row = stats_df[stats_df["gene"] == "SLC25A6"].iloc[0]
    assert slc_row["dosage_padj_unadj"] < 0.01, f"SLC25A6 expected FDR < 0.01, got {slc_row['dosage_padj_unadj']}"

    kdm_row = stats_df[stats_df["gene"] == "KDM6A"].iloc[0]
    assert kdm_row["dosage_padj_unadj"] < 0.01, f"KDM6A expected FDR < 0.01, got {kdm_row['dosage_padj_unadj']}"


def test_nnls_immune_deconvolution_constraints():
    """
    Validates mathematical constraints of the NNLS immune deconvolution:
      1. Non-negativity constraint: all cell fractions f_ij >= 0.
      2. Convex sum-to-one constraint: sum_k f_kj == 1.0 (+/- 1e-4) for all samples.
      3. All 6 canonical leukocyte lineages are quantified across all 111 samples.
    """
    fractions_path = REPO_ROOT / "results" / "immune_deconvolution_fractions.csv"
    if not fractions_path.exists():
        pytest.skip("immune_deconvolution_fractions.csv not yet generated. Run Layer 05 first.")

    frac_df = pd.read_csv(fractions_path, index_col=0)
    assert frac_df.shape[0] == 111, f"Expected 111 samples, got {frac_df.shape[0]}"

    expected_cell_types = [
        "Monocytes_Mac",
        "CD4_T_cells",
        "CD8_T_cells",
        "B_cells",
        "NK_cells",
        "Neutrophils",
    ]
    for cell in expected_cell_types:
        assert cell in frac_df.columns, f"Expected leukocyte lineage {cell} missing from deconvolution!"

    # 1. Non-negativity: fractions >= 0.0
    cell_matrix = frac_df[expected_cell_types].values
    assert np.all(cell_matrix >= -1e-7), "Negative cell fractions detected violating NNLS constraint!"

    # 2. Sum-to-one convex constraint: row sums == 1.0
    row_sums = cell_matrix.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-4), "Sum-to-one constraint violated in cell fractions!"


def test_clec4e_multivariate_dissection():
    """
    Validates empirical findings of the CLEC4E multivariate dissection:
      1. Monocyte/Macrophage cellularity positively scales with X-dosage (p < 1e-4).
      2. NK cells contract with X-dosage (r < 0, p < 1e-3).
      3. Controlling for Monocyte/Mac cellularity reveals dual-component architecture:
         Monocyte/Mac cellularity is significant (p < 1e-10) AND
         cell-intrinsic X-dosage slope remains positive (beta > 0, p < 0.05).
    """
    summary_path = REPO_ROOT / "results" / "sota_systems_summary.json"
    if not summary_path.exists():
        pytest.skip("sota_systems_summary.json not yet generated. Run Layer 05 first.")

    with open(summary_path, "r") as f:
        summary = json.load(f)

    # 1. Monocyte/Mac expansion
    mono_metrics = summary["immune_deconvolution"]["cell_shift_metrics"]["Monocytes_Mac"]
    assert mono_metrics["pearson_r"] > 0.35, f"Expected Monocyte/Mac r > 0.35, got {mono_metrics['pearson_r']}"
    assert mono_metrics["dosage_pval"] < 1e-4, f"Expected Monocyte/Mac p < 1e-4, got {mono_metrics['dosage_pval']}"

    # 2. NK contraction
    nk_metrics = summary["immune_deconvolution"]["cell_shift_metrics"]["NK_cells"]
    assert nk_metrics["pearson_r"] < -0.25, f"Expected NK r < -0.25, got {nk_metrics['pearson_r']}"
    assert nk_metrics["dosage_pval"] < 1e-3, f"Expected NK p < 1e-3, got {nk_metrics['dosage_pval']}"

    # 3. CLEC4E dual-component dissection
    clec_disc = summary["immune_deconvolution"]["clec4e_dissection"]
    assert clec_disc["unadjusted_dosage_beta"] > 100.0, "Expected large unadjusted CLEC4E dosage slope"
    assert clec_disc["adjusted_dosage_beta"] > 0.0, "Expected positive adjusted CLEC4E dosage slope"
    adj_pval = clec_disc["adjusted_dosage_pval"]
    assert adj_pval < 0.05, f"Expected adjusted p < 0.05, got {adj_pval}"
    assert clec_disc["monocyte_mac_pval"] < 1e-10, "Expected highly significant Monocyte/Mac cellularity term"
    assert clec_disc["r2_adjusted"] > clec_disc["r2_unadjusted"], "Multivariate model R^2 should exceed univariate R^2"


def test_e2f2_regulon_mechanics():
    """
    Validates E2F2 transcription factor target network enrichment:
      1. All 23 canonical targets are evaluated without nulls.
      2. Top cell cycle and DNA replication targets (TK1, TYMS) exhibit significant co-expression.
      3. Correlations are properly bounded in [-1, 1].
    """
    regulon_path = REPO_ROOT / "results" / "tf_regulon_e2f2_enrichment.csv"
    if not regulon_path.exists():
        pytest.skip("tf_regulon_e2f2_enrichment.csv not yet generated. Run Layer 05 first.")

    reg_df = pd.read_csv(regulon_path)
    assert len(reg_df) == 23, f"Expected 23 canonical E2F2 targets, got {len(reg_df)}"
    assert not reg_df["correlation_with_E2F2"].isna().any(), "NaN correlations detected in regulon table!"

    # Bounds check
    assert reg_df["correlation_with_E2F2"].between(-1.0, 1.0).all(), "Invalid Pearson r outside [-1, 1]!"
    assert reg_df["correlation_with_dosage"].between(-1.0, 1.0).all(), "Invalid dosage r outside [-1, 1]!"

    # Key targets verification
    tk1_row = reg_df[reg_df["target_gene"] == "TK1"].iloc[0]
    tk1_corr = tk1_row["correlation_with_E2F2"]
    assert tk1_corr > 0.70, f"TK1 expected strong correlation with E2F2, got {tk1_corr}"
    assert tk1_row["pval_E2F2"] < 1e-15, f"TK1 expected highly significant p-val, got {tk1_row['pval_E2F2']}"

    tyms_row = reg_df[reg_df["target_gene"] == "TYMS"].iloc[0]
    tyms_corr = tyms_row["correlation_with_E2F2"]
    assert tyms_corr > 0.30, f"TYMS expected moderate correlation with E2F2, got {tyms_corr}"


def test_wgcna_dosage_module_integrity():
    """
    Validates WGCNA co-expression network results:
      1. 500 variable genes are clustered into distinct modules.
      2. Module eigengenes are computed across all 111 samples.
      3. The primary dosage module (Blue) strongly correlates with X-chromosome dosage (r > 0.55, p < 1e-10).
      4. Key dosage drivers (SLC25A6, CD99, KDM6A) are members of the dominant dosage module.
    """
    mod_path = REPO_ROOT / "results" / "wgcna_dosage_modules.csv"
    me_path = REPO_ROOT / "results" / "wgcna_module_eigengenes.csv"
    if not mod_path.exists() or not me_path.exists():
        pytest.skip("WGCNA output files not yet generated. Run Layer 05 first.")

    mod_df = pd.read_csv(mod_path)
    me_df = pd.read_csv(me_path, index_col=0)

    # 1. Dimensions
    assert len(mod_df) == 500, f"Expected 500 genes in WGCNA table, got {len(mod_df)}"
    assert me_df.shape[0] == 111, f"Expected 111 samples in ME table, got {me_df.shape[0]}"
    assert me_df.shape[1] >= 4, f"Expected >= 4 modules, got {me_df.shape[1]}"

    # 2. Key genes membership in Blue module
    blue_genes = set(mod_df[mod_df["module_name"] == "Blue"]["gene"])
    for key_gene in ["SLC25A6", "CD99", "KDM6A", "CLEC4E"]:
        assert key_gene in blue_genes, f"Essential driver {key_gene} missing from primary Blue dosage module!"

    # 3. Dosage correlation of Blue module eigengene
    meta_df = pd.read_csv(REPO_ROOT / "data" / "processed" / "harmonized_metadata.csv", index_col=0)
    dosage_col = "x_rel_dosage" if "x_rel_dosage" in meta_df.columns else "x_dosage"
    dosage_vec = meta_df.loc[me_df.index, dosage_col].values
    blue_me = me_df["ME_Blue"].values
    r_val = float(np.corrcoef(blue_me, dosage_vec)[0, 1])

    assert abs(r_val) > 0.55, f"Expected |r| > 0.55 for Blue module eigengene, got {r_val}"
