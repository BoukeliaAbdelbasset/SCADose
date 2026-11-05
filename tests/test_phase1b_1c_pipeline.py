#!/usr/bin/env python3
"""
tests/test_phase1b_1c_pipeline.py
=================================
Automated pytest suite for statistical modeling, meta-analysis, systems biology,
machine learning, and simulation (Tests T6 - T12 per Master Prompt).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import nnls
import scipy.stats as stats
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"


# ---------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------
@pytest.fixture(scope="module")
def de_results():
    studies = ["GSE42331", "GSE46687", "GSE47584", "GSE126712"]
    res = {}
    for s in studies:
        p = RESULTS_DIR / f"de_{s}.tsv"
        assert p.exists(), f"Missing DE file {p}"
        res[s] = pd.read_csv(p, sep="\t", index_col=0)
    return res


@pytest.fixture(scope="module")
def meta_results():
    p = RESULTS_DIR / "meta_analysis_results.tsv"
    assert p.exists(), "Missing meta_analysis_results.tsv"
    return pd.read_csv(p, sep="\t", index_col=0)


@pytest.fixture(scope="module")
def systems_summary():
    p = RESULTS_DIR / "calibrated_systems_summary.json"
    assert p.exists(), "Missing calibrated_systems_summary.json"
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------
# T6: Simulation FDR Calibration & Legacy Defect Documented
# ---------------------------------------------------------
def test_t6_simulation_fdr_calibration(systems_summary):
    sim = systems_summary["simulation"]
    # Final pipeline controls FDR strictly under the null
    assert sim["meta_analysis_false_positive_pct"] <= 5.0, (
        f"Meta-analysis false positive rate too high: {sim['meta_analysis_false_positive_pct']}%"
    )
    assert sim["joint_model_false_positive_pct"] <= 5.0, (
        f"Joint model false positive rate too high: {sim['joint_model_false_positive_pct']}%"
    )
    # Naive unadjusted pooled OLS exhibits massive false positive inflation (legacy defect)
    assert sim["naive_ols_false_positive_pct"] > 50.0, (
        f"Naive pooled regression failed to demonstrate expected false-positive inflation: {sim['naive_ols_false_positive_pct']}%"
    )


# ---------------------------------------------------------
# T7: Positive and Negative Controls
# ---------------------------------------------------------
def test_t7_positive_and_negative_controls(de_results, meta_results):
    # Positive control 1: KDM6A (XCI escape) must be significantly upregulated in TS and KS
    de_42331 = de_results["GSE42331"]
    de_46687 = de_results["GSE46687"]
    de_rna = de_results["GSE126712"]

    assert "KDM6A" in de_42331.index, "KDM6A missing in GSE42331"
    assert de_42331.loc["KDM6A", "log2FC"] > 0, "KDM6A must have positive log2FC in KS"
    assert de_42331.loc["KDM6A", "padj"] < 0.05, "KDM6A must be significant in KS"

    assert "KDM6A" in de_46687.index, "KDM6A missing in GSE46687"
    assert de_46687.loc["KDM6A", "log2FC"] > 0, "KDM6A must have positive log2FC in TS"
    assert de_46687.loc["KDM6A", "padj"] < 0.05, "KDM6A must be significant in TS"

    # Positive control 2: SLC25A6 (PAR1) must be positive in TS, KS, and RNA-seq
    assert "SLC25A6" in de_42331.index
    assert de_42331.loc["SLC25A6", "log2FC"] > 0
    assert de_42331.loc["SLC25A6", "padj"] < 0.05

    assert "SLC25A6" in de_46687.index
    assert de_46687.loc["SLC25A6", "log2FC"] > 0
    assert de_46687.loc["SLC25A6", "padj"] < 0.05

    # Positive control 3: In cross-study models, KDM6A and SLC25A6 must show significant positive X-dosage
    assert "KDM6A" in meta_results.index
    assert meta_results.loc["KDM6A", "beta_meta"] > 0
    assert meta_results.loc["KDM6A", "pvalue_z"] < 0.05

    joint_results = pd.read_csv(RESULTS_DIR / "joint_study_stratified_model.tsv", sep="\t", index_col=0)
    assert joint_results.loc["KDM6A", "beta_joint_X"] > 0
    assert joint_results.loc["KDM6A", "padj_joint_X"] < 0.01

    # Negative control: CLEC4E and E2F2 must NOT be significant in GSE42331
    assert "CLEC4E" in de_42331.index
    assert de_42331.loc["CLEC4E", "padj"] > 0.05, "CLEC4E should NOT be significant in GSE42331"

    assert "E2F2" in de_42331.index
    assert de_42331.loc["E2F2", "padj"] > 0.05, "E2F2 should NOT be significant in GSE42331"


# ---------------------------------------------------------
# T8: Meta-Analysis Implementation Verification
# ---------------------------------------------------------
def test_t8_meta_analysis_math(de_results, meta_results):
    # Verify random-effects weighting on a test gene
    test_gene = "KDM6A"
    studies = ["GSE42331", "GSE46687", "GSE47584"]
    betas = np.array([de_results[s].loc[test_gene, "log2FC"] for s in studies])
    ses = np.array([de_results[s].loc[test_gene, "se"] for s in studies])

    meta_row = meta_results.loc[test_gene]
    tau2 = meta_row["tau2"]

    weights = 1.0 / (ses ** 2 + tau2 + 1e-12)
    manual_re_beta = float(np.sum(weights * betas) / np.sum(weights))

    # Summary estimate should match manual calculation within numerical tolerance
    assert np.isclose(meta_row["beta_meta"], manual_re_beta, atol=1e-3), (
        f"Meta beta mismatch: {meta_row['beta_meta']} vs {manual_re_beta}"
    )


# ---------------------------------------------------------
# T9: Deconvolution Recovery on Synthetic Mixtures & Compositional Closure
# ---------------------------------------------------------
def test_t9_deconvolution_synthetic_recovery():
    # Load signature matrix
    sig_path = REPO_ROOT / "data" / "reference" / "TIL10_signature.txt"
    assert sig_path.exists(), "TIL10_signature.txt missing"
    sig = pd.read_csv(sig_path, sep="\t", index_col=0)

    # Generate synthetic mixture with known weights
    np.random.seed(42)
    n_cell_types = sig.shape[1]
    true_weights = np.array([0.15, 0.05, 0.05, 0.20, 0.25, 0.10, 0.10, 0.05, 0.03, 0.02])
    true_weights = true_weights / np.sum(true_weights)

    synth_expr = sig.values @ true_weights

    # Deconvolve via NNLS
    est_w, _ = nnls(sig.values, synth_expr)
    est_w = est_w / np.sum(est_w)

    # Assert compositional closure: sum to 1 and all >= 0
    assert np.isclose(np.sum(est_w), 1.0, atol=1e-5), "Estimated fractions do not sum to 1"
    assert np.all(est_w >= 0.0), "Negative fractions detected"

    # Assert recovery correlation > 0.95
    corr = np.corrcoef(true_weights, est_w)[0, 1]
    assert corr > 0.95, f"Deconvolution recovery correlation too low: {corr:.3f}"


# ---------------------------------------------------------
# T10: Mediation Bootstrap Coverage on Synthetic Data
# ---------------------------------------------------------
def test_t10_mediation_bootstrap_coverage():
    # Simulate known mediation: X -> M -> Y with indirect effect a*b = 0.5 * 0.8 = 0.40
    np.random.seed(42)
    n = 100
    X = np.random.binomial(1, 0.5, size=n)
    M = 0.5 * X + np.random.normal(0, 0.3, size=n)
    Y = 0.2 * X + 0.8 * M + np.random.normal(0, 0.3, size=n)

    # Bootstrap indirect effect
    n_boot = 500
    boot_ab = []
    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        mod_m = sm.OLS(M[idx], sm.add_constant(X[idx])).fit()
        a_b = mod_m.params[1]

        mod_y = sm.OLS(Y[idx], sm.add_constant(np.column_stack([X[idx], M[idx]]))).fit()
        b_b = mod_y.params[2]

        boot_ab.append(a_b * b_b)

    ci_lower = np.percentile(boot_ab, 2.5)
    ci_upper = np.percentile(boot_ab, 97.5)

    true_indirect = 0.5 * 0.8
    # Assert true indirect effect is within 95% bootstrap CI
    assert ci_lower <= true_indirect <= ci_upper, (
        f"True indirect effect {true_indirect} outside bootstrap CI: [{ci_lower:.3f}, {ci_upper:.3f}]"
    )


# ---------------------------------------------------------
# T11: ML Leakage Test (Permutation Null AUC ~ 0.5)
# ---------------------------------------------------------
def test_t11_ml_leakage(systems_summary):
    null_auc = systems_summary["machine_learning"]["permutation_null_auc_mean"]
    # Label permutation under nested CV must give AUC close to chance (0.50)
    assert 0.45 <= null_auc <= 0.60, (
        f"Permutation null AUC {null_auc} deviates significantly from 0.50, indicating data leakage!"
    )


# ---------------------------------------------------------
# T12: Reproducibility & Result Files Integrity
# ---------------------------------------------------------
def test_t12_result_files_integrity():
    required_files = [
        "cell_composition_fractions_til10.tsv",
        "cell_composition_clr_til10.tsv",
        "cell_composition_stats.tsv",
        "cell_deconvolution_concordance.tsv",
        "mediation_analysis_results.tsv",
        "tf_activity_scores.tsv",
        "tf_dosage_associations.tsv",
        "e2f2_regulon_targets.tsv",
        "e2f2_leverage_diagnostics.tsv",
        "wgcna_power_scan.tsv",
        "wgcna_module_assignments.tsv",
        "wgcna_module_eigengenes.tsv",
        "wgcna_module_trait_correlations.tsv",
        "wgcna_module_preservation.tsv",
        "ml_performance_loso.tsv",
        "ml_cross_syndrome_transfer.tsv",
        "ml_permutation_null.tsv",
        "simulation_fdr_comparison.tsv",
        "simulation_power_curves.tsv",
        "calibrated_systems_summary.json"
    ]
    for fname in required_files:
        p = RESULTS_DIR / fname
        assert p.exists(), f"Missing required result file: {fname}"
        assert p.stat().st_size > 0, f"Result file {fname} is empty"


# ---------------------------------------------------------
# T13: Manuscript Number Traceability (No Hardcoded Stats)
# ---------------------------------------------------------
def test_t13_manuscript_number_macros():
    macros_path = REPO_ROOT / "latex" / "macros.tex"
    assert macros_path.exists(), "latex/macros.tex does not exist! Run generate_manuscript_macros.py."
    content = macros_path.read_text()
    assert "\\newcommand{\\TotalSamples}{165}" in content
    assert "\\newcommand{\\DiscoverySamples}{111}" in content
    assert "\\newcommand{\\KDMSixABetaJoint}" in content
    assert "\\newcommand{\\SLCBetaJoint}" in content
    assert "\\newcommand{\\SimulationLegacyFDR}" in content


# ---------------------------------------------------------
# T14: Reference Verification (All DOIs and Metadata Resolved)
# ---------------------------------------------------------
def test_t14_reference_audit():
    audit_path = REPO_ROOT / "REFERENCE_AUDIT.csv"
    assert audit_path.exists(), "REFERENCE_AUDIT.csv missing!"
    df = pd.read_csv(audit_path)
    assert len(df) >= 40, f"Too few verified references: {len(df)}"
    assert (df["verification_status"] == "VERIFIED_CROSSREF").all(), "Unverified references found in audit table"
    assert (df["doi"].str.len() > 5).all(), "Detected invalid or missing DOIs in audit table"
    # Ensure Raznahan 2018 is correctly attributed
    raz_row = df[df["citation_key"] == "raznahan2018dosage"]
    assert len(raz_row) == 1
    assert "Sex-chromosome dosage" in raz_row["bib_title"].values[0]


# ---------------------------------------------------------
# T15: Figure Existence & Format Consistency
# ---------------------------------------------------------
def test_t15_figure_existence():
    required_figures = [
        "Fig1_Study_Design",
        "Fig2_QC_Sex_Concordance",
        "Fig3_Within_Study_Volcano",
        "Fig4_Chromosome_Dosage_Map",
        "Fig5_Meta_Forest_Concordance",
        "Fig6_Gene_Class_Model_Selection",
        "Fig7_RNA_Replication",
        "Fig8_Cell_Composition_Deconvolution",
        "Fig9_Mediation_Bootstrap",
        "Fig10_TF_Regulon_E2F2",
        "Fig11_Signed_WGCNA",
        "Fig12_Machine_Learning",
        "FigS1_Simulation_FDR_Power"
    ]
    for base in required_figures:
        for ext in [".pdf", ".png"]:
            fig_p = REPO_ROOT / "figures" / f"{base}{ext}"
            assert fig_p.exists(), f"Missing figure file: {fig_p}"
            assert fig_p.stat().st_size > 0, f"Figure file {fig_p} is empty"
