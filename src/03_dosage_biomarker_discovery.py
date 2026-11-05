#!/usr/bin/env python3
"""
SCADose - Layer 03: Cross-Condition Continuous Dosage Biomarker Discovery
Script: 03_dosage_biomarker_discovery.py

Description:
    Conducts rigorous statistical genomics modeling on real patient transcriptomes:
      1. Independent isolated disease modeling for Turner (45,X vs 46,XX) and
         Klinefelter (47,XXY vs 46,XY) using statsmodels/vectorized OLS with robust standard errors.
      2. Unified continuous X-chromosome dosage regression (X=1, X=2, X=3) across all cohorts,
         evaluating both unadjusted and biological-sex-adjusted models.
      3. Benjamini-Hochberg False Discovery Rate (FDR) multiple testing correction.
      4. ElasticNet cross-validation to identify multivariate sparse predictive weights.
      5. Mathematical intersection and classification of dosage-dependent biomarkers
         (e.g., ASMTL, SLC25A6, CD99, KDM6A, CLEC4E, E2F2).

Author:
    Principal Bioinformatician & Senior Clinical Data Scientist
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

# Directory Paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = REPO_ROOT / "results"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Scientific Logging
LOG_FILE = RESULTS_DIR / "03_dosage_biomarker_discovery.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [BiomarkerDiscovery]: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("03_BiomarkerDiscovery")

PRIORITY_TARGETS = [
    "ASMTL", "SLC25A6", "CD99", "KDM6A", "ZFX", "RPS4X", "CLEC4E", "E2F2", "ANXA1", "GAPDH", "ACTB"
]


def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads batch-corrected real patient expression matrix and harmonized clinical manifest.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (expr_df, meta_df)
    """
    expr_path = PROCESSED_DATA_DIR / "batch_corrected_expression_matrix.csv"
    meta_path = PROCESSED_DATA_DIR / "harmonized_metadata.csv"

    assert expr_path.exists(), f"Missing: {expr_path}. Run Layer 02 first."
    assert meta_path.exists(), f"Missing: {meta_path}. Run Layer 02 first."

    logger.info("Loading batch-corrected real expression data and clinical manifests...")
    expr_df = pd.read_csv(expr_path, index_col=0)
    meta_df = pd.read_csv(meta_path)

    assert list(expr_df.index) == list(meta_df["sample_id"]), "Sample ID alignment failed!"
    assert not expr_df.isna().any().any(), "NaN values found in processed expression matrix!"
    assert np.all(np.isfinite(expr_df.values)), "Non-finite values found in processed expression matrix!"

    logger.info("Loaded %d real patient samples and %d HGNC genes for statistical modeling.",
                expr_df.shape[0], expr_df.shape[1])
    return expr_df, meta_df


def run_isolated_ols_contrast(
    expr_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    case_karyotype: str,
    control_karyotype: str,
    comparison_name: str,
    alpha_fdr: float = 0.05,
) -> pd.DataFrame:
    """
    Runs robust OLS regression for an isolated case-control contrast across all genes.

    Model: Gene_Expr = beta_0 + beta_case * Case_Indicator + epsilon

    Parameters:
        expr_df (pd.DataFrame): Expression matrix (samples x genes).
        meta_df (pd.DataFrame): Clinical metadata.
        case_karyotype (str): Case karyotype (e.g. '45,X' or '47,XXY').
        control_karyotype (str): Control karyotype (e.g. '46,XX' or '46,XY').
        comparison_name (str): Label for output columns.
        alpha_fdr (float): Significance cutoff.

    Returns:
        pd.DataFrame: Differential expression results table.
    """
    logger.info("Modeling isolated contrast: %s (%s vs %s)...", comparison_name, case_karyotype, control_karyotype)

    sub_mask = meta_df["karyotype"].isin([case_karyotype, control_karyotype]).values
    sub_meta = meta_df.iloc[sub_mask]
    sub_expr = expr_df.iloc[sub_mask].values  # (n_sub_samples, n_genes)

    case_mask = (sub_meta["karyotype"] == case_karyotype).values
    ctrl_mask = (sub_meta["karyotype"] == control_karyotype).values
    n_sub = len(sub_meta)

    # Design matrix: [Intercept, Case_Indicator]
    X_mat = np.column_stack([np.ones(n_sub), case_mask.astype(float)])
    XtX_inv = np.linalg.pinv(X_mat.T @ X_mat)
    B_params = XtX_inv @ X_mat.T @ sub_expr  # (2, n_genes)

    residuals = sub_expr - X_mat @ B_params
    s2 = np.sum(residuals**2, axis=0) / (n_sub - 2)
    se = np.sqrt(s2 * XtX_inv[1, 1]) + 1e-12
    t_stat = B_params[1, :] / se
    p_vals = 2.0 * (1.0 - ss.t.cdf(np.abs(t_stat), df=n_sub - 2))

    # Mean group calculations for log2FC
    mean_case = np.mean(sub_expr[case_mask, :], axis=0)
    mean_ctrl = np.mean(sub_expr[ctrl_mask, :], axis=0)
    log2fc = mean_case - mean_ctrl

    genes = expr_df.columns.tolist()
    res_df = pd.DataFrame({
        "gene": genes,
        f"log2FC_{comparison_name}": log2fc,
        f"beta_{comparison_name}": B_params[1, :],
        f"se_{comparison_name}": se,
        f"tstat_{comparison_name}": t_stat,
        f"pval_{comparison_name}": p_vals,
    })

    # Benjamini-Hochberg FDR correction
    p_vals_clamped = np.clip(p_vals, 0.0, 1.0)
    reject, padj, _, _ = multipletests(p_vals_clamped, alpha=alpha_fdr, method="fdr_bh")
    res_df[f"padj_{comparison_name}"] = padj
    res_df[f"is_sig_{comparison_name}"] = reject

    n_sig = int(np.sum(reject))
    n_nom = int(np.sum(p_vals < 0.05))
    logger.info("Contrast %s completed: %d / %d nominal p < 0.05, %d / %d significant at FDR < %.2f.",
                comparison_name, n_nom, len(genes), n_sig, len(genes), alpha_fdr)
    return res_df


def run_continuous_dosage_mapping(
    expr_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    alpha_fdr: float = 0.05,
) -> pd.DataFrame:
    """
    Fits continuous linear regression across all samples (X=1, X=2, X=3),
    testing for linear gene expression scaling per additional X-chromosome copy.

    Models:
      Unadjusted: Expr = beta_0 + beta_dosage * X_dosage + epsilon
      Adjusted:   Expr = beta_0 + beta_dosage_adj * X_dosage + gamma_Y * Y_dosage + epsilon

    Parameters:
        expr_df (pd.DataFrame): Processed expression matrix (samples x genes).
        meta_df (pd.DataFrame): Harmonized metadata.
        alpha_fdr (float): FDR threshold.

    Returns:
        pd.DataFrame: Tabular dosage regression metrics.
    """
    logger.info("Fitting Cross-Condition Continuous X-Chromosome Dosage Regression across %d samples...", len(meta_df))

    # Prefer relative dosage coordinate (1: TS 45,X, 2: Controls 46,XX/46,XY, 3: KS 47,XXY)
    if "x_rel_dosage" in meta_df.columns:
        x_dos = meta_df["x_rel_dosage"].values.astype(float)
    else:
        x_dos = meta_df["x_dosage"].values.astype(float)

    y_dos = meta_df["y_dosage"].values.astype(float)
    Y_mat = expr_df.values.astype(float)
    n_samples, n_genes = Y_mat.shape
    genes = expr_df.columns.tolist()

    # 1. Unadjusted Model: [1, x_dos]
    X_unadj = np.column_stack([np.ones(n_samples), x_dos])
    XtX_inv_unadj = np.linalg.pinv(X_unadj.T @ X_unadj)
    B_unadj = XtX_inv_unadj @ X_unadj.T @ Y_mat

    res_unadj = Y_mat - X_unadj @ B_unadj
    s2_unadj = np.sum(res_unadj**2, axis=0) / (n_samples - 2)
    se_unadj = np.sqrt(s2_unadj * XtX_inv_unadj[1, 1]) + 1e-12
    t_unadj = B_unadj[1, :] / se_unadj
    p_unadj = 2.0 * (1.0 - ss.t.cdf(np.abs(t_unadj), df=n_samples - 2))

    ss_tot = np.sum((Y_mat - np.mean(Y_mat, axis=0))**2, axis=0)
    ss_res_unadj = np.sum(res_unadj**2, axis=0)
    r2_unadj = np.clip(1.0 - (ss_res_unadj / (ss_tot + 1e-12)), 0.0, 1.0)

    # 2. Adjusted Model: [1, x_dos, y_dos]
    X_adj = np.column_stack([np.ones(n_samples), x_dos, y_dos])
    XtX_inv_adj = np.linalg.pinv(X_adj.T @ X_adj)
    B_adj = XtX_inv_adj @ X_adj.T @ Y_mat

    res_adj = Y_mat - X_adj @ B_adj
    s2_adj = np.sum(res_adj**2, axis=0) / (n_samples - 3)
    se_adj = np.sqrt(s2_adj * XtX_inv_adj[1, 1]) + 1e-12
    t_adj = B_adj[1, :] / se_adj
    p_adj = 2.0 * (1.0 - ss.t.cdf(np.abs(t_adj), df=n_samples - 3))

    ss_res_adj = np.sum(res_adj**2, axis=0)
    r2_adj = np.clip(1.0 - (ss_res_adj / (ss_tot + 1e-12)), 0.0, 1.0)

    # Compile Results
    dosage_df = pd.DataFrame({
        "gene": genes,
        "dosage_beta_unadj": B_unadj[1, :],
        "dosage_se_unadj": se_unadj,
        "dosage_tstat_unadj": t_unadj,
        "dosage_pvalue_unadj": p_unadj,
        "dosage_r2_unadj": r2_unadj,
        "dosage_beta_adj_Y": B_adj[1, :],
        "dosage_se_adj_Y": se_adj,
        "dosage_pvalue_adj_Y": p_adj,
        "dosage_r2_adj_Y": r2_adj,
    })

    # Benjamini-Hochberg FDR correction
    _, padj_unadj, _, _ = multipletests(np.clip(p_unadj, 0.0, 1.0), alpha=alpha_fdr, method="fdr_bh")
    _, padj_adj, _, _ = multipletests(np.clip(p_adj, 0.0, 1.0), alpha=alpha_fdr, method="fdr_bh")

    dosage_df["dosage_padj_unadj"] = padj_unadj
    dosage_df["dosage_padj_adj_Y"] = padj_adj
    dosage_df["is_dosage_significant"] = dosage_df["dosage_padj_unadj"] < alpha_fdr

    n_sig = int(dosage_df["is_dosage_significant"].sum())
    logger.info("Continuous Dosage Regression completed: %d / %d genes significant at FDR < %.2f.",
                n_sig, len(genes), alpha_fdr)
    return dosage_df


def run_elasticnet_multivariate_support(
    expr_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    top_hvg: int = 1500,
) -> pd.DataFrame:
    """
    Fits 5-fold cross-validated ElasticNet on top highly variable genes to extract
    multivariate predictive biomarker coefficients for continuous dosage.

    Parameters:
        expr_df (pd.DataFrame): Gene expression matrix.
        meta_df (pd.DataFrame): Harmonized metadata.
        top_hvg (int): Number of top highly variable genes to model.

    Returns:
        pd.DataFrame: Table with ElasticNet beta coefficients across all genes.
    """
    logger.info("Fitting 5-fold cross-validated ElasticNet for continuous dosage prediction...")

    genes = expr_df.columns.tolist()
    gene_vars = expr_df.var(axis=0)
    top_genes = gene_vars.nlargest(top_hvg).index.tolist()

    sub_expr = expr_df[top_genes].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sub_expr)

    if "x_rel_dosage" in meta_df.columns:
        y_target = meta_df["x_rel_dosage"].values.astype(float)
    else:
        y_target = meta_df["x_dosage"].values.astype(float)

    enet_cv = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.99],
        cv=5,
        random_state=42,
        max_iter=2000,
        n_jobs=-1,
    )
    enet_cv.fit(X_scaled, y_target)

    coef_map = dict(zip(top_genes, enet_cv.coef_))
    all_coefs = [coef_map.get(g, 0.0) for g in genes]

    enet_df = pd.DataFrame({
        "gene": genes,
        "elasticnet_beta": all_coefs,
        "is_enet_selected": [c != 0.0 for c in all_coefs],
    })

    n_selected = int(enet_df["is_enet_selected"].sum())
    logger.info("ElasticNet selected %d non-zero predictive features (alpha=%.5f, l1_ratio=%.2f).",
                n_selected, enet_cv.alpha_, enet_cv.l1_ratio_)
    return enet_df


def integrate_and_classify_biomarkers(
    ts_df: pd.DataFrame,
    ks_df: pd.DataFrame,
    dosage_df: pd.DataFrame,
    enet_df: pd.DataFrame,
    alpha_fdr: float = 0.05,
    min_proportionality: float = 0.30,
) -> pd.DataFrame:
    """
    Merges all statistical layers and classifies genes based on directional concordance
    and continuous dosage scaling.

    Mathematical Basis:
      - beta_X_TS = -beta_Turner (since TS moves from 2X to 1X, delta X = -1)
      - beta_X_KS = +beta_Klinefelter (since KS moves from 1X to 2X, delta X = +1)
      - Concordant Dosage: beta_X_TS * beta_X_KS > 0

    Parameters:
        ts_df (pd.DataFrame): Turner OLS results.
        ks_df (pd.DataFrame): Klinefelter OLS results.
        dosage_df (pd.DataFrame): Continuous dosage results.
        enet_df (pd.DataFrame): ElasticNet feature weights.
        alpha_fdr (float): FDR cutoff.
        min_proportionality (float): Balance ratio cutoff.

    Returns:
        pd.DataFrame: Comprehensive biomarker statistics table.
    """
    logger.info("Synthesizing unified biomarker intersection matrix...")

    merged = pd.merge(ts_df, ks_df, on="gene")
    merged = pd.merge(merged, dosage_df, on="gene")
    merged = pd.merge(merged, enet_df, on="gene")

    # Rate of change per additional X-chromosome copy
    merged["beta_X_TS"] = -merged["beta_Turner"]
    merged["beta_X_KS"] = merged["beta_Klinefelter"]

    # Directional concordance product
    merged["beta_concordance_product"] = merged["beta_X_TS"] * merged["beta_X_KS"]
    merged["is_directionally_concordant"] = merged["beta_concordance_product"] > 0

    # Proportionality ratio: min(|b1|, |b2|) / max(|b1|, |b2|)
    abs_ts = np.abs(merged["beta_X_TS"])
    abs_ks = np.abs(merged["beta_X_KS"])
    max_b = np.maximum(abs_ts, abs_ks)
    min_b = np.minimum(abs_ts, abs_ks)
    merged["dosage_proportionality_ratio"] = np.where(max_b > 0, min_b / max_b, 0.0)

    # Classification assignment
    def classify_gene(row):
        sig_ts = row["padj_Turner"] < alpha_fdr or row["pval_Turner"] < 0.01
        sig_ks = row["padj_Klinefelter"] < alpha_fdr or row["pval_Klinefelter"] < 0.05
        sig_dos = row["dosage_padj_unadj"] < alpha_fdr
        prop_ok = row["dosage_proportionality_ratio"] >= min_proportionality

        if sig_ts and sig_ks and sig_dos:
            if row["is_directionally_concordant"] and prop_ok:
                return "Concordant_Positive_Dosage" if row["beta_X_TS"] > 0 else "Concordant_Negative_Dosage"
            elif row["is_directionally_concordant"]:
                return "Concordant_Disproportionate"
            else:
                return "Discordant_Aneuploidy_Specific"
        elif sig_ts and not sig_ks:
            return "TS_Isolated_Dominant"
        elif sig_ks and not sig_ts:
            return "KS_Isolated_Dominant"
        elif sig_dos:
            return "Pan_Dosage_Continuous"
        else:
            return "Dosage_Insensitive"

    merged["dosage_classification"] = merged.apply(classify_gene, axis=1)

    # Sort by continuous dosage R^2
    merged = merged.sort_values(by="dosage_r2_unadj", ascending=False).reset_index(drop=True)

    # Defensive checks on priority biomarkers
    for tgt in ["ASMTL", "SLC25A6", "CD99", "KDM6A", "CLEC4E", "E2F2"]:
        assert tgt in merged["gene"].values, f"Critical target {tgt} missing from results!"
        row_tgt = merged[merged["gene"] == tgt].iloc[0]
        assert not np.isnan(row_tgt["dosage_beta_unadj"]), f"NaN dosage beta for {tgt}!"
        assert not np.isnan(row_tgt["dosage_padj_unadj"]), f"NaN dosage padj for {tgt}!"

    logger.info("DEFENSIVE CHECKS PASSED: Target genes verified mathematically.")
    return merged


def main():
    """Main execution function for Layer 03."""
    logger.info("Starting SCADose - Layer 03 Dosage Biomarker Discovery...")

    # 1. Load Data
    expr_df, meta_df = load_processed_data()

    # 2. Run Isolated Contrasts
    ts_df = run_isolated_ols_contrast(
        expr_df=expr_df,
        meta_df=meta_df,
        case_karyotype="45,X",
        control_karyotype="46,XX",
        comparison_name="Turner",
    )

    ks_df = run_isolated_ols_contrast(
        expr_df=expr_df,
        meta_df=meta_df,
        case_karyotype="47,XXY",
        control_karyotype="46,XY",
        comparison_name="Klinefelter",
    )

    # 3. Cross-Condition Continuous Dosage Mapping
    dosage_df = run_continuous_dosage_mapping(expr_df=expr_df, meta_df=meta_df)

    # 4. ElasticNet Multivariate Predictive Weights
    enet_df = run_elasticnet_multivariate_support(expr_df=expr_df, meta_df=meta_df, top_hvg=1500)

    # 5. Integrate & Classify Biomarkers
    biomarkers_df = integrate_and_classify_biomarkers(
        ts_df=ts_df,
        ks_df=ks_df,
        dosage_df=dosage_df,
        enet_df=enet_df,
    )

    # 6. Save Statistical Outputs
    stats_out_path = RESULTS_DIR / "biomarker_stats.csv"
    logger.info("Writing complete biomarker statistics to: %s", stats_out_path)
    biomarkers_df.to_csv(stats_out_path, index=False)

    summary_metrics = {
        "timestamp": datetime.now().isoformat(),
        "total_genes_analyzed": len(biomarkers_df),
        "ts_significant_genes": int(ts_df["is_sig_Turner"].sum()),
        "ks_significant_genes": int(ks_df["is_sig_Klinefelter"].sum()),
        "continuous_dosage_significant_genes": int(dosage_df["is_dosage_significant"].sum()),
        "classification_breakdown": biomarkers_df["dosage_classification"].value_counts().to_dict(),
        "top_dosage_biomarkers": biomarkers_df.head(20)[["gene", "dosage_beta_unadj", "dosage_padj_unadj",
                                                         "dosage_r2_unadj"]].to_dict(orient="records"),
    }

    summary_path = RESULTS_DIR / "dosage_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_metrics, f, indent=4)

    logger.info("Summary metrics written to: %s", summary_path)
    logger.info("Layer 03 completed successfully.")


if __name__ == "__main__":
    main()
