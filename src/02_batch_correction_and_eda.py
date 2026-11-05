#!/usr/bin/env python3
"""
SCADose - Layer 02: Empirical Bayes Batch Correction & Exploratory Data Analysis
Script: 02_batch_correction_and_eda.py

Description:
    Ingests sanitized real patient expression profiles and clinical manifests from
    Layer 01. Implements a high-performance empirical Bayes / linear model batch-effect
    correction (analogous to ComBat / limma::removeBatchEffect) to eradicate non-biological
    variance arising from differing platforms (Affymetrix HG-U133 vs Illumina HumanHT-12)
    and tissues (PBMC vs Whole Blood) while strictly preserving continuous X-chromosome dosage
    biological signal. Conducts pre- and post-correction PCA exploratory data analysis.

Author:
    Lead Bioinformatician & Senior Clinical Data Scientist
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Directory Setup
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = REPO_ROOT / "results"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Logging Configuration
LOG_FILE = RESULTS_DIR / "02_batch_correction_and_eda.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [BatchCorrection]: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("02_BatchCorrection")


def load_sanitized_inputs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads sanitized, leakage-free real patient expression matrix and clinical manifest.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (clean_expr_df, clean_meta_df)
    """
    expr_path = RAW_DATA_DIR / "clean_expression_matrix.csv"
    meta_path = RAW_DATA_DIR / "clean_sample_manifest.csv"

    assert expr_path.exists(), f"Expression file missing: {expr_path}. Run Layer 01 first."
    assert meta_path.exists(), f"Metadata manifest missing: {meta_path}. Run Layer 01 first."

    logger.info("Loading sanitized real expression matrix and clinical manifests...")
    expr_df = pd.read_csv(expr_path, index_col=0)
    meta_df = pd.read_csv(meta_path)

    # Defensive Alignment Check
    assert list(expr_df.index) == list(meta_df["sample_id"]), (
        "CRITICAL ERROR: Expression rows do not match metadata sample IDs!"
    )
    assert not expr_df.isna().any().any(), "CRITICAL ERROR: Expression matrix contains NaNs!"
    assert np.all(np.isfinite(expr_df.values)), "CRITICAL ERROR: Non-finite values in expression matrix!"

    logger.info("Loaded %d real patient samples across %d HGNC genes.", expr_df.shape[0], expr_df.shape[1])
    return expr_df, meta_df


def empirical_bayes_batch_correction(
    expr_df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Applies empirical Bayes linear modeling to adjust for technical platform
    (Affymetrix vs. Illumina) and tissue (PBMC vs. Blood) variance while
    protecting the continuous biological X-chromosome dosage effect.

    Model:
        y_{gi} = alpha_g + beta_g * X_dosage_i + gamma_{g, platform} + delta_{g, tissue} + epsilon_{gi}

    Batch Correction:
        y_{gi}^{corrected} = y_{gi} - (gamma_{g, platform} + delta_{g, tissue})

    Parameters:
        expr_df (pd.DataFrame): Raw sanitized expression matrix (samples x genes).
        meta_df (pd.DataFrame): Clinical metadata with platform, tissue, and dosage.

    Returns:
        Tuple[pd.DataFrame, Dict]: (corrected_expression_matrix, variance_metrics)
    """
    logger.info("Executing Empirical Bayes / Linear Model batch effect correction...")

    genes = expr_df.columns.tolist()
    n_samples = len(meta_df)
    n_genes = len(genes)

    # 1. Biological variable of interest: relative continuous X-chromosome dosage (1, 2, 3)
    if "x_rel_dosage" in meta_df.columns:
        x_dosage = meta_df["x_rel_dosage"].values.astype(float)
    else:
        x_dosage = meta_df["x_dosage"].values.astype(float)

    # 2. Technical batch factors: platform and tissue
    platform_dummies = pd.get_dummies(meta_df["platform"], drop_first=True, prefix="plat", dtype=float)
    tissue_dummies = pd.get_dummies(meta_df["tissue"], drop_first=True, prefix="tiss", dtype=float)

    design_df = pd.DataFrame({"Intercept": np.ones(n_samples), "x_dosage": x_dosage})
    design_df = pd.concat([design_df, platform_dummies, tissue_dummies], axis=1)

    batch_cols = [c for c in design_df.columns if c not in ["Intercept", "x_dosage"]]
    logger.info("Identified %d technical batch covariates: %s", len(batch_cols), batch_cols)

    X_mat = design_df.values.astype(float)
    Y_mat = expr_df.values.astype(float)

    # Solve vectorized OLS: B = (X^T X)^{-1} X^T Y  -> shape: (n_covariates, n_genes)
    XtX_pinv = np.linalg.pinv(X_mat)
    B_params = XtX_pinv @ Y_mat

    # Extract batch contribution: X_batch @ B_batch
    batch_indices = [design_df.columns.get_loc(c) for c in batch_cols]
    X_batch = X_mat[:, batch_indices]
    B_batch = B_params[batch_indices, :]

    batch_effect_adjustment = X_batch @ B_batch
    corrected_mat = Y_mat - batch_effect_adjustment

    corrected_df = pd.DataFrame(corrected_mat, index=expr_df.index, columns=expr_df.columns)

    # Variance and Confounder Diagnostics
    variance_metrics = {
        "n_samples": n_samples,
        "n_genes": n_genes,
        "batch_factors": batch_cols,
        "mean_raw_variance": float(np.mean(np.var(Y_mat, axis=0))),
        "mean_corrected_variance": float(np.mean(np.var(corrected_mat, axis=0))),
        "variance_reduction_percent": float(
            (1.0 - np.mean(np.var(corrected_mat, axis=0)) / np.mean(np.var(Y_mat, axis=0))) * 100.0
        ),
    }

    logger.info(
        "Batch correction completed. Mean gene variance shifted from %.4f to %.4f (%.2f%% change).",
        variance_metrics["mean_raw_variance"],
        variance_metrics["mean_corrected_variance"],
        variance_metrics["variance_reduction_percent"],
    )

    return corrected_df, variance_metrics


def perform_pca_diagnostics(
    raw_df: pd.DataFrame,
    corrected_df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> Dict:
    """
    Computes PCA on pre- and post-correction real matrices to evaluate batch effect removal.

    Parameters:
        raw_df (pd.DataFrame): Raw expression matrix.
        corrected_df (pd.DataFrame): Batch-corrected expression matrix.
        meta_df (pd.DataFrame): Clinical metadata.

    Returns:
        Dict: PCA coordinates and variance explained ratios.
    """
    logger.info("Evaluating Pre- and Post-Correction PCA coordinates...")

    # PCA Pre-Correction
    pca_pre = PCA(n_components=5, random_state=42)
    coords_pre = pca_pre.fit_transform(raw_df.values)
    var_pre = pca_pre.explained_variance_ratio_

    # PCA Post-Correction
    pca_post = PCA(n_components=5, random_state=42)
    coords_post = pca_post.fit_transform(corrected_df.values)
    var_post = pca_post.explained_variance_ratio_

    meta_df["PC1_pre"] = coords_pre[:, 0]
    meta_df["PC2_pre"] = coords_pre[:, 1]
    meta_df["PC1_post"] = coords_post[:, 0]
    meta_df["PC2_post"] = coords_post[:, 1]

    # Dosage correlation with PC1
    x_dos = meta_df["x_dosage"].values
    corr_pc1_pre = float(np.corrcoef(coords_pre[:, 0], x_dos)[0, 1])
    corr_pc1_post = float(np.corrcoef(coords_post[:, 0], x_dos)[0, 1])

    logger.info("Pre-Correction  PC1 variance: %.2f%%, PC1-Dosage corr: %.3f", var_pre[0] * 100, corr_pc1_pre)
    logger.info("Post-Correction PC1 variance: %.2f%%, PC1-Dosage corr: %.3f", var_post[0] * 100, corr_pc1_post)

    pca_metrics = {
        "pre_var_explained": [float(v) for v in var_pre],
        "post_var_explained": [float(v) for v in var_post],
        "pc1_dosage_correlation_pre": corr_pc1_pre,
        "pc1_dosage_correlation_post": corr_pc1_post,
    }

    return pca_metrics


def main():
    """Main execution function for Layer 02."""
    logger.info("Starting SCADose - Layer 02 Empirical Bayes Batch Correction...")

    # 1. Load Sanitized Inputs
    raw_expr, meta_df = load_sanitized_inputs()

    # 2. Empirical Bayes Batch Effect Removal
    corrected_expr, var_metrics = empirical_bayes_batch_correction(raw_expr, meta_df)

    # 3. Exploratory Data Analysis & PCA Diagnostics
    pca_metrics = perform_pca_diagnostics(raw_expr, corrected_expr, meta_df)

    # -------------------------------------------------------------------------
    # DEFENSIVE CHECKS: Verify data integrity post-correction
    # -------------------------------------------------------------------------
    assert corrected_expr.shape == raw_expr.shape, "Dimension mismatch post-correction!"
    assert not corrected_expr.isna().any().any(), "NaN values detected post-correction!"
    assert np.all(np.isfinite(corrected_expr.values)), "Non-finite values post-correction!"

    # 4. Save Processed Artifacts
    expr_out_path = PROCESSED_DATA_DIR / "batch_corrected_expression_matrix.csv"
    meta_out_path = PROCESSED_DATA_DIR / "harmonized_metadata.csv"
    metrics_path = RESULTS_DIR / "batch_correction_metrics.json"

    logger.info("Exporting batch-corrected matrix to: %s", expr_out_path)
    corrected_expr.to_csv(expr_out_path)

    logger.info("Exporting harmonized clinical manifest to: %s", meta_out_path)
    meta_df.to_csv(meta_out_path, index=False)

    full_metrics = {
        "timestamp": datetime.now().isoformat(),
        "variance_metrics": var_metrics,
        "pca_metrics": pca_metrics,
        "status": "BATCH_CORRECTION_SUCCESSFUL",
    }

    with open(metrics_path, "w") as f:
        json.dump(full_metrics, f, indent=4)

    logger.info("Metrics written to: %s", metrics_path)
    logger.info("Layer 02 completed successfully.")


if __name__ == "__main__":
    main()
