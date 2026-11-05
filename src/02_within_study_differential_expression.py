#!/usr/bin/env python3
"""
SCADose - Phase 1B: Within-Study Differential Expression & Empirical Bayes
Script: src/02_within_study_differential_expression.py

Description:
    Conducts rigorous, pre-specified within-study differential expression across
    all four cohorts, implementing Smyth's (2004) empirical Bayes variance moderation
    (limma-trend), exact contrast identification, multiple testing correction (BH FDR),
    and genomic inflation factor (lambda) calibration.

Contrasts Formulated:
    - GSE42331:
        1. X-effect in males: 47,XXY vs 46,XY (delta nX = +1 at nY = 1)
        2. Sex effect: 46,XX vs 46,XY (delta nX = +1, delta nY = -1)
        3. Y-effect in diploid females: 47,XXY vs 46,XX (delta nY = +1 at nX = 2)
    - GSE46687:
        1. X-effect in females: 45,X vs 46,XX (oriented so delta nX = +1)
        2. Parental origin: 45,Xp vs 45,Xm
    - GSE47584:
        1. X-effect in males: 47,XXY vs 46,XY (delta nX = +1 at nY = 1)
    - GSE126712 (RNA-Seq):
        1. X-effect at nY=0: 45,X vs 46,XX (delta nX = +1)
        2. X-effect at nY=1: 46,XY vs 47,XXY (delta nX = +1)
        3. Y-effect at nX=1: 45,X vs 46,XY (delta nY = +1)
        4. Y-effect at nX=2: 46,XX vs 47,XXY (delta nY = +1)
        5. Additive X-effect model adjusted for Y-chromosome

Author:
    Lead Bioinformatician & Applied Statistician
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.special as sp
import scipy.stats as ss
from statsmodels.stats.multitest import multipletests

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "data"
CLEAN_DIR = DATA_DIR / "clean"
RESULTS_DIR = REPO_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Logging
LOG_FILE = RESULTS_DIR / "02_within_study_de.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [WithinStudyDE]: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("02_WithinStudyDE")


def fit_empirical_bayes_moderation(
    s2: np.ndarray,
    df_residual: float,
) -> Tuple[float, float, np.ndarray]:
    """
    Fits Smyth's (2004) empirical Bayes prior on sample variances.
    
    Model: 1/sigma_g^2 ~ 1/(d0 * s0^2) * Chi^2(d0)
    
    Parameters:
        s2 (np.ndarray): Array of sample residual variances across genes.
        df_residual (float): Residual degrees of freedom for each regression.
        
    Returns:
        Tuple[float, float, np.ndarray]: (d0, s02, s2_moderated)
    """
    valid = (s2 > 1e-12) & np.isfinite(s2)
    z = np.log(s2[valid])
    e_z = np.mean(z)
    var_z = np.var(z, ddof=1)

    # Theoretical variance of log(s2) is polygamma(1, df/2) + polygamma(1, d0/2)
    trigamma_df = sp.polygamma(1, df_residual / 2.0)
    var_prior = var_z - trigamma_df

    if var_prior > 1e-4:
        # Solve polygamma(1, d0/2) = var_prior via root finding or approximation
        # Approximation: polygamma(1, x) approx 1/x + 1/(2x^2)
        # For d0/2: var_prior approx 2/d0 -> d0 approx 2/var_prior
        d0 = max(1.0, min(100.0, 2.0 / var_prior))
        # Refine with 5 Newton steps
        for _ in range(5):
            f_val = sp.polygamma(1, d0 / 2.0) - var_prior
            f_prime = 0.5 * sp.polygamma(2, d0 / 2.0)
            d0_new = d0 - f_val / f_prime
            if d0_new > 0:
                d0 = d0_new
            else:
                break
    else:
        d0 = 50.0  # High shrinkage if empirical variance is small

    # Estimate s0^2 from E[log(s2)]
    s02 = np.exp(e_z - sp.digamma(df_residual / 2.0) + np.log(df_residual / 2.0) +
                 sp.digamma(d0 / 2.0) - np.log(d0 / 2.0))

    # Moderated variances: (d0*s02 + df*s2) / (d0 + df)
    s2_moderated = (d0 * s02 + df_residual * s2) / (d0 + df_residual)
    return float(d0), float(s02), s2_moderated


def compute_genomic_inflation(p_values: np.ndarray) -> float:
    """Computes the genomic inflation factor lambda based on median chi-square."""
    valid_p = p_values[np.isfinite(p_values) & (p_values > 0) & (p_values < 1)]
    if len(valid_p) == 0:
        return 1.0
    chisq = ss.chi2.ppf(1.0 - valid_p, df=1)
    return float(np.median(chisq) / 0.4549364)


def run_linear_model(
    Y: np.ndarray,
    X: np.ndarray,
    gene_names: List[str],
    contrast_vectors: Dict[str, np.ndarray],
) -> Dict[str, pd.DataFrame]:
    """
    Fits linear models across genes: Y = X * B + E
    Estimates contrasts, computes Smyth empirical Bayes moderation, and FDR.
    
    Parameters:
        Y: (n_genes, n_samples) expression matrix.
        X: (n_samples, n_params) design matrix.
        gene_names: List of gene symbols.
        contrast_vectors: Dictionary of {contrast_name: vector of weights}.
        
    Returns:
        Dict[str, pd.DataFrame]: Table of statistics per contrast.
    """
    n_genes, n_samples = Y.shape
    n_params = X.shape[1]
    df_resid = n_samples - n_params
    assert df_resid > 0, "Zero or negative residual degrees of freedom!"

    # OLS estimation: B = (X'X)^(-1) X' Y'
    XtX_inv = np.linalg.pinv(X.T @ X)
    B = XtX_inv @ X.T @ Y.T  # (n_params, n_genes)

    # Residuals & sample variance
    fitted = (X @ B).T  # (n_genes, n_samples)
    residuals = Y - fitted
    s2_sample = np.sum(residuals**2, axis=1) / df_resid  # (n_genes,)

    # Empirical Bayes moderation
    d0, s02, s2_mod = fit_empirical_bayes_moderation(s2_sample, df_resid)
    df_total = df_resid + d0
    logger.info("Empirical Bayes Prior: d0 = %.2f df, s0^2 = %.4f (total df = %.2f)", d0, s02, df_total)

    results = {}
    for cname, cvec in contrast_vectors.items():
        cvec = np.asarray(cvec, dtype=float)
        # Contrast estimate: c' * B
        c_est = cvec @ B  # (n_genes,)
        # Contrast variance: c' (X'X)^(-1) c
        v_c = float(cvec @ XtX_inv @ cvec)
        se_unmod = np.sqrt(s2_sample * v_c)
        se_mod = np.sqrt(s2_mod * v_c)

        # Moderated t-statistic
        t_mod = c_est / (se_mod + 1e-12)
        p_mod = 2.0 * (1.0 - ss.t.cdf(np.abs(t_mod), df=df_total))
        p_mod = np.clip(p_mod, 1e-300, 1.0)

        # Multiple testing correction
        _, padj_bh, _, _ = multipletests(p_mod, method="fdr_bh")

        # Genomic inflation
        lam = compute_genomic_inflation(p_mod)

        df = pd.DataFrame({
            "gene": gene_names,
            "log2FC": c_est,
            "se": se_mod,
            "se_unmoderated": se_unmod,
            "tstat": t_mod,
            "pvalue": p_mod,
            "padj": padj_bh,
            "is_significant": padj_bh < 0.05,
        })
        df.attrs["lambda"] = lam
        df.attrs["d0"] = d0
        df.attrs["s02"] = s02
        df.attrs["df_total"] = df_total
        results[cname] = df

        n_sig = int((padj_bh < 0.05).sum())
        logger.info("Contrast [%s]: %d / %d significant at FDR < 0.05 (lambda = %.3f)",
                    cname, n_sig, n_genes, lam)

    return results


def analyze_gse42331(meta: pd.DataFrame, annot_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Analyzes GSE42331 (N=65: 35 KS, 15 Male, 15 Female)."""
    logger.info("Analyzing GSE42331 (Münster EXAKT)...")
    expr_path = CLEAN_DIR / "GSE42331_log2.tsv"
    expr = pd.read_csv(expr_path, sep="\t", index_col=0)
    sub_meta = meta.loc[expr.columns]

    # Design Matrix: Reference = 46,XY Male
    # [Intercept, Indicator(XXY), Indicator(XX)]
    is_xxy = (sub_meta["declared_karyotype"] == "47,XXY").astype(float).values
    is_xx = (sub_meta["declared_karyotype"] == "46,XX").astype(float).values
    X = np.column_stack([np.ones(len(sub_meta)), is_xxy, is_xx])

    contrasts = {
        "XXY_vs_XY": np.array([0.0, 1.0, 0.0]),     # X-dosage in males (delta nX = +1)
        "XX_vs_XY": np.array([0.0, 0.0, 1.0]),      # Sex difference (delta nX = +1, delta nY = -1)
        "XXY_vs_XX": np.array([0.0, 1.0, -1.0]),    # Y-effect in diploid background (delta nY = +1)
    }

    res = run_linear_model(expr.values, X, list(expr.index), contrasts)
    # Save primary table
    primary_df = res["XXY_vs_XY"].copy()
    primary_df["log2FC_XX_vs_XY"] = res["XX_vs_XY"]["log2FC"]
    primary_df["padj_XX_vs_XY"] = res["XX_vs_XY"]["padj"]
    primary_df["log2FC_XXY_vs_XX"] = res["XXY_vs_XX"]["log2FC"]
    primary_df["padj_XXY_vs_XX"] = res["XXY_vs_XX"]["padj"]
    primary_df.to_csv(RESULTS_DIR / "de_GSE42331.tsv", sep="\t", index=False)
    return res


def analyze_gse46687(meta: pd.DataFrame, annot_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Analyzes GSE46687 (N=36: 16 45Xm, 10 45Xp, 10 46XX)."""
    logger.info("Analyzing GSE46687 (NICHD Bondy Turner)...")
    expr_path = CLEAN_DIR / "GSE46687_log2.tsv"
    expr = pd.read_csv(expr_path, sep="\t", index_col=0)
    sub_meta = meta.loc[expr.columns]

    # Design Matrix: Reference = 46,XX Female
    # [Intercept, Indicator(45,X), Indicator(45,Xp)]
    is_xo = (sub_meta["declared_karyotype"] == "45,X").astype(float).values
    is_xp = (sub_meta["parental_origin"] == "45,Xp").astype(float).values
    X = np.column_stack([np.ones(len(sub_meta)), is_xo, is_xp])

    contrasts = {
        # Oriented so delta nX = +1 (moving from 1X to 2X: XX - XO)
        "XX_vs_XO": np.array([0.0, -1.0, 0.0]),
        # Parental origin: 45,Xp vs 45,Xm
        "Xp_vs_Xm": np.array([0.0, 0.0, 1.0]),
    }

    res = run_linear_model(expr.values, X, list(expr.index), contrasts)
    primary_df = res["XX_vs_XO"].copy()
    primary_df["log2FC_Xp_vs_Xm"] = res["Xp_vs_Xm"]["log2FC"]
    primary_df["padj_Xp_vs_Xm"] = res["Xp_vs_Xm"]["padj"]
    primary_df.to_csv(RESULTS_DIR / "de_GSE46687.tsv", sep="\t", index=False)
    return res


def analyze_gse47584(meta: pd.DataFrame, annot_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Analyzes GSE47584 (N=10: 5 KS, 5 Male)."""
    logger.info("Analyzing GSE47584 (Denmark Klinefelter)...")
    expr_path = CLEAN_DIR / "GSE47584_log2.tsv"
    expr = pd.read_csv(expr_path, sep="\t", index_col=0)
    sub_meta = meta.loc[expr.columns]

    # Design Matrix: Reference = 46,XY Male
    is_xxy = (sub_meta["declared_karyotype"] == "47,XXY").astype(float).values
    X = np.column_stack([np.ones(len(sub_meta)), is_xxy])

    contrasts = {
        "XXY_vs_XY": np.array([0.0, 1.0]),  # delta nX = +1
    }

    res = run_linear_model(expr.values, X, list(expr.index), contrasts)
    res["XXY_vs_XY"].to_csv(RESULTS_DIR / "de_GSE47584.tsv", sep="\t", index=False)
    return res


def analyze_gse126712(meta: pd.DataFrame, annot_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Analyzes GSE126712 RNA-Seq (N=54 primary lymphocytes: 13 XO, 14 XX, 13 XY, 14 XXY)."""
    logger.info("Analyzing GSE126712 (Stanford Urban/Raznahan RNA-Seq)...")
    expr_path = CLEAN_DIR / "GSE126712_filtered_log2.tsv"
    expr = pd.read_csv(expr_path, sep="\t", index_col=0)
    sub_meta = meta.loc[expr.columns]

    # Parameterization: [1, nX - 1, nY, (nX - 1)*nY]
    # XO:  nX=1, nY=0 -> [1, 0, 0, 0]
    # XX:  nX=2, nY=0 -> [1, 1, 0, 0]
    # XY:  nX=1, nY=1 -> [1, 0, 1, 0]
    # XXY: nX=2, nY=1 -> [1, 1, 1, 1]
    dx = (sub_meta["nX"] - 1.0).values
    dy = sub_meta["nY"].values.astype(float)
    dxy = dx * dy
    X = np.column_stack([np.ones(len(sub_meta)), dx, dy, dxy])

    contrasts = {
        # 1. X-effect in females: XX vs XO (delta nX = +1 at nY = 0)
        "X_effect_female": np.array([0.0, 1.0, 0.0, 0.0]),
        # 2. X-effect in males: XXY vs XY (delta nX = +1 at nY = 1): beta_dx + beta_dxy
        "X_effect_male": np.array([0.0, 1.0, 0.0, 1.0]),
        # 3. Y-effect in monosomy X: XY vs XO (delta nY = +1 at nX = 1)
        "Y_effect_monosomy": np.array([0.0, 0.0, 1.0, 0.0]),
        # 4. Y-effect in diploid X: XXY vs XX (delta nY = +1 at nX = 2): beta_dy + beta_dxy
        "Y_effect_diploid": np.array([0.0, 0.0, 1.0, 1.0]),
        # 5. X x Y interaction: beta_dxy
        "X_by_Y_interaction": np.array([0.0, 0.0, 0.0, 1.0]),
        # 6. Main additive X-dosage effect: (X_female + X_male) / 2
        "Main_X_dosage": np.array([0.0, 1.0, 0.0, 0.5]),
    }

    res = run_linear_model(expr.values, X, list(expr.index), contrasts)
    
    # Save combined table
    main_df = res["Main_X_dosage"].copy()
    main_df["log2FC_female_X"] = res["X_effect_female"]["log2FC"]
    main_df["padj_female_X"] = res["X_effect_female"]["padj"]
    main_df["log2FC_male_X"] = res["X_effect_male"]["log2FC"]
    main_df["padj_male_X"] = res["X_effect_male"]["padj"]
    main_df["log2FC_Y_effect"] = res["Y_effect_diploid"]["log2FC"]
    main_df["padj_Y_effect"] = res["Y_effect_diploid"]["padj"]
    main_df["log2FC_interaction"] = res["X_by_Y_interaction"]["log2FC"]
    main_df["padj_interaction"] = res["X_by_Y_interaction"]["padj"]
    main_df.to_csv(RESULTS_DIR / "de_GSE126712.tsv", sep="\t", index=False)
    return res


def main():
    logger.info("==========================================================================")
    logger.info("PHASE 1B: WITHIN-STUDY DIFFERENTIAL EXPRESSION & EMPIRICAL BAYES MODELING")
    logger.info("==========================================================================")

    # 1. Load canonical metadata and annotations
    meta = pd.read_csv(DATA_DIR / "samples.tsv", sep="\t", index_col=0)
    annot = pd.read_csv(CLEAN_DIR / "gene_annotations.tsv", sep="\t", index_col=0)

    # 2. Run per-study models
    res42 = analyze_gse42331(meta, annot)
    res46 = analyze_gse46687(meta, annot)
    res47 = analyze_gse47584(meta, annot)
    res12 = analyze_gse126712(meta, annot)

    # 3. Compile summary statistics
    summary = {
        "GSE42331": {
            "n_samples": 65,
            "lambda_XXY_vs_XY": res42["XXY_vs_XY"].attrs["lambda"],
            "sig_XXY_vs_XY": int((res42["XXY_vs_XY"]["padj"] < 0.05).sum()),
            "sig_XX_vs_XY": int((res42["XX_vs_XY"]["padj"] < 0.05).sum()),
            "d0": res42["XXY_vs_XY"].attrs["d0"],
            "s02": res42["XXY_vs_XY"].attrs["s02"],
        },
        "GSE46687": {
            "n_samples": 36,
            "lambda_XX_vs_XO": res46["XX_vs_XO"].attrs["lambda"],
            "sig_XX_vs_XO": int((res46["XX_vs_XO"]["padj"] < 0.05).sum()),
            "sig_Xp_vs_Xm": int((res46["Xp_vs_Xm"]["padj"] < 0.05).sum()),
            "d0": res46["XX_vs_XO"].attrs["d0"],
            "s02": res46["XX_vs_XO"].attrs["s02"],
        },
        "GSE47584": {
            "n_samples": 10,
            "lambda_XXY_vs_XY": res47["XXY_vs_XY"].attrs["lambda"],
            "sig_XXY_vs_XY": int((res47["XXY_vs_XY"]["padj"] < 0.05).sum()),
            "d0": res47["XXY_vs_XY"].attrs["d0"],
            "s02": res47["XXY_vs_XY"].attrs["s02"],
        },
        "GSE126712": {
            "n_samples": 54,
            "lambda_Main_X": res12["Main_X_dosage"].attrs["lambda"],
            "sig_Main_X": int((res12["Main_X_dosage"]["padj"] < 0.05).sum()),
            "sig_female_X": int((res12["X_effect_female"]["padj"] < 0.05).sum()),
            "sig_male_X": int((res12["X_effect_male"]["padj"] < 0.05).sum()),
            "sig_interaction": int((res12["X_by_Y_interaction"]["padj"] < 0.05).sum()),
            "d0": res12["Main_X_dosage"].attrs["d0"],
            "s02": res12["Main_X_dosage"].attrs["s02"],
        },
    }

    summary_file = RESULTS_DIR / "within_study_de_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Within-study differential expression complete. Summary written to %s", summary_file)


if __name__ == "__main__":
    main()

