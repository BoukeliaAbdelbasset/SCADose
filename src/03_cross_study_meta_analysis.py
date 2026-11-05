#!/usr/bin/env python3
"""
SCADose - Phase 1B: Random-Effects Meta-Analysis & Independent Replication
Script: src/03_cross_study_meta_analysis.py

Description:
    Implements random-effects meta-analysis (Hartung-Knapp REML) across discovery
    microarray cohorts (GSE42331, GSE46687, GSE47584), evaluates directional sign
    concordance between Turner and Klinefelter syndromes, performs independent replication
    in GSE126712 RNA-Seq, conducts gene-class BIC model selection (flat, nX, nY, nX+nY),
    and executes leave-one-study-out sensitivity analyses.

Author:
    Lead Bioinformatician & Senior Biostatistician
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.stats as ss
from statsmodels.stats.multitest import multipletests

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
DATA_DIR = REPO_ROOT / "data"
CLEAN_DIR = DATA_DIR / "clean"
RESULTS_DIR = REPO_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Logging
LOG_FILE = RESULTS_DIR / "03_meta_analysis.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [MetaAnalysis]: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("03_MetaAnalysis")


def reml_heterogeneity(theta: np.ndarray, s2: np.ndarray, max_iter: int = 100) -> float:
    """Estimates between-study heterogeneity variance tau^2 via REML."""
    k = len(theta)
    if k <= 1:
        return 0.0

    tau2 = max(0.0, float(np.var(theta, ddof=1) - np.mean(s2)))
    for _ in range(max_iter):
        w = 1.0 / (s2 + tau2)
        sum_w = np.sum(w)
        sum_w2 = np.sum(w**2)
        theta_w = np.sum(w * theta) / sum_w

        # REML score and derivative
        # Score = -0.5 * sum(w) + 0.5 * sum(w^2 * (theta - theta_w)^2) + 0.5 * (sum_w2 / sum_w)
        q = np.sum(w**2 * (theta - theta_w)**2)
        tau2_new = max(0.0, (q - np.sum(w * (1.0 - w / sum_w) * s2)) / (np.sum(w**2 * (1.0 - 2.0 * w / sum_w + (sum_w2 / (sum_w**2)))) + 1e-12))
        
        if abs(tau2_new - tau2) < 1e-6:
            tau2 = tau2_new
            break
        tau2 = 0.5 * (tau2 + tau2_new)

    return float(tau2)


def fit_hartung_knapp_meta(
    theta: np.ndarray,
    se: np.ndarray,
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Fits random-effects meta-analysis with Hartung-Knapp standard error adjustment.
    
    Returns:
        (summary_beta, se_hk, t_hk, pval_hk, tau2, Q, I2)
    """
    k = len(theta)
    s2 = se**2

    # Fixed-effects weights for Cochran Q
    w_fe = 1.0 / (s2 + 1e-12)
    theta_fe = np.sum(w_fe * theta) / np.sum(w_fe)
    Q = float(np.sum(w_fe * (theta - theta_fe)**2))
    df_Q = k - 1
    I2 = max(0.0, float((Q - df_Q) / (Q + 1e-12))) * 100.0 if Q > df_Q else 0.0

    # Between-study variance
    tau2 = reml_heterogeneity(theta, s2)

    # Random-effects weights
    w_re = 1.0 / (s2 + tau2 + 1e-12)
    sum_w_re = np.sum(w_re)
    theta_re = float(np.sum(w_re * theta) / sum_w_re)

    # Hartung-Knapp adjustment
    if k > 1:
        q_hk = np.sum(w_re * (theta - theta_re)**2) / (k - 1)
        se_hk = float(np.sqrt(max(1.0, q_hk) / sum_w_re))
        df_hk = k - 1
        t_hk = theta_re / (se_hk + 1e-12)
        pval_hk = 2.0 * (1.0 - ss.t.cdf(abs(t_hk), df=df_hk))
    else:
        se_hk = float(np.sqrt(1.0 / sum_w_re))
        t_hk = theta_re / (se_hk + 1e-12)
        pval_hk = 2.0 * (1.0 - ss.norm.cdf(abs(t_hk)))

    pval_hk = float(np.clip(pval_hk, 1e-300, 1.0))
    return theta_re, se_hk, t_hk, pval_hk, tau2, Q, I2


def run_meta_analysis() -> Tuple[pd.DataFrame, Dict]:
    """Runs meta-analysis across discovery microarray cohorts on common genes."""
    logger.info("Loading per-study differential expression results...")
    de42 = pd.read_csv(RESULTS_DIR / "de_GSE42331.tsv", sep="\t").set_index("gene")
    de46 = pd.read_csv(RESULTS_DIR / "de_GSE46687.tsv", sep="\t").set_index("gene")
    de47 = pd.read_csv(RESULTS_DIR / "de_GSE47584.tsv", sep="\t").set_index("gene")
    annot = pd.read_csv(CLEAN_DIR / "gene_annotations.tsv", sep="\t").set_index("gene")

    # Common genes across microarrays
    common = sorted(list(de42.index.intersection(de46.index).intersection(de47.index)))
    logger.info("Evaluating meta-analysis across %d shared microarray genes...", len(common))

    meta_rows = []
    concordant_count = 0

    for gene in common:
        # Contrasts oriented so delta nX = +1:
        # GSE42331: XXY vs XY (delta nX = +1)
        # GSE46687: XX vs XO (delta nX = +1)
        # GSE47584: XXY vs XY (delta nX = +1)
        b42, se42 = de42.loc[gene, "log2FC"], de42.loc[gene, "se"]
        b46, se46 = de46.loc[gene, "log2FC"], de46.loc[gene, "se"]
        b47, se47 = de47.loc[gene, "log2FC"], de47.loc[gene, "se"]

        theta = np.array([b42, b46, b47])
        se = np.array([se42, se46, se47])

        b_meta, se_meta, t_meta, p_meta, tau2, Q, I2 = fit_hartung_knapp_meta(theta, se)

        # Directional concordance: TS direction (b46) vs KS direction (b42)
        # Both represent delta nX = +1
        sign_prod = np.sign(b46) * np.sign(b42)
        is_concordant = bool(sign_prod > 0)
        if is_concordant:
            concordant_count += 1

        chr_info = annot.loc[gene, "chromosome"] if gene in annot.index else "Unknown"
        xci_info = annot.loc[gene, "xci_status"] if gene in annot.index else "Unknown"

        meta_rows.append({
            "gene": gene,
            "chromosome": chr_info,
            "xci_status": xci_info,
            "beta_meta": b_meta,
            "se_meta": se_meta,
            "tstat_meta": t_meta,
            "pvalue_meta": p_meta,
            "tau2": tau2,
            "cochran_Q": Q,
            "I2": I2,
            "beta_GSE42331_KS": b42,
            "se_GSE42331_KS": se42,
            "pval_GSE42331_KS": de42.loc[gene, "pvalue"],
            "beta_GSE46687_TS": b46,
            "se_GSE46687_TS": se46,
            "pval_GSE46687_TS": de46.loc[gene, "pvalue"],
            "beta_GSE47584_KS": b47,
            "se_GSE47584_KS": se47,
            "pval_GSE47584_KS": de47.loc[gene, "pvalue"],
            "is_sign_concordant_TS_KS": is_concordant,
        })

    meta_df = pd.DataFrame(meta_rows).set_index("gene")

    # 1. Hartung-Knapp p-values (conservative)
    p_meta_hk = meta_df["pvalue_meta"].values
    _, padj_all_hk, _, _ = multipletests(p_meta_hk, method="fdr_bh")

    # 2. Standard Random-Effects Z-score p-values
    z_stat = meta_df["beta_meta"].values / (meta_df["se_meta"].values + 1e-12)
    p_meta_z = 2.0 * (1.0 - ss.norm.cdf(np.abs(z_stat)))
    p_meta_z = np.clip(p_meta_z, 1e-300, 1.0)
    _, padj_all_z, _, _ = multipletests(p_meta_z, method="fdr_bh")

    meta_df["pvalue_meta_hk"] = p_meta_hk
    meta_df["pvalue_z"] = p_meta_z
    meta_df["padj_meta_hk"] = padj_all_hk
    meta_df["padj_meta_z"] = padj_all_z
    meta_df["padj_meta"] = padj_all_z  # primary standard random-effects FDR
    meta_df["is_meta_significant"] = meta_df["padj_meta_z"] < 0.05

    # Family-wise FDR for X-linked and PAR genes
    x_par_mask = meta_df["chromosome"].isin(["chrX", "PAR1", "PAR2"])
    if x_par_mask.sum() > 0:
        _, padj_x, _, _ = multipletests(meta_df.loc[x_par_mask, "pvalue_z"], method="fdr_bh")
        meta_df.loc[x_par_mask, "padj_family_X_PAR"] = padj_x
    else:
        meta_df["padj_family_X_PAR"] = np.nan

    # True X-Dosage Concordant hits:
    # 1. padj_meta_z < 0.05
    # 2. is_sign_concordant_TS_KS == True
    # 3. Nominal p < 0.05 in both GSE42331 and GSE46687
    meta_df["is_true_dosage_concordant"] = (
        (meta_df["padj_meta_z"] < 0.05) &
        (meta_df["is_sign_concordant_TS_KS"]) &
        (meta_df["pval_GSE42331_KS"] < 0.05) &
        (meta_df["pval_GSE46687_TS"] < 0.05)
    )

    # Binomial sign test across all genes
    binom_p = ss.binomtest(concordant_count, len(common), p=0.5).pvalue

    n_sig_meta = int((meta_df["padj_meta_z"] < 0.05).sum())
    n_concordant_hits = int(meta_df["is_true_dosage_concordant"].sum())
    logger.info("Meta-Analysis: %d / %d significant at FDR < 0.05 (Z-test), %d under HK.",
                n_sig_meta, len(common), int((padj_all_hk < 0.05).sum()))
    logger.info("True Cross-Condition Concordant Dosage Hits: %d genes.", n_concordant_hits)
    logger.info("Genome-wide Sign Concordance: %d / %d (%.2f%%, binomial p = %.2e)",
                concordant_count, len(common), concordant_count / len(common) * 100, binom_p)

    meta_df.to_csv(RESULTS_DIR / "meta_analysis_results.tsv", sep="\t")

    metrics = {
        "n_genes_tested": len(common),
        "n_fdr_significant": n_sig_meta,
        "n_true_dosage_concordant": n_concordant_hits,
        "concordant_genes": meta_df[meta_df["is_true_dosage_concordant"]].index.tolist(),
        "sign_concordance_rate": float(concordant_count / len(common)),
        "binomial_sign_test_p": float(binom_p),
    }
    return meta_df, metrics


def evaluate_rna_seq_replication(meta_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Evaluates discovery meta-analysis hits in GSE126712 RNA-Seq replication cohort."""
    logger.info("Evaluating independent replication in GSE126712 RNA-Seq cohort...")
    de12 = pd.read_csv(RESULTS_DIR / "de_GSE126712.tsv", sep="\t").set_index("gene")

    rep_genes = [g for g in meta_df.index if g in de12.index]
    logger.info("Found %d genes shared between microarray meta-analysis and GSE126712 RNA-Seq.", len(rep_genes))

    rep_df = pd.DataFrame(index=rep_genes)
    rep_df["beta_meta_microarray"] = meta_df.loc[rep_genes, "beta_meta"]
    rep_df["padj_meta_microarray"] = meta_df.loc[rep_genes, "padj_meta"]
    rep_df["is_meta_sig"] = meta_df.loc[rep_genes, "is_meta_significant"]
    rep_df["is_true_concordant"] = meta_df.loc[rep_genes, "is_true_dosage_concordant"]
    rep_df["chromosome"] = meta_df.loc[rep_genes, "chromosome"]

    rep_df["beta_rna_seq"] = de12.loc[rep_genes, "log2FC"]
    rep_df["se_rna_seq"] = de12.loc[rep_genes, "se"]
    rep_df["pval_rna_seq"] = de12.loc[rep_genes, "pvalue"]
    rep_df["padj_rna_seq"] = de12.loc[rep_genes, "padj"]
    rep_df["is_rna_seq_sig"] = de12.loc[rep_genes, "is_significant"]

    # Correlation across all shared genes
    r_all, p_r_all = ss.pearsonr(rep_df["beta_meta_microarray"], rep_df["beta_rna_seq"])
    rho_all, p_rho_all = ss.spearmanr(rep_df["beta_meta_microarray"], rep_df["beta_rna_seq"])

    # Correlation among significant meta hits
    sig_sub = rep_df[rep_df["is_meta_sig"]]
    if len(sig_sub) > 2:
        r_sig, p_r_sig = ss.pearsonr(sig_sub["beta_meta_microarray"], sig_sub["beta_rna_seq"])
        sign_agree_sig = float((np.sign(sig_sub["beta_meta_microarray"]) == np.sign(sig_sub["beta_rna_seq"])).mean())
    else:
        r_sig, p_r_sig, sign_agree_sig = 0.0, 1.0, 0.0

    # Storey's pi1 estimate of true positive replication
    # pi0 = min(1.0, 2.0 * mean(p > 0.5)); pi1 = 1 - pi0
    sig_pvals = sig_sub["pval_rna_seq"].values
    if len(sig_pvals) > 0:
        pi0 = min(1.0, 2.0 * float(np.mean(sig_pvals > 0.5)))
        pi1 = max(0.0, 1.0 - pi0)
    else:
        pi1 = 0.0

    logger.info("Replication Effect Size Correlation (all genes): r = %.3f (p = %.2e), rho = %.3f",
                r_all, p_r_all, rho_all)
    logger.info("Replication Effect Size Correlation (meta hits): r = %.3f (p = %.2e), sign agreement = %.2f%%",
                r_sig, p_r_sig, sign_agree_sig * 100)
    logger.info("Storey's pi1 Replication Rate: %.2f%%", pi1 * 100)

    rep_df.to_csv(RESULTS_DIR / "replication_gse126712.tsv", sep="\t")

    metrics = {
        "n_shared_genes": len(rep_genes),
        "r_all_genes": float(r_all),
        "rho_all_genes": float(rho_all),
        "r_meta_significant_genes": float(r_sig),
        "sign_agreement_meta_significant": float(sign_agree_sig),
        "storey_pi1_replication_rate": float(pi1),
    }
    return rep_df, metrics


def fit_gene_class_model_selection() -> pd.DataFrame:
    """
    Fits 4 competing models for every X-linked and PAR gene across all 165 samples:
      M_flat: y = a + Study + e
      M_X:    y = a + b_X * nX + Study + e
      M_Y:    y = a + b_Y * nY + Study + e
      M_PAR:  y = a + b_PAR * (nX + nY) + Study + e
    Selects optimal model by BIC.
    """
    logger.info("Performing gene-class model selection (flat vs nX vs nY vs nX+nY) via BIC...")
    samples = pd.read_csv(DATA_DIR / "samples.tsv", sep="\t", index_col=0)
    annot = pd.read_csv(CLEAN_DIR / "gene_annotations.tsv", sep="\t", index_col=0)

    x_par_genes = annot[annot["chromosome"].isin(["chrX", "PAR1", "PAR2"])].index.tolist()

    # Ingest clean expression matrices
    m42 = pd.read_csv(CLEAN_DIR / "GSE42331_log2.tsv", sep="\t", index_col=0)
    m46 = pd.read_csv(CLEAN_DIR / "GSE46687_log2.tsv", sep="\t", index_col=0)
    m47 = pd.read_csv(CLEAN_DIR / "GSE47584_log2.tsv", sep="\t", index_col=0)
    m12 = pd.read_csv(CLEAN_DIR / "GSE126712_common_all_log2.tsv", sep="\t", index_col=0)

    shared_x = sorted(list(set(x_par_genes).intersection(m42.index).intersection(m46.index).intersection(m47.index).intersection(m12.index)))
    logger.info("Modeling %d X/PAR genes present across all 4 cohorts...", len(shared_x))

    # Combine expression and metadata across all 165 samples
    expr_all = pd.concat([m42.loc[shared_x], m46.loc[shared_x], m47.loc[shared_x], m12.loc[shared_x]], axis=1)
    sub_samples = samples.loc[expr_all.columns]

    n_obs = len(sub_samples)
    study_dummies = pd.get_dummies(sub_samples["study_id"], drop_first=True, dtype=float).values
    nx = sub_samples["nX"].values.astype(float)
    ny = sub_samples["nY"].values.astype(float)
    n_tot = nx + ny

    # Design matrices
    X_flat = np.column_stack([np.ones(n_obs), study_dummies])                       # k = 1 + 3 = 4
    X_X = np.column_stack([np.ones(n_obs), nx, study_dummies])                      # k = 5
    X_Y = np.column_stack([np.ones(n_obs), ny, study_dummies])                      # k = 5
    X_PAR = np.column_stack([np.ones(n_obs), n_tot, study_dummies])                # k = 5

    def compute_bic(X: np.ndarray, y: np.ndarray) -> float:
        k = X.shape[1]
        b = np.linalg.pinv(X.T @ X) @ X.T @ y
        rss = np.sum((y - X @ b)**2)
        s2 = rss / n_obs
        log_lik = -0.5 * n_obs * (np.log(2.0 * np.pi * s2) + 1.0)
        return float(k * np.log(n_obs) - 2.0 * log_lik)

    model_records = []
    for gene in shared_x:
        y = expr_all.loc[gene].values.astype(float)
        bic_flat = compute_bic(X_flat, y)
        bic_X = compute_bic(X_X, y)
        bic_Y = compute_bic(X_Y, y)
        bic_par = compute_bic(X_PAR, y)

        bics = {"Flat": bic_flat, "X_Dosage": bic_X, "Y_Linked": bic_Y, "PAR_Sum": bic_par}
        best_model = min(bics, key=bics.get)
        delta_bic = bic_flat - bics[best_model]

        model_records.append({
            "gene": gene,
            "chromosome": annot.loc[gene, "chromosome"],
            "xci_status": annot.loc[gene, "xci_status"],
            "best_model": best_model,
            "BIC_Flat": bic_flat,
            "BIC_X_Dosage": bic_X,
            "BIC_Y_Linked": bic_Y,
            "BIC_PAR_Sum": bic_par,
            "delta_BIC_vs_Flat": delta_bic,
        })

    model_df = pd.DataFrame(model_records).set_index("gene")
    model_df.to_csv(RESULTS_DIR / "gene_class_model_selection.tsv", sep="\t")

    model_counts = model_df["best_model"].value_counts().to_dict()
    logger.info("Gene-Class Model Selection Results: %s", model_counts)
    return model_df


def run_joint_study_stratified_model() -> pd.DataFrame:
    """
    Fits joint study-stratified linear model across all 111 discovery microarray samples:
      y_gi = alpha_g + beta_X * (nX - 1) + gamma_46 * Study_46 + gamma_47 * Study_47 + e_gi
    Degrees of freedom = 111 - 4 = 107.
    """
    logger.info("Fitting Joint Study-Stratified Model across all 111 discovery microarray samples...")
    samples = pd.read_csv(DATA_DIR / "samples.tsv", sep="\t", index_col=0)
    m42 = pd.read_csv(CLEAN_DIR / "GSE42331_common_array_log2.tsv", sep="\t", index_col=0)
    m46 = pd.read_csv(CLEAN_DIR / "GSE46687_common_array_log2.tsv", sep="\t", index_col=0)
    m47 = pd.read_csv(CLEAN_DIR / "GSE47584_common_array_log2.tsv", sep="\t", index_col=0)
    annot = pd.read_csv(CLEAN_DIR / "gene_annotations.tsv", sep="\t", index_col=0)

    expr_comb = pd.concat([m42, m46, m47], axis=1)
    meta_comb = samples.loc[expr_comb.columns]

    dx = (meta_comb["nX"] - 1.0).values
    s46 = (meta_comb["study_id"] == "GSE46687").astype(float).values
    s47 = (meta_comb["study_id"] == "GSE47584").astype(float).values
    X = np.column_stack([np.ones(len(meta_comb)), dx, s46, s47])

    n_obs = len(meta_comb)
    df_resid = n_obs - 4
    XtX_inv = np.linalg.pinv(X.T @ X)

    Y = expr_comb.values
    B = XtX_inv @ X.T @ Y.T
    residuals = Y - (X @ B).T
    s2 = np.sum(residuals**2, axis=1) / df_resid

    # Empirical Bayes moderation
    import importlib
    de_mod = importlib.import_module("02_within_study_differential_expression")
    d0, s02, s2_mod = de_mod.fit_empirical_bayes_moderation(s2, df_resid)
    df_total = df_resid + d0

    se_dx = np.sqrt(s2_mod * XtX_inv[1, 1])
    t_dx = B[1, :] / (se_dx + 1e-12)
    p_dx = 2.0 * (1.0 - ss.t.cdf(np.abs(t_dx), df=df_total))
    p_dx = np.clip(p_dx, 1e-300, 1.0)
    _, padj_dx, _, _ = multipletests(p_dx, method="fdr_bh")

    genes = expr_comb.index.tolist()
    chrs = [annot.loc[g, "chromosome"] if g in annot.index else "Unknown" for g in genes]
    xcis = [annot.loc[g, "xci_status"] if g in annot.index else "Unknown" for g in genes]

    res_df = pd.DataFrame({
        "gene": genes,
        "chromosome": chrs,
        "xci_status": xcis,
        "beta_joint_X": B[1, :],
        "se_joint_X": se_dx,
        "tstat_joint_X": t_dx,
        "pvalue_joint_X": p_dx,
        "padj_joint_X": padj_dx,
        "is_joint_sig": padj_dx < 0.05,
    }).set_index("gene")

    res_df.to_csv(RESULTS_DIR / "joint_study_stratified_model.tsv", sep="\t")
    n_sig = int((padj_dx < 0.05).sum())
    logger.info("Joint Study-Stratified Model: %d / %d significant at FDR < 0.05 (df = %.1f)",
                n_sig, len(genes), df_total)
    return res_df


def run_loso_meta_analysis(meta_df: pd.DataFrame) -> pd.DataFrame:
    """Performs Leave-One-Study-Out (LOSO) meta-analysis for sensitivity."""
    logger.info("Executing Leave-One-Study-Out (LOSO) meta-analysis sensitivity...")
    studies = ["GSE42331", "GSE46687", "GSE47584"]
    
    loso_records = []
    for gene, row in meta_df.iterrows():
        b_vec = np.array([row["beta_GSE42331_KS"], row["beta_GSE46687_TS"], row["beta_GSE47584_KS"]])
        se_vec = np.array([row["se_GSE42331_KS"], row["se_GSE46687_TS"], row["se_GSE47584_KS"]])

        rec = {"gene": gene, "beta_full_meta": row["beta_meta"], "padj_full_meta": row["padj_meta"]}
        for i, omit_study in enumerate(studies):
            idx_keep = [j for j in range(3) if j != i]
            b_loso, se_loso, t_loso, p_loso, _, _, _ = fit_hartung_knapp_meta(b_vec[idx_keep], se_vec[idx_keep])
            rec[f"beta_omit_{omit_study}"] = b_loso
            rec[f"pval_omit_{omit_study}"] = p_loso

        loso_records.append(rec)

    loso_df = pd.DataFrame(loso_records).set_index("gene")
    loso_df.to_csv(RESULTS_DIR / "loso_meta_analysis.tsv", sep="\t")
    logger.info("LOSO meta-analysis complete. Results saved to results/loso_meta_analysis.tsv")
    return loso_df
    """Performs Leave-One-Study-Out (LOSO) meta-analysis for sensitivity."""
    logger.info("Executing Leave-One-Study-Out (LOSO) meta-analysis sensitivity...")
    studies = ["GSE42331", "GSE46687", "GSE47584"]
    
    loso_records = []
    for gene, row in meta_df.iterrows():
        b_vec = np.array([row["beta_GSE42331_KS"], row["beta_GSE46687_TS"], row["beta_GSE47584_KS"]])
        se_vec = np.array([row["se_GSE42331_KS"], row["se_GSE46687_TS"], row["se_GSE47584_KS"]])

        rec = {"gene": gene, "beta_full_meta": row["beta_meta"], "padj_full_meta": row["padj_meta"]}
        for i, omit_study in enumerate(studies):
            idx_keep = [j for j in range(3) if j != i]
            b_loso, se_loso, t_loso, p_loso, _, _, _ = fit_hartung_knapp_meta(b_vec[idx_keep], se_vec[idx_keep])
            rec[f"beta_omit_{omit_study}"] = b_loso
            rec[f"pval_omit_{omit_study}"] = p_loso

        loso_records.append(rec)

    loso_df = pd.DataFrame(loso_records).set_index("gene")
    loso_df.to_csv(RESULTS_DIR / "loso_meta_analysis.tsv", sep="\t")
    logger.info("LOSO meta-analysis complete. Results saved to results/loso_meta_analysis.tsv")
    return loso_df


def main():
    logger.info("==========================================================================")
    logger.info("PHASE 1B: RANDOM-EFFECTS META-ANALYSIS & INDEPENDENT REPLICATION")
    logger.info("==========================================================================")

    # 1. Meta-analysis across discovery cohorts
    meta_df, meta_metrics = run_meta_analysis()

    # 2. Independent replication in GSE126712 RNA-Seq
    rep_df, rep_metrics = evaluate_rna_seq_replication(meta_df)

    # 3. Gene-class model selection via BIC
    model_df = fit_gene_class_model_selection()

    # 4. Joint study-stratified model
    joint_df = run_joint_study_stratified_model()

    # 5. Leave-one-study-out sensitivity
    loso_df = run_loso_meta_analysis(meta_df)

    # 6. Save master summary JSON
    summary = {
        "discovery_meta_analysis": meta_metrics,
        "joint_study_stratified_model": {
            "n_samples": 111,
            "n_fdr_significant": int(joint_df["is_joint_sig"].sum()),
        },
        "replication_gse126712": rep_metrics,
        "gene_class_model_counts": model_df["best_model"].value_counts().to_dict(),
        "key_genes_summary": {},
    }

    key_genes = ["SLC25A6", "KDM6A", "CD99", "ASMTL", "P2RY8", "EIF1AX", "ZFX", "CLEC4E", "E2F2", "ANXA1"]
    for g in key_genes:
        if g in meta_df.index:
            r_meta = meta_df.loc[g]
            r_rep = rep_df.loc[g] if g in rep_df.index else {}
            r_model = model_df.loc[g] if g in model_df.index else {}
            summary["key_genes_summary"][g] = {
                "chromosome": str(r_meta.get("chromosome", "")),
                "beta_meta": float(r_meta.get("beta_meta", 0.0)),
                "se_meta": float(r_meta.get("se_meta", 0.0)),
                "pvalue_meta": float(r_meta.get("pvalue_meta", 1.0)),
                "padj_meta": float(r_meta.get("padj_meta", 1.0)),
                "is_true_dosage_concordant": bool(r_meta.get("is_true_dosage_concordant", False)),
                "beta_rna_seq": float(r_rep.get("beta_rna_seq", 0.0)) if len(r_rep) > 0 else None,
                "padj_rna_seq": float(r_rep.get("padj_rna_seq", 1.0)) if len(r_rep) > 0 else None,
                "best_model_bic": str(r_model.get("best_model", "Unknown")),
            }

    with open(RESULTS_DIR / "meta_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Meta-analysis & replication pipeline completed successfully.")


if __name__ == "__main__":
    main()
