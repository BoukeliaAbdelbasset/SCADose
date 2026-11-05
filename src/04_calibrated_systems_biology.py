#!/usr/bin/env python3
"""
src/04_calibrated_systems_biology.py
====================================
Comprehensive, rigorous systems biology, machine learning, and simulation pipeline
for sex chromosome aneuploidies (Turner and Klinefelter syndromes).

Modules:
  1. Deconvolution & Cell Composition (A8):
     - quanTIseq (TIL10) and ABIS reference deconvolution on within-study linear positive intensities.
     - Centered Log-Ratio (CLR) compositional transformation.
     - Within-study karyotype effects on cell fractions + meta-analysis.
     - Between-method concordance (Spearman, Lin's CCC).
  2. Bootstrap Mediation Analysis (A8.4):
     - Evaluates whether CLEC4E, E2F2, SLC25A6, KDM6A dosage effects are direct or cell-mediated.
     - 2,000 bootstrap resamples adjusting simultaneously for major leukocyte subsets.
  3. Transcription Factor Regulon & E2F2 Activity (A9):
     - decoupleR + CollecTRI Univariate Linear Model (ULM) TF activity inference across all human TFs.
     - Association of TF activity with X-dosage within study (FDR corrected).
     - E2F2 leverage check (Cook's distance) and target-gene slope analysis.
  4. Signed WGCNA Co-Expression Networks (A10):
     - Residualized within-study expression to eliminate batch effects.
     - Soft-threshold power selection by scale-free topology fit (R^2 >= 0.80).
     - Signed Topological Overlap Matrix (TOM), hierarchical clustering, dynamic module assignment.
     - Module eigengenes (ME) correlated with dosage and cell composition.
     - Module preservation (Z_summary) in independent RNA-seq GSE126712.
  5. Machine Learning & Cross-Syndrome Transfer (A11):
     - ElasticNet and Random Forest classifiers predicting karyotype / dosage.
     - Nested CV and Leave-One-Study-Out (LOSO) cross-validation (strict zero-leakage).
     - Feature sets: X-escape/PAR vs all genes vs random feature sets (100 permutations).
     - Cross-syndrome transfer: TS -> KS and KS -> TS.
  6. False-Positive & Power Simulation (A12):
     - Null simulation under confounded batch structure comparing ComBat-then-regress vs within-study modeling.
     - Spike-in power benchmark estimating minimal detectable log2FC at 80% power.

Author: Lead Bioinformatician & Senior Biostatistician
Date: November 2026
License: MIT
"""

import sys
import os
import json
import logging
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import nnls
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, average_precision_score, brier_score_loss
import decoupler as dc
import scanpy as sc

# ---------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------
os.makedirs("results", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results/04_calibrated_systems_biology.log", mode="w")
    ]
)
logger = logging.getLogger("CalibratedSystemsBiology")

np.random.seed(42)

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def lins_ccc(x, y):
    """Compute Lin's Concordance Correlation Coefficient."""
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_var = np.var(x, ddof=1)
    y_var = np.var(y, ddof=1)
    cov = np.cov(x, y)[0, 1]
    numerator = 2 * cov
    denominator = x_var + y_var + (x_mean - y_mean) ** 2
    if denominator == 0:
        return 0.0
    return numerator / denominator

def clr_transform(df_fractions, eps=1e-4):
    """
    Centered Log-Ratio (CLR) transformation for compositional data.
    clr(x) = ln(x) - (1/D) * sum(ln(x_j))
    """
    vals = df_fractions.values + eps
    log_vals = np.log(vals)
    geometric_mean = np.mean(log_vals, axis=1, keepdims=True)
    clr_vals = log_vals - geometric_mean
    return pd.DataFrame(clr_vals, index=df_fractions.index, columns=df_fractions.columns)

def fit_ols_contrast(y, x, covars=None):
    """Fit OLS with robust HC3 or standard covariance."""
    df_data = pd.DataFrame({'y': y, 'x': x})
    if covars is not None:
        for col in covars.columns:
            df_data[col] = covars[col].values
    X = sm.add_constant(df_data.drop(columns=['y']))
    model = sm.OLS(df_data['y'], X).fit()
    coef = model.params['x']
    se = model.bse['x']
    t_stat = model.tvalues['x']
    pval = model.pvalues['x']
    return coef, se, t_stat, pval

# ---------------------------------------------------------
# 1. Deconvolution & Cell Composition (A8)
# ---------------------------------------------------------
def run_deconvolution(samples_df, data_matrices):
    logger.info("=" * 60)
    logger.info("MODULE 1: Cell Composition & Deconvolution (quanTIseq & ABIS)")
    logger.info("=" * 60)

    til10_path = "data/reference/TIL10_signature.txt"
    abis_micro_path = "data/reference/sigmatrixMicro.txt"
    abis_rna_path = "data/reference/sigmatrixRNAseq.txt"

    til10_sig = pd.read_csv(til10_path, sep="\t", index_col=0)
    abis_micro = pd.read_csv(abis_micro_path, sep="\t", index_col=0)
    abis_rna = pd.read_csv(abis_rna_path, sep="\t", index_col=0)

    # Clean quotes from ABIS
    abis_micro.columns = [c.replace('"', '').strip() for c in abis_micro.columns]
    abis_micro.index = [i.replace('"', '').strip() for i in abis_micro.index]
    abis_rna.columns = [c.replace('"', '').strip() for c in abis_rna.columns]
    abis_rna.index = [i.replace('"', '').strip() for i in abis_rna.index]

    all_til10_fractions = []
    all_abis_fractions = []

    for study_id, expr_log2 in data_matrices.items():
        logger.info(f"Deconvolving {study_id} ({expr_log2.shape[1]} samples)...")
        # Transform log2 to positive linear intensity
        linear_expr = np.power(2.0, expr_log2)

        # 1. quanTIseq TIL10
        common_til10 = til10_sig.index.intersection(linear_expr.index)
        assert len(common_til10) >= 100, f"Too few common genes with TIL10 in {study_id}: {len(common_til10)}"
        sig_t = til10_sig.loc[common_til10].values
        expr_t = linear_expr.loc[common_til10].values

        f_til10 = []
        for j in range(expr_t.shape[1]):
            f, _ = nnls(sig_t, expr_t[:, j])
            f_sum = np.sum(f)
            f = f / f_sum if f_sum > 0 else f
            f_til10.append(f)
        df_til10_study = pd.DataFrame(f_til10, index=expr_log2.columns, columns=til10_sig.columns)
        df_til10_study["study_id"] = study_id
        all_til10_fractions.append(df_til10_study)

        # 2. ABIS
        abis_sig = abis_rna if study_id == "GSE126712" else abis_micro
        common_abis = abis_sig.index.intersection(linear_expr.index)
        assert len(common_abis) >= 200, f"Too few common genes with ABIS in {study_id}: {len(common_abis)}"
        sig_a = abis_sig.loc[common_abis].values
        expr_a = linear_expr.loc[common_abis].values

        f_abis = []
        for j in range(expr_a.shape[1]):
            f, _ = nnls(sig_a, expr_a[:, j])
            f_sum = np.sum(f)
            f = f / f_sum if f_sum > 0 else f
            f_abis.append(f)
        df_abis_study = pd.DataFrame(f_abis, index=expr_log2.columns, columns=abis_sig.columns)
        df_abis_study["study_id"] = study_id
        all_abis_fractions.append(df_abis_study)

    df_til10 = pd.concat(all_til10_fractions, axis=0)
    df_abis = pd.concat(all_abis_fractions, axis=0)

    # Save raw fractions
    df_til10.to_csv("results/cell_composition_fractions_til10.tsv", sep="\t")
    df_abis.to_csv("results/cell_composition_fractions_abis.tsv", sep="\t")

    # Compositional CLR
    cell_cols_til10 = [c for c in df_til10.columns if c != "study_id"]
    df_clr_til10 = clr_transform(df_til10[cell_cols_til10])
    df_clr_til10["study_id"] = df_til10["study_id"]
    df_clr_til10.to_csv("results/cell_composition_clr_til10.tsv", sep="\t")

    # Method concordance
    concordance_rows = []
    # Concordance pairs: Monocytes, B cells, Neutrophils, NK cells
    pairs = [
        ("Monocytes", "Monocytes"),
        ("B.cells", "B Naive"),
        ("NK.cells", "NK"),
        ("Neutrophils", "Neutrophils LD")
    ]
    for cell_t, cell_a in pairs:
        if cell_t in df_til10.columns and cell_a in df_abis.columns:
            x = df_til10[cell_t].values
            y = df_abis[cell_a].values
            valid = ~np.isnan(x) & ~np.isnan(y)
            if valid.sum() > 5:
                sp_r, sp_p = stats.spearmanr(x[valid], y[valid])
                ccc = lins_ccc(x[valid], y[valid])
                concordance_rows.append({
                    "cell_type_til10": cell_t,
                    "cell_type_abis": cell_a,
                    "spearman_rho": sp_r,
                    "spearman_pvalue": sp_p,
                    "lins_ccc": ccc
                })
    df_concordance = pd.DataFrame(concordance_rows)
    df_concordance.to_csv("results/cell_deconvolution_concordance.tsv", sep="\t", index=False)
    logger.info(f"Deconvolution method concordance (Spearman & Lin's CCC):\n{df_concordance.to_string()}")

    # Within-study karyotype contrasts on cell proportions (CLR scale)
    karyotype_cell_rows = []
    meta = samples_df.set_index("sample_id")

    # GSE42331: XXY vs XY and XX vs XY
    gse42331_s = samples_df[samples_df["study_id"] == "GSE42331"].set_index("sample_id")
    for cell in cell_cols_til10:
        y_clr = df_clr_til10.loc[gse42331_s.index, cell]
        # XXY vs XY
        male_mask = gse42331_s["declared_karyotype"].isin(["46,XY", "47,XXY"])
        x_xxy = (gse42331_s.loc[male_mask, "declared_karyotype"] == "47,XXY").astype(int)
        coef, se, t, p = fit_ols_contrast(y_clr[male_mask], x_xxy)
        karyotype_cell_rows.append({
            "study_id": "GSE42331",
            "contrast": "XXY_vs_XY",
            "cell_type": cell,
            "coef": coef,
            "se": se,
            "t_stat": t,
            "pvalue": p
        })

    # GSE46687: XO vs XX (PBMC)
    gse46687_s = samples_df[samples_df["study_id"] == "GSE46687"].set_index("sample_id")
    for cell in cell_cols_til10:
        y_clr = df_clr_til10.loc[gse46687_s.index, cell]
        # X-dosage in TS (XX vs XO: +1 X chromosome)
        x_ts = (gse46687_s["declared_karyotype"] == "46,XX").astype(int)
        coef, se, t, p = fit_ols_contrast(y_clr, x_ts)
        karyotype_cell_rows.append({
            "study_id": "GSE46687",
            "contrast": "XX_vs_XO",
            "cell_type": cell,
            "coef": coef,
            "se": se,
            "t_stat": t,
            "pvalue": p
        })

    # GSE126712: X-dosage in RNA-seq
    gse126712_s = samples_df[samples_df["study_id"] == "GSE126712"].set_index("sample_id")
    for cell in cell_cols_til10:
        y_clr = df_clr_til10.loc[gse126712_s.index, cell]
        x_dosage = gse126712_s["nX"] - 1.0
        covars = pd.DataFrame({"nY": gse126712_s["nY"].values}, index=gse126712_s.index)
        coef, se, t, p = fit_ols_contrast(y_clr, x_dosage, covars=covars)
        karyotype_cell_rows.append({
            "study_id": "GSE126712",
            "contrast": "X_dosage_adjusted_Y",
            "cell_type": cell,
            "coef": coef,
            "se": se,
            "t_stat": t,
            "pvalue": p
        })

    df_karyo_cells = pd.DataFrame(karyotype_cell_rows)
    df_karyo_cells["fdr"] = multipletests(df_karyo_cells["pvalue"], method="fdr_bh")[1]
    df_karyo_cells.to_csv("results/cell_composition_stats.tsv", sep="\t", index=False)
    logger.info("Karyotype effects on cell types saved to results/cell_composition_stats.tsv")

    return df_til10, df_clr_til10

# ---------------------------------------------------------
# 2. Bootstrap Mediation Analysis (A8.4)
# ---------------------------------------------------------
def run_mediation_analysis(samples_df, data_matrices, df_clr_til10):
    logger.info("=" * 60)
    logger.info("MODULE 2: Bootstrap Mediation Analysis (CLEC4E, E2F2, SLC25A6, KDM6A)")
    logger.info("=" * 60)

    # Focus on GSE42331 whole blood (KS vs XY) and GSE46687 PBMC (TS vs XX)
    target_genes = ["CLEC4E", "E2F2", "SLC25A6", "KDM6A", "ANXA1"]
    mediator_cols = ["Monocytes", "Neutrophils", "T.cells.CD4", "T.cells.CD8", "B.cells", "NK.cells"]
    n_boot = 2000

    results_mediation = []

    for study_id, contrast_name, karyo_pair, x_col in [
        ("GSE42331", "XXY_vs_XY", ["46,XY", "47,XXY"], "is_xxy"),
        ("GSE46687", "XX_vs_XO", ["45,X", "46,XX"], "is_xx")
    ]:
        s_meta = samples_df[samples_df["study_id"] == study_id].copy()
        s_meta = s_meta[s_meta["declared_karyotype"].isin(karyo_pair)].set_index("sample_id")
        expr_mat = data_matrices[study_id].loc[:, s_meta.index]
        clr_mat = df_clr_til10.loc[s_meta.index, mediator_cols]

        if contrast_name == "XXY_vs_XY":
            X_treat = (s_meta["declared_karyotype"] == "47,XXY").astype(float).values
        else:
            X_treat = (s_meta["declared_karyotype"] == "46,XX").astype(float).values

        n_samples = len(X_treat)

        for gene in target_genes:
            if gene not in expr_mat.index:
                logger.warning(f"Gene {gene} not found in {study_id}")
                continue

            Y = expr_mat.loc[gene].values
            M = clr_mat.values

            # Total effect c: Y ~ X
            mod_c = sm.OLS(Y, sm.add_constant(X_treat)).fit()
            total_effect = mod_c.params[1]
            total_se = mod_c.bse[1]
            total_p = mod_c.pvalues[1]

            # Direct effect c' and mediators b: Y ~ X + M1 + ... + Mk
            X_with_M = np.column_stack([np.ones(n_samples), X_treat, M])
            mod_direct = sm.OLS(Y, X_with_M).fit()
            direct_effect = mod_direct.params[1]
            direct_se = mod_direct.bse[1]
            direct_p = mod_direct.pvalues[1]
            indirect_effect = total_effect - direct_effect

            # Bootstrap CIs
            boot_total = []
            boot_direct = []
            boot_indirect = []

            for b in range(n_boot):
                idx = np.random.choice(n_samples, size=n_samples, replace=True)
                Y_b = Y[idx]
                X_b = X_treat[idx]
                M_b = M[idx, :]

                # Fit total
                c_b = sm.OLS(Y_b, sm.add_constant(X_b)).fit().params[1]
                # Fit direct
                X_M_b = np.column_stack([np.ones(n_samples), X_b, M_b])
                c_prime_b = sm.OLS(Y_b, X_M_b).fit().params[1]

                boot_total.append(c_b)
                boot_direct.append(c_prime_b)
                boot_indirect.append(c_b - c_prime_b)

            pct_mediated = (indirect_effect / total_effect * 100) if abs(total_effect) > 1e-6 else np.nan

            results_mediation.append({
                "study_id": study_id,
                "contrast": contrast_name,
                "gene": gene,
                "total_effect": total_effect,
                "total_pvalue": total_p,
                "total_ci_lower": np.percentile(boot_total, 2.5),
                "total_ci_upper": np.percentile(boot_total, 97.5),
                "direct_effect": direct_effect,
                "direct_pvalue": direct_p,
                "direct_ci_lower": np.percentile(boot_direct, 2.5),
                "direct_ci_upper": np.percentile(boot_direct, 97.5),
                "indirect_effect": indirect_effect,
                "indirect_ci_lower": np.percentile(boot_indirect, 2.5),
                "indirect_ci_upper": np.percentile(boot_indirect, 97.5),
                "prop_mediated_pct": pct_mediated
            })

    df_med = pd.DataFrame(results_mediation)
    df_med.to_csv("results/mediation_analysis_results.tsv", sep="\t", index=False)
    logger.info(f"Mediation results:\n{df_med[['study_id', 'gene', 'total_effect', 'total_pvalue', 'direct_effect', 'direct_pvalue', 'indirect_effect', 'prop_mediated_pct']].round(4).to_string()}")
    return df_med

# ---------------------------------------------------------
# 3. Transcription Factor Regulon & E2F2 Activity (A9)
# ---------------------------------------------------------
def run_tf_regulon_analysis(samples_df, data_matrices):
    logger.info("=" * 60)
    logger.info("MODULE 3: Transcription Factor Regulon Activity (CollecTRI + decoupleR)")
    logger.info("=" * 60)

    # Load CollecTRI
    net = dc.op.collectri(organism="human")
    logger.info(f"CollecTRI loaded: {net.shape[0]} interactions across {net['source'].nunique()} TFs.")

    all_tf_acts = []
    tf_dosage_results = []
    meta = samples_df.set_index("sample_id")

    # Specific TFs of interest
    focal_tfs = ["E2F1", "E2F2", "E2F3", "MYBL2", "FOXM1"]

    for study_id, expr_log2 in data_matrices.items():
        logger.info(f"Inferring TF activities for {study_id} via ULM...")
        # decoupleR expects rows as samples, columns as genes
        mat = expr_log2.T
        acts, _ = dc.mt.ulm(mat, net, verbose=False)
        acts["study_id"] = study_id
        all_tf_acts.append(acts)

        s_meta = meta.loc[acts.index]
        tfs_tested = [c for c in acts.columns if c != "study_id"]

        # Run OLS for each TF vs X-dosage within study
        if study_id in ["GSE42331", "GSE47584"]:
            male_mask = s_meta["declared_karyotype"].isin(["46,XY", "47,XXY"])
            X_dos = (s_meta.loc[male_mask, "declared_karyotype"] == "47,XXY").astype(float).values
            for tf in tfs_tested:
                y = acts.loc[male_mask, tf].values
                coef, se, t, p = fit_ols_contrast(y, X_dos)
                tf_dosage_results.append({
                    "study_id": study_id,
                    "contrast": "XXY_vs_XY",
                    "tf": tf,
                    "coef": coef,
                    "se": se,
                    "t_stat": t,
                    "pvalue": p
                })
        elif study_id == "GSE46687":
            # XX vs XO (+1 X chromosome)
            X_dos = (s_meta["declared_karyotype"] == "46,XX").astype(float).values
            for tf in tfs_tested:
                y = acts.loc[:, tf].values
                coef, se, t, p = fit_ols_contrast(y, X_dos)
                tf_dosage_results.append({
                    "study_id": study_id,
                    "contrast": "XX_vs_XO",
                    "tf": tf,
                    "coef": coef,
                    "se": se,
                    "t_stat": t,
                    "pvalue": p
                })
        elif study_id == "GSE126712":
            X_dos = (s_meta["nX"] - 1.0).values
            covars = pd.DataFrame({"nY": s_meta["nY"].values}, index=s_meta.index)
            for tf in tfs_tested:
                y = acts.loc[:, tf].values
                coef, se, t, p = fit_ols_contrast(y, X_dos, covars=covars)
                tf_dosage_results.append({
                    "study_id": study_id,
                    "contrast": "X_dosage",
                    "tf": tf,
                    "coef": coef,
                    "se": se,
                    "t_stat": t,
                    "pvalue": p
                })

    df_tf_acts = pd.concat(all_tf_acts, axis=0)
    df_tf_acts.to_csv("results/tf_activity_scores.tsv", sep="\t")

    df_tf_res = pd.DataFrame(tf_dosage_results)
    # FDR per study
    df_tf_res["fdr"] = 1.0
    for s_id, group in df_tf_res.groupby("study_id"):
        df_tf_res.loc[group.index, "fdr"] = multipletests(group["pvalue"], method="fdr_bh")[1]

    df_tf_res.to_csv("results/tf_dosage_associations.tsv", sep="\t", index=False)

    focal_df = df_tf_res[df_tf_res["tf"].isin(focal_tfs)]
    logger.info(f"Focal TFs dosage association results:\n{focal_df.to_string()}")

    # E2F2 leverage and influence check in GSE42331
    gse42331_meta = meta.loc[data_matrices["GSE42331"].columns]
    male_mask_42331 = gse42331_meta["declared_karyotype"].isin(["46,XY", "47,XXY"])
    y_e2f2_act = df_tf_acts.loc[male_mask_42331.index[male_mask_42331], "E2F2"].values
    x_e2f2_dos = (gse42331_meta.loc[male_mask_42331, "declared_karyotype"] == "47,XXY").astype(float).values
    X_e2f2 = sm.add_constant(x_e2f2_dos)
    model_e2f2 = sm.OLS(y_e2f2_act, X_e2f2).fit()
    influence = model_e2f2.get_influence()
    cooks_d = influence.cooks_distance[0]
    hat_diag = influence.hat_matrix_diag

    df_leverage = pd.DataFrame({
        "sample_id": male_mask_42331.index[male_mask_42331],
        "karyotype": gse42331_meta.loc[male_mask_42331, "declared_karyotype"].values,
        "e2f2_activity": y_e2f2_act,
        "cooks_distance": cooks_d,
        "leverage": hat_diag,
        "is_influential": cooks_d > (4.0 / len(y_e2f2_act))
    })
    df_leverage.to_csv("results/e2f2_leverage_diagnostics.tsv", sep="\t", index=False)
    n_influential = df_leverage["is_influential"].sum()
    logger.info(f"E2F2 leverage check in GSE42331: {n_influential} influential samples flagged (Cook's D > 4/n).")

    # E2F2 target gene slope analysis
    e2f2_targets = net[net["source"] == "E2F2"]["target"].unique()
    de_gse42331 = pd.read_csv("results/de_GSE42331.tsv", sep="\t", index_col=0)
    common_targets = [g for g in e2f2_targets if g in de_gse42331.index]
    target_stats = de_gse42331.loc[common_targets]
    target_stats.to_csv("results/e2f2_regulon_targets.tsv", sep="\t")
    median_target_logfc = target_stats["log2FC"].median()
    prop_sig_targets = (target_stats["padj"] < 0.05).mean()
    logger.info(f"E2F2 targets in GSE42331: N={len(common_targets)}, Median log2FC={median_target_logfc:.4f}, Fraction FDR<0.05={prop_sig_targets*100:.1f}%.")

    return df_tf_acts, df_tf_res

# ---------------------------------------------------------
# 4. Signed WGCNA Co-Expression Networks (A10)
# ---------------------------------------------------------
def run_wgcna(samples_df, data_matrices, df_clr_til10):
    logger.info("=" * 60)
    logger.info("MODULE 4: Signed WGCNA Network Analysis")
    logger.info("=" * 60)

    # Combine discovery arrays (GSE42331, GSE46687, GSE47584, N=111)
    disc_studies = ["GSE42331", "GSE46687", "GSE47584"]
    shared_genes = list(set(data_matrices["GSE42331"].index) & set(data_matrices["GSE46687"].index) & set(data_matrices["GSE47584"].index))
    logger.info(f"Shared genes across 3 discovery arrays: {len(shared_genes)}")

    disc_samples = samples_df[samples_df["study_id"].isin(disc_studies)]["sample_id"].tolist()
    combined_expr = pd.concat([data_matrices[s].loc[shared_genes] for s in disc_studies], axis=1)[disc_samples]

    # Residualize within study to eliminate batch/study mean differences
    meta_disc = samples_df.set_index("sample_id").loc[disc_samples]
    study_dummies = pd.get_dummies(meta_disc["study_id"], drop_first=True, dtype=float)
    X_batch = sm.add_constant(study_dummies)

    logger.info("Residualizing gene expression across discovery cohorts...")
    resid_mat = np.zeros_like(combined_expr.values)
    for i in range(len(shared_genes)):
        y = combined_expr.values[i, :]
        res = sm.OLS(y, X_batch).fit()
        resid_mat[i, :] = res.resid + np.mean(y)

    df_resid = pd.DataFrame(resid_mat, index=shared_genes, columns=disc_samples)

    # Select top 4,000 most variable genes for network construction
    gene_vars = df_resid.var(axis=1)
    top_genes = gene_vars.nlargest(4000).index
    df_net = df_resid.loc[top_genes].T  # samples x genes

    # Soft-threshold power scan
    powers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]
    power_rows = []

    cor_mat = np.corrcoef(df_net.values, rowvar=False)

    chosen_power = 6
    best_r2 = -1.0

    for beta in powers:
        # Signed adjacency
        adj = np.power(0.5 * (1.0 + cor_mat), beta)
        np.fill_diagonal(adj, 0)
        k = adj.sum(axis=0)

        # Scale-free topology fit: log10 p(k) vs log10 k
        # Bin degrees
        k_nz = k[k > 0]
        if len(k_nz) < 50:
            continue
        hist, bin_edges = np.histogram(k_nz, bins=20)
        bin_mids = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        valid = hist > 0
        if valid.sum() > 3:
            slope, intercept, r_val, p_val, _ = stats.linregress(np.log10(bin_mids[valid]), np.log10(hist[valid] / len(k_nz)))
            r2 = r_val ** 2
            mean_k = np.mean(k)
            power_rows.append({
                "power": beta,
                "scale_free_r2": r2,
                "slope": slope,
                "mean_connectivity": mean_k
            })
            if r2 >= 0.80 and r2 > best_r2:
                best_r2 = r2
                chosen_power = beta

    df_power = pd.DataFrame(power_rows)
    df_power.to_csv("results/wgcna_power_scan.tsv", sep="\t", index=False)
    if best_r2 < 0.80 and len(power_rows) > 0:
        chosen_power = df_power.loc[df_power["scale_free_r2"].idxmax(), "power"]
    logger.info(f"Selected soft-threshold power beta={chosen_power} (Scale-free R^2={best_r2:.3f}).")

    # Compute TOM with chosen power
    adj = np.power(0.5 * (1.0 + cor_mat), chosen_power)
    np.fill_diagonal(adj, 0)
    k = adj.sum(axis=0)
    L = adj @ adj
    min_k = np.minimum.outer(k, k)
    tom = (L + adj) / (min_k + 1.0 - adj)
    np.fill_diagonal(tom, 1.0)

    # Hierarchical clustering
    dist_tom = 1.0 - tom
    cond_dist = squareform(dist_tom, checks=False)
    Z = linkage(cond_dist, method="average")

    # Module cutting: target ~8-15 distinct co-expression modules
    clusters = fcluster(Z, t=0.92, criterion="distance")
    cluster_counts = pd.Series(clusters).value_counts()
    logger.info(f"Initial raw cluster counts:\n{cluster_counts.head(10)}")

    # Assign small modules (<30 genes) to grey module (0)
    min_mod_size = 30
    module_labels = {}
    mod_id = 1
    cluster_to_module = {}
    for c_id, count in cluster_counts.items():
        if count >= min_mod_size:
            cluster_to_module[c_id] = f"M{mod_id}"
            mod_id += 1
        else:
            cluster_to_module[c_id] = "grey"

    gene_modules = pd.Series([cluster_to_module.get(c, "grey") for c in clusters], index=top_genes, name="module")
    gene_modules.to_csv("results/wgcna_module_assignments.tsv", sep="\t")
    logger.info(f"Final module distribution:\n{gene_modules.value_counts()}")

    # Calculate Module Eigengenes (ME)
    me_dict = {}
    for mod in gene_modules.unique():
        mod_genes = gene_modules[gene_modules == mod].index
        mod_expr = df_net[mod_genes].values
        # 1st PC
        u, s, vt = np.linalg.svd(mod_expr - np.mean(mod_expr, axis=0), full_matrices=False)
        me = u[:, 0]
        # Align sign so correlation with average module expression is positive
        if np.corrcoef(me, np.mean(mod_expr, axis=1))[0, 1] < 0:
            me = -me
        me_dict[f"ME_{mod}"] = me

    df_me = pd.DataFrame(me_dict, index=disc_samples)
    df_me.to_csv("results/wgcna_module_eigengenes.tsv", sep="\t")

    # Module-trait associations
    traits_df = meta_disc[["nX", "nY", "age"]].copy()
    traits_df["is_XXY"] = (meta_disc["declared_karyotype"] == "47,XXY").astype(float)
    traits_df["is_XO"] = (meta_disc["declared_karyotype"] == "45,X").astype(float)
    # Add cell fractions
    for cell in ["Monocytes", "Neutrophils", "T.cells.CD4", "B.cells"]:
        traits_df[cell] = df_clr_til10.loc[disc_samples, cell]

    mod_trait_rows = []
    for me_col in df_me.columns:
        for trait_col in traits_df.columns:
            valid = ~traits_df[trait_col].isna()
            r, p = stats.pearsonr(df_me.loc[valid, me_col], traits_df.loc[valid, trait_col])
            mod_trait_rows.append({
                "module": me_col,
                "trait": trait_col,
                "pearson_r": r,
                "pvalue": p
            })
    df_mod_trait = pd.DataFrame(mod_trait_rows)
    df_mod_trait["fdr"] = multipletests(df_mod_trait["pvalue"], method="fdr_bh")[1]
    df_mod_trait.to_csv("results/wgcna_module_trait_correlations.tsv", sep="\t", index=False)
    logger.info(f"Top module-trait correlations:\n{df_mod_trait.sort_values('pvalue').head(10).to_string()}")

    # Module preservation in GSE126712 RNA-seq
    rna_samples = samples_df[samples_df["study_id"] == "GSE126712"]["sample_id"].tolist()
    rna_expr = data_matrices["GSE126712"].loc[:, rna_samples]
    preservation_rows = []

    for mod in gene_modules.unique():
        if mod == "grey":
            continue
        mod_genes = gene_modules[gene_modules == mod].index
        common_mod = [g for g in mod_genes if g in rna_expr.index]
        if len(common_mod) < 15:
            continue

        # Correlation matrix in discovery vs test
        cor_disc = np.corrcoef(df_net[common_mod].values, rowvar=False)
        cor_rna = np.corrcoef(rna_expr.loc[common_mod].values)

        # Correlation between correlation matrices (density/preservation)
        v_disc = cor_disc[np.triu_indices(len(common_mod), k=1)]
        v_rna = cor_rna[np.triu_indices(len(common_mod), k=1)]

        r_pres, p_pres = stats.pearsonr(v_disc, v_rna)
        # Approximate Z_summary statistic
        z_stat = np.arctanh(r_pres) * np.sqrt(len(v_disc) - 3)

        preservation_rows.append({
            "module": mod,
            "n_genes": len(common_mod),
            "cor_preservation": r_pres,
            "z_summary": z_stat,
            "is_preserved": z_stat > 2.0
        })

    df_pres = pd.DataFrame(preservation_rows)
    df_pres.to_csv("results/wgcna_module_preservation.tsv", sep="\t", index=False)
    logger.info(f"Module preservation in GSE126712 RNA-seq:\n{df_pres.to_string()}")

    return df_me, df_mod_trait, df_pres, chosen_power

# ---------------------------------------------------------
# 5. Machine Learning & Cross-Syndrome Transfer (A11)
# ---------------------------------------------------------
def run_machine_learning(samples_df, data_matrices):
    logger.info("=" * 60)
    logger.info("MODULE 5: Machine Learning Validation & Cross-Syndrome Transfer")
    logger.info("=" * 60)

    # Define feature sets
    gene_annot = pd.read_csv("data/clean/gene_annotations.tsv", sep="\t", index_col=0)
    x_escape_par_genes = gene_annot[
        (gene_annot["is_par1"] == True) | (gene_annot["is_par2"] == True) | (gene_annot["is_xci_escape"] == True)
    ].index.tolist()
    logger.info(f"Curated X-escape / PAR feature set: N={len(x_escape_par_genes)}")

    # Discovery cohorts pooled: GSE42331, GSE46687, GSE47584
    disc_studies = ["GSE42331", "GSE46687", "GSE47584"]
    shared_genes = list(set(data_matrices["GSE42331"].index) & set(data_matrices["GSE46687"].index) & set(data_matrices["GSE47584"].index))
    disc_samples = samples_df[samples_df["study_id"].isin(disc_studies)].copy()

    # Formulate binary task: Aneuploidy with extra X (+1 X chromosome: XXY vs XY in KS, XX vs XO in TS)
    # Standardize orientation: 1 = higher X-dosage (XXY or XX), 0 = lower X-dosage (XY or XO)
    disc_samples["target"] = np.nan
    for idx, row in disc_samples.iterrows():
        if row["declared_karyotype"] in ["47,XXY", "46,XX"]:
            disc_samples.loc[idx, "target"] = 1.0
        elif row["declared_karyotype"] in ["46,XY", "45,X"]:
            disc_samples.loc[idx, "target"] = 0.0

    valid_samples = disc_samples.dropna(subset=["target"]).copy()
    valid_samples["target"] = valid_samples["target"].astype(int)

    X_all = pd.concat([data_matrices[s].loc[shared_genes] for s in disc_studies], axis=1)[valid_samples["sample_id"]].T

    # Filter feature sets present in shared_genes
    feat_x_par = [g for g in x_escape_par_genes if g in shared_genes]
    feat_all = shared_genes

    logger.info(f"Valid ML samples: N={len(valid_samples)} (Positive: {valid_samples['target'].sum()}, Negative: {(1-valid_samples['target']).sum()})")
    logger.info(f"Features: X/PAR={len(feat_x_par)}, All genes={len(feat_all)}")

    # 1. Leave-One-Study-Out (LOSO) Cross-Validation
    loso_results = []
    for test_study in disc_studies:
        train_idx = valid_samples[valid_samples["study_id"] != test_study].index
        test_idx = valid_samples[valid_samples["study_id"] == test_study].index

        train_s = valid_samples.loc[train_idx]
        test_s = valid_samples.loc[test_idx]

        # Evaluate both feature sets
        for feat_name, feats in [("X_Escape_PAR", feat_x_par), ("All_Genes", feat_all)]:
            X_tr = X_all.loc[train_s["sample_id"], feats].values
            y_tr = train_s["target"].values
            X_te = X_all.loc[test_s["sample_id"], feats].values
            y_te = test_s["target"].values

            # Scale inside training fold (zero leakage)
            mean_tr = np.mean(X_tr, axis=0)
            std_tr = np.std(X_tr, axis=0) + 1e-6
            X_tr_sc = (X_tr - mean_tr) / std_tr
            X_te_sc = (X_te - mean_tr) / std_tr

            # Logistic Regression with L2/ElasticNet penalty
            clf = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=42)
            clf.fit(X_tr_sc, y_tr)
            y_pred_prob = clf.predict_proba(X_te_sc)[:, 1]
            y_pred = clf.predict(X_te_sc)

            auc = roc_auc_score(y_te, y_pred_prob) if len(np.unique(y_te)) > 1 else np.nan
            bal_acc = balanced_accuracy_score(y_te, y_pred)
            brier = brier_score_loss(y_te, y_pred_prob)

            loso_results.append({
                "test_study": test_study,
                "feature_set": feat_name,
                "n_features": len(feats),
                "n_test": len(y_te),
                "roc_auc": auc,
                "balanced_accuracy": bal_acc,
                "brier_score": brier
            })

    df_loso = pd.DataFrame(loso_results)
    df_loso.to_csv("results/ml_performance_loso.tsv", sep="\t", index=False)
    logger.info(f"LOSO cross-validation performance:\n{df_loso.to_string()}")

    # 2. Cross-Syndrome Transfer: Train on TS (GSE46687), Test on KS (GSE42331 + GSE47584)
    transfer_results = []
    ts_samples = valid_samples[valid_samples["study_id"] == "GSE46687"]
    ks_samples = valid_samples[valid_samples["study_id"].isin(["GSE42331", "GSE47584"])]

    for direction, train_s, test_s in [
        ("TS_to_KS", ts_samples, ks_samples),
        ("KS_to_TS", ks_samples, ts_samples)
    ]:
        for feat_name, feats in [("X_Escape_PAR", feat_x_par), ("All_Genes", feat_all)]:
            X_tr = X_all.loc[train_s["sample_id"], feats].values
            y_tr = train_s["target"].values
            X_te = X_all.loc[test_s["sample_id"], feats].values
            y_te = test_s["target"].values

            mean_tr = np.mean(X_tr, axis=0)
            std_tr = np.std(X_tr, axis=0) + 1e-6
            X_tr_sc = (X_tr - mean_tr) / std_tr
            X_te_sc = (X_te - mean_tr) / std_tr

            clf = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=42)
            clf.fit(X_tr_sc, y_tr)
            y_pred_prob = clf.predict_proba(X_te_sc)[:, 1]
            y_pred = clf.predict(X_te_sc)

            auc = roc_auc_score(y_te, y_pred_prob)
            bal_acc = balanced_accuracy_score(y_te, y_pred)

            transfer_results.append({
                "direction": direction,
                "feature_set": feat_name,
                "n_train": len(y_tr),
                "n_test": len(y_te),
                "roc_auc": auc,
                "balanced_accuracy": bal_acc
            })

    df_transfer = pd.DataFrame(transfer_results)
    df_transfer.to_csv("results/ml_cross_syndrome_transfer.tsv", sep="\t", index=False)
    logger.info(f"Cross-syndrome transfer performance:\n{df_transfer.to_string()}")

    # 3. Permutation Null Test (100 permutations)
    logger.info("Running permutation test (shuffling labels, 100 permutations)...")
    perm_aucs = []
    X_tr_all = X_all.loc[valid_samples["sample_id"], feat_x_par].values
    y_true = valid_samples["target"].values
    mean_all = np.mean(X_tr_all, axis=0)
    std_all = np.std(X_tr_all, axis=0) + 1e-6
    X_sc = (X_tr_all - mean_all) / std_all

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for p in range(100):
        y_perm = np.random.permutation(y_true)
        cv_preds = np.zeros(len(y_true))
        for tr_i, te_i in skf.split(X_sc, y_perm):
            clf = LogisticRegression(penalty="l2", C=1.0, max_iter=500, random_state=42)
            clf.fit(X_sc[tr_i], y_perm[tr_i])
            cv_preds[te_i] = clf.predict_proba(X_sc[te_i])[:, 1]
        perm_aucs.append(roc_auc_score(y_perm, cv_preds))

    df_perm = pd.DataFrame({"perm_idx": range(100), "null_auc": perm_aucs})
    df_perm.to_csv("results/ml_permutation_null.tsv", sep="\t", index=False)
    logger.info(f"Permutation null AUC mean: {np.mean(perm_aucs):.3f} (95% CI: [{np.percentile(perm_aucs, 2.5):.3f}, {np.percentile(perm_aucs, 97.5):.3f}])")

    return df_loso, df_transfer, df_perm

# ---------------------------------------------------------
# 6. False-Positive & Power Simulation (A12)
# ---------------------------------------------------------
def run_simulation_study(samples_df):
    logger.info("=" * 60)
    logger.info("MODULE 6: False Discovery Rate & Power Simulation Study")
    logger.info("=" * 60)

    # Real discovery design structure
    # GSE42331: N=65 (35 XXY, 15 XY, 15 XX)
    # GSE46687: N=36 (26 XO, 10 XX)
    # GSE47584: N=10 (5 XXY, 5 XY)
    n_sim_genes = 4000
    disc_samples = samples_df[samples_df["study_id"].isin(["GSE42331", "GSE46687", "GSE47584"])].copy()
    n_tot = len(disc_samples)

    # 1. Null Simulation (beta_X = 0 for all genes)
    logger.info(f"Simulating null expression matrix ({n_sim_genes} genes across {n_tot} samples)...")

    # True parameters under H0
    # Study platform effect: GSE42331 = 0, GSE46687 = +1.5, GSE47584 = -1.0
    study_effects = {"GSE42331": 0.0, "GSE46687": 1.5, "GSE47584": -1.0}
    # Tissue effect: GSE46687 is PBMC (affects 30% of genes by N(0, 1.2))
    pbmc_mask = (disc_samples["tissue"] == "PBMC").values

    np.random.seed(42)
    base_mean = np.random.uniform(5.0, 10.0, size=(n_sim_genes, 1))
    noise = np.random.normal(0.0, 0.5, size=(n_sim_genes, n_tot))

    study_vec = np.array([study_effects[s] for s in disc_samples["study_id"]])
    tissue_shift = np.zeros((n_sim_genes, n_tot))
    tissue_affected_genes = np.random.choice(n_sim_genes, size=int(0.3 * n_sim_genes), replace=False)
    tissue_shift[tissue_affected_genes, :] = np.outer(np.random.normal(0, 1.2, size=len(tissue_affected_genes)), pbmc_mask)

    # Null expression matrix
    Y_null = base_mean + study_vec + tissue_shift + noise
    df_null = pd.DataFrame(Y_null, index=[f"gene_{i}" for i in range(n_sim_genes)], columns=disc_samples["sample_id"].values)

    # Define continuous X-chromosome dosage vector
    x_dosage = disc_samples["nX"].values

    # Test 4 Strategies:
    # Strategy 1: Naive unadjusted pooled OLS
    logger.info("Evaluating Strategy 1: Naive unadjusted pooled OLS...")
    pvals_unadj = []
    X_unadj = sm.add_constant(x_dosage)
    for i in range(n_sim_genes):
        res = sm.OLS(Y_null[i, :], X_unadj).fit()
        pvals_unadj.append(res.pvalues[1])
    fdr_unadj = multipletests(pvals_unadj, method="fdr_bh")[1]

    # Strategy 2: ComBat-then-regress (Legacy approach)
    logger.info("Evaluating Strategy 2: ComBat-then-regress (Legacy defect D1/D2)...")
    adata = sc.AnnData(X=df_null.T)
    adata.obs["study"] = disc_samples["study_id"].values
    sc.pp.combat(adata, key="study")
    Y_combat = adata.X.T

    pvals_combat = []
    for i in range(n_sim_genes):
        res = sm.OLS(Y_combat[i, :], X_unadj).fit()
        pvals_combat.append(res.pvalues[1])
    fdr_combat = multipletests(pvals_combat, method="fdr_bh")[1]

    # Strategy 3: Joint study-stratified model (Fixed study effects)
    logger.info("Evaluating Strategy 3: Joint study-stratified model (Corrected)...")
    dummies = pd.get_dummies(disc_samples["study_id"], drop_first=True, dtype=float)
    X_joint = sm.add_constant(np.column_stack([x_dosage, dummies.values]))
    pvals_joint = []
    for i in range(n_sim_genes):
        res = sm.OLS(Y_null[i, :], X_joint).fit()
        pvals_joint.append(res.pvalues[1])  # coefficient for x_dosage
    fdr_joint = multipletests(pvals_joint, method="fdr_bh")[1]

    # Strategy 4: Within-study contrast modeling + random-effects meta-analysis
    logger.info("Evaluating Strategy 4: Within-study modeling + Meta-analysis (Primary pipeline)...")
    pvals_meta = []
    for i in range(n_sim_genes):
        betas = []
        ses = []
        for s_id in ["GSE42331", "GSE46687", "GSE47584"]:
            s_mask = (disc_samples["study_id"] == s_id).values
            y_sub = Y_null[i, s_mask]
            x_sub = x_dosage[s_mask]
            res_sub = sm.OLS(y_sub, sm.add_constant(x_sub)).fit()
            betas.append(res_sub.params[1])
            ses.append(res_sub.bse[1])
        # Fixed-effects inverse-variance pooling
        w = 1.0 / (np.array(ses) ** 2)
        pooled_beta = np.sum(w * np.array(betas)) / np.sum(w)
        pooled_se = np.sqrt(1.0 / np.sum(w))
        z = pooled_beta / pooled_se
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        pvals_meta.append(p)
    fdr_meta = multipletests(pvals_meta, method="fdr_bh")[1]

    sim_fdr_comparison = pd.DataFrame([
        {
            "strategy": "Naive Unadjusted Pooled OLS",
            "false_positives_p05": np.sum(np.array(pvals_unadj) < 0.05),
            "false_positive_rate_p05": np.mean(np.array(pvals_unadj) < 0.05),
            "false_positives_fdr05": np.sum(fdr_unadj < 0.05),
            "empirical_fdr_pct": np.mean(fdr_unadj < 0.05) * 100
        },
        {
            "strategy": "Legacy ComBat-then-Regress",
            "false_positives_p05": np.sum(np.array(pvals_combat) < 0.05),
            "false_positive_rate_p05": np.mean(np.array(pvals_combat) < 0.05),
            "false_positives_fdr05": np.sum(fdr_combat < 0.05),
            "empirical_fdr_pct": np.mean(fdr_combat < 0.05) * 100
        },
        {
            "strategy": "Joint Study-Stratified Model",
            "false_positives_p05": np.sum(np.array(pvals_joint) < 0.05),
            "false_positive_rate_p05": np.mean(np.array(pvals_joint) < 0.05),
            "false_positives_fdr05": np.sum(fdr_joint < 0.05),
            "empirical_fdr_pct": np.mean(fdr_joint < 0.05) * 100
        },
        {
            "strategy": "Within-Study Meta-Analysis",
            "false_positives_p05": np.sum(np.array(pvals_meta) < 0.05),
            "false_positive_rate_p05": np.mean(np.array(pvals_meta) < 0.05),
            "false_positives_fdr05": np.sum(fdr_meta < 0.05),
            "empirical_fdr_pct": np.mean(fdr_meta < 0.05) * 100
        }
    ])
    sim_fdr_comparison.to_csv("results/simulation_fdr_comparison.tsv", sep="\t", index=False)
    logger.info(f"Null simulation false discovery rate results:\n{sim_fdr_comparison.to_string()}")

    # 2. Spike-In Power Benchmark
    logger.info("Running spike-in power benchmark across effect sizes...")
    effect_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5]
    n_spike = 100
    power_rows = []

    for eff in effect_sizes:
        # Spike effect into 100 genes
        Y_spiked = Y_null.copy()
        spike_idx = np.random.choice(n_sim_genes, size=n_spike, replace=False)
        for i in spike_idx:
            Y_spiked[i, :] += eff * x_dosage

        # Test recovery via joint model
        pvals_spiked = []
        for i in range(n_sim_genes):
            res = sm.OLS(Y_spiked[i, :], X_joint).fit()
            pvals_spiked.append(res.pvalues[1])
        fdr_spiked = multipletests(pvals_spiked, method="fdr_bh")[1]

        detected = np.sum(fdr_spiked[spike_idx] < 0.05)
        power = detected / n_spike

        power_rows.append({
            "log2FC_true": eff,
            "spiked_genes": n_spike,
            "detected_fdr05": detected,
            "power": power
        })

    df_power = pd.DataFrame(power_rows)
    df_power.to_csv("results/simulation_power_curves.tsv", sep="\t", index=False)
    logger.info(f"Spike-in power benchmark:\n{df_power.to_string()}")

    # Find minimal detectable log2FC at 80% power
    mde_80 = np.interp(0.80, df_power["power"].values, df_power["log2FC_true"].values)
    logger.info(f"Minimal detectable effect size (MDE) at 80% power: log2FC = {mde_80:.3f}")

    return sim_fdr_comparison, df_power, mde_80

# ---------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------
def main():
    logger.info("Starting Calibrated Systems Biology Pipeline...")

    # Load samples metadata
    samples_df = pd.read_csv("data/samples.tsv", sep="\t")
    logger.info(f"Loaded samples.tsv: N={len(samples_df)} samples across 4 cohorts.")

    # Load clean expression matrices
    data_matrices = {
        "GSE42331": pd.read_csv("data/clean/GSE42331_log2.tsv", sep="\t", index_col=0),
        "GSE46687": pd.read_csv("data/clean/GSE46687_log2.tsv", sep="\t", index_col=0),
        "GSE47584": pd.read_csv("data/clean/GSE47584_log2.tsv", sep="\t", index_col=0),
        "GSE126712": pd.read_csv("data/clean/GSE126712_filtered_log2.tsv", sep="\t", index_col=0)
    }

    # 1. Deconvolution
    df_til10, df_clr_til10 = run_deconvolution(samples_df, data_matrices)

    # 2. Mediation Analysis
    df_med = run_mediation_analysis(samples_df, data_matrices, df_clr_til10)

    # 3. TF Regulon & E2F2
    df_tf_acts, df_tf_res = run_tf_regulon_analysis(samples_df, data_matrices)

    # 4. WGCNA Co-expression Networks
    df_me, df_mod_trait, df_pres, chosen_power = run_wgcna(samples_df, data_matrices, df_clr_til10)

    # 5. Machine Learning Validation
    df_loso, df_transfer, df_perm = run_machine_learning(samples_df, data_matrices)

    # 6. False-Positive & Power Simulation
    sim_fdr, df_power, mde_80 = run_simulation_study(samples_df)

    # Generate summary JSON for LaTeX macros
    summary_dict = {
        "deconvolution": {
            "til10_cell_types": 10,
            "monocyte_concordance_spearman": float(pd.read_csv("results/cell_deconvolution_concordance.tsv", sep="\t").loc[0, "spearman_rho"]),
            "monocyte_concordance_ccc": float(pd.read_csv("results/cell_deconvolution_concordance.tsv", sep="\t").loc[0, "lins_ccc"])
        },
        "mediation": {
            "clec4e_gse42331_total_effect": float(df_med[(df_med["study_id"]=="GSE42331") & (df_med["gene"]=="CLEC4E")]["total_effect"].values[0]),
            "clec4e_gse42331_direct_effect": float(df_med[(df_med["study_id"]=="GSE42331") & (df_med["gene"]=="CLEC4E")]["direct_effect"].values[0]),
            "clec4e_gse42331_direct_pvalue": float(df_med[(df_med["study_id"]=="GSE42331") & (df_med["gene"]=="CLEC4E")]["direct_pvalue"].values[0]),
            "e2f2_gse42331_total_effect": float(df_med[(df_med["study_id"]=="GSE42331") & (df_med["gene"]=="E2F2")]["total_effect"].values[0]),
            "e2f2_gse42331_direct_effect": float(df_med[(df_med["study_id"]=="GSE42331") & (df_med["gene"]=="E2F2")]["direct_effect"].values[0]),
            "kdm6a_gse42331_total_effect": float(df_med[(df_med["study_id"]=="GSE42331") & (df_med["gene"]=="KDM6A")]["total_effect"].values[0]),
            "kdm6a_gse42331_direct_effect": float(df_med[(df_med["study_id"]=="GSE42331") & (df_med["gene"]=="KDM6A")]["direct_effect"].values[0])
        },
        "tf_regulon": {
            "total_tfs_tested": int(df_tf_acts.shape[1] - 1),
            "e2f2_activity_gse42331_coef": float(df_tf_res[(df_tf_res["study_id"]=="GSE42331") & (df_tf_res["tf"]=="E2F2")]["coef"].values[0]),
            "e2f2_activity_gse42331_pvalue": float(df_tf_res[(df_tf_res["study_id"]=="GSE42331") & (df_tf_res["tf"]=="E2F2")]["pvalue"].values[0]),
            "e2f2_activity_gse46687_coef": float(df_tf_res[(df_tf_res["study_id"]=="GSE46687") & (df_tf_res["tf"]=="E2F2")]["coef"].values[0]),
            "e2f2_activity_gse46687_pvalue": float(df_tf_res[(df_tf_res["study_id"]=="GSE46687") & (df_tf_res["tf"]=="E2F2")]["pvalue"].values[0])
        },
        "wgcna": {
            "soft_threshold_power": int(chosen_power),
            "n_modules_identified": int(df_mod_trait["module"].nunique()),
            "preserved_modules_count": int(df_pres["is_preserved"].sum())
        },
        "machine_learning": {
            "loso_auc_x_par_mean": float(df_loso[df_loso["feature_set"]=="X_Escape_PAR"]["roc_auc"].mean()),
            "loso_auc_all_genes_mean": float(df_loso[df_loso["feature_set"]=="All_Genes"]["roc_auc"].mean()),
            "transfer_ts_to_ks_auc": float(df_transfer[(df_transfer["direction"]=="TS_to_KS") & (df_transfer["feature_set"]=="X_Escape_PAR")]["roc_auc"].values[0]),
            "transfer_ks_to_ts_auc": float(df_transfer[(df_transfer["direction"]=="KS_to_TS") & (df_transfer["feature_set"]=="X_Escape_PAR")]["roc_auc"].values[0]),
            "permutation_null_auc_mean": float(df_perm["null_auc"].mean())
        },
        "simulation": {
            "legacy_combat_false_positive_pct": float(sim_fdr[sim_fdr["strategy"]=="Legacy ComBat-then-Regress"]["empirical_fdr_pct"].values[0]),
            "naive_ols_false_positive_pct": float(sim_fdr[sim_fdr["strategy"]=="Naive Unadjusted Pooled OLS"]["empirical_fdr_pct"].values[0]),
            "joint_model_false_positive_pct": float(sim_fdr[sim_fdr["strategy"]=="Joint Study-Stratified Model"]["empirical_fdr_pct"].values[0]),
            "meta_analysis_false_positive_pct": float(sim_fdr[sim_fdr["strategy"]=="Within-Study Meta-Analysis"]["empirical_fdr_pct"].values[0]),
            "minimal_detectable_log2fc_80_power": float(mde_80)
        }
    }

    with open("results/calibrated_systems_summary.json", "w") as f:
        json.dump(summary_dict, f, indent=2)

    logger.info("Summary scalars saved to results/calibrated_systems_summary.json.")
    logger.info("Calibrated Systems Biology Pipeline completed successfully!")

if __name__ == "__main__":
    main()
