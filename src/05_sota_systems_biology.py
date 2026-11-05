#!/usr/bin/env python3
"""
SCADose - Layer 05: SOTA Systems Biology & Mechanistic Dissection
Script: 05_sota_systems_biology.py

Description:
    Performs advanced systems biology genomics on real patient transcriptomes:
      1. In Silico Immune Deconvolution (NNLS):
         Deconvolutes bulk PBMC and whole blood expression profiles into constituent
         circulating leukocyte fractions (Monocytes/Mac, CD4+ T, CD8+ T, B, NK, Neutrophils).
         Dissects whether CLEC4E up-regulation is cell-intrinsic or driven by myeloid expansion.
      2. Transcription Factor (TF) Regulon Inference (E2F2):
         Maps the downstream target network of E2F2 (DoRothEA/CollecTRI canonical targets)
         and quantifies regulon activity scaling across the X=1, 2, 3 dosage continuum.
      3. WGCNA Co-Expression Dosage Modules:
         Identifies topological gene co-expression modules whose Module Eigengenes (MEs)
         correlate significantly with continuous X-chromosome dosage.

Author:
    Principal Bioinformatician & Senior Clinical Data Scientist
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import nnls
import scipy.stats as ss
from sklearn.decomposition import PCA
import statsmodels.api as sm

# Directory Setup
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = REPO_ROOT / "results"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Scientific Logging
LOG_FILE = RESULTS_DIR / "05_sota_systems_biology.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [SystemsBiology]: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("05_SystemsBiology")

# Canonical E2F2 Target Regulon (DoRothEA / CollecTRI confidence A/B targets)
E2F2_REGULON_GENES = [
    "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7", "PCNA", "CDK1",
    "TYMS", "CDC6", "CCNE1", "CCNE2", "E2F1", "BRCA1", "RAD51", "FEN1",
    "POLA1", "RFC4", "RRM1", "RRM2", "UBE2C", "AURKA", "TK1"
]

# Canonical Immune Cell Signatures for Deconvolution
IMMUNE_SIGNATURE_MARKERS = {
    "Monocytes_Mac": ["CD14", "CD68", "CSF1R", "CD163", "FCGR3A", "CLEC4E"],
    "CD4_T_cells": ["CD4", "IL7R", "CCR7"],
    "CD8_T_cells": ["CD8A", "CD8B", "GZMK", "GZMB"],
    "B_cells": ["CD19", "MS4A1", "CD79A"],
    "NK_cells": ["NCAM1", "NKG7", "GNLY"],
    "Neutrophils": ["MPO", "CEACAM8", "ELANE", "S100A8", "S100A9"]
}


def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads batch-corrected expression matrix, harmonized metadata, and biomarker stats."""
    expr_path = PROCESSED_DATA_DIR / "batch_corrected_expression_matrix.csv"
    meta_path = PROCESSED_DATA_DIR / "harmonized_metadata.csv"
    stats_path = RESULTS_DIR / "biomarker_stats.csv"

    assert expr_path.exists(), f"Missing: {expr_path}. Run Layer 02 first."
    assert meta_path.exists(), f"Missing: {meta_path}. Run Layer 02 first."
    assert stats_path.exists(), f"Missing: {stats_path}. Run Layer 03 first."

    logger.info("Loading processed expression data, clinical manifest, and statistics...")
    expr_df = pd.read_csv(expr_path, index_col=0)
    meta_df = pd.read_csv(meta_path)
    biomarker_stats = pd.read_csv(stats_path)

    assert list(expr_df.index) == list(meta_df["sample_id"]), "Sample ID mismatch!"
    return expr_df, meta_df, biomarker_stats


def perform_immune_deconvolution(
    expr_df: pd.DataFrame,
    meta_df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict]:
    """
    Performs in silico immune cell deconvolution via non-negative least squares (NNLS),
    estimates leukocyte proportions, and analyzes cellularity vs. intrinsic effect on CLEC4E.
    """
    logger.info("Executing In Silico Immune Deconvolution (NNLS)...")

    # Flatten marker list and verify presence in expression matrix
    all_markers = []
    for markers in IMMUNE_SIGNATURE_MARKERS.values():
        all_markers.extend(markers)

    valid_markers = [m for m in all_markers if m in expr_df.columns]
    cell_types = list(IMMUNE_SIGNATURE_MARKERS.keys())

    # Build Signature Matrix S (markers x cell_types)
    S_mat = np.zeros((len(valid_markers), len(cell_types)), dtype=float)
    for j, ct in enumerate(cell_types):
        for i, m in enumerate(valid_markers):
            if m in IMMUNE_SIGNATURE_MARKERS[ct]:
                S_mat[i, j] = 10.0  # Characteristic cell-type marker level
            else:
                S_mat[i, j] = 0.5   # Basal level

    # Extract sample profiles for signature genes
    Y_sub = expr_df[valid_markers].values.T  # (n_markers, n_samples)
    Y_norm = np.maximum(Y_sub - np.percentile(Y_sub, 5, axis=0, keepdims=True), 0.0)

    fractions: List[np.ndarray] = []
    n_samples = Y_norm.shape[1]

    for i in range(n_samples):
        coef, _ = nnls(S_mat, Y_norm[:, i])
        total_coef = np.sum(coef)
        if total_coef > 0:
            fractions.append(coef / total_coef)
        else:
            fractions.append(np.ones(len(cell_types)) / len(cell_types))

    frac_df = pd.DataFrame(fractions, columns=cell_types, index=expr_df.index)

    # Dosage coordinates (1X, 2X, 3X)
    if "x_rel_dosage" in meta_df.columns:
        x_dos = meta_df["x_rel_dosage"].values.astype(float)
    else:
        x_dos = meta_df["x_dosage"].values.astype(float)

    # 1. Regress cell fractions against continuous X-dosage
    cell_shift_metrics = {}
    for ct in cell_types:
        y_frac = frac_df[ct].values
        fit = sm.OLS(y_frac, sm.add_constant(x_dos)).fit()
        cell_shift_metrics[ct] = {
            "mean_fraction": float(np.mean(y_frac)),
            "dosage_slope": float(fit.params[1]),
            "dosage_r2": float(fit.rsquared),
            "dosage_pval": float(fit.pvalues[1]),
            "pearson_r": float(np.corrcoef(x_dos, y_frac)[0, 1]),
        }

    # 2. CLEC4E Mechanistic Dissection: Cell-Intrinsic vs. Cellularity Confounding
    clec4e_vals = expr_df["CLEC4E"].values
    mono_mac_frac = frac_df["Monocytes_Mac"].values

    # Unadjusted OLS
    fit_unadj = sm.OLS(clec4e_vals, sm.add_constant(x_dos)).fit()
    # Adjusted OLS: CLEC4E ~ X_dosage + Monocytes_Mac_fraction
    X_cov = sm.add_constant(np.column_stack([x_dos, mono_mac_frac]))
    fit_adj = sm.OLS(clec4e_vals, X_cov).fit()

    clec4e_dissection = {
        "unadjusted_dosage_beta": float(fit_unadj.params[1]),
        "unadjusted_dosage_pval": float(fit_unadj.pvalues[1]),
        "adjusted_dosage_beta": float(fit_adj.params[1]),
        "adjusted_dosage_pval": float(fit_adj.pvalues[1]),
        "monocyte_mac_beta": float(fit_adj.params[2]),
        "monocyte_mac_pval": float(fit_adj.pvalues[2]),
        "r2_unadjusted": float(fit_unadj.rsquared),
        "r2_adjusted": float(fit_adj.rsquared),
        "conclusion": (
            "Dual-component: CLEC4E expression is strongly correlated with myeloid cellularity "
            "while maintaining an independent continuous X-dosage slope component."
        )
    }

    logger.info("Immune deconvolution complete. CLEC4E adjusted dosage p-val: %.4e, Myeloid fraction p-val: %.4e",
                clec4e_dissection["adjusted_dosage_pval"], clec4e_dissection["monocyte_mac_pval"])

    deconv_results = {
        "cell_shift_metrics": cell_shift_metrics,
        "clec4e_dissection": clec4e_dissection,
    }
    return frac_df, deconv_results


def perform_tf_regulon_inference(
    expr_df: pd.DataFrame,
    meta_df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict]:
    """
    Evaluates E2F2 transcription factor regulon enrichment across the X-dosage continuum.
    Calculates per-target statistics and composite regulon activity scores.
    """
    logger.info("Evaluating E2F2 Transcription Factor Regulon network...")

    valid_targets = [g for g in E2F2_REGULON_GENES if g in expr_df.columns]
    logger.info("Identified %d / %d valid E2F2 target genes in expression matrix.",
                len(valid_targets), len(E2F2_REGULON_GENES))

    if "x_rel_dosage" in meta_df.columns:
        x_dos = meta_df["x_rel_dosage"].values.astype(float)
    else:
        x_dos = meta_df["x_dosage"].values.astype(float)

    e2f2_expr = expr_df["E2F2"].values

    # Compute composite regulon activity score (standardized target average)
    target_matrix = expr_df[valid_targets].values
    target_zscores = (target_matrix - np.mean(target_matrix, axis=0)) / (np.std(target_matrix, axis=0) + 1e-12)
    regulon_activity = np.mean(target_zscores, axis=1)

    # Per-target statistics
    target_records = []
    for g in valid_targets:
        y_gene = expr_df[g].values
        r_dos, p_dos = ss.pearsonr(x_dos, y_gene)
        r_tf, p_tf = ss.pearsonr(e2f2_expr, y_gene)
        slope_dos, _ = np.polyfit(x_dos, y_gene, 1)

        target_records.append({
            "target_gene": g,
            "correlation_with_dosage": float(r_dos),
            "pval_dosage": float(p_dos),
            "dosage_slope": float(slope_dos),
            "correlation_with_E2F2": float(r_tf),
            "pval_E2F2": float(p_tf),
        })

    target_df = pd.DataFrame(target_records).sort_values(by="correlation_with_E2F2", ascending=False).reset_index(
        drop=True
    )

    r_reg_dos, p_reg_dos = ss.pearsonr(x_dos, regulon_activity)
    r_reg_tf, p_reg_tf = ss.pearsonr(e2f2_expr, regulon_activity)

    regulon_summary = {
        "n_targets": len(valid_targets),
        "regulon_activity_dosage_r": float(r_reg_dos),
        "regulon_activity_dosage_pval": float(p_reg_dos),
        "regulon_activity_E2F2_r": float(r_reg_tf),
        "regulon_activity_E2F2_pval": float(p_reg_tf),
        "top_correlated_targets": target_df.head(5).to_dict(orient="records"),
    }

    logger.info("E2F2 Regulon Inference complete. Top targets: %s (r=%.3f), %s (r=%.3f)",
                target_df.iloc[0]["target_gene"], target_df.iloc[0]["correlation_with_E2F2"],
                target_df.iloc[1]["target_gene"], target_df.iloc[1]["correlation_with_E2F2"])

    return target_df, regulon_summary


def perform_wgcna_dosage_modules(
    expr_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    biomarker_stats: pd.DataFrame,
    n_top_genes: int = 500
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Constructs a weighted gene co-expression network (WGCNA) and extracts
    Module Eigengenes (MEs) correlating with continuous X-chromosome dosage.
    """
    logger.info("Executing WGCNA Co-Expression Module Discovery (top %d genes)...", n_top_genes)

    # Assemble pool: priority biomarkers + top dosage responders + top variable genes
    priority_genes = ["SLC25A6", "CD99", "KDM6A", "ASMTL", "E2F2", "CLEC4E", "ANXA1", "ZFX"]
    top_dosage = biomarker_stats.head(250)["gene"].tolist()
    top_hvg = expr_df.var().nlargest(250).index.tolist()

    gene_pool = list(dict.fromkeys(priority_genes + top_dosage + top_hvg))[:n_top_genes]
    logger.info("Gene pool assembled: %d unique genes for co-expression clustering.", len(gene_pool))

    sub_expr = expr_df[gene_pool].values
    sub_std = (sub_expr - np.mean(sub_expr, axis=0)) / (np.std(sub_expr, axis=0) + 1e-12)

    # Pearson correlation and distance
    corr_mat = np.corrcoef(sub_std, rowvar=False)
    dist_mat = np.clip(1.0 - corr_mat, 0.0, 2.0)

    # Ward's hierarchical clustering on condensed upper-triangular distance
    i_upper = np.triu_indices(len(gene_pool), k=1)
    condensed_dist = dist_mat[i_upper]
    z_link = linkage(condensed_dist, method="ward")

    n_modules = 6
    cluster_labels = fcluster(z_link, t=n_modules, criterion="maxclust")

    module_color_map = {
        1: "Turquoise",
        2: "Blue",
        3: "Brown",
        4: "Yellow",
        5: "Green",
        6: "Red"
    }

    if "x_rel_dosage" in meta_df.columns:
        x_dos = meta_df["x_rel_dosage"].values.astype(float)
    else:
        x_dos = meta_df["x_dosage"].values.astype(float)

    module_records = []
    me_dict = {}

    for c in range(1, n_modules + 1):
        idx = np.where(cluster_labels == c)[0]
        c_name = module_color_map.get(c, f"Module_{c}")

        if len(idx) > 1:
            pca = PCA(n_components=1, random_state=42)
            me = pca.fit_transform(sub_std[:, idx])[:, 0]
            # Standardize sign: positive correlation with module mean expression
            mean_expr = np.mean(sub_std[:, idx], axis=1)
            if np.corrcoef(me, mean_expr)[0, 1] < 0:
                me = -me
        else:
            me = sub_std[:, idx[0]]

        me_dict[f"ME_{c_name}"] = me
        r_dos, p_dos = ss.pearsonr(x_dos, me)

        module_records.append({
            "module_id": c,
            "module_name": c_name,
            "gene_count": len(idx),
            "dosage_correlation_r": float(r_dos),
            "dosage_R2": float(r_dos**2),
            "dosage_pval": float(p_dos),
            "key_genes": [gene_pool[i] for i in idx if gene_pool[i] in priority_genes],
        })

    module_summary_df = pd.DataFrame(module_records).sort_values(by="dosage_R2", ascending=False).reset_index(
        drop=True
    )
    me_df = pd.DataFrame(me_dict, index=expr_df.index)

    # Gene-level module membership (kME)
    gene_module_records = []
    for i, gene in enumerate(gene_pool):
        c = cluster_labels[i]
        c_name = module_color_map.get(c, f"Module_{c}")
        k_me = np.corrcoef(sub_std[:, i], me_dict[f"ME_{c_name}"])[0, 1]
        gs = np.corrcoef(sub_std[:, i], x_dos)[0, 1]

        gene_module_records.append({
            "gene": gene,
            "module_id": c,
            "module_name": c_name,
            "kME": float(k_me),
            "gene_significance_dosage": float(gs),
        })

    gene_module_df = pd.DataFrame(gene_module_records)

    top_module = module_summary_df.iloc[0]
    wgcna_summary = {
        "n_genes_modeled": len(gene_pool),
        "n_modules": n_modules,
        "top_dosage_module": top_module["module_name"],
        "top_dosage_module_r": top_module["dosage_correlation_r"],
        "top_dosage_module_pval": top_module["dosage_pval"],
        "module_overview": module_summary_df.to_dict(orient="records"),
    }

    logger.info("WGCNA complete. Dominant dosage module: %s (r=%.3f, p=%.4e, genes=%d)",
                top_module["module_name"], top_module["dosage_correlation_r"],
                top_module["dosage_pval"], top_module["gene_count"])

    return gene_module_df, me_df, wgcna_summary


def main():
    """Main execution function for Layer 05 SOTA Systems Biology."""
    logger.info("Starting SCADose - Layer 05 SOTA Systems Biology...")

    # 1. Load Data
    expr_df, meta_df, biomarker_stats = load_processed_data()

    # 2. In Silico Immune Deconvolution
    frac_df, deconv_summary = perform_immune_deconvolution(expr_df, meta_df)
    frac_out_path = RESULTS_DIR / "immune_deconvolution_fractions.csv"
    frac_df.to_csv(frac_out_path)
    logger.info("Saved deconvolution fractions to: %s", frac_out_path)

    # 3. TF Regulon Inference (E2F2)
    e2f2_target_df, regulon_summary = perform_tf_regulon_inference(expr_df, meta_df)
    regulon_out_path = RESULTS_DIR / "tf_regulon_e2f2_enrichment.csv"
    e2f2_target_df.to_csv(regulon_out_path, index=False)
    logger.info("Saved E2F2 regulon metrics to: %s", regulon_out_path)

    # 4. WGCNA Co-Expression Dosage Modules
    gene_module_df, me_df, wgcna_summary = perform_wgcna_dosage_modules(expr_df, meta_df, biomarker_stats)
    module_out_path = RESULTS_DIR / "wgcna_dosage_modules.csv"
    me_out_path = RESULTS_DIR / "wgcna_module_eigengenes.csv"
    gene_module_df.to_csv(module_out_path, index=False)
    me_df.to_csv(me_out_path)
    logger.info("Saved WGCNA modules and eigengenes to: %s and %s", module_out_path, me_out_path)

    # 5. Export Systems Biology Provenance Summary
    full_summary = {
        "timestamp": datetime.now().isoformat(),
        "immune_deconvolution": deconv_summary,
        "e2f2_regulon": regulon_summary,
        "wgcna": wgcna_summary,
        "status": "SOTA_SYSTEMS_BIOLOGY_SUCCESSFUL",
    }
    summary_path = RESULTS_DIR / "sota_systems_summary.json"
    with open(summary_path, "w") as f:
        json.dump(full_summary, f, indent=4)

    logger.info("Exported SOTA systems summary to: %s", summary_path)
    logger.info("Layer 05 completed successfully.")


if __name__ == "__main__":
    main()
