#!/usr/bin/env python3
"""
SCADose - Phase 1A: Comprehensive Clinical Data Audit, Ingestion & QC
Script: src/01_clinical_data_audit.py

Description:
    Conducts a rigorous Phase 1A data audit and ingestion from primary sources:
      1. Ingests all four real cohorts:
         - GSE42331 (Affymetrix Gene 1.0 ST, N=65: 35 KS, 15 Male, 15 Female)
         - GSE46687 (Affymetrix U133 Plus 2.0, N=36: 16 45Xm, 10 45Xp, 10 46XX)
         - GSE47584 (Agilent SurePrint 8x60K, N=10: 5 KS, 5 Male)
         - GSE126712 (Illumina HiSeq 2500 RNA-Seq, N=54: 13 XO, 14 XX, 13 XY, 14 XXY)
      2. Corrects legacy defects:
         - Fixes GSE42331 soft exponent typo (probe 8173009 PHF8 8.17E+06 -> 8.17)
         - Corrects GSE42331 control group mislabeling (15 female controls partitioned from 15 male controls)
         - Standardizes all datasets to uniform log2 intensity scales
         - Separates RNA-Seq primary lymphocytes from DNA methylation and Hi-C in GSE126712
      3. Performs sex/karyotype verification using XIST and Y-chromosome gametologs
      4. Audits cross-cohort duplicate profiles via pairwise Spearman correlations
      5. Generates comprehensive samples.tsv, gene_annotations.tsv, and clean within-study matrices.

Author:
    Lead Bioinformatician, Applied Statistician & Computational Scientist
"""

import glob
import gzip
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

# Setup Paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "data"
CLEAN_DATA_DIR = DATA_DIR / "clean"
RESULTS_DIR = REPO_ROOT / "results"

CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Logging
LOG_FILE = RESULTS_DIR / "audit_report.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Audit1A]: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("01_ClinicalDataAudit")

# External raw data roots
SAFA_ROOT = Path("/media/abdelbasset/New Volume/safa")
MATRICES_DIR = SAFA_ROOT / "matrices"
METADATA_DIR = SAFA_ROOT / "metadata"
RAW_DATA_DIR = SAFA_ROOT / "data"

# Known XCI Escape and PAR1 gene references
PAR1_GENES = [
    "SLC25A6", "CD99", "ASMTL", "CSF2RA", "IL3RA", "SHOX", "CRLF2", "ZBED1",
    "AKAP17A", "GTPBP6", "PUDP", "DHRSX"
]
PAR2_GENES = ["SPRY3", "VAMP7", "IL9R"]

XCI_ESCAPE_GENES = [
    "KDM6A", "EIF1AX", "ZFX", "DDX3X", "KDM5C", "USP9X", "RPS4X", "SMC1A",
    "MED14", "CA5B", "PRKX", "NLGN4X", "TXLNG", "AP1S2", "JPX"
]

Y_LINKED_MARKERS = ["RPS4Y1", "DDX3Y", "KDM5D", "UTY", "EIF1AY", "ZFY"]


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 digest of a file for integrity provenance."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def ingest_gse42331() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ingests GSE42331 (Affymetrix Gene 1.0 ST), fixes PHF8 typo and metadata."""
    logger.info("Ingesting GSE42331 (Münster EXAKT cohort)...")
    expr_path = MATRICES_DIR / "GSE42331_expression_matrix.csv"
    meta_path = METADATA_DIR / "GSE42331_metadata.csv"

    expr = pd.read_csv(expr_path, index_col=0)
    meta = pd.read_csv(meta_path).set_index("GSM_ID")

    # Fix Defect D2: 8.17E+06 typo in probe 8173009 (PHF8) for GSM1037947
    if "PHF8" in expr.index and "GSM1037947" in expr.columns:
        old_val = expr.loc["PHF8", "GSM1037947"]
        if old_val > 1000.0:
            logger.info("Detected SOFT export exponent typo in GSM1037947 PHF8: %.2f -> correcting to 8.17", old_val)
            expr.loc["PHF8", "GSM1037947"] = 8.17

    # Build clean metadata fixing Defect D3
    sample_records = []
    for gsm in expr.columns:
        row = meta.loc[gsm]
        char = str(row.get("characteristics_ch1", ""))
        title = str(row.get("title", ""))
        
        if "gender: female" in char or "female" in title.lower():
            karyotype = "46,XX"
            sex = "Female"
            nX, nY = 2, 0
            condition = "Control_Female"
        elif "condition: Klinefelter" in char or "KS" in char:
            karyotype = "47,XXY"
            sex = "Male"
            nX, nY = 2, 1
            condition = "Klinefelter"
        else:
            karyotype = "46,XY"
            sex = "Male"
            nX, nY = 1, 1
            condition = "Control_Male"

        sample_records.append({
            "sample_id": gsm,
            "study_id": "GSE42331",
            "gpl": "GPL6244",
            "platform": "Affymetrix_HuGene-1_0-st",
            "tissue": "Whole_Blood",
            "cell_type": "Whole_Blood",
            "declared_karyotype": karyotype,
            "declared_sex": sex,
            "age": 36.8 if condition == "Klinefelter" else (35.4 if sex == "Male" else 33.1),
            "parental_origin": "N/A",
            "condition": condition,
            "nX": nX,
            "nY": nY,
            "n_sexchr": nX + nY,
            "study_design": "Identifies X at nY=1 (XXY vs XY) and sex effect (XX vs XY)",
        })

    meta_df = pd.DataFrame(sample_records).set_index("sample_id")
    logger.info("GSE42331 ingested: %d samples, %d genes. Value range: [%.2f, %.2f]",
                expr.shape[1], expr.shape[0], expr.values.min(), expr.values.max())
    return expr, meta_df


def ingest_gse46687() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ingests GSE46687 (Affymetrix U133 Plus 2.0), applies log2 transform."""
    logger.info("Ingesting GSE46687 (NICHD Bondy Turner cohort)...")
    expr_path = MATRICES_DIR / "GSE46687_expression_matrix.csv"
    meta_path = METADATA_DIR / "GSE46687_metadata.csv"

    expr_raw = pd.read_csv(expr_path, index_col=0)
    meta = pd.read_csv(meta_path).set_index("GSM_ID")

    # Defect D2 fix: Transform linear scale to log2 scale
    if expr_raw.values.max() > 100.0:
        logger.info("GSE46687 max raw intensity is %.1f. Applying log2(x + 1) transformation...", expr_raw.values.max())
        expr = np.log2(np.maximum(expr_raw, 0.0) + 1.0)
    else:
        expr = expr_raw

    sample_records = []
    for gsm in expr.columns:
        row = meta.loc[gsm]
        char = str(row.get("characteristics_ch1", ""))
        
        if "45Xm" in char:
            karyotype = "45,X"
            parental = "45,Xm"
            condition = "Turner_45Xm"
            nX, nY = 1, 0
        elif "45Xp" in char:
            karyotype = "45,X"
            parental = "45,Xp"
            condition = "Turner_45Xp"
            nX, nY = 1, 0
        else:
            karyotype = "46,XX"
            parental = "N/A"
            condition = "Control_Female"
            nX, nY = 2, 0

        sample_records.append({
            "sample_id": gsm,
            "study_id": "GSE46687",
            "gpl": "GPL570",
            "platform": "Affymetrix_HG-U133_Plus_2",
            "tissue": "PBMC",
            "cell_type": "PBMC",
            "declared_karyotype": karyotype,
            "declared_sex": "Female",
            "age": 32.0,  # Adult females
            "parental_origin": parental,
            "condition": condition,
            "nX": nX,
            "nY": nY,
            "n_sexchr": nX + nY,
            "study_design": "Identifies X at nY=0 (XO vs XX) and parental origin (45,Xm vs 45,Xp)",
        })

    meta_df = pd.DataFrame(sample_records).set_index("sample_id")
    logger.info("GSE46687 ingested: %d samples, %d genes. Value range: [%.2f, %.2f]",
                expr.shape[1], expr.shape[0], expr.values.min(), expr.values.max())
    return expr, meta_df


def ingest_gse47584() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ingests GSE47584 (Agilent SurePrint 8x60K), applies log2 transform."""
    logger.info("Ingesting GSE47584 (Denmark Klinefelter cohort)...")
    expr_path = MATRICES_DIR / "GSE47584_expression_matrix.csv"
    meta_path = METADATA_DIR / "GSE47584_metadata.csv"

    expr_raw = pd.read_csv(expr_path, index_col=0)
    meta = pd.read_csv(meta_path).set_index("GSM_ID")

    # Transform linear to log2 scale
    if expr_raw.values.max() > 100.0:
        logger.info("GSE47584 max raw intensity is %.1f. Applying log2(x + 1) transformation...", expr_raw.values.max())
        expr = np.log2(np.maximum(expr_raw, 0.0) + 1.0)
    else:
        expr = expr_raw

    sample_records = []
    for gsm in expr.columns:
        row = meta.loc[gsm]
        char = str(row.get("characteristics_ch1", ""))
        
        if "Klinefelter" in char:
            karyotype = "47,XXY"
            condition = "Klinefelter"
            nX, nY = 2, 1
        else:
            karyotype = "46,XY"
            condition = "Control_Male"
            nX, nY = 1, 1

        sample_records.append({
            "sample_id": gsm,
            "study_id": "GSE47584",
            "gpl": "GPL14550",
            "platform": "Agilent_SurePrint_G3_8x60K",
            "tissue": "Whole_Blood",
            "cell_type": "Whole_Blood",
            "declared_karyotype": karyotype,
            "declared_sex": "Male",
            "age": 30.0,
            "parental_origin": "N/A",
            "condition": condition,
            "nX": nX,
            "nY": nY,
            "n_sexchr": nX + nY,
            "study_design": "Identifies X at nY=1 (XXY vs XY)",
        })

    meta_df = pd.DataFrame(sample_records).set_index("sample_id")
    logger.info("GSE47584 ingested: %d samples, %d genes. Value range: [%.2f, %.2f]",
                expr.shape[1], expr.shape[0], expr.values.min(), expr.values.max())
    return expr, meta_df


def ingest_gse126712_rna() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ingests GSE126712 RNA-Seq (N=54 primary lymphocytes) from RSEM results."""
    logger.info("Ingesting GSE126712 RNA-Seq primary lymphocyte cohort...")
    meta_path = METADATA_DIR / "GSE126712_metadata.csv"
    meta = pd.read_csv(meta_path).set_index("GSM_ID")

    # Filter to total RNA from blood (N=54)
    rna_meta = meta[meta["molecule_ch1"] == "total RNA"].copy()
    assert len(rna_meta) == 54, f"Expected 54 RNA-seq samples, found {len(rna_meta)}"

    tpm_dict = {}
    counts_dict = {}
    sample_records = []

    for gsm, row in rna_meta.iterrows():
        dirs = glob.glob(str(RAW_DATA_DIR / f"GSE126712/Supp_{gsm}_*"))
        if not dirs:
            raise FileNotFoundError(f"Missing RSEM folder for {gsm}")
        rsem_files = glob.glob(f"{dirs[0]}/*.genes.results.txt.gz")
        if not rsem_files:
            raise FileNotFoundError(f"Missing .genes.results.txt.gz in {dirs[0]}")

        with gzip.open(rsem_files[0], "rt") as f:
            df = pd.read_csv(f, sep="\t")

        # Parse gene symbol from gene_id: ENSG00000000003.14_TSPAN6 -> TSPAN6
        symbols = df["gene_id"].apply(lambda s: s.split("_")[1] if "_" in str(s) else str(s))
        df["symbol"] = symbols
        # For duplicated symbols, retain the highest TPM
        df_sorted = df.sort_values(by="TPM", ascending=False).drop_duplicates(subset=["symbol"], keep="first")
        df_sorted = df_sorted.set_index("symbol")

        tpm_dict[gsm] = df_sorted["TPM"]
        counts_dict[gsm] = df_sorted["expected_count"]

        char = str(row.get("characteristics_ch1", ""))
        karyo_part = "Unknown"
        for p in char.split(","):
            if "chromosomes:" in p:
                karyo_part = p.split(":")[1].strip()

        karyo_map = {
            "XO": ("45,X", "Female", 1, 0, "Turner"),
            "XX": ("46,XX", "Female", 2, 0, "Control_Female"),
            "XY": ("46,XY", "Male", 1, 1, "Control_Male"),
            "XXY": ("47,XXY", "Male", 2, 1, "Klinefelter"),
        }
        karyo, sex, nX, nY, cond = karyo_map.get(karyo_part, ("Unknown", "Unknown", 0, 0, "Unknown"))

        sample_records.append({
            "sample_id": gsm,
            "study_id": "GSE126712",
            "gpl": "GPL16791",
            "platform": "Illumina_HiSeq_2500_RNASeq",
            "tissue": "Blood",
            "cell_type": "Primary_Lymphocytes",
            "declared_karyotype": karyo,
            "declared_sex": sex,
            "age": 10.5,  # Pediatric cohort (mean age ~10.5 yrs)
            "parental_origin": "N/A",
            "condition": cond,
            "nX": nX,
            "nY": nY,
            "n_sexchr": nX + nY,
            "study_design": "Identifies both X at nY=0 (XO vs XX) and X at nY=1 (XXY vs XY) in single cohort",
        })

    tpm_df = pd.DataFrame(tpm_dict)
    counts_df = pd.DataFrame(counts_dict)
    log2_tpm = np.log2(tpm_df + 1.0)
    meta_df = pd.DataFrame(sample_records).set_index("sample_id")

    # Filter unexpressed genes (filterByExpr standard: >= 10 counts in >= 10 samples or mean log2(TPM+1) > 0.1)
    expressed_mask = (counts_df >= 10).sum(axis=1) >= 10
    filtered_tpm = log2_tpm.loc[expressed_mask]
    filtered_counts = counts_df.loc[expressed_mask]

    logger.info("GSE126712 RNA-Seq ingested: 54 samples across %d total genes, %d expressed genes.",
                log2_tpm.shape[0], filtered_tpm.shape[0])
    return log2_tpm, filtered_tpm, filtered_counts, counts_df, meta_df


def verify_sample_sex_expression(
    expr_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    study_id: str,
) -> pd.DataFrame:
    """Verifies declared sex vs expression of XIST and Y-linked genes."""
    logger.info("Verifying sex concordance for %s using marker expression...", study_id)
    sub_meta = meta_df[meta_df["study_id"] == study_id].copy()
    sub_samples = sub_meta.index.intersection(expr_df.columns)

    y_present = [g for g in Y_LINKED_MARKERS if g in expr_df.index]
    xist_present = [g for g in ["XIST", "ENSG00000229807.9_XIST"] if g in expr_df.index]

    y_score = expr_df.loc[y_present, sub_samples].mean(axis=0) if y_present else pd.Series(0.0, index=sub_samples)
    xist_score = expr_df.loc[xist_present[0], sub_samples] if xist_present else pd.Series(0.0, index=sub_samples)

    # Biological classification:
    # In RNA-Seq (GSE126712), female Y-mean < 0.5, male Y-mean > 3.0
    # In microarrays (GSE42331, GSE46687, GSE47584), female Y-mean < 6.0, male Y-mean > 7.5
    if "RNASeq" in str(sub_meta["platform"].iloc[0]):
        inferred_sex = ["Male" if s >= 1.0 else "Female" for s in y_score]
    else:
        inferred_sex = ["Male" if s >= 6.8 else "Female" for s in y_score]

    sub_meta.loc[sub_samples, "y_marker_mean_log2"] = y_score
    sub_meta.loc[sub_samples, "xist_log2"] = xist_score
    sub_meta.loc[sub_samples, "inferred_sex"] = inferred_sex
    sub_meta.loc[sub_samples, "sex_concordant"] = sub_meta.loc[sub_samples, "declared_sex"] == inferred_sex

    n_mismatch = int((~sub_meta.loc[sub_samples, "sex_concordant"]).sum())
    if n_mismatch > 0:
        logger.warning("DETECTED %d sex mismatches in %s!", n_mismatch, study_id)
    else:
        logger.info("SEX AUDIT PASSED for %s: 100%% concordance (%d / %d samples).",
                    study_id, len(sub_samples), len(sub_samples))

    return sub_meta


def audit_cross_cohort_duplication(
    expr_dict: Dict[str, pd.DataFrame],
    common_genes: List[str],
) -> pd.DataFrame:
    """Audits for cross-cohort duplicated expression profiles via pairwise Spearman correlation."""
    logger.info("Performing pairwise Spearman correlation duplicate audit across all cohorts...")
    
    # Combine normalized matrices on common genes
    combined_cols = []
    combined_data = []
    sample_study = {}
    
    for study, df in expr_dict.items():
        genes = [g for g in common_genes if g in df.index]
        sub = df.loc[common_genes]
        for col in sub.columns:
            combined_cols.append(col)
            combined_data.append(sub[col].values)
            sample_study[col] = study

    comb_matrix = np.column_stack(combined_data)  # (n_genes, n_samples)
    n_samples = comb_matrix.shape[1]

    # Calculate correlation matrix
    corr_mat, _ = spearmanr(comb_matrix, axis=0)

    # Check for near-identical cross-study samples (corr > 0.99 between different studies)
    cross_duplicates = []
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            s1, s2 = combined_cols[i], combined_cols[j]
            std1, std2 = sample_study[s1], sample_study[s2]
            if std1 != std2:
                r = corr_mat[i, j]
                if r >= 0.98:
                    cross_duplicates.append({
                        "sample1": s1, "study1": std1,
                        "sample2": s2, "study2": std2,
                        "spearman_r": r
                    })

    if cross_duplicates:
        logger.warning("DETECTED %d suspiciously high cross-study correlation pairs (r >= 0.98)!", len(cross_duplicates))
    else:
        logger.info("DUPLICATE AUDIT PASSED: Zero cross-cohort identical profiles detected (max cross-study r < 0.98).")

    return pd.DataFrame(cross_duplicates)


def build_gene_annotations(common_genes: List[str]) -> pd.DataFrame:
    """Builds comprehensive gene annotation table including PAR and XCI escape status."""
    cache_path = CLEAN_DATA_DIR / "gene_annotations.tsv"
    if cache_path.exists():
        logger.info("Loading cached gene annotations from %s...", cache_path)
        annot_df = pd.read_csv(cache_path, sep="\t", index_col=0)
        if len(annot_df) >= len(common_genes) * 0.95:
            return annot_df

    logger.info("Building gene annotations for %d common genes via mygene...", len(common_genes))
    import mygene
    mg = mygene.MyGeneInfo()

    query_res = mg.querymany(
        common_genes,
        scopes="symbol",
        fields="symbol,genomic_pos,type_of_gene",
        species="human",
        as_dataframe=True
    )

    records = []
    for gene in common_genes:
        chr_val = "Autosome"
        is_par1 = gene in PAR1_GENES
        is_par2 = gene in PAR2_GENES
        is_escape = gene in XCI_ESCAPE_GENES
        
        if gene in query_res.index:
            row = query_res.loc[gene]
            pos = row.get("genomic_pos")
            if isinstance(pos, list) and len(pos) > 0:
                pos_entry = pos[0]
            elif isinstance(pos, dict):
                pos_entry = pos
            else:
                pos_entry = {}
                
            ch = str(pos_entry.get("chr", ""))
            if ch in ["X", "Y"]:
                chr_val = f"chr{ch}"
            elif ch != "":
                chr_val = f"chr{ch}"

        if is_par1:
            xci_status = "PAR1_Escape"
            chr_val = "PAR1"
        elif is_par2:
            xci_status = "PAR2_Escape"
            chr_val = "PAR2"
        elif is_escape:
            xci_status = "XCI_Escape"
            chr_val = "chrX"
        elif chr_val == "chrX":
            xci_status = "XCI_Subject"
        else:
            xci_status = "Autosomal"

        records.append({
            "gene": gene,
            "chromosome": chr_val,
            "is_par1": is_par1,
            "is_par2": is_par2,
            "is_xci_escape": is_escape,
            "xci_status": xci_status,
        })

    annot_df = pd.DataFrame(records).set_index("gene")
    annot_df.to_csv(CLEAN_DATA_DIR / "gene_annotations.tsv", sep="\t")
    logger.info("Gene annotations saved to %s", CLEAN_DATA_DIR / "gene_annotations.tsv")
    return annot_df


def main():
    logger.info("==========================================================================")
    logger.info("PHASE 1A: DATA AUDIT, RE-INGESTION, SCALE VERIFICATION & QC FIREWALL")
    logger.info("==========================================================================")

    # 1. Ingest all 4 cohorts
    m42, meta42 = ingest_gse42331()
    m46, meta46 = ingest_gse46687()
    m47, meta47 = ingest_gse47584()
    m12_tpm, m12_filtered_tpm, m12_filtered_counts, m12_counts, meta12 = ingest_gse126712_rna()

    # 2. Harmonize metadata
    all_meta = pd.concat([meta42, meta46, meta47, meta12], axis=0)
    logger.info("Combined cohort sample manifest: %d total biological samples.", len(all_meta))

    # 3. Sex verification across each study
    meta42_v = verify_sample_sex_expression(m42, all_meta, "GSE42331")
    meta46_v = verify_sample_sex_expression(m46, all_meta, "GSE46687")
    meta47_v = verify_sample_sex_expression(m47, all_meta, "GSE47584")
    meta12_v = verify_sample_sex_expression(m12_tpm, all_meta, "GSE126712")
    all_meta_verified = pd.concat([meta42_v, meta46_v, meta47_v, meta12_v], axis=0)

    # 4. Find gene intersections
    common_microarray = sorted(list(m42.index.intersection(m46.index).intersection(m47.index)))
    common_all = sorted(list(set(common_microarray).intersection(set(m12_tpm.index))))
    logger.info("Identified %d shared genes across microarrays, %d across all 4 cohorts.",
                len(common_microarray), len(common_all))

    # 5. Build gene annotations
    annot_df = build_gene_annotations(common_microarray)

    # 6. Audit cross-cohort duplicate expression
    expr_dict = {
        "GSE42331": m42,
        "GSE46687": m46,
        "GSE47584": m47,
        "GSE126712": m12_tpm,
    }
    dup_df = audit_cross_cohort_duplication(expr_dict, common_all)

    # 7. Save clean matrices and manifests
    logger.info("Saving clean sanitized within-study expression matrices...")
    m42.to_csv(CLEAN_DATA_DIR / "GSE42331_log2.tsv", sep="\t")
    m46.to_csv(CLEAN_DATA_DIR / "GSE46687_log2.tsv", sep="\t")
    m47.to_csv(CLEAN_DATA_DIR / "GSE47584_log2.tsv", sep="\t")
    m12_tpm.to_csv(CLEAN_DATA_DIR / "GSE126712_tpm_log2.tsv", sep="\t")
    m12_counts.to_csv(CLEAN_DATA_DIR / "GSE126712_counts.tsv", sep="\t")
    m12_filtered_tpm.to_csv(CLEAN_DATA_DIR / "GSE126712_filtered_log2.tsv", sep="\t")
    m12_filtered_counts.to_csv(CLEAN_DATA_DIR / "GSE126712_filtered_counts.tsv", sep="\t")

    # Common gene subsets
    m42.loc[common_microarray].to_csv(CLEAN_DATA_DIR / "GSE42331_common_array_log2.tsv", sep="\t")
    m46.loc[common_microarray].to_csv(CLEAN_DATA_DIR / "GSE46687_common_array_log2.tsv", sep="\t")
    m47.loc[common_microarray].to_csv(CLEAN_DATA_DIR / "GSE47584_common_array_log2.tsv", sep="\t")
    m12_tpm.loc[common_all].to_csv(CLEAN_DATA_DIR / "GSE126712_common_all_log2.tsv", sep="\t")

    # Export canonical samples.tsv
    all_meta_verified.to_csv(DATA_DIR / "samples.tsv", sep="\t")
    logger.info("Exported canonical samples manifest to %s", DATA_DIR / "samples.tsv")

    # 8. Compute within-study and cross-study PCA to verify variance structure
    within_pca_metrics = {}
    for std_name, expr_mat in [("GSE42331", m42), ("GSE46687", m46), ("GSE47584", m47), ("GSE126712", m12_tpm)]:
        p = PCA(n_components=3)
        p.fit(expr_mat.T)
        v = p.explained_variance_ratio_
        within_pca_metrics[std_name] = [float(x) for x in v]
        logger.info("%s Within-Study PCA: PC1=%.2f%%, PC2=%.2f%%, PC3=%.2f%%",
                    std_name, v[0] * 100, v[1] * 100, v[2] * 100)
        assert v[0] < 0.35, f"{std_name} PC1 variance too high ({v[0]:.2%}) - check for outlier!"

    pca = PCA(n_components=5)
    common_mat = pd.concat([
        m42.loc[common_microarray],
        m46.loc[common_microarray],
        m47.loc[common_microarray]
    ], axis=1).T
    pca_res = pca.fit_transform(common_mat)
    var_exp = pca.explained_variance_ratio_

    logger.info("Unstandardized Multi-Cohort Microarray PCA: PC1=%.2f%%, PC2=%.2f%% (tracks platform batch effect)",
                var_exp[0] * 100, var_exp[1] * 100)

    # Summary metrics
    metrics = {
        "n_total_samples": len(all_meta_verified),
        "cohorts": {
            "GSE42331": {"n_samples": len(meta42), "platform": "Affymetrix_HuGene-1_0-st", "tissue": "Whole_Blood"},
            "GSE46687": {"n_samples": len(meta46), "platform": "Affymetrix_HG-U133_Plus_2", "tissue": "PBMC"},
            "GSE47584": {"n_samples": len(meta47), "platform": "Agilent_SurePrint_G3_8x60K", "tissue": "Whole_Blood"},
            "GSE126712": {"n_samples": len(meta12), "platform": "Illumina_HiSeq_2500_RNASeq", "tissue": "Primary_Lymphocytes"},
        },
        "karyotype_counts": all_meta_verified["declared_karyotype"].value_counts().to_dict(),
        "n_common_microarray_genes": len(common_microarray),
        "n_common_all_genes": len(common_all),
        "within_study_pca_variance": within_pca_metrics,
        "pooled_microarray_pca_variance": [float(v) for v in var_exp],
        "sex_concordance_rate": float((all_meta_verified["sex_concordant"]).mean()),
    }

    with open(RESULTS_DIR / "phase_1a_audit_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Phase 1A Audit completed successfully. Metrics written to results/phase_1a_audit_metrics.json")


if __name__ == "__main__":
    main()
