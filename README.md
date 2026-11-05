# SCADose: Sex Chromosome Aneuploidy Dosage Analysis Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Pytest 25 Passed](https://img.shields.io/badge/tests-25%20passed%20in%201.08s-brightgreen.svg)](https://docs.pytest.org/)
[![Code Style: PEP-8](https://img.shields.io/badge/code%20style-PEP--8%20clean-success.svg)](https://pep8.org/)
[![Bioinformatics: Genomics](https://img.shields.io/badge/domain-Systems%20Genomics-9cf.svg)](https://www.ncbi.nlm.nih.gov/geo/)

**SCADose** is an audited, reproducible computational genomics framework for investigating sex-chromosome dosage compensation, pseudoautosomal region 1 (PAR1) expression scaling, X-chromosome inactivation (XCI) escape, and transcriptomic trans-regulatory stability across Turner syndrome ($45,\mathrm{X}$) and Klinefelter syndrome ($47,\mathrm{XXY}$) using multi-cohort human blood transcriptomes.

---

### Accompanying Manuscript

**"Sex-Chromosome Dosage Effects on the Human Blood Transcriptome: Within-Study Modeling, Independent RNA-Seq Replication, and Trans-Regulatory Stability in Turner and Klinefelter Syndromes"**

**Authors:**  
**Safa Nair**$^{1,2,\dagger}$ (`s.nair@crbt.dz`), **Abdelbasset Boukelia**$^{2,*,\dagger}$ (`a.boukelia@crbt.dz`), **Anouar Boucheham**$^3$ (`anouar.boucheham@univ-constantine3.dz`), **Ferial Sellam**$^2$ (`f.sellam@crbt.dz`)

$^1$ Department of Biology and Plant Ecology, Faculty of Natural and Life Sciences, Université Frères Mentouri - Constantine 1, Route de Aïn El Bey, 25017 Constantine, Algeria  
$^2$ Centre de Recherche en Biotechnologie (C.R.Bt), Nouvelle Ville Ali Mendjli, 25000 Constantine, Algeria  
$^3$ University of Constantine 3, Nouvelle Ville Ali Mendjli, 25000 Constantine, Algeria  
$^\dagger$ Co-first authors (equal contribution)  
$^*$ Corresponding author: Abdelbasset Boukelia (`a.boukelia@crbt.dz`)  

*Target Journal:* **Molecular Genetics and Metabolism** (Elsevier)  
*CRediT Authorship Contributions:*  
- **S. Nair & A. Boukelia:** Conceptualization, Methodology, Formal analysis, Software, Validation, Data curation, Writing - original draft, Visualization.
- **A. Boucheham:** Review & proofreading.
- **F. Sellam:** Conceptualization, Project administration, Supervision, Writing - review & editing (Directed the study).

---

## 🧬 Biological Rationale & Scientific Summary

Sex chromosome aneuploidies represent classic human models for deciphering gene dosage compensation and aneuploidy pathophysiology. However, multi-cohort transcriptomic syntheses in rare genetic conditions are vulnerable to severe technical confounding when disparate microarray and sequencing batches are naively pooled across asymmetric karyotype distributions.

**SCADose** establishes a pre-specified within-study modeling architecture across $N=165$ biological transcriptomes from four independent series (`GSE42331`, `GSE46687`, `GSE47584`, and `GSE126712` RNA-Seq), comprising:
- $39 \times 45,\mathrm{X}$ (Turner syndrome)
- $39 \times 46,\mathrm{XX}$ (Female controls)
- $33 \times 46,\mathrm{XY}$ (Male controls)
- $54 \times 47,\mathrm{XXY}$ (Klinefelter syndrome)

### Key Empirical Findings

1. **Stoichiometric Scaling of Mitochondrial ADP/ATP Translocase SLC25A6 (ANT3):**
   Canonical Pseudoautosomal Region 1 (PAR1) loci escape XCI and display strictly linear copy-number increments per active sex chromosome ($n_X + n_Y$). The inner mitochondrial membrane carrier **SLC25A6** ($\beta_{\text{joint}} = +0.526, p_{\text{adj}} = 1.71 \times 10^{-6}$; independent primary lymphocyte RNA-Seq replication $\beta = +0.739, p_{\text{adj}} = 3.57 \times 10^{-12}$) and cell-adhesion antigen **CD99** ($\beta_{\text{joint}} = +0.430, p_{\text{adj}} = 4.12 \times 10^{-5}$) show dosage-dependent expression, directly connecting sex-chromosome copy number to mitochondrial adenine nucleotide exchange and cellular bioenergetics.

2. **Epigenetic Regulation via Histone Demethylase KDM6A:**
   The non-PAR XCI-escape locus **KDM6A** (UTX) displays robust X-dosage sensitivity ($\beta_{\text{joint}} = +0.776, p_{\text{adj}} = 7.46 \times 10^{-12}$) that independently replicates in sequence-level RNA-Seq ($\beta = +0.782, p_{\text{adj}} = 2.59 \times 10^{-11}$). Alterations in KDM6A abundance modulate H3K27me3 demethylation dynamics across developmental and metabolic targets.

3. **Refutation of Putative Autosomal "Master Regulators":**
   - **CLEC4E (Mincle):** Shows no intrinsic X-dosage association in whole blood ($\log_2\mathrm{FC} = -0.133, p_{\text{adj}} = 0.7958$; joint model $\beta = +0.239, p_{\text{adj}} = 0.3579$). Formal causal bootstrap mediation (2,000 resamples) demonstrates that $62.4\%$ (95% CI: $[31.8\%, 94.2\%]$, $p < 0.001$) of its apparent elevation in PBMC reflects leukocyte composition shifts (elevated circulating monocyte fractions) rather than genuine aneuploidy-driven trans-regulation.
   - **E2F2:** DecoupleR regulon activity inference across 770 human transcription factors using CollecTRI refutes putative transcriptional hyper-activation ($p = 0.7828$), with 50 downstream target genes exhibiting flat expression (median $\log_2\mathrm{FC} = -0.022$).

4. **Trans-Regulatory Stability and Cellular Buffering:**
   Steady-state peripheral blood maintains substantial trans-regulatory stability against sex-chromosome aneuploidy. Genome-wide autosomal trans-effects are absent in blood, bounded by our statistical power threshold (Minimum Detectable Effect size $\text{MDE} = \log_2\mathrm{FC} \ge 0.47$ at 80% power). This finding highlights the biological hypothesis of mature blood tissue buffering compared to clinically vulnerable solid target tissues (e.g., ovarian follicles, aortic vascular endothelium, and testicular tubules).

5. **Cross-Syndrome & Cross-Platform Machine Learning Generalizability:**
   Machine learning classifiers trained on 36 curated X-escape and PAR1 loci achieve robust Leave-One-Study-Out generalization (mean LOSO $\mathrm{AUC} = 0.87$, range $0.852$ to $0.900$) and bidirectional cross-syndrome transfer:
   - Turner-to-Klinefelter ($45,\mathrm{X} \to 47,\mathrm{XXY}$): $\mathrm{AUC} = 0.890$
   - Klinefelter-to-Turner ($47,\mathrm{XXY} \to 45,\mathrm{X}$): $\mathrm{AUC} = 0.902$  
   In contrast, unconstrained whole-transcriptome models drop to $\mathrm{AUC} = 0.770$, demonstrating that genome-wide feature spaces overfit to platform-specific technical noise.

6. **Empirical Error Calibration (Simulation):**
   Synthetic null simulations ($M=4,000$ genes) prove that naive multi-cohort pooling under unbalanced karyotype allocations produces $92.5\%$ false discoveries, whereas within-study linear modeling controls empirical FDR strictly at $0.0\%$.

---

## 📊 Core Validated Biomarkers

| Gene Symbol | Chromosome | Locus Class | Joint Model $\beta$ | Joint Model $p_{\text{adj}}$ | RNA-Seq Replication $\beta$ | RNA-Seq $p_{\text{adj}}$ | Primary Function |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **KDM6A** | chrX (Escape) | X-Dosage | $+0.776$ | $7.46 \times 10^{-12}$ | $+0.782$ | $2.59 \times 10^{-11}$ | Histone H3K27me3 demethylase |
| **SLC25A6** | chrX/Y (PAR1) | Total Sex ($n_X + n_Y$) | $+0.526$ | $1.71 \times 10^{-6}$ | $+0.739$ | $3.57 \times 10^{-12}$ | Inner mitochondrial ADP/ATP translocase 3 |
| **CD99** | chrX/Y (PAR1) | Total Sex ($n_X + n_Y$) | $+0.430$ | $4.12 \times 10^{-5}$ | $+0.264$ | $0.315$ | Leukocyte adhesion / migration antigen |
| **CLEC4E** | chr12 (Autosome) | Cell Confounded | $+0.239$ | $0.358$ | $-0.103$ | $0.887$ | Mincle receptor ($62.4\%$ monocyte mediated) |
| **E2F2** | chr1 (Autosome) | Invariant Factor | $+0.174$ | $0.586$ | $-0.041$ | $0.945$ | Transcription factor (activity $p = 0.7828$) |

---

## 🏗️ Repository Architecture

```text
SCADose/
├── Makefile                       # Top-level orchestration: make all, test, pipeline, manuscript
├── README.md                      # Comprehensive project documentation
├── LICENSE                        # MIT License
├── environment.yml                # Conda virtual environment specification
├── requirements.txt               # Pinned pip requirements
├── scripts/                       # Shell automation wrappers
│   ├── run_pipeline.sh            # Runs full analytical workflow (Stages 01-05)
│   ├── run_tests.sh               # Executes automated pytest verification suite (25/25)
│   └── setup_env.sh               # Dependency installer
├── src/                           # Audited Python analytical pipeline (Python 3.11+)
│   ├── 01_clinical_data_audit.py  # Data ingestion, probe calibration, and sex concordance
│   ├── 02_within_study_differential_expression.py # Empirical Bayes linear models (limma-trend)
│   ├── 03_cross_study_meta_analysis.py # Random-effects meta-analysis & RNA-Seq replication
│   ├── 04_calibrated_systems_biology.py # Deconvolution, mediation, TF regulon, WGCNA, ML
│   ├── 05_publication_figures.py  # High-resolution figure renderer (adjustText, 300 DPI)
│   └── main.py                    # Master CLI orchestrator
├── tests/                         # Pytest test suite (25 unit & mathematical assertions)
│   ├── test_dosage_math.py        # Continuous regression, FDR penalty, NNLS deconvolution
│   ├── test_phase1a_audit.py      # Sample sheet integrity, sex checks, scale plausibility
│   └── test_phase1b_1c_pipeline.py# Empirical power, mediation coverage, ML leakage tests
├── data/                          # Sanitized expression matrices and reference signatures
│   ├── clean/                     # Normalized log2 matrices (GSE42331, GSE46687, GSE47584, GSE126712)
│   ├── reference/                 # Reference signatures (quanTIseq, ABIS, Ensembl HGNC mappings)
│   └── samples.tsv                # Master clinical sample manifest (N=165)
├── results/                       # Verifiable result matrices, meta-analyses, and logs
│   ├── meta_analysis_results.tsv  # Random-effects meta-analysis summary
│   ├── joint_study_stratified_model.tsv # Joint stratified discovery model
│   ├── cell_composition_fractions_til10.tsv # Deconvolution immune fractions
│   ├── mediation_analysis_results.tsv # 2,000 bootstrap mediation parameters
│   ├── tf_dosage_associations.tsv # decoupleR ULM TF activity associations
│   ├── wgcna_module_eigengenes.tsv # Signed WGCNA eigengenes and module preservation
│   ├── ml_performance_loso.tsv    # Leave-One-Study-Out cross-validation metrics
│   ├── ml_cross_syndrome_transfer.tsv # Cross-syndrome bidirectional transfer metrics
│   └── simulation_power_curves.tsv# Empirical power and false-discovery simulations
├── figures/                       # Publication vector PDFs and 300 DPI raster PNGs (Figs 1-12, Fig S1)
```

---

## ⚡ Quickstart

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/BoukeliaAbdelbasset/SCADose.git
cd SCADose

# Run environment setup (requires Conda or Python 3.10+)
bash scripts/setup_env.sh
```

### 2. Automated Pipeline Execution via Makefile
```bash
# Run end-to-end workflow (pipeline, test suite, and manuscript compilation)
make all

# Or run individual stages:
make test        # Run automated pytest verification suite (25 tests)
make pipeline    # Run 5-stage computational genomics workflow
make manuscript  # Compile manuscript PDF (LaTeX), Word DOCX, and Markdown
```

### 3. Execution via Shell Scripts
```bash
bash scripts/run_pipeline.sh      # Execute analytical pipeline
bash scripts/run_tests.sh         # Execute verification suite
bash scripts/build_manuscript.sh  # Build publication documents
```

---

## 🧪 Automated Pytest Verification Suite

The repository contains 25 automated mathematical and data integrity tests:

```bash
pytest tests/ -v
```

```text
============================= test session starts ==============================
platform linux -- Python 3.13.7, pytest-8.4.2, pluggy-1.5.0
collecting ... collected 25 items

tests/test_dosage_math.py::test_continuous_dosage_regression_math PASSED [  4%]
tests/test_dosage_math.py::test_benjamini_hochberg_fdr_penalty PASSED    [  8%]
tests/test_dosage_math.py::test_directional_concordance_logic PASSED    [ 12%]
tests/test_dosage_math.py::test_quarantine_leakage_detection PASSED      [ 16%]
tests/test_dosage_math.py::test_real_data_integrity PASSED              [ 20%]
tests/test_dosage_math.py::test_real_dosage_biomarker_results PASSED    [ 24%]
tests/test_dosage_math.py::test_nnls_immune_deconvolution_constraints PASSED [ 28%]
tests/test_dosage_math.py::test_clec4e_multivariate_dissection PASSED    [ 32%]
tests/test_dosage_math.py::test_e2f2_regulon_mechanics PASSED           [ 36%]
tests/test_dosage_math.py::test_wgcna_dosage_module_integrity PASSED    [ 40%]
tests/test_phase1a_audit.py::test_t1_sample_sheet_integrity PASSED       [ 44%]
tests/test_phase1a_audit.py::test_t2_sex_concordance PASSED             [ 48%]
tests/test_phase1a_audit.py::test_t3_no_degenerate_features PASSED      [ 52%]
tests/test_phase1a_audit.py::test_t4_scale_plausibility PASSED          [ 56%]
tests/test_phase1a_audit.py::test_t5_design_estimability_and_rank PASSED[ 60%]
tests/test_phase1b_1c_pipeline.py::test_t6_simulation_fdr_calibration PASSED [ 64%]
tests/test_phase1b_1c_pipeline.py::test_t7_positive_and_negative_controls PASSED [ 68%]
tests/test_phase1b_1c_pipeline.py::test_t8_meta_analysis_math PASSED    [ 72%]
tests/test_phase1b_1c_pipeline.py::test_t9_deconvolution_synthetic_recovery PASSED [ 76%]
tests/test_phase1b_1c_pipeline.py::test_t10_mediation_bootstrap_coverage PASSED [ 80%]
tests/test_phase1b_1c_pipeline.py::test_t11_ml_leakage PASSED           [ 84%]
tests/test_phase1b_1c_pipeline.py::test_t12_result_files_integrity PASSED [ 88%]
tests/test_phase1b_1c_pipeline.py::test_t13_manuscript_number_macros PASSED [ 92%]
tests/test_phase1b_1c_pipeline.py::test_t14_reference_audit PASSED      [ 96%]
tests/test_phase1b_1c_pipeline.py::test_t15_figure_existence PASSED     [100%]

============================== 25 passed in 1.08s ==============================
```

---

## 📄 License & Open Science Compliance

This project is licensed under the [MIT License](LICENSE).  
All transcriptomic datasets analyzed originate from public NCBI GEO accessions:
- [GSE42331](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE42331) (Affymetrix Human Gene 1.0 ST)
- [GSE46687](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE46687) (Affymetrix Human Genome U133 Plus 2.0)
- [GSE47584](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE47584) (Agilent-028004 SurePrint G3 Human GE 8x60K)
- [GSE126712](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE126712) (Illumina HiSeq 4000 RNA-Seq)

For correspondence regarding the data, code, or findings, contact **Abdelbasset Boukelia** (`a.boukelia@crbt.dz`).
