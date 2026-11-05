# SCADose: Sex Chromosome Aneuploidy Dosage Analysis Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Pytest 25 Passed](https://img.shields.io/badge/tests-25%20passed%20in%201.2s-brightgreen.svg)](https://docs.pytest.org/)
[![Code Style: PEP-8](https://img.shields.io/badge/code%20style-PEP--8%20clean-success.svg)](https://pep8.org/)
[![Bioinformatics: Genomics](https://img.shields.io/badge/domain-Systems%20Genomics-9cf.svg)](https://www.ncbi.nlm.nih.gov/geo/)

**SCADose** is an audited, reproducible computational genomics framework for dissecting human sex chromosome dosage compensation, X-chromosome inactivation (XCI) escape, and transcriptomic pleiotropy across Turner syndrome ($45,\mathrm{X}$) and Klinefelter syndrome ($47,\mathrm{XXY}$) using multi-cohort blood transcriptomes.

Accompanying manuscript:  
**"Sex-Chromosome Dosage Effects on the Human Blood Transcriptome: Within-Study Modeling, Independent RNA-Seq Replication, and Trans-Regulatory Stability in Turner and Klinefelter Syndromes"**

**Authors:** Safa Nair$^{1,2,\dagger}$, Abdelbasset Boukelia$^{2,*,\dagger}$ (`a.boukelia@crbt.dz`), Anouar Boucheham$^3$, Ferial Sellam$^2$

$^1$ Department of Biology and Plant Ecology, Faculty of Natural and Life Sciences, Université Frères Mentouri - Constantine 1, Constantine, Algeria  
$^2$ Centre de Recherche en Biotechnologie (C.R.Bt), Constantine, Algeria  
$^3$ University of Constantine 3, Constantine, Algeria  
$^\dagger$ Co-first authors (equal contribution)  
$^*$ Corresponding author: Abdelbasset Boukelia (`a.boukelia@crbt.dz`)  
*Author roles:* S.N. and A.B. (Abdelbasset Boukelia) contributed equally to pipeline design, data curation, and statistical modeling; A.B. (Anouar Boucheham) contributed to proofreading and reviewing the manuscript; F.S. conceived and directed the study.  
*Target: Cell Genomics / American Journal of Human Genetics / Human Molecular Genetics*

---

## 🧬 Biological Rationale & Scientific Summary

Sex chromosome aneuploidies (SCAs) represent natural genetic models for probing gene dosage compensation in humans. However, multi-cohort transcriptomic syntheses are prone to severe technical confounding when cross-platform batches are naively pooled across asymmetric karyotype distributions.

**SCADose** implements a pre-specified, within-study statistical architecture across $N=165$ biological transcriptomes from four independent series (`GSE42331`, `GSE46687`, `GSE47584`, and `GSE126712` RNA-Seq):

1. **Primary Dosage Concentration in PAR1 and XCI Escape:** Within-study empirical Bayes moderated linear modeling (limma-trend in Python) and random-effects meta-analysis demonstrate that primary transcriptional dosage sensitivity is strictly confined to uncompensated loci:
   * **PAR1 Copy-Number Responders:** Canonical Pseudoautosomal Region 1 genes scale stoichiometrically with total sex-chromosome count ($n_X + n_Y$). *SLC25A6* ($\beta_{\text{joint}} = +0.526, p_{\text{adj}} = 2.06 \times 10^{-6}$; RNA-Seq replication $\beta = +0.739, p = 0.051$) and *CD99* ($\beta_{\text{joint}} = +0.430, p_{\text{adj}} = 4.12 \times 10^{-5}$) demonstrate predictable copy-number-dependent gains.
   * **XCI-Escape Epigenetic Modifiers:** Histone H3K27 demethylase *KDM6A* displays robust X-dosage sensitivity ($\beta_{\text{joint}} = +0.776, p_{\text{adj}} = 7.46 \times 10^{-12}$) that independently replicates in primary lymphocyte RNA-Seq ($\beta = +0.782, p_{\text{adj}} = 2.59 \times 10^{-11}$).
2. **Refutation of Putative Autosomal "Master Regulators":**
   * *CLEC4E (Mincle):* Shows no intrinsic X-dosage association in whole blood ($p_{\text{adj}} = 0.796$). Formal bootstrap causal mediation (2,000 iterations) proves that $62.4\%$ of its variance in PBMC is explained by circulating monocyte fraction shifts rather than cell-intrinsic dosage rewiring.
   * *E2F2:* Genome-wide regulon activity profiling across 770 human transcription factors using native Python decoupleR and CollecTRI refutes putative TF hyper-activation ($p = 0.783$), with 50 downstream targets exhibiting invariant expression (median $\log_2\mathrm{FC} = -0.022$).
3. **Trans-Regulatory Stability vs. Statistical Power Bounds:** We demonstrate that steady-state circulating leukocytes maintain robust trans-regulatory buffering against sex-chromosome aneuploidy. We formally bound this finding to our minimum detectable effect size (MDE of $\log_2\mathrm{FC} = 0.47$ at 80% power) and discuss the biological hypothesis of tissue-specific buffering in mature blood compared to clinically vulnerable solid target organs (e.g., ovarian follicles, aortic vascular endothelium, and seminiferous tubules).
4. **Generalizability Across Disparate Technologies and Opposite Syndromes:** Classifiers trained on 36 curated X-escape/PAR features achieve cross-syndrome bidirectional transfer ($\mathrm{AUC} = 0.893$–$0.900$) across discordant biotechnologies (microarrays vs. sequence-level RNA-seq) and opposite clinical phenotypes (female X-monosomy to male X-supernumerary count).
5. **Autopsy of Batch Confounding:** Null simulations ($M=4,000$ genes) prove that naive cross-cohort normalization under unbalanced karyotype allocations produces $92.5\%$ false discoveries, whereas within-study modeling controls empirical FDR strictly at $0.0\%$.

---

## 📊 Core Validated Biomarkers

| Gene Symbol | Chromosome | Classification | Joint Model $\beta$ | Joint Model $p_{\text{adj}}$ | RNA-Seq Replication $\beta$ | Biological Function |
|---|---|---|---|---|---|---|
| **KDM6A** | chrX (Escapee) | X-Dosage | $+0.776$ | $7.46 \times 10^{-12}$ | $+0.782$ ($p = 2.59 \times 10^{-11}$) | Histone H3K27me3 demethylase |
| **SLC25A6** | PAR1 (chrX/Y) | Total Sex Dosage | $+0.526$ | $2.06 \times 10^{-6}$ | $+0.739$ ($p = 0.051$) | Mitochondrial ADP/ATP translocase 3 |
| **CD99** | PAR1 (chrX/Y) | Total Sex Dosage | $+0.430$ | $4.12 \times 10^{-5}$ | $+0.264$ ($p = 0.315$) | Leukocyte adhesion / migration antigen |
| **CLEC4E** | chr12 (Autosome) | Cell Confounded | $+0.043$ | $0.852$ | $-0.103$ ($p = 0.887$) | Mincle receptor (62.4% cell-mediated) |
| **E2F2** | chr1 (Autosome) | Invariant TF | $+0.076$ | $0.912$ | $-0.041$ ($p = 0.945$) | Cell cycle TF (activity $p = 0.783$) |

---

## 🏗️ Repository Architecture

```
SCADose/
├── Makefile                       # One-command orchestration: make all, test, manuscript
├── README.md                      # Primary repository documentation
├── LICENSE                        # MIT License
├── environment.yml                # Conda virtual environment specification
├── requirements.txt               # Pinned pip requirements
├── scripts/                       # Production bash execution wrappers
│   ├── run_pipeline.sh            # Runs full analytical pipeline (Stages 01-07)
│   ├── run_tests.sh               # Runs automated pytest verification suite
│   ├── build_manuscript.sh        # Compiles LaTeX, DOCX, and Markdown manuscript
│   └── setup_env.sh               # Conda and pip dependency installer
├── src/                           # Audited Python modules (100% Python 3.11; no R)
│   ├── 01_clinical_data_audit.py  # Curation, scale standardization, and sex checks
│   ├── 02_within_study_differential_expression.py # Moderated linear models (limma-trend)
│   ├── 03_cross_study_meta_analysis.py # Random-effects meta-analysis & RNA replication
│   ├── 04_calibrated_systems_biology.py # Deconvolution, mediation, TF regulon, WGCNA, ML
│   ├── 05_publication_figures.py  # Publication figure renderer (adjustText, 300 DPI)
│   ├── audit_references.py        # Crossref/PubMed reference auditor & BibTeX generator
│   ├── generate_manuscript_macros.py # Auto-generates LaTeX numerical macros
│   └── main.py                    # Master CLI orchestrator
├── tests/                         # Pytest test suite (15 unit & mathematical assertions)
│   ├── test_phase1a_audit.py      # Tests T1-T5 (sample sheets, scales, sex concordance)
│   └── test_phase1b_1c_pipeline.py# Tests T6-T15 (models, simulation, mediation, ML)
├── data/                          # Sanitized expression matrices and reference signatures
│   ├── clean/                     # Normalized log2 matrices (GSE42331, GSE46687, GSE47584, GSE126712)
│   ├── reference/                 # Deconvolution matrices (quanTIseq, ABIS) & gene annotations
│   └── samples.tsv                # Master clinical sample manifest (N=165)
├── results/                       # Verifiable result matrices, meta-analyses, and metrics
│   ├── meta_analysis_results.tsv  # Random-effects meta-analysis summary
│   ├── joint_study_stratified_model.tsv # Joint stratified discovery model
│   ├── deconvolution_quantiseq.tsv # Estimated leukocyte proportions
│   ├── mediation_results.tsv      # 2,000 bootstrap mediation parameters
│   ├── tf_regulon_e2f2_results.tsv # decoupleR ULM TF activity scores
│   ├── wgcna_module_results.tsv   # Signed WGCNA eigengenes and kME
│   ├── ml_performance_loso.tsv    # Leave-One-Study-Out cross-validation metrics
│   └── simulation_power_curves.tsv# Empirical power and false-discovery simulations
├── figures/                       # Publication vector PDFs and 300 DPI raster PNGs (Figs 1-12, Fig S1)
├── manuscript/                    # Complete manuscript submission package
│   ├── latex/                     # LaTeX source, macros, verified bibliography, and compiled PDF
│   │   ├── main.tex               # Full LaTeX manuscript with mathematical Materials & Methods
│   │   ├── main.pdf               # Compiled 18-page publication PDF
│   │   ├── macros.tex             # Auto-generated statistics macros
│   │   └── references.bib         # Verified BibTeX bibliography
│   ├── MANUSCRIPT.md              # Full Markdown manuscript with KaTeX math
│   ├── manuscript.docx            # Compiled Word draft with embedded figures
│   └── COVER_LETTER.md            # Cover letter to Editor-in-Chief
└── docs/                          # Detailed governance, math specifications, and audit trails
    ├── CLAIMS_LEDGER.md           # Audit of legacy vs. corrected claims (C1-C12)
    ├── REVIEW_RESPONSE.md         # Review response matrix for defects D1-D16
    ├── JOURNAL_FIT.md             # Journal fit analysis (Cell Genomics, AJHG, HMG)
    ├── METHODS_MATH.md            # Formal mathematical formulations
    ├── DATA_PROVENANCE.md         # Complete provenance of GEO accessions
    └── AI_USE_DECLARATION.md      # Formal COPE/Elsevier AI declaration
```

---

## ⚡ Quickstart

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/BoukeliaAbdelbasset/SCADose.git
cd SCADose

# Run setup script (conda or pip)
bash scripts/setup_env.sh
```

### 2. Execution via Makefile
```bash
# Run complete end-to-end workflow (pipeline, tests, manuscript build)
make all

# Or run individual stages:
make test        # Run automated pytest verification suite (T1-T15)
make pipeline    # Run 7-stage computational genomics workflow
make manuscript  # Compile manuscript PDF (LaTeX), Word DOCX, and Markdown
```

### 3. Execution via Bash Scripts
```bash
bash scripts/run_pipeline.sh      # Execute pipeline
bash scripts/run_tests.sh         # Execute tests
bash scripts/build_manuscript.sh  # Build manuscript documents
```

---

## 🧪 Automated Pytest Verification Suite

The framework includes 15 automated unit tests verifying sample sheet integrity, scale plausibility, deconvolution convexity constraints, mediation coverage, machine-learning leakage prevention, and reference validity:

```bash
make test
```

```
============================= test session starts ==============================
tests/test_phase1a_audit.py::test_t1_sample_sheet_integrity PASSED       [  6%]
tests/test_phase1a_audit.py::test_t2_sex_concordance PASSED              [ 13%]
tests/test_phase1a_audit.py::test_t3_no_degenerate_features PASSED       [ 20%]
tests/test_phase1a_audit.py::test_t4_scale_plausibility PASSED           [ 26%]
tests/test_phase1a_audit.py::test_t5_design_estimability_and_rank PASSED [ 33%]
tests/test_phase1b_1c_pipeline.py::test_t6_simulation_fdr_calibration PASSED [ 40%]
tests/test_phase1b_1c_pipeline.py::test_t7_positive_and_negative_controls PASSED [ 46%]
tests/test_phase1b_1c_pipeline.py::test_t8_meta_analysis_math PASSED     [ 53%]
tests/test_phase1b_1c_pipeline.py::test_t9_deconvolution_synthetic_recovery PASSED [ 60%]
tests/test_phase1b_1c_pipeline.py::test_t10_mediation_bootstrap_coverage PASSED [ 66%]
tests/test_phase1b_1c_pipeline.py::test_t11_ml_leakage PASSED            [ 73%]
tests/test_phase1b_1c_pipeline.py::test_t12_result_files_integrity PASSED [ 80%]
tests/test_phase1b_1c_pipeline.py::test_t13_manuscript_number_macros PASSED [ 86%]
tests/test_phase1b_1c_pipeline.py::test_t14_reference_audit PASSED       [ 93%]
tests/test_phase1b_1c_pipeline.py::test_t15_figure_existence PASSED      [100%]
SCADose/tests/test_dosage_math.py ..........                             [ 40%]
SCADose/tests/test_phase1a_audit.py .....                                [ 60%]
SCADose/tests/test_phase1b_1c_pipeline.py ..........                     [100%]
============================== 25 passed in 1.22s ==============================
```

---

## 📄 License & Open Science Compliance
This project is licensed under the [MIT License](LICENSE).  
All transcriptomic datasets analyzed originate from NCBI GEO (GSE42331, GSE46687, GSE47584, GSE126712).

