#!/usr/bin/env python3
"""
format_manuscript_md.py
======================
Post-processes the pandoc-generated Markdown file to ensure that full
publication headers, authors, affiliations, correspondence, abstract,
highlights, and references are beautifully formatted and fully intact
in Elsevier format.
"""

import sys
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parent.parent
    manuscript_dir = repo_root / "manuscript"
    target_md = manuscript_dir / "MANUSCRIPT.md"

    header = """# Sex-Chromosome Dosage Effects on the Human Blood Transcriptome: Within-Study Modeling, Independent RNA-Seq Replication, and Trans-Regulatory Stability in Turner and Klinefelter Syndromes

**Safa Nair**$^{1,2,\\dagger}$ (`s.nair@crbt.dz`), **Abdelbasset Boukelia**$^{2,*,\\dagger}$ (`a.boukelia@crbt.dz`), **Anouar Boucheham**$^3$ (`anouar.boucheham@univ-constantine3.dz`), **Ferial Sellam**$^2$ (`f.sellam@crbt.dz`)

$^1$ Department of Biology and Plant Ecology, Faculty of Natural and Life Sciences, Université Frères Mentouri - Constantine 1, Route de Aïn El Bey, 25017 Constantine, Algeria  
$^2$ Centre de Recherche en Biotechnologie (C.R.Bt), Nouvelle Ville Ali Mendjli, 25000 Constantine, Algeria  
$^3$ University of Constantine 3, Nouvelle Ville Ali Mendjli, 25000 Constantine, Algeria  
$^\\dagger$ These authors contributed equally to this work.  
$^*$ Corresponding author: Abdelbasset Boukelia (`a.boukelia@crbt.dz`)

*Journal:* **Molecular Genetics and Metabolism** (Elsevier)

---

## Highlights

- Sex-chromosome dosage responses in blood localize strictly to PAR1 and escapees.
- Mitochondrial ADP/ATP translocase SLC25A6 scales with active sex chromosome dose.
- Histone demethylase KDM6A displays reproducible X-dosage sensitivity in RNA-Seq.
- Autosomal candidate regulators (CLEC4E, E2F2) reflect leukocyte composition shifts.
- Blood buffers autosomal expression, bounding trans-effects by $\\log_2\\mathrm{FC} \\ge 0.47$.

---

## Abstract

Sex chromosome aneuploidies alter gene copy number across human sex chromosomes and associate with pronounced endocrine and metabolic disturbances. In Turner syndrome ($45,\\mathrm{X}$) and Klinefelter syndrome ($47,\\mathrm{XXY}$), distinct sex-chromosome complements offer a natural experiment to assess dosage compensation, mitochondrial transport, and epigenetic regulation. Cross-platform transcriptomic studies in rare disorders frequently confound biological differences with technical batch effects when karyotypes segregate unevenly across cohorts. Here, we analyze $N=165$ blood transcriptomes across four independent series (GSE42331, GSE46687, GSE47584, and GSE126712 RNA-Seq) using pre-specified within-study linear models. Primary transcriptional dosage responses localize almost exclusively to pseudoautosomal region 1 (PAR1) and authentic X-chromosome inactivation (XCI) escape genes. Canonical PAR1 loci scale with total sex-chromosome count ($n_X + n_Y$): the inner mitochondrial membrane ADP/ATP translocase **SLC25A6** ($\\beta_{\\text{joint}} = +0.526, p_{\\text{adj}} = 1.71 \\times 10^{-6}$) and **CD99** ($\\beta_{\\text{joint}} = +0.430, p_{\\text{adj}} = 4.12 \\times 10^{-5}$) show predictable linear increments per active chromosome dose. The epigenetic histone H3K27 demethylase **KDM6A** displays consistent X-dosage sensitivity ($\\beta_{\\text{joint}} = +0.776, p_{\\text{adj}} = 7.46 \\times 10^{-12}$) that replicates in independent RNA-Seq ($\\beta = +0.782, p_{\\text{adj}} = 2.59 \\times 10^{-11}$). Previously reported autosomal master regulators fail replication. **CLEC4E** exhibits no intrinsic X-dosage association in whole blood ($\\log_2\\mathrm{FC} = -0.133, p_{\\text{adj}} = 0.7958$); bootstrap mediation indicates that 62.4% of its apparent change in PBMC reflects myeloid fraction shifts. Regulatory activity for transcription factor **E2F2** remains unchanged across karyotypes ($p = 0.7828$), and downstream target genes show near-zero slopes (median $\\log_2\\mathrm{FC} = -0.022$). Synthetic simulations show that naive pooling under unbalanced sampling yields 92.5% false discoveries, whereas within-study modeling controls empirical error at 0.0%. Classifiers trained on curated escape and PAR loci generalize across syndromes ($\\mathrm{AUC} = 0.89$--$0.90$), surpassing unconstrained whole-transcriptome models ($\\mathrm{AUC} = 0.77$). These results indicate that human peripheral blood buffers autosomal expression against sex-chromosome aneuploidy, confining primary dosage effects to uncompensated mitochondrial and epigenetic sex-linked loci.

**Keywords:** Sex chromosome aneuploidy; Turner syndrome; Klinefelter syndrome; Mitochondrial ADP/ATP translocase; X-inactivation escape; Transcriptome replication

---

"""

    if not target_md.exists():
        print(f"Error: {target_md} does not exist.")
        sys.exit(1)

    with open(target_md, "r", encoding="utf-8") as f:
        content = f.read()

    # If header already present, replace it or keep body
    if "# Introduction" in content:
        body = content[content.index("# Introduction"):]
    elif "## Introduction" in content:
        body = content[content.index("## Introduction"):]
    else:
        body = content

    # Ensure a proper '# References' header exists before the references block
    if '<div id="refs"' in body and '# References' not in body:
        body = body.replace('<div id="refs"', '# References\n\n<div id="refs"')

    full_md = header + body

    with open(target_md, "w", encoding="utf-8") as f:
        f.write(full_md)

    # Also sync to TheLastRescue if directory exists
    rescue_md = repo_root.parent / "TheLastRescue" / "MANUSCRIPT.md"
    if rescue_md.parent.exists():
        with open(rescue_md, "w", encoding="utf-8") as f:
            f.write(full_md)

    print(f"Successfully formatted: {target_md}")

if __name__ == "__main__":
    main()
