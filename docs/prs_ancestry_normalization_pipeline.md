# PRS Ancestry-Aware Normalization Pipeline

## Purpose

This document specifies a reproducible pipeline to derive ancestry-aware PRS normalization features from genotype data.

The pipeline starts from target genotype data and a 1000 Genomes reference panel, then produces:

1. **Hard-label ancestry normalization with shrinkage**
2. **Soft-label distance-based ancestry normalization with shrinkage**

The intended downstream use is PRS ensemble modeling.

---

## Core Design

```text
target genotype
→ harmonize with 1000G
→ fit PCA on 1000G only
→ project both 1000G and target into the 1000G PC space
→ compute target-to-1000G-center distances
→ derive hard labels and soft similarity weights
→ normalize each raw PRS using hard-label or soft-label ancestry-aware normalization
```

Default settings:

```text
Reference panel: 1000 Genomes
Ancestry groups: AFR, AMR, EAS, EUR, SAS
PCA space: 1000G-only PCA
Number of PCs used for distances: PC1–PC10
Distance metric: Mahalanobis distance
Hard-label ambiguity margin: 0.5
Shrinkage tau: 100
Soft-label RBF tau: 1.0
```

Recommended starting point:

```text
target genotype variant universe
→ target ∩ 1000G by CHR:BP:unordered allele pair
→ nonambiguous autosomal A/C/G/T biallelic SNPs
→ 1000G MAF/missingness QC
→ high-LD exclusion
→ LD pruning on 1000G
```

Do not restrict the ancestry PCA marker set to disease-specific PRS or GWAS-significant SNPs.
The ancestry marker set should be genome-wide, common, nonambiguous, and LD-pruned.

---

## Leakage Rule

For model selection:

```text
Fit normalization parameters on TRAIN only.
Apply normalization parameters to VALIDATION.
```

For final submission, if competition rules allow using train + validation:

```text
Fit normalization parameters on TRAIN + VALIDATION.
Apply normalization parameters to HOLDOUT TEST.
```

Do **not** estimate PRS mean, PRS SD, group-specific normalization parameters, shrinkage parameters, or ensemble weights using holdout test.

Distance to 1000G centers can be computed for all target samples because it is genotype-only and reference-defined. However, PRS normalization parameters must be estimated only from the allowed fitting split.

---

# 1. Expected Inputs

## 1.1 Target genotype

Either PLINK2 format:

```text
target.pgen
target.pvar
target.psam
```

or PLINK1 format:

```text
target.bed
target.bim
target.fam
```

## 1.2 1000G reference genotype

Either PLINK2 format:

```text
1000g.pgen
1000g.pvar
1000g.psam
```

or PLINK1 format:

```text
1000g.bed
1000g.bim
1000g.fam
```

## 1.3 1000G sample panel

File:

```text
1000g_panel.tsv
```

Required columns:

```text
IID    pop    super_pop
```

where `super_pop` is one of:

```text
AFR, AMR, EAS, EUR, SAS
```

## 1.4 Raw PRS table

File:

```text
raw_prs.tsv
```

Required columns:

```text
FID    IID    PRS_1    PRS_2    ...    PRS_K
```

## 1.5 Split file

File:

```text
split.tsv
```

Required columns:

```text
FID    IID    split
```

where `split` is one of:

```text
train, val, test
```

---

# 2. Genotype Harmonization

## 2.1 Convert target and 1000G to PLINK2 format

If target is already in PLINK2 format:

```bash
plink2 \
  --pfile target \
  --autosome \
  --snps-only just-acgt \
  --max-alleles 2 \
  --make-pgen \
  --out target.acgt
```

If target is in PLINK1 format:

```bash
plink2 \
  --bfile target \
  --autosome \
  --snps-only just-acgt \
  --max-alleles 2 \
  --make-pgen \
  --out target.acgt
```

For 1000G:

```bash
plink2 \
  --pfile 1000g \
  --autosome \
  --snps-only just-acgt \
  --max-alleles 2 \
  --make-pgen \
  --out 1000g.acgt
```

If 1000G is in PLINK1 format:

```bash
plink2 \
  --bfile 1000g \
  --autosome \
  --snps-only just-acgt \
  --max-alleles 2 \
  --make-pgen \
  --out 1000g.acgt
```

---

## 2.2 Create genome-wide nonambiguous overlap files

Create a script:

```text
scripts/make_ancestry_overlap.py
```

Purpose:

```text
Find genome-wide variants shared between target and 1000G that are safe for PCA projection.
```

Recommended conservative rules:

1. Keep autosomal biallelic SNPs only.
2. Keep only A/C/G/T alleles.
3. Remove strand-ambiguous SNPs:
   - A/T
   - T/A
   - C/G
   - G/C
4. Keep variants present in both target and 1000G.
5. Keep variants with matching unordered allele pairs.
6. Remove allele-mismatched variants.
7. Match by `CHR:BP:unordered_alleles`, not only by variant ID.
8. Do not attempt aggressive allele flipping unless it is explicitly implemented and logged.

Important:

```text
Target and 1000G variant IDs may differ.
After matching, rename target variants to the matched 1000G IDs before PCA projection.
This lets PLINK2 use the 1000G PCA allele-weight file to score/project target samples.
```

Output:

```text
ancestry_harmonization/genome/genome.matched_variants.tsv
ancestry_harmonization/genome/genome.target_extract.snplist
ancestry_harmonization/genome/genome.1000g_extract.snplist
ancestry_harmonization/genome/genome.target_rename_to_1000g.tsv
ancestry_harmonization/genome/genome.summary.json
```

Command:

```bash
python scripts/make_ancestry_overlap.py \
  --target-variants target.acgt.pvar \
  --reference-variants 1000g.acgt.pvar \
  --out-dir ancestry_harmonization/genome \
  --prefix genome
```

The script supports `.pvar` and `.bim` inputs and writes an audit summary.

---

## 2.3 Extract shared nonambiguous SNPs

```bash
plink2 \
  --pfile target.acgt \
  --extract ancestry_harmonization/genome/genome.target_extract.snplist \
  --make-pgen \
  --out target.overlap.raw_ids

plink2 \
  --pfile target.overlap.raw_ids \
  --update-name ancestry_harmonization/genome/genome.target_rename_to_1000g.tsv 2 1 1 \
  --make-pgen \
  --out target.overlap

plink2 \
  --pfile 1000g.acgt \
  --extract ancestry_harmonization/genome/genome.1000g_extract.snplist \
  --make-pgen \
  --out 1000g.overlap
```

---

# 3. Fit 1000G Reference PCA

## 3.1 Optional high-LD region exclusion

Create:

```text
resources/high_ld_regions.txt
```

Minimal version:

```text
6 25000000 34000000 MHC
```

---

## 3.2 LD pruning on 1000G only

```bash
plink2 \
  --pfile 1000g.overlap \
  --maf 0.01 \
  --geno 0.02 \
  --exclude range resources/high_ld_regions.txt \
  --indep-pairwise 200 50 0.1 \
  --out 1000g.ldprune
```

Extract pruned SNPs from both reference and target:

```bash
plink2 \
  --pfile 1000g.overlap \
  --extract 1000g.ldprune.prune.in \
  --make-pgen \
  --out 1000g.pca.snps

plink2 \
  --pfile target.overlap \
  --extract 1000g.ldprune.prune.in \
  --make-pgen \
  --out target.pca.snps
```

---

## 3.3 Fit PCA on 1000G only

Compute 10 PCs. [我觉得可以采用20个PCs，但是最后算距离的时候还是算10个PCs]

```bash
plink2 \
  --pfile 1000g.pca.snps \
  --freq counts \
  --keep-founders \
  --pca 10 allele-wts vcols=chrom,ref,alt \
  --out 1000g.ref_pca
```

Expected outputs:

```text
1000g.ref_pca.eigenvec
1000g.ref_pca.eigenval
1000g.ref_pca.eigenvec.allele
1000g.ref_pca.acount
```

---

# 4. Project 1000G and Target Into the Same PC Space

Use the PCA allele weights from 1000G.

Important: project both 1000G and target using the same projection recipe. Downstream distances should be computed using the projected 1000G scores and projected target scores.

Codex should inspect the header of:

```text
1000g.ref_pca.eigenvec.allele
```

and verify the correct column indices for:

```text
variant ID column
allele column
PC1–PC10 columns
```

The commands below assume:

```text
variant ID column = 2
allele column = 5
PC columns = 6–15
```

## 4.1 Project 1000G

```bash
plink2 \
  --pfile 1000g.pca.snps \
  --read-freq 1000g.ref_pca.acount \
  --score 1000g.ref_pca.eigenvec.allele 2 5 header-read no-mean-imputation variance-standardize \
  --score-col-nums 6-15 \
  --out 1000g.projected
```

## 4.2 Project target

```bash
plink2 \
  --pfile target.pca.snps \
  --read-freq 1000g.ref_pca.acount \
  --score 1000g.ref_pca.eigenvec.allele 2 5 header-read no-mean-imputation variance-standardize \
  --score-col-nums 6-15 \
  --out target.projected
```

Expected outputs:

```text
1000g.projected.sscore
target.projected.sscore
```

Expected PC columns:

```text
PC1_AVG, PC2_AVG, ..., PC10_AVG
```

---

# 5. Compute 1000G Centers and Target Distances

Create:

```text
scripts/compute_ancestry_distances.py
```

Inputs:

```text
1000g.projected.sscore
target.projected.sscore
1000g_panel.tsv
```

Output:

```text
ancestry/target_1000G_distances_pc10.tsv
ancestry/target_1000G_hard_labels_pc10_margin0.5.tsv
ancestry/target_1000G_soft_weights_pc10_tau1.tsv
ancestry/target_1000G_PC1_PC2_1000G_centers_hard_assignments.svg
```

## 5.1 Distance computation

Use projected 1000G PCs to compute super-population centers:

```text
center_AFR
center_AMR
center_EAS
center_EUR
center_SAS
```

Then compute target-to-center distances.

Default distance metric:

```text
Pooled Mahalanobis distance with Ledoit-Wolf covariance shrinkage
```

Also compute scaled Euclidean distance for sanity checks.

## 5.2 Python implementation

Implemented command:

```bash
uv run --extra dev python scripts/compute_ancestry_distances.py \
  --ref-sscore 1000g.projected.sscore \
  --target-sscore target.projected.sscore \
  --panel 1000g_panel.tsv \
  --out-dir ancestry \
  --prefix target_1000G
```

The SVG audit plot shows 1000G samples colored by `super_pop`, target samples colored by
hard 1000G-like assignment, and one cross per **1000G super-population center**.

```python
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

GROUPS = ["AFR", "AMR", "EAS", "EUR", "SAS"]
PC_COLS = [f"PC{i}_AVG" for i in range(1, 11)]

def load_sscore(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+")
    df = df.rename(columns={"#FID": "FID"})
    return df

def compute_distances(
    ref_sscore: str,
    target_sscore: str,
    panel_file: str,
    out_file: str,
):
    ref = load_sscore(ref_sscore)
    target = load_sscore(target_sscore)

    panel = pd.read_csv(panel_file, sep=r"\s+")
    if "sample" in panel.columns and "IID" not in panel.columns:
        panel = panel.rename(columns={"sample": "IID"})

    ref = ref.merge(panel[["IID", "pop", "super_pop"]], on="IID", how="inner")
    ref = ref[ref["super_pop"].isin(GROUPS)].copy()

    centers = ref.groupby("super_pop")[PC_COLS].mean()

    out = target[["FID", "IID"]].copy()

    # Scaled Euclidean distance
    ref_sd = ref[PC_COLS].std(axis=0).replace(0, np.nan)

    for g in GROUPS:
        diff = (target[PC_COLS].values - centers.loc[g, PC_COLS].values) / ref_sd.values
        out[f"d_euclid_{g}"] = np.sqrt(np.nansum(diff ** 2, axis=1))

    # Pooled Mahalanobis distance
    X_ref = ref[PC_COLS].values
    lw = LedoitWolf().fit(X_ref)
    inv_cov = np.linalg.inv(lw.covariance_)

    for g in GROUPS:
        mu = centers.loc[g, PC_COLS].values
        diff = target[PC_COLS].values - mu
        d2 = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
        out[f"d_mahal_{g}"] = np.sqrt(np.maximum(d2, 0))

    mahal_cols = [f"d_mahal_{g}" for g in GROUPS]

    out["nearest_1000G_superpop"] = (
        out[mahal_cols]
        .idxmin(axis=1)
        .str.replace("d_mahal_", "", regex=False)
    )

    sorted_d = np.sort(out[mahal_cols].values, axis=1)
    out["min_mahal"] = sorted_d[:, 0]
    out["second_min_mahal"] = sorted_d[:, 1]
    out["mahal_margin"] = sorted_d[:, 1] - sorted_d[:, 0]

    out.to_csv(out_file, sep="\t", index=False)

if __name__ == "__main__":
    compute_distances(
        ref_sscore="1000g.projected.sscore",
        target_sscore="target.projected.sscore",
        panel_file="1000g_panel.tsv",
        out_file="ancestry/target_1000G_distances_pc10.tsv",
    )
```

---

# 6. Generate Hard Labels

Create:

```text
scripts/make_hard_labels.py
```

Input:

```text
ancestry/target_1000G_distances_pc10.tsv
```

Output:

```text
ancestry/target_1000G_hard_labels_pc10.tsv
```

## 6.1 Hard-label rule

Default:

```text
hard_label = nearest_1000G_superpop
```

Mark ambiguous if the nearest and second-nearest distances are too close:

```text
if mahal_margin < 0.5:
    hard_label = "AMBIGUOUS"
```

Recommended label names:

```text
AFR-like
AMR-like
EAS-like
EUR-like
SAS-like
AMBIGUOUS
```

Do not describe these as true ancestry labels. They are 1000G genetic-similarity labels.

## 6.2 Python implementation

```python
import pandas as pd

MARGIN_THRESHOLD = 0.5

def make_hard_labels(dist_file, out_file):
    df = pd.read_csv(dist_file, sep="\t")

    df["hard_label"] = df["nearest_1000G_superpop"] + "-like"
    df.loc[df["mahal_margin"] < MARGIN_THRESHOLD, "hard_label"] = "AMBIGUOUS"

    keep_cols = [
        "FID", "IID",
        "nearest_1000G_superpop",
        "hard_label",
        "min_mahal",
        "second_min_mahal",
        "mahal_margin",
        "d_mahal_AFR",
        "d_mahal_AMR",
        "d_mahal_EAS",
        "d_mahal_EUR",
        "d_mahal_SAS",
    ]

    df[keep_cols].to_csv(out_file, sep="\t", index=False)

if __name__ == "__main__":
    make_hard_labels(
        dist_file="ancestry/target_1000G_distances_pc10.tsv",
        out_file="ancestry/target_1000G_hard_labels_pc10.tsv",
    )
```

---

# 7. Generate Soft Labels From Distances

Create:

```text
scripts/make_soft_weights.py
```

Input:

```text
ancestry/target_1000G_distances_pc10.tsv
```

Output:

```text
ancestry/target_1000G_soft_weights_pc10_tau1.0.tsv
```

## 7.1 Distance-to-weight conversion

Use RBF/Gaussian-kernel softmax.

For individual `i` and group `g`:

```text
s_ig = exp(-0.5 * (d_ig / tau)^2)
w_ig = s_ig / sum_g s_ig
```

Default:

```text
distance = d_mahal_g
tau = 1.0
```

These weights are **genetic-similarity weights**, not true ancestry proportions.

## 7.2 Python implementation

```python
import numpy as np
import pandas as pd

GROUPS = ["AFR", "AMR", "EAS", "EUR", "SAS"]
TAU = 1.0

def make_soft_weights(dist_file, out_file, tau=TAU):
    df = pd.read_csv(dist_file, sep="\t")

    D = df[[f"d_mahal_{g}" for g in GROUPS]].values.astype(float)

    log_s = -0.5 * (D / tau) ** 2
    log_s = log_s - log_s.max(axis=1, keepdims=True)

    S = np.exp(log_s)
    W = S / S.sum(axis=1, keepdims=True)

    out = df[["FID", "IID"]].copy()

    for j, g in enumerate(GROUPS):
        out[f"w_{g}"] = W[:, j]

    out["max_weight"] = W.max(axis=1)
    out["soft_nearest_group"] = [GROUPS[j] + "-like" for j in W.argmax(axis=1)]
    out["weight_entropy"] = -(W * np.log(W + 1e-12)).sum(axis=1)

    out.to_csv(out_file, sep="\t", index=False)

if __name__ == "__main__":
    make_soft_weights(
        dist_file="ancestry/target_1000G_distances_pc10.tsv",
        out_file="ancestry/target_1000G_soft_weights_pc10_tau1.0.tsv",
        tau=1.0,
    )
```

---

# 8. Normalization Setting 1: Hard Label + Shrinkage

Create:

```text
scripts/normalize_prs_hard.py
```

Inputs:

```text
raw_prs.tsv
split.tsv
ancestry/target_1000G_hard_labels_pc10.tsv
```

Outputs:

```text
prs_normalized/train_hard_pc10_margin0.5_shrink100.tsv
prs_normalized/val_hard_pc10_margin0.5_shrink100.tsv
prs_normalized/test_hard_pc10_margin0.5_shrink100.tsv
normalization_params/hard_pc10_margin0.5_shrink100.json
```

## 8.1 Formula

For raw PRS `x_ik`:

```text
i = individual
k = PRS method
g = hard ancestry/similarity group
```

Global mean and variance on fitting split:

```text
mu_k_global
sigma2_k_global
```

Group-specific mean and variance:

```text
mu_kg
sigma2_kg
```

Shrinkage factor:

```text
lambda_g = n_g / (n_g + shrink_tau)
```

Default:

```text
shrink_tau = 100
```

Shrunk parameters:

```text
mu_kg_shrunk = lambda_g * mu_kg + (1 - lambda_g) * mu_k_global

sigma2_kg_shrunk = lambda_g * sigma2_kg + (1 - lambda_g) * sigma2_k_global
```

Normalized PRS:

```text
z_ik_hard = (x_ik - mu_kg_shrunk) / sqrt(sigma2_kg_shrunk)
```

For `AMBIGUOUS` samples:

```text
Use global mean and global SD.
```

## 8.2 Python implementation

```python
import json
import numpy as np
import pandas as pd
from pathlib import Path

SHRINK_TAU = 100.0
AMBIGUOUS_LABEL = "AMBIGUOUS"

def get_prs_cols(df):
    return [c for c in df.columns if c not in ["FID", "IID", "split", "hard_label"]]

def fit_hard_norm_params(fit_df, prs_cols, label_col="hard_label", shrink_tau=SHRINK_TAU):
    params = {}
    groups = sorted([g for g in fit_df[label_col].dropna().unique() if g != AMBIGUOUS_LABEL])

    for prs in prs_cols:
        x = fit_df[prs].astype(float).values
        global_mu = float(np.nanmean(x))
        global_var = float(np.nanvar(x))

        params[prs] = {
            "global_mu": global_mu,
            "global_var": max(global_var, 1e-12),
            "groups": {},
        }

        for g in groups:
            xv = fit_df.loc[fit_df[label_col] == g, prs].astype(float).dropna().values
            n_g = len(xv)

            if n_g == 0:
                continue

            mu_g = float(np.mean(xv))
            var_g = float(np.var(xv))

            lam = n_g / (n_g + shrink_tau)

            mu_shrunk = lam * mu_g + (1.0 - lam) * global_mu
            var_shrunk = lam * var_g + (1.0 - lam) * global_var

            params[prs]["groups"][g] = {
                "n": int(n_g),
                "lambda": float(lam),
                "mu_raw": mu_g,
                "var_raw": var_g,
                "mu": float(mu_shrunk),
                "var": float(max(var_shrunk, 1e-12)),
            }

    return params

def apply_hard_norm(df, params, prs_cols, label_col="hard_label"):
    out = df[["FID", "IID"]].copy()

    for prs in prs_cols:
        p = params[prs]
        global_mu = p["global_mu"]
        global_sd = np.sqrt(p["global_var"])

        z = []

        for _, row in df.iterrows():
            x = row[prs]
            g = row[label_col]

            if pd.isna(x):
                z.append(np.nan)
                continue

            if g in p["groups"]:
                mu = p["groups"][g]["mu"]
                sd = np.sqrt(p["groups"][g]["var"])
            else:
                mu = global_mu
                sd = global_sd

            z.append((x - mu) / sd)

        out[f"{prs}_hardnorm"] = z

    return out

def main():
    raw = pd.read_csv("raw_prs.tsv", sep="\t")
    split = pd.read_csv("split.tsv", sep="\t")
    labels = pd.read_csv("ancestry/target_1000G_hard_labels_pc10.tsv", sep="\t")

    df = raw.merge(split, on=["FID", "IID"], how="inner")
    df = df.merge(labels[["FID", "IID", "hard_label"]], on=["FID", "IID"], how="left")

    prs_cols = [c for c in raw.columns if c not in ["FID", "IID"]]

    fit_df = df[df["split"] == "train"].copy()
    params = fit_hard_norm_params(fit_df, prs_cols)

    Path("normalization_params").mkdir(exist_ok=True)
    with open("normalization_params/hard_pc10_margin0.5_shrink100.json", "w") as f:
        json.dump(params, f, indent=2)

    Path("prs_normalized").mkdir(exist_ok=True)

    for split_name in ["train", "val", "test"]:
        sub = df[df["split"] == split_name].copy()
        norm = apply_hard_norm(sub, params, prs_cols)
        norm.to_csv(
            f"prs_normalized/{split_name}_hard_pc10_margin0.5_shrink100.tsv",
            sep="\t",
            index=False,
        )

if __name__ == "__main__":
    main()
```

---

# 9. Normalization Setting 2: Soft Label + Distance + Shrinkage

Create:

```text
scripts/normalize_prs_soft.py
```

Inputs:

```text
raw_prs.tsv
split.tsv
ancestry/target_1000G_soft_weights_pc10_tau1.0.tsv
```

Outputs:

```text
prs_normalized/train_soft_pc10_tau1.0_shrink100.tsv
prs_normalized/val_soft_pc10_tau1.0_shrink100.tsv
prs_normalized/test_soft_pc10_tau1.0_shrink100.tsv
normalization_params/soft_pc10_tau1.0_shrink100.json
```

## 9.1 Formula

Given soft weights:

```text
w_ig = similarity weight between individual i and 1000G group g
```

Weighted group-specific mean:

```text
mu_kg = sum_i w_ig x_ik / sum_i w_ig
```

Weighted group-specific variance:

```text
sigma2_kg = sum_i w_ig (x_ik - mu_kg)^2 / sum_i w_ig
```

Effective sample size:

```text
n_eff_g = (sum_i w_ig)^2 / sum_i w_ig^2
```

Shrinkage factor:

```text
lambda_g = n_eff_g / (n_eff_g + shrink_tau)
```

Default:

```text
shrink_tau = 100
```

Shrunk parameters:

```text
mu_kg_shrunk = lambda_g * mu_kg + (1 - lambda_g) * mu_k_global

sigma2_kg_shrunk = lambda_g * sigma2_kg + (1 - lambda_g) * sigma2_k_global
```

Individual-specific expected PRS mean:

```text
mu_ik = sum_g w_ig * mu_kg_shrunk
```

Individual-specific SD via log-SD interpolation:

```text
log_sigma_ik = sum_g w_ig * log(sigma_kg_shrunk)
sigma_ik = exp(log_sigma_ik)
```

Normalized PRS:

```text
z_ik_soft = (x_ik - mu_ik) / sigma_ik
```

Use log-SD interpolation by default. Do not use mixture variance as the default, because mixture variance can overinflate SD for admixed individuals and overly compress their PRS.

## 9.2 Python implementation

```python
import json
import numpy as np
import pandas as pd
from pathlib import Path

GROUPS = ["AFR", "AMR", "EAS", "EUR", "SAS"]
SHRINK_TAU = 100.0

def fit_soft_norm_params(fit_df, prs_cols, groups=GROUPS, shrink_tau=SHRINK_TAU):
    params = {}
    W = fit_df[[f"w_{g}" for g in groups]].values.astype(float)

    for prs in prs_cols:
        x = fit_df[prs].astype(float).values
        valid = np.isfinite(x)

        global_mu = float(np.nanmean(x))
        global_var = float(np.nanvar(x))

        params[prs] = {
            "global_mu": global_mu,
            "global_var": max(global_var, 1e-12),
            "groups": {},
        }

        for j, g in enumerate(groups):
            w = W[:, j]
            wv = w[valid]
            xv = x[valid]

            w_sum = np.sum(wv)
            if w_sum <= 0:
                continue

            mu_g = float(np.sum(wv * xv) / w_sum)
            var_g = float(np.sum(wv * (xv - mu_g) ** 2) / w_sum)

            n_eff = float((w_sum ** 2) / np.sum(wv ** 2))
            lam = n_eff / (n_eff + shrink_tau)

            mu_shrunk = lam * mu_g + (1.0 - lam) * global_mu
            var_shrunk = lam * var_g + (1.0 - lam) * global_var

            params[prs]["groups"][g] = {
                "n_eff": n_eff,
                "lambda": float(lam),
                "mu_raw": mu_g,
                "var_raw": var_g,
                "mu": float(mu_shrunk),
                "var": float(max(var_shrunk, 1e-12)),
            }

    return params

def apply_soft_norm(df, params, prs_cols, groups=GROUPS):
    out = df[["FID", "IID"]].copy()
    W = df[[f"w_{g}" for g in groups]].values.astype(float)

    for prs in prs_cols:
        p = params[prs]

        mu_g = []
        sd_g = []

        for g in groups:
            if g in p["groups"]:
                mu_g.append(p["groups"][g]["mu"])
                sd_g.append(np.sqrt(p["groups"][g]["var"]))
            else:
                mu_g.append(p["global_mu"])
                sd_g.append(np.sqrt(p["global_var"]))

        mu_g = np.asarray(mu_g)
        sd_g = np.asarray(sd_g)

        x = df[prs].astype(float).values

        mu_i = W @ mu_g
        log_sd_i = W @ np.log(sd_g + 1e-12)
        sd_i = np.exp(log_sd_i)

        out[f"{prs}_softnorm"] = (x - mu_i) / sd_i

    return out

def main():
    raw = pd.read_csv("raw_prs.tsv", sep="\t")
    split = pd.read_csv("split.tsv", sep="\t")
    weights = pd.read_csv("ancestry/target_1000G_soft_weights_pc10_tau1.0.tsv", sep="\t")

    df = raw.merge(split, on=["FID", "IID"], how="inner")
    df = df.merge(weights, on=["FID", "IID"], how="left")

    prs_cols = [c for c in raw.columns if c not in ["FID", "IID"]]

    fit_df = df[df["split"] == "train"].copy()
    params = fit_soft_norm_params(fit_df, prs_cols)

    Path("normalization_params").mkdir(exist_ok=True)
    with open("normalization_params/soft_pc10_tau1.0_shrink100.json", "w") as f:
        json.dump(params, f, indent=2)

    Path("prs_normalized").mkdir(exist_ok=True)

    for split_name in ["train", "val", "test"]:
        sub = df[df["split"] == split_name].copy()
        norm = apply_soft_norm(sub, params, prs_cols)
        norm.to_csv(
            f"prs_normalized/{split_name}_soft_pc10_tau1.0_shrink100.tsv",
            sep="\t",
            index=False,
        )

if __name__ == "__main__":
    main()
```

---

# 10. Minimal QC

Create:

```text
scripts/qc_ancestry_normalization.py
```

## 10.1 PCA projection plot

Generate:

```text
QC/projected_1000G_target_PC1_PC2.png
QC/projected_1000G_target_PC1_PC3.png
```

Plot:

```text
1000G colored by super_pop
target as gray or black points
```

## 10.2 Hard-label count

Output:

```text
QC/hard_label_counts.tsv
```

Columns:

```text
hard_label
n
fraction
```

## 10.3 Soft-weight summary

Output:

```text
QC/soft_weight_summary_tau1.0.tsv
```

Include:

```text
mean max_weight
median max_weight
mean entropy
median entropy
fraction max_weight > 0.8
fraction max_weight > 0.9
```

## 10.4 PRS distribution summary after normalization

Output:

```text
QC/normalized_prs_group_summary.tsv
```

For each hard group and each PRS:

```text
mean
sd
n
```

Expected behavior:

```text
Hard normalization:
    In the train split, large groups should have PRS mean close to 0 and SD close to 1.

Soft normalization:
    Distribution should be more balanced across groups, but not necessarily exactly mean 0 / SD 1 within hard groups.
```

## 10.5 PRS-PC correlation before and after normalization

Output:

```text
QC/prs_pc_correlation_before_after.tsv
```

For each raw and normalized PRS:

```text
corr(PRS, PC1)
corr(PRS, PC2)
...
corr(PRS, PC10)
```

Expected behavior:

```text
Ancestry-aware normalization should reduce extreme PRS-PC correlation.
It does not need to remove all PC correlation.
```

---

# 11. Final Output Structure

```text
ancestry/
  target_1000G_distances_pc10.tsv
  target_1000G_hard_labels_pc10.tsv
  target_1000G_soft_weights_pc10_tau1.0.tsv

normalization_params/
  hard_pc10_margin0.5_shrink100.json
  soft_pc10_tau1.0_shrink100.json

prs_normalized/
  train_hard_pc10_margin0.5_shrink100.tsv
  val_hard_pc10_margin0.5_shrink100.tsv
  test_hard_pc10_margin0.5_shrink100.tsv

  train_soft_pc10_tau1.0_shrink100.tsv
  val_soft_pc10_tau1.0_shrink100.tsv
  test_soft_pc10_tau1.0_shrink100.tsv

QC/
  projected_1000G_target_PC1_PC2.png
  projected_1000G_target_PC1_PC3.png
  hard_label_counts.tsv
  soft_weight_summary_tau1.0.tsv
  normalized_prs_group_summary.tsv
  prs_pc_correlation_before_after.tsv
```

---

# 12. Validation Interface

The final validation scripts support both wide and long PRS inputs.

Use **long input** when comparing multiple normalization settings in one run:

```text
sampleid    disease    prs    model    normalization
S-100024    T2D        ...    pt_r20p8_p5e-4    raw
S-100024    T2D        ...    pt_r20p8_p5e-4    hard_pc10_margin0.5_shrink100
S-100024    T2D        ...    pt_r20p8_p5e-4    soft_pc10_tau1.0_shrink100
```

Build these files from wide raw/normalized score tables with:

```bash
uv run --extra dev python scripts/build_ancestry_norm_validation_inputs.py \
  --disease T2D \
  --setting raw raw_prs_train.tsv raw_prs_validation.tsv \
  --setting hard_pc10_margin0.5_shrink100 prs_normalized/train_hard_pc10_margin0.5_shrink100.tsv prs_normalized/val_hard_pc10_margin0.5_shrink100.tsv \
  --setting soft_pc10_tau1.0_shrink100 prs_normalized/train_soft_pc10_tau1.0_shrink100.tsv prs_normalized/val_soft_pc10_tau1.0_shrink100.tsv \
  --out-dir outputs/ancestry_norm/T2D
```

Required command pattern:

```bash
uv run --extra dev python scripts/validate_official_model.py \
  --model-name ancestry_norm_grid \
  --train-prs-long-scores outputs/ancestry_norm/T2D/T2D.train_long.csv \
  --validation-prs-long-scores outputs/ancestry_norm/T2D/T2D.validation_long.csv \
  --prs-param-cols model \
  --out-dir outputs/official_validation_ancestry_norm_grid
```

Use enough `--prs-param-cols` to identify exactly one score per patient/disease. If multiple
base PRS hyperparameter settings are present, include those parameter columns too, or encode
them into `model`.

Wide input is acceptable for one normalization setting at a time. Its columns must match the
existing validation convention:

```text
sampleid    prs_cad    prs_bc    prs_T2D
```

Then run one validation output directory per normalization setting.

```bash
uv run --extra dev python scripts/validate_official_model.py \
  --model-name ancestry_norm_soft \
  --train-prs-scores outputs/ancestry_norm/soft/train_wide.csv \
  --validation-prs-scores outputs/ancestry_norm/soft/validation_wide.csv \
  --out-dir outputs/official_validation_ancestry_norm_soft
```

For ensemble fitting with `scripts/build_ensemble_prs.py`, long inputs are also supported.
In that case, make `model` unique per base PRS and normalization setting, for example
`pt_r20p8_p5e-4__soft_pc10_tau1.0_shrink100`, because ensemble pivoting uses `model` as
the feature column name.

---

# 13. Recommended First Production Candidates

Use these two normalized PRS datasets first:

```text
hard_pc10_margin0.5_shrink100
soft_pc10_tau1.0_shrink100
```

Then compare against the unnormalized or globally standardized PRS baseline in the official-style validation pipeline.

Selection criteria:

```text
1. strong overall validation performance
2. no collapse in AFR-like subgroup
3. reduced extreme PRS-PC correlation
4. stable PRS distribution across ancestry-like groups
5. no evidence that small groups drive unstable normalization
```

---

# 14. Terminology

Use:

```text
1000G genetic-similarity group
EUR-like
AFR-like
EAS-like
AMR-like
SAS-like
AMBIGUOUS
```

Avoid:

```text
true ancestry
race
self-identified ethnicity
```

The labels and weights here are technical constructs derived from 1000G reference PCA projection and distances to 1000G super-population centers. They are intended for PRS normalization and subgroup robustness checks, not for identity inference.
