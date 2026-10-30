# PRS distillation v3 — fixed prototypes, positive scales, sign-aware age adjustment

## 0. Decision and scope

Implement ONE candidate student with three transparent components:

1. An ancestry-only softmax over the existing, fixed PRSs.
2. One positive magnitude `r_k` per PRS.
3. One nonnegative, sign-aware age response `d_k` per PRS, plus a small monotone baseline age function.

Do not learn new PRS prototypes, change SNP/GWAS weights, add `v_kj`, add an MLP gate, add another shared multiplicative age scale, or add a monotonicity penalty. Retain the existing teacher, splits, approved preprocessing and evaluation pipeline.

This is a candidate structural prior, NOT an established biological mechanism or a proven performance improvement. Overall age–risk monotonicity does not logically require every PRS contribution to be age-monotone. The sign-aware candidate imposes the stronger condition and must be tested. A PRS of zero is a chosen reference, not a biological threshold. PRS–age associations can be weaker at older ages in published data [S1]. Do not present the exact sign-based transform as a published genetic law.

Use the existing binary disease definitions and eligibility rules. Fit each endpoint separately; do not infer that a prior useful for one disease is valid for all diseases or all age ranges. This is not a survival model or a population absolute-risk calculator [S4].

### Precedence over earlier specifications

This file supersedes the STUDENT STRUCTURE and related initialization/constraint parts of v1/v2. All compatible data isolation, privacy, local execution, provenance and evaluation requirements remain.

| Earlier setting | v3 decision |
|---|---|
| Shared multiplicative `s(age)` | Removed; constant magnitude is included in `r_k` |
| `sum(log r_k)=0` / relative scales | Removed; positive `r_k` have free common magnitude |
| `sum(d_k)=0` / signed age deviations | Removed; now `d_k >= 0` are absolute strengths |
| Free baseline age spline | Replaced by a three-segment monotone linear function |
| `exp(d_k*t)*Z_k` | Replaced by `exp(d_k*t*sign(Z_k))*Z_k` |
| New prototype mixing `v_kj` | Still excluded; fixed identity matrix |
| Claimed exact warm start from v1 | Not valid in general; use the v3 no-interaction arm for exact nesting |

Keep old results and models, including the unconstrained-age comparator, unchanged.

## 1. Exact model

For patient i, let Z_i be the fixed corrected, risk-oriented PRS vector, A_i the existing ancestry coordinates, C_i the existing non-age covariates, and a_i age.

```text
pi(A) = softmax(W A + c)                  # temperature 1
r_k = exp(log_r_k)                       # positive; no sum constraint

a_used = clip(a, training_min_age, training_max_age)
t(a) = (a_used - reference_age) / age_unit

phi_k(Z_k, a) = Z_k * exp(d_k * t(a) * sign(Z_k))

eta = alpha + gamma^T A + delta^T C + f_up(a)
      + sum_k pi_k(A) * r_k * phi_k(Z_k, a)
p = sigmoid(eta)
```

Constraints:

```text
d_k >= 0
f_up(reference_age) = 0
f_up is nondecreasing
pi uses ancestry only, not age, PRSs, or labels
```

Do NOT renormalize `pi * r * age_multiplier` in the predictive forward pass. Its total amplitude is part of the model. A normalized coefficient share may be exported for description only.

### Component meanings

- `pi_k(A)`: model mixing weight, not ancestry fraction or causal risk percentage.
- `r_k`: scale of fixed PRS k at the reference age, NOT its heritability or unique biological importance.
- `d_k`: strength of this deliberately constrained age modification, NOT a causal aging coefficient.
- `f_up`: baseline age dependence even when all corrected PRSs are zero.

The original PRS is never overwritten. `phi_k` is a derived prediction feature, not a change in the patient's genotype or underlying genetic score.

### Equivalent positive/negative decomposition

```text
z_pos = max(z, 0)
z_neg = max(-z, 0)
phi = z_pos * exp(d*t) - z_neg * exp(-d*t)
```

At reference age `phi=z`; at `d=0`, `phi=z`; at `z=0`, `phi=0` for all ages. The function is continuous at zero but generally has different left and right slopes. No artificial smoothing or learned sign boundary in this experiment.

### Why the direction is guaranteed

Within the supported age range, holding corrected PRSs, ancestry and non-age covariates fixed:

```text
d phi_k / d a = (d_k / age_unit) * abs(Z_k)
               * exp(d_k*t*sign(Z_k)) >= 0
```

Consequently `d eta / d a >= 0` and `d p / d a >= 0`, away from piecewise-linear knots; continuity extends monotonicity across knots. With the declared constant extrapolation, monotonicity is non-strict outside the range. Larger age never implies that ANY older individual outranks ANY younger individual: this is conditional on other inputs being fixed.

At each fixed age/ancestry, phi is also increasing in each PRS. The resulting function is piecewise linear in a PRS with a kink at its zero reference; it is not a globally linear coefficient model across both signs.

The guarantee is for the DIRECT student. It does not automatically transfer through an unconstrained downstream Stage-2 regression.

## 2. Inputs, age units and baseline implementation

### 2.1 Freeze existing inputs

Read repository AGENTS.md, feature manifests, prior configs, teacher cache provenance and split manifests. Reuse current PRS correction; do not introduce a new normalization, refit GWAS weights, change score orientation based on assessment labels or rename all models as ancestry-specific when their source metadata do not establish that.

Ancestry-dependent PRS mean/variance adjustment has research precedent [S2], but the project's existing approved transform takes precedence. Freeze the zero/reference convention for each PRS. Sign is with respect to that reference, not an absolute protective/harmful label. Preprocessing must not change when drawing an age response for a fixed patient.

C must exclude age and unconstrained age-derived terms. Age enters only the two declared branches. No hidden duplicate age path is permitted. Retain existing non-age covariates and endpoint eligibility; in particular, do not redefine the BC cohort.

### 2.2 Confirm age units before using a decade

- If real age in years is available: `age_unit=10`, `reference_age=training median`.
- If only standardized age is available and a documented invertible transform exists, recover years from that documented transform.
- If only standardized age exists and the original scale is unavailable: use `age_unit=1` in the documented standardized input units, set the reference to the training median in those units, and label d per one such unit. Do NOT call it a per-decade coefficient or fabricate ages in years. Preserve the fixed sign/orientation that larger input means older age.

All reference ages, age bounds, knots and newly fitted feature transformations use the current fit scope only. Refit them inside each appropriate fold and serialize them. No validation/test distribution is used to set the range or reference.

### 2.3 Small monotone baseline

Take training-only age breakpoints at quantiles `[0, 1/3, 2/3, 1]`, remove exact duplicates, and form J intervals. Typically J=3. All patients use the same baseline curve for an endpoint; ancestry and other metadata enter additively as above.

Let boundary coordinates be `b_j=(age_boundary_j-reference_age)/age_unit` and interval width be `w_j=b_(j+1)-b_j`. Define

```text
B_j(t) = clip(t - b_j, 0, w_j) - clip(-b_j, 0, w_j)
f_up(a) = sum_j m_j * B_j(t(a)),  m_j >= 0
```

Each m_j is a baseline log-odds slope per age_unit, not an increment on the probability scale. This construction allows increasing, decreasing or equal slopes across segments, but never a negative slope. It adds at most three age parameters. At the reference age every basis equals zero.

If all ages are identical, there is no estimable age effect: set t and f to zero and omit d/m from fitting; document the special case. If too few distinct ages exist, use the distinct boundaries available; no forced knot jitter.

Use the same clipped age in BOTH branches. Flag out-of-support records without dropping or reallocating them. Constant extrapolation is a numerical/modeling policy, not evidence about extreme ages.

## 3. Minimal comparisons

Keep the previously trained v1/v2 model and teacher as historical/paired references on the SAME eligible evaluation patients. v3 is not a strict superset of v1: changing its age prior can lose as well as gain expressivity.

Within v3 compare only:

| ID | Per-PRS r | Age correction | Purpose |
|---|---|---|---|
| M0 | Free positive r_k | d_k=0 exactly | Are ancestry mixing, scale and age baseline sufficient? |
| M1 | Free positive r_k | One common d>=0 | Is a shared sign-aware age interaction useful? |
| M2 (candidate main) | Free positive r_k | Individual d_k>=0 | Do PRS-specific age responses add held-out value? |

The baseline age function, gate, cohort, teacher targets, preprocessing and stopping rules are identical in M0–M2. No separate nonlinear age network per PRS. No separate positive/negative d values; both signs share the same d_k for now.

Use both the existing locked KD weight and lambda_KD=0 for the same structures. This distinguishes capacity/prior improvements from an actual distillation benefit. M0 must be allowed to win; d need not be nonzero or heterogeneous.

## 4. Training

Keep teacher checkpoint/version, context/query settings, positive class and information set unchanged.

```text
y_soft = (1 - lambda_KD) * y + lambda_KD * q_teacher_oof
loss_data = mean(BCEWithLogits(eta, y_soft))
loss = loss_data + existing_comparable_regularization
       + kappa * mean(log_r ** 2)
       + kappa * mean(d_effective ** 2)
```

`BCEWithLogits` receives logits, not probabilities; soft targets are in [0,1] [S3, S5]. The added log-r penalty is an explicit shrinkage toward r=1 on the frozen score scale, not a biological assertion. The d penalty shrinks toward no interaction. For M1 compute the d penalty on the single/effectively repeated common value; for M0 it is exactly zero. Do not penalize the same parameters again through optimizer weight decay without documenting it.

Suggested small new penalty grid: `kappa in {1e-4, 1e-3, 1e-2}`. This is an initial numerical grid, not a known optimum. All arms receive the same selection budget. Retain the current lambda, learning rate, optimizer and stopping policy for the first structural comparison; do not silently re-tune the teacher or replace the primary evaluation metric.

### Nonnegative optimization

A simple exact-boundary implementation is projected optimization:

```python
# d and age_slopes are directly trainable, initially nonnegative.
optimizer.step()
with torch.no_grad():
    d.clamp_(min=0.0)
    age_slopes.clamp_(min=0.0)
```

For M0, d is a fixed zero buffer. For M1, d is one scalar expanded to K entries. For M2, d is a K-vector. Zero initialization is valid because the derivative w.r.t. d of the transform need not be zero. Do not use d=raw**2 at raw=0, which would suppress its initial gradient. A softplus alternative is allowed only with documented near-zero initialization and the acknowledgment that it cannot attain exact zero at finite raw parameter values.

Parameterize r using unconstrained log_r, with r=exp(log_r). Do NOT enforce sum(log_r)=0 unless a separate free overall scale is explicitly introduced; no such scale is part of this version. Do NOT enforce sum(d)=0.

Center gate parameters over K in forward and regularization, or use the existing equivalent identifiable representation. K=1 yields pi=1 and has no meaningful gate parameters. This removes common softmax shifts, not all gate/scale ambiguity.

### Initialization

Fit M0 using only the appropriate training scope. Initialize r=1, gate near uniform (or a compatible same-scope gate), and baseline from the existing allowed training procedure; no held-out prevalence initialization. Train all components jointly.

Initialize M1/M2 from M0 with d exactly zero and assert identical forward outputs before optimization. Then jointly optimize every component. Repeat the existing seeds (17, 29, 43) rather than interpreting one favorable initialization as a robust improvement.

Do not claim exact recovery of v1 at d=0: its free shared s(age) is absent and its baseline may not have been monotone. Copying a v1 gate alone is permissible only under identical preprocessing/scope, but is not a full exact warm start.

### Numerical safety

Age clipping prevents unbounded extrapolation, not arbitrarily large in-range parameters. Keep regularization and finite-value checks. Track max |d*t| and scale magnitudes. Fail/report non-finite runs; do not silently clip logits, winsorize PRSs, or cap d to improve a result. A predeclared bounded transform would be another model, not an invisible numerical fix.

## 5. OOF isolation and evaluation

Teacher targets can be reused ONLY if teacher version, feature transforms, PRS panel, labels, fit scope and context indices match exactly. A matching filename is insufficient.

For a student inner training split U and tuning split H, the teacher context used to generate targets in U must not include H labels. Within U, use the existing OOF teacher protocol. After locking choices, refit on the permitted development training pool. Keep the assessment set out of all selection, teacher contexts and preprocessing estimation.

If the existing formal report uses the fixed Stage-2 logistic wrapper, retain the correct nested Stage-1 pipeline OOF protocol for every student. Never use global teacher-OOF labels that were indirectly informed by an outer held-out fold to train that fold's student. Do not compare direct student probabilities with Stage-2 teacher probabilities.

Report direct-student monotonicity separately from the wrapper. A negative Stage-2 score coefficient, an unrestricted age coefficient or another age path can break it. Audit and report the wrapper's derivative/finite differences; do not silently constrain or replace the historical evaluator and call it unchanged.

Any evaluation set that has already guided repeated model choices is developmental/post-hoc evidence. Preserve an untouched holdout or new outer evaluation for confirmation. Never promise a minimum improvement or claim a biological discovery from improved AUC alone.

Required outcomes, reusing existing metric definitions:

- Main task performance: predeclared primary endpoint metric, plus log loss, AUROC, Brier and calibration where appropriate. Retain historical OR90 / Nagelkerke R² definitions when running the official wrapper.
- Teacher fidelity: held-out probability error, distinct from true-label performance.
- Incremental PRS value: retain the metadata-only reference.
- Existing age/ancestry subgroups with counts, events and joint support.
- Matched-seed stability; paired uncertainty under the existing valid procedure. Seeds are not confidence intervals.

A higher-fidelity model is not automatically a better predictor. A monotone model can disagree with a nonmonotone teacher; assess real labels and report the fidelity trade-off.

## 6. Exact patient explanation

Export the unnormalized effective factor and contribution:

```text
m_ik = exp(d_k * t_i * sign(Z_ik))
beta_ik = pi_ik * r_k * m_ik
C_ik = beta_ik * Z_ik
C_ref_ik = pi_ik * r_k * Z_ik
Delta_age_i = f_up(a_i) + sum_k(C_ik - C_ref_ik)
eta_ref_i = alpha + gamma^T A_i + delta^T C_i + sum_k C_ref_ik
eta_i = eta_ref_i + Delta_age_i
```

This last identity is the simplest age explanation: holding the other inputs fixed, `Delta_age>=0` above the reference age, `<=0` below it, and exactly zero at the reference age.

`beta_ik` now depends on the sign of the individual's score, not only age and ancestry. Do not report exp(beta_ik) as a global one-SD OR across a change crossing zero. The normalized `beta/sum(beta)` is a coefficient share, not a patient's percentage risk attribution. Logit contributions are not additive probabilities.

Export: direct p/logit, base-reference logit, baseline age term, reference-age logit, age delta, pi, r, d, original and transformed PRSs, multipliers, effective coefficients, contributions, support flags. All patient-level outputs stay in the approved local environment.

Make only the essential aggregate plots: pi across observed ancestry, r/d per PRS, sign-aware curves for z=-1/0/+1 across supported ages, and direct student/teacher calibration and fidelity. Show sample support. Highly correlated PRSs can yield unstable pi/r/d despite stable predictions; report this limitation and effective contributions, not a uniquely recovered genetic mechanism.

## 7. Required tests and delivery

The attached reference core is stateless, not an integrated training pipeline. It illustrates the exact forward function and monotone baseline. Its synthetic tests have been run locally; this does NOT test clinical efficacy, model fitting, cache isolation or repository integration.

Preserve and test:

1. pi sums to one and is nonnegative; r positive; d and baseline slopes nonnegative.
2. Reference-age identity; exact d=0 reduction; exact zero PRS contribution.
3. Positive, zero and negative PRSs all have nondecreasing age contribution.
4. Total direct probability/logit is nondecreasing with age at fixed Z/A/C.
5. Increasing each PRS at fixed age/ancestry cannot decrease direct prediction.
6. Reference-age decomposition and contribution reconstruction agree numerically.
7. K=1, no extra covariates, tied/constant ages, out-of-range age policy, correct age units.
8. M0 -> M1/M2 equality at zero d, save/load parity, finite parameter gradients and projected constraints.
9. Identical evaluation IDs, scope-correct teacher caches, no hidden age covariates, no silent label/score flipping.
10. Direct vs Stage-2 outputs and monotonicity claims remain clearly separated.

Deliver actual code changes, tests, frozen config, scope/provenance checks, M0/M1/M2 comparison, direct-supervision comparisons, local models and interpretation outputs. Produce a short decision: does M2 improve over M1 and M0 on allowed held-out data, and does KD beat the same supervised structure?

No real patient data were inspected or used to train this version here. No predictive improvement is claimed by this specification.

## Sources

[S1] Guo et al. Polygenic risk score for type 2 diabetes shows context-dependent effects across populations. Nature Communications (2025). `https://www.nature.com/articles/s41467-025-63546-4`. Supports age/context variation, not this exact sign-aware function.

[S2] Lennon et al. Selection, optimization and validation of ten chronic disease polygenic risk scores for clinical implementation in diverse US populations. Nature Medicine (2024). `https://www.nature.com/articles/s41591-024-02796-z`. Ancestry-based distribution calibration precedent, not a requirement to replace existing project preprocessing.

[S3] Hinton, Vinyals and Dean. Distilling the Knowledge in a Neural Network (2015). `https://arxiv.org/abs/1503.02531`. Soft-label distillation background.

[S4] Pal Choudhury et al. iCARE: An R package to build, validate and apply absolute risk models. PLOS ONE (2020). `https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0228198`. Population absolute-risk inputs differ from a binary classifier.

[S5] PyTorch official BCEWithLogitsLoss documentation. `https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html`. Soft targets/logits and numerical implementation. Use the repository's locked environment; no automatic upgrades.

Project lineage: v1 `PRS_Prototype_Distillation_Codex_Plan_v1.md` and v2 `PRS_Distillation_v2_Scale_Age_Addendum.md` supplied in this conversation. All new model choices and numerical settings in this file are proposed designs, not validated biological conclusions.
