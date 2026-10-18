# Experimental Protocol

**Status: DRAFT v0.2 (2026-10-18) — NOT FROZEN.** Freeze requires (a) the dataset audit on the real BAF files and
(b) the literature gap check. Every change after freezing must be appended to the change log with date and reason.

Evidence labels: **[V]** verified from an official source read in this session, **[U]** unverified / to be checked on real data.

## 1. Dataset
- Feedzai Bank Account Fraud (BAF) suite, NeurIPS 2022. Six files of 1M rows each: Base, Type I–V. **[V: datasheet Q5, paper Table 1]**
- Distribution channel: Kaggle (`sgpjesus/bank-account-fraud-dataset-neurips-2022`). **[V: official README]** Not reachable from the
  development sandbox used so far → **no BAF file has been inspected yet**; all numeric dataset facts below are from documentation only.
- License: **CC BY-NC-ND 4.0** (datasheet Q44). Raw data and *row-level derived files* (predictions with labels) are not redistributed. **[V]**
- Primary variant: Base. Other variants (RQ4) only after the Base pipeline is validated.

## 2. Target, time, features
- Label `fraud_bool` (1 = fraudulent application). **[V]**
- Time column `month`, integers 0–7, documented as Feb–Sep application month. **[V]** Observed encoding to be confirmed by audit **[U]**.
- Within-month order carries **no** information (datasheet: instances independent). `days_since_request` is a *feature*, not an event time.
- Features: all columns except `fraud_bool`, `month`, `row_id`. This **includes `customer_age`**, as in the benchmark's own
  LightGBM study. Sensitivity run excluding it is planned (config `exclude_features`). Categorical: payment_type, employment_status,
  housing_status, source, device_os. `-1`/negative values are missing sentinels and are left as numeric values (tree models handle them).
- `device_fraud_count` is documented as range [0,1] and may be (near-)constant; the audit reports constant columns; no manual removal.

## 3. Chronological split (proposed; compatibility with data to be confirmed [U])
| Partition | Months | Use |
|---|---|---|
| train | 0–3 | fit classifier + preprocessing |
| val | 4 | hyper-parameter selection, early stopping, operating threshold (FPR 5%) |
| cal | 5 | conformal calibration only |
| test | 6, 7 | evaluation only |

**Deviation from the benchmark's own split** (train 0–5, test 6–7, datasheet Q11): ours spends months 4–5 on validation/calibration,
so the classifier sees less data than the benchmark default. This keeps calibration temporally adjacent to test (the setting in which a
drift-induced coverage gap is most interpretable). The benchmark-default split is a planned robustness check (D-003), not the primary.
The split is enforced by `SplitSpec` (non-empty, disjoint, strictly chronological, months exist) and by automated leakage checks.

## 4. Classifiers
LightGBM (primary) and Logistic Regression (secondary). Preprocessing (category vocabulary, scaling, encoding) is fitted on **train rows only**.
LightGBM: explicit 6-config grid in `configs/methods.yaml`, early stopping and selection on **validation average precision**; no test use.
Logistic regression: C × class-weight grid, same selection rule. Seed 20260101; all seeds/versions recorded per run.
Operating point for hard-label metrics: threshold giving ≤5% FPR on **validation negatives**, applied unchanged to test.

## 5. Conformal methods (initial scope)
- Score: LAC, `s(x,y)=1−p̂(y|x)`.
- Quantile: `q = s_(k)`, `k=⌈(n+1)(1−α)⌉`; `+∞` if `k>n` (all labels included, flagged in `calibration.json`).
- Set: include label y iff `s(x,y) ≤ q_y` (deterministic, ties included, no randomisation).
- **Marginal**: single q from all calibration scores. **Class-conditional (Mondrian on Y)**: q_y from calibration points with label y.
  Rationale: with ≈1% fraud, marginal LAC calibration is dominated by legitimate cases and guarantees nothing for the fraud class (D-005).
- α = 0.10 primary; 0.20 and 0.05 robustness. Deferred until the core run is validated: APS, group-Mondrian, ACI, weighted CP (D-007).

**What is and is not guaranteed.** Under exchangeability of calibration and test points, marginal coverage ≥ 1−α (class-conditional: per label).
Months 5 and 6–7 are *not* exchangeable under drift; therefore observed test coverage is an **empirical measurement**, not a guarantee. The
drift-free sanity check is the unit test on simulated exchangeable data.

## 6. Metrics
Classification: ROC-AUC; average precision (sklearn definition, not trapezoidal PR-AUC); precision/recall/F1/confusion at the validation-chosen
5%-FPR threshold; realised test FPR reported. Conformal: marginal, fraud-class and legitimate-class coverage; set-size distribution
(empty/singleton/both); per test month and pooled. Review policy (experimental, **not** a validated bank policy): `{0}` approve, `{1}` escalate,
`{0,1}` or `{}` human review; review/escalate/approve rates; fraud approved / reviewed / escalated. No monetary cost model is assumed.

## 7. Groups
Primary: `age_ge_50` (customer_age ≥ 50 vs < 50), per datasheet Q17. Sensitivity: `age_gt_50` (> 50) as coded in the official generation
notebook. The two documents disagree; ages are rounded to decades so the definitions differ by the age-50 stratum (D-004). Fixed before data inspection.
Group tables report n, fraud n, coverage (overall, fraud, legitimate), Wilson 95% CI, review rate; gaps use Newcombe 95% CIs.
`interpretable_pos` = fraud count ≥ 30 in the cell: a screening warning, not a reliability guarantee. Income/employment groups: exploratory only.

## 8. Uncertainty and statistics
Wilson intervals for proportions; Newcombe hybrid-score for differences; bootstrap for AP/AUC if reported with CIs. Records are treated as independent
(documented BAF property) — intervals capture sampling noise only, **not** between-month drift variability; with two test months no
inference about "drift in general" is claimed. Group-gap p-values, if any, are exploratory and Holm-adjusted over the declared family.

## 9. Audit vs confirmatory
The dataset audit is descriptive and uses full-data labels (prevalence by month/group). No design choice (features, grid, groups, α, split) may be changed
because of test-month *outcomes* (coverage, performance). Audit-driven changes allowed only for feasibility (e.g. a group cell too small) and must be logged.

## 10. Exclusions / failure handling
Failed or infinite-threshold calibrations are reported, never silently replaced by a global threshold. Unreported experiments are listed in the manifest.

## 11. Planned (not yet run)
Logistic baseline on same split; α∈{0.2,0.05}; benchmark-default split; rolling-origin windows (secondary, to address the 2-test-month limit — D-006);
exclude-age sensitivity; Types I–V (RQ4); then APS / group-Mondrian / ACI / weighted CP only if assumptions can be defended.

## Change log
| Date | Change | Reason | Confirmatory impact |
|---|---|---|---|
| 2026-10-18 | v0.2 draft: corrected to documented schema; added class-conditional method, group-definition pre-registration, benchmark-default-split deviation note | Official datasheet/notebooks read | none (nothing run on real data) |
