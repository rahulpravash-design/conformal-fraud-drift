# Experimental protocol v1 (prepared 2026-10-18) — STATUS: NOT YET FROZEN

This file is the executable specification derived from `docs/protocol.md` (v0.2, kept as history). **It is not frozen** because the
freeze conditions below require the real BAF audit, which has not been possible (data unreachable). Nothing here has been run on real data.

## Freeze conditions (all required, in order)
1. Real `Base` file present in `data/raw/`; `python scripts/audit_dataset.py --variant Base` completes; SHA-256, size, acquisition date recorded in `docs/dataset_audit.md`.
2. Audit confirms: `fraud_bool` binary; `month` ∈ {0..7}; 1,000,000 rows (datasheet figure); `customer_age` present; every partition non-empty with ≥30 fraud cases per partition (feasibility rule, below).
3. Audit-driven changes (feasibility only) logged in the change log, dated, **before** any model is fitted.
4. Commit the frozen file (tag `protocol-v1.0`) **before** running `train_baselines.py`. Test-month outcomes must not have been viewed.

## 1. Research questions and hypotheses
RQ1 temporal effect on marginal coverage; RQ2 group heterogeneity; RQ3 method comparison (initial scope: marginal vs label-conditional LAC); RQ4 variants (later).
Hypotheses (all untested; stated before data): H1 marginal LAC under-covers the fraud class (prevalence ≈1%); H2 observed marginal test coverage may differ from 90% because month 5 and months 6–7 are not exchangeable (direction not assumed); H3 coverage differs between age groups. Null/no-difference outcomes are reportable results.

## 2. Data and target
BAF Base, Feedzai, NeurIPS 2022; Kaggle distribution; CC BY-NC-ND 4.0 (from datasheet; re-verify on the Kaggle page). Target `fraud_bool`. Synthetic data: conclusions concern the benchmark, not real banks. Raw files never modified or committed.

## 3. Features
All columns except `fraud_bool`, `month`, `row_id`; includes `customer_age`. Categoricals: payment_type, employment_status, housing_status, source, device_os. Missing sentinels (-1) left as values. No data-driven feature selection. Sensitivity: exclude `customer_age`.

## 4. Split (to be validated on real data)
train months 0–3 · val 4 · cal 5 · test 6–7. Validity rules checked by `SplitSpec`/`leakage_checks`: disjoint, strictly chronological, months exist, row-id disjoint, label/month/row_id not features, no cross-partition exact-duplicate feature vectors. **Feasibility rule:** each of val/cal/test contains ≥30 fraud cases overall and in each reported group; otherwise the group is reported as "not interpretable" (not dropped, split not tuned to outcomes). If months 0–7 are not all present or ordered, stop and propose a documented alternative; months are never fabricated.

## 5. Preprocessing and model
Preprocessing fitted on train rows only (category vocabulary; unseen levels → NaN). LightGBM grid and early stopping per `configs/methods.yaml` (6 configs, selection by validation average precision, tie → first listed). Seed 20260101; `deterministic=true`. Secondary: logistic regression (C × class_weight). Operating point: 5% FPR threshold from validation negatives, fixed for test.

## 6. Conformal procedure
LAC score `1−p̂(y|x)`. Calibration on month 5 only. `k=⌈(n+1)(1−α)⌉`, q = k-th smallest score, +∞ if k>n (reported, never silently replaced). Sets `{y : s(x,y) ≤ q_y}`, ties inclusive. Two variants: marginal; label-conditional (Mondrian on Y). α=0.10 primary (nominal coverage 90%); 0.20, 0.05 secondary. Thresholds fixed before test scoring. No online updating, no reweighting.
Assumption statement: guarantee requires exchangeability of calibration and test; under temporal drift it is not guaranteed, so observed coverage is an empirical measurement to be reported against the nominal target without claiming validity.

## 7. Metrics
Classification: ROC-AUC, average precision (sklearn; called "PR-AUC (average precision)" in text), precision/recall/F1 and realised FPR at the fixed threshold. Conformal: marginal, fraud-class, legit-class coverage; set-size mean and shares of empty/singleton/{0,1}; per month and pooled. Review: approve / escalate / human-review rates and fraud captured in each. Groups: `age_ge_50` primary, `age_gt_50` sensitivity (fixed before data inspection; D-004 — not to be changed after seeing results; any change is a dated, separately labelled deviation).

## 8. Uncertainty
Wilson 95% CI for proportions; Newcombe for group differences; bootstrap (seeded) for AUC/AP. CIs reflect sampling noise only, not drift variability across months; two test months do not support general claims about drift. Exploratory p-values, if any, Holm-adjusted and labelled exploratory.

## 9. Reproducibility run
Full pipeline run twice from the same config; compare predictions.parquet and calibration.json hashes; report identical/non-identical. Record package versions, git commit, seed, data SHA-256 (`run_metadata_*.json`).

## 10. Planned sensitivity analyses (after the primary run, not before)
logistic baseline; α∈{0.2,0.05}; benchmark-default split (0–5/6–7); rolling-origin windows; exclude `customer_age`; `age_gt_50`; Types I–V.

## 11. Stopping conditions
Stop and report if: data/audit assumptions fail; leakage check fails; a partition fails the feasibility rule for the primary analysis; calibration quantile is +∞ for any label. Do not add APS/ACI/weighted/group-Mondrian until the primary run is validated and reproduced.

## 12. Limitations (declared in advance)
Synthetic benchmark; two test months; single seed for the split (no resampling of time); label-noise/delay absent; review policy illustrative; no cost model; marginal guarantee ≠ group or label guarantee.

## Change log
| Date | Change | Reason | Impact |
|---|---|---|---|
| 2026-10-18 | v1 prepared from v0.2; added freeze conditions, feasibility rule, hypotheses H1–H3, reproducibility run | Phase 5 request | none; no real data seen |
