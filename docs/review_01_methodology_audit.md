# Independent methodology review #1 (self-review of protocol v0.2 + implementation at first commit)
Scope: code and docs in this repository. **No BAF data were available, so no output of the real pipeline was inspected;** the review covers
design and the synthetic-fixture test run only. Passing tests ≠ approval. Severity: H = can invalidate a claim, M = weakens it, L = polish.

| # | Sev | Issue | Why it matters | Required correction / status |
|---|---|---|---|---|
| 1 | H | **Core RQ1 inference rests on two test months.** Pooled "all" row mixes months 6–7. | Coverage differences between 2 months cannot support statements about "drift" in general; CIs reflect sampling noise only. | Report per month (done); add rolling-origin windows as prespecified secondary analysis (D-006, **not yet implemented**); phrase RQ1 as observation on this benchmark. |
| 2 | H | **Marginal LAC says little about the fraud class at ~1% prevalence.** | Marginal 90% coverage can be met while fraud-class coverage is far lower; headline "coverage holds" would mislead. | Class-conditional method implemented; protocol requires reporting `coverage_pos` next to marginal coverage. Hypothesis H1 (marginal under-covers fraud) is **untested**. |
| 3 | H | **Guarantee scope.** Months 5→6/7 violate exchangeability by construction. | Any sentence "conformal guarantees 90%" under drift is false. | Protocol §5 states this; paper text must say "empirical coverage". Unit test only exercises the exchangeable case, correctly labelled. |
| 4 | M | **Group definition conflict** (datasheet `≥50` vs notebook `>50`; ages rounded to decades). | Different groups ⇒ different gaps. | Pre-registered primary/sensitivity before data view (D-004). Audit must confirm the ~80/20 share. |
| 5 | M | **`customer_age` is both a feature and the group variable.** | Coverage-by-group of a model that uses the attribute differs from a blind model; fairness interpretation changes. | Matches benchmark; exclude-age sensitivity planned (D-008), **not yet run**. Coverage parity ≠ fairness (stated). |
| 6 | M | **CIs assume independent records and no multiplicity control.** Many cells (months × groups × metrics × α). | Overstated precision; false positives among many gaps. | Wilson/Newcombe are descriptive; any p-values exploratory with Holm over a declared family (§8). `interpretable_pos` ≥30 is only a warning. |
| 7 | M | **Hard-failing duplicate-feature-vector leakage check** may trigger on legitimate synthetic duplicates in real data. | Would block a valid run, tempting an ad-hoc bypass. | Decide rule after audit reports the duplicate counts; log in decisions.md. Do not weaken silently. |
| 8 | M | **Validation month used for early stopping, model selection and the FPR threshold.** | Validation metrics are optimistic; temporal gap val→test unmodelled. | Acceptable (conformal validity depends only on cal/test), but report val and test side by side; never report val as performance. |
| 9 | M | **Weighted CP for covariate shift assumes P(Y\|X) unchanged**; the BAF paper reports month-varying prevalence, and Type IV/V inject conditional shift deliberately. | Weighted method may be invalid on exactly the data studied. | Deferred (D-007); implement only if assumption defence is written; report ESS and weights. |
| 10 | M | **ACI needs an observation order; BAF has none within month.** | Randomised order is a simulation convention only. | Deferred; if added: many permutations, seeds recorded, no streaming claims. |
| 11 | L | `row_id` = file position; used only for disjointness. | Not an event time. | Documented. |
| 12 | L | Unseen categorical levels in later months become NaN at inference. | Silent handling of vocabulary drift. | By design (train-only vocabulary); add count of unseen levels to run metadata (TODO). |
| 13 | L | AP is sklearn step-wise average precision, not trapezoidal PR-AUC. | Naming ambiguity in paper. | Say "average precision" everywhere. |
| 14 | L | Ties handled deterministically (≤), no randomisation; slightly conservative for discrete scores. | LightGBM scores are continuous, so negligible. | Documented; ties unit-tested. |
| 15 | L | Reproducibility: SHA of real data not yet recorded; thread-count nondeterminism possible in LightGBM. | Bit-exact reruns not guaranteed across machines. | `deterministic=True`, `force_row_wise`, seed fixed; compare metrics to tolerance, not bits. |

## Code-level checks performed (on synthetic fixture)
- Quantile convention hand-verified (n=9, α=0.1 → max; n=5 → +∞; not an interpolated percentile).
- Set construction hand-verified including boundary ties; a floating-point boundary issue (1−0.7 ≠ 0.3) was found in my *test*, not the code, because
  calibration and test scores use the identical expression.
- Exchangeable simulation: mean marginal and fraud-class coverage ≥ 1−α−0.005 over 300 repeats.
- Label-scrambling invariance test: scrambling test-month labels leaves model, validation threshold and conformal thresholds bit-identical (so test labels
  cannot leak into fitting/calibration) while reported test metrics change.
## Verdict
Mechanics: **plausible, not yet validated on real data.** Methodology: **conditionally acceptable** subject to items 1–3 being respected in the write-up. Not approved for results.
