# [Provisional title] Does Conformal Fraud Detection Keep Its Promise Under Drift?
**DRAFT SKELETON — NO RESULTS EXIST.** Every `[PENDING: ...]` marker is an unfilled claim. Template/page limit: unverified (IJCNN CFP not read).

## Abstract
[PENDING: written last, from frozen results only.]

## 1 Introduction
Fraud scores feed human-review decisions; conformal prediction (CP) offers finite-sample coverage under exchangeability, which temporal drift can violate. We ask whether split-conformal LAC sets built on month-5 calibration keep nominal 90% coverage on months 6–7 of BAF Base, whether the aggregate hides label- and age-group differences, and what review workload results. [PENDING: contribution statement — cannot claim novelty until Gate A passes.]

## 2 Related work
Foundations: BAF (Jesus et al., NeurIPS 2022 — verified); adaptive CP under shift, weighted CP under covariate shift, APS [cite after verification: L02–L04 unverified]. CP for fraud: [PENDING: L09, L11 and others — read before describing].

## 3 Research gap and contributions
[PENDING Gate A.]

## 4 Dataset and problem formulation
BAF Base (synthetic, CC BY-NC-ND 4.0), `fraud_bool`, months 0–7. [PENDING: counts, prevalence from the audit.]

## 5 Method
LAC score, finite-sample quantile k=⌈(n+1)(1−α)⌉, marginal and label-conditional thresholds, sets {y: s(x,y) ≤ q_y}; coverage guarantee holds under exchangeability only (see protocol §6). Review policy: {0} approve, {1} escalate, {0,1}/{} human review (illustrative).

## 6 Experimental setup
Split 0–3 / 4 / 5 / 6–7; LightGBM 6-config grid selected on validation AP; 5%-FPR operating point; groups age ≥50 vs <50 (sensitivity >50); Wilson/Newcombe CIs. See `docs/experimental_protocol_v1.md`.

## 7 Results
[PENDING: none. Tables to be generated from results/ by script.]

## 8 Discussion
[PENDING.] Must separate observed coverage from guarantee; discuss H1–H3 including null outcomes.

## 9 Limitations and threats to validity
Synthetic benchmark; two test months; non-exchangeability; illustrative review policy; age is both feature and group variable; label-conditional ≠ group-conditional guarantee.

## 10 Conclusion
[PENDING.]

## References
Jesus et al., "Turning the Tables…", NeurIPS 2022 (see literature/references.bib). Others added only after verification.
