# Research Questions

## RQ1
How does temporal distribution shift affect marginal conformal prediction coverage on later test months?

## RQ2
Does aggregate coverage conceal differences in coverage across sufficiently represented demographic groups?

## RQ3
How do selected conformal methods compare in coverage reliability, prediction-set size, fraud-detection performance, and estimated human-review workload?

## RQ4
How consistent are the observed results across the available variants of the Feedzai Bank Account Fraud benchmark?

## Status (2026-10-18)
Provisional. Answerability on the real data is unchecked (dataset audit not run). Notes from the first review (`docs/review_01_methodology_audit.md`):
RQ1 rests on only two test months (rolling-origin analysis planned); RQ2 should report fraud-class coverage separately from marginal coverage;
RQ4 is conditional on Types I–V being protocol-compatible. Hypothesis H1 (untested): marginal LAC calibration under-covers the fraud class at ~1% prevalence.
