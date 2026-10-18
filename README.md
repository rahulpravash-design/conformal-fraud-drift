# Conformal Fraud Detection Under Temporal Drift

A reproducible machine-learning research project on how well conformal prediction sets keep their coverage under temporal distribution shift in
fraud detection, using the Feedzai Bank Account Fraud (BAF) benchmark (synthetic data, NeurIPS 2022).

## Research questions
* **RQ1** How does temporal distribution shift affect marginal conformal coverage on later test months?
* **RQ2** Does aggregate coverage conceal differences across sufficiently represented demographic groups?
* **RQ3** How do selected conformal methods compare in coverage reliability, set size, fraud-detection performance and review workload?
* **RQ4** How consistent are the results across the BAF variants?

The research gap is **not yet verified** (see `docs/research_gap.md`). No novelty claim is made. BAF is synthetic: results say nothing direct about real banks.

## Dataset
Official repo: https://github.com/feedzai/bank-account-fraud — data on Kaggle, licensed **CC BY-NC-ND 4.0**, not stored in Git. See `data/README.md`.

## Protocol (draft v0.2, `docs/protocol.md`)
| Partition | Months | Purpose |
|---|---|---|
| Training | 0–3 | fit classifier + preprocessing |
| Validation | 4 | model selection, early stopping, 5%-FPR operating threshold |
| Calibration | 5 | conformal calibration only |
| Test | 6–7 | evaluation only |

Models: LightGBM (primary), logistic regression (secondary). Conformal: LAC score, finite-sample quantile, **marginal** and **class-conditional**
split conformal at α = 0.10 (0.20/0.05 later). Review policy `{0}` approve, `{1}` escalate, `{0,1}`/`{}` human review (experimental).
APS, group-Mondrian, ACI and weighted CP are deferred until the core run is validated.

## Quick start
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e . pytest
python -m pytest                      # 37 tests on a synthetic fixture (mechanics only)
python scripts/reproduce_all.py       # needs BAF files in data/raw
```
Layout: `configs/` · `src/conformal_fraud/{data,models,conformal,evaluation,statistics,visualization}` · `scripts/` · `tests/` · `docs/` · `literature/` · `results/` (git-ignored) · `paper/` · `submission/`.

## Current status (2026-10-18)
Infrastructure, protocol draft, audit script, LightGBM + split-conformal pipeline and tests are written; **tests pass on synthetic data only.**
**No BAF result exists yet** (dataset not obtainable in the development sandbox). See `docs/STATUS.md`.

## Research integrity
No fabricated results or citations · no test-label use in fitting/selection/calibration (tested) · assumptions and limits stated · tables/figures generated from result files.

## Publication target
Provisional: IEEE IJCNN 2027 (Cape Town, 14–18 June 2027 per search listing). Deadline conflicting in sources (15 vs 31 Jan 2027) — unverified. See `docs/conference_checklist.md`. No acceptance or indexing is assumed.

## License
Software license to be chosen by the author. The BAF dataset has its own license (CC BY-NC-ND 4.0).
