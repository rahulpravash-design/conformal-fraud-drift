# Dataset audit

**Status: NOT RUN on real data.** The Kaggle host is blocked in the development sandbox (proxy policy), so no BAF file was available.
Nothing in this file is an observed statistic.

## Documented facts (official datasheet / paper; to be confirmed against files)
- 6 files × 1M rows [V]; 8 months, `month` ∈ [0,7] (Feb–Sep) [V]; no missing information except −1 sentinels [V].
- Monthly fraud prevalence reported by the paper as 0.85%–1.5% (higher in later months); monthly volume share 9.5%–15% [V, paper §3].
- ~80/20 split of applications between age <50 and ≥50 groups; older group fraud prevalence ≈ 2× [V, datasheet Q17] — note this ≈20% share
  is the audit check that discriminates `≥50` from `>50` using group sizes only.
- Implication to verify: ~1M × (9.5–15%) ≈ 95k–150k rows/month, ≈ 1k+ fraud cases/month; calibration month needs ≥ ⌈(1−α)/α⌉ = 9 fraud cases for finite α=0.1 thresholds [derived, trivial].

## To produce (command)
`python scripts/audit_dataset.py --variant Base TypeI TypeII TypeIII TypeIV TypeV` → `results/tables/audit_*`, `results/figures/audit_*`, `results/metrics/audit_*.json` (rows, columns, hashes, duplicates, sentinel counts, monthly prevalence + CI, group×month counts, KS/TV shift vs training months, split feasibility).
Shift descriptors cover covariate/label shift only; concept drift P(Y|X) is not identified by them.
