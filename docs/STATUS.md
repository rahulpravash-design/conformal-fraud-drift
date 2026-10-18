# Project status (updated 2026-10-18)
**Stage:** 0–4 partially complete (infrastructure + protocol draft); Stage 5–6 code written and mechanically tested on synthetic data; **no real-data run.**

| Gate | State | Evidence / blocker |
|---|---|---|
| A research gap | **NOT passed** | candidates not opened (scholarly hosts unreachable); `docs/research_gap.md` |
| B reproducible core experiment | **NOT passed** | BAF files unavailable here (Kaggle blocked); code + 37 tests pass on synthetic fixture only |
| C methodological validity | partial | quantile/set/coverage unit-tested; real-data validation pending |
| D results freeze / E submission | not started | — |

## Blockers
1. **BAF data**: download from Kaggle on a machine with access, place in `data/raw/`, then run the audit (researcher action or enable Kaggle access for this environment).
2. **Literature verification** and **IJCNN CFP verification**: scholarly sites and ijcnn.org unreachable; manual reading needed (list in `research_gap.md`, `conference_checklist.md`). Deadline conflict 15 vs 31 Jan 2027.
3. Software license undecided (no LICENSE file added).

## Next three actions
1. Obtain `Base.*`, run `audit_dataset.py`; fill `dataset_audit.md`; confirm `month` encoding, column names, age-group shares (settles `≥50` vs `>50`), duplicates; then freeze protocol v1.0.
2. Run `train_baselines.py` + `run_split_conformal.py`; inspect leakage checks, calibration.json, per-month coverage (core experiment, Gate B).
3. Read L09–L11 and run the literature queries; complete the gap table (Gate A).
