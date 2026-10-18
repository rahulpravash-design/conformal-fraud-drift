# Reproducibility
```bash
python3 -m venv .venv && source .venv/bin/activate      # or: uv venv && source .venv/bin/activate
pip install -e . pytest                                  # versions used so far: requirements-lock.txt
python -m pytest                                         # 37 tests, synthetic fixture only
# --- requires real data in data/raw (see data/README.md) ---
python scripts/audit_dataset.py --variant Base
python scripts/train_baselines.py --variant Base --model lightgbm
python scripts/run_split_conformal.py --variant Base --model lightgbm
# or: python scripts/reproduce_all.py
```
Outputs (git-ignored): `results/metrics/Base_lightgbm_seed20260101/{predictions.parquet, baseline_metrics.csv, model_selection.csv,
conformal_summary.csv, group_metrics.csv, group_gaps.csv, calibration.json, test_prediction_sets.parquet, run_metadata_*.json}`.
Row-level files derive from CC BY-NC-ND data: do not redistribute.
Expected on success: all leakage checks `passed: true` in `run_metadata_train.json`; `calibration.json` has finite `qhat` unless `qhat_infinite` is flagged.
**Status: real-data commands have NOT been executed (dataset unavailable in this environment).**
