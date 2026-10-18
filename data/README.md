# Data acquisition

Raw BAF files are **not** in Git (license CC BY-NC-ND 4.0; also `.gitignore`).

1. Source: https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022 (linked from the official repo README,
   https://github.com/feedzai/bank-account-fraud). Read the license terms on that page.
2. Download (requires a free Kaggle account/API token; never commit the token):
   ```bash
   pip install kaggle
   kaggle datasets download -d sgpjesus/bank-account-fraud-dataset-neurips-2022 -p data/raw --unzip
   ```
3. The loader expects `data/raw/<Variant>.parquet` or `.csv` with `Variant ∈ {Base, TypeI, TypeII, TypeIII, TypeIV, TypeV}`
   (names used by the official notebook). **Kaggle's actual file names were not verified** — rename if they differ.
4. Run `python scripts/audit_dataset.py --variant Base`; it records byte size and SHA-256 in `results/metrics/audit_Base.json`.
   Record the download date and Kaggle dataset version in `docs/dataset_audit.md`.
