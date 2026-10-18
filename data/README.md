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

## Windows instructions (researcher machine) — added 2026-10-18
Source of truth: the official repo README (https://github.com/feedzai/bank-account-fraud) points to the Kaggle dataset above; it gives
no filenames and says data comes as `parquet` or `csv` (zipped; decompress first). Filenames are therefore **unverified** until you look.
```powershell
cd path\to\conformal-fraud-drift
# Option A (browser): download the zip from the Kaggle page (log in, accept terms), extract into data\raw\
# Option B (CLI): put your own token at %USERPROFILE%\.kaggle\kaggle.json (never in the repo)
pip install kaggle
kaggle datasets download -d sgpjesus/bank-account-fraud-dataset-neurips-2022 -p data\raw --unzip
dir data\raw
Get-FileHash data\raw\* -Algorithm SHA256
```
If names differ from `Base.parquet` / `Base.csv`, rename or tell me the listing. Then: `python scripts/audit_dataset.py --variant Base`
(or `python scripts/reproduce_all.py`) and share `results/metrics/audit_Base.json`; do not upload the raw data anywhere (CC BY-NC-ND 4.0).
