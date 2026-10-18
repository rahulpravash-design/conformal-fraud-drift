#!/usr/bin/env python
"""Stage 1: split, leakage checks, train + select on train/val, save predictions. Usage:
python scripts/train_baselines.py --variant Base --model lightgbm"""
import argparse

from conformal_fraud.data.load_data import find_variant_file, read_variant
from conformal_fraud.pipeline import train_stage
from conformal_fraud.utils.config import load_configs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=None)
    ap.add_argument("--model", default="lightgbm", choices=["lightgbm", "logreg"])
    ap.add_argument("--config-dir", default="configs")
    ap.add_argument("--data-dir", default=None)
    a = ap.parse_args()
    cfgs = load_configs(a.config_dir)
    v = a.variant or cfgs["data"]["primary_variant"]
    path = find_variant_file(a.data_dir or cfgs["data"]["data_dir"], v, cfgs["data"]["file_format"])
    run_dir = train_stage(read_variant(path), cfgs, v, a.model, cfgs["experiment"]["output_dir"], data_path=path)
    print(f"train stage complete -> {run_dir}")


if __name__ == "__main__":
    main()
