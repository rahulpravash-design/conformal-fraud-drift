#!/usr/bin/env python
"""Stage 2: conformal calibration on the calibration month and evaluation on test months.
Usage: python scripts/run_split_conformal.py --variant Base --model lightgbm"""
import argparse

from conformal_fraud.pipeline import conformal_stage, run_id
from conformal_fraud.utils.config import load_configs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=None)
    ap.add_argument("--model", default="lightgbm")
    ap.add_argument("--config-dir", default="configs")
    a = ap.parse_args()
    cfgs = load_configs(a.config_dir)
    v = a.variant or cfgs["data"]["primary_variant"]
    rd = f"{cfgs['experiment']['output_dir']}/metrics/{run_id(v, a.model, cfgs['experiment']['seed'])}"
    conformal_stage(rd, cfgs)
    print(f"conformal stage complete -> {rd}")


if __name__ == "__main__":
    main()
