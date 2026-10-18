#!/usr/bin/env python
"""Core reproduction: audit -> train -> conformal for the primary variant (Base, LightGBM).
Usage: python scripts/reproduce_all.py [--variant Base] [--model lightgbm]
Expects raw data in data/raw (see data/README.md). Fails loudly if data are absent."""
import argparse
import subprocess
import sys

STEPS = [["scripts/audit_dataset.py"], ["scripts/train_baselines.py"], ["scripts/run_split_conformal.py"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="Base")
    ap.add_argument("--model", default="lightgbm")
    a = ap.parse_args()
    for step in STEPS:
        cmd = [sys.executable, *step, "--variant", a.variant] + (["--model", a.model] if "audit" not in step[0] else [])
        print("$", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
