#!/usr/bin/env python
"""Audit one or more BAF variants. Usage: python scripts/audit_dataset.py --variant Base [--data-dir data/raw]"""
import argparse
import json
from pathlib import Path

from conformal_fraud.data.audit_data import audit_variant
from conformal_fraud.data.load_data import find_variant_file, read_variant, validate_schema
from conformal_fraud.utils.config import load_configs
from conformal_fraud.utils.provenance import environment_info, git_commit, sha256_file, utc_now, write_json
from conformal_fraud.visualization.plots import plot_monthly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", nargs="+", default=None, help="default: primary variant only")
    ap.add_argument("--config-dir", default="configs")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    cfgs = load_configs(a.config_dir)
    d, e = cfgs["data"], cfgs["experiment"]
    for v in a.variant or [d["primary_variant"]]:
        path = find_variant_file(a.data_dir or d["data_dir"], v, d["file_format"])
        df = read_variant(path)
        validate_schema(df, d["label"], d["time_column"], d["categorical_features"],
                        sorted({g["column"] for g in d["groups"].values()}))
        res = audit_variant(df, d, e)
        out = Path(a.out)
        for k, val in res.items():
            if hasattr(val, "to_csv"):
                (out / "tables").mkdir(parents=True, exist_ok=True)
                val.to_csv(out / "tables" / f"audit_{v}_{k}.csv", index=False)
        plot_monthly(res["monthly"], out / "figures", v, d["time_column"])
        write_json({k: val for k, val in res.items() if not hasattr(val, "to_csv")} |
                   {"file": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path),
                    "audited_utc": utc_now(), "git": git_commit(), "environment": environment_info()},
                   out / "metrics" / f"audit_{v}.json")
        print(f"[{v}] rows={len(df)} fraud={res['overview']['fraud_prevalence']:.4%} months={res['overview']['months']}")
        print(res["split_feasibility"].to_string(index=False))


if __name__ == "__main__":
    main()
