"""Chronological split with explicit validation and leakage checks."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .load_data import ROW_ID


@dataclass(frozen=True)
class SplitSpec:
    train: tuple[int, ...]
    val: tuple[int, ...]
    cal: tuple[int, ...]
    test: tuple[int, ...]

    @classmethod
    def from_config(cls, cfg: dict) -> "SplitSpec":
        s = cfg["split"]
        return cls(tuple(s["train_months"]), tuple(s["val_months"]), tuple(s["cal_months"]), tuple(s["test_months"]))

    def roles(self) -> dict[str, tuple[int, ...]]:
        return {"train": self.train, "val": self.val, "cal": self.cal, "test": self.test}

    def as_dict(self) -> dict:
        return {k: list(v) for k, v in asdict(self).items()}


def validate_spec(spec: SplitSpec, available_months: set[int]) -> None:
    """Spec-level checks: non-empty, disjoint, strictly chronological, months exist."""
    roles = spec.roles()
    for name, months in roles.items():
        if len(months) == 0:
            raise ValueError(f"Partition {name!r} has no months")
    all_months = [m for ms in roles.values() for m in ms]
    if len(all_months) != len(set(all_months)):
        raise ValueError("Months are assigned to more than one partition")
    order = ["train", "val", "cal", "test"]
    for a, b in zip(order, order[1:]):
        if max(roles[a]) >= min(roles[b]):
            raise ValueError(f"Chronology violated: max({a})={max(roles[a])} >= min({b})={min(roles[b])}")
    absent = sorted(set(all_months) - available_months)
    if absent:
        raise ValueError(f"Spec months {absent} not present in data (available: {sorted(available_months)})")


def make_split(df: pd.DataFrame, spec: SplitSpec, time_column: str) -> dict[str, pd.DataFrame]:
    months = set(int(m) for m in pd.unique(df[time_column]))
    validate_spec(spec, months)
    t = df[time_column].astype(int)
    parts = {name: df[t.isin(ms)].copy() for name, ms in spec.roles().items()}
    for name, p in parts.items():
        if p.empty:
            raise ValueError(f"Partition {name!r} is empty")
    return parts


def leakage_checks(parts: dict[str, pd.DataFrame], spec: SplitSpec, features: list[str], label: str, time_column: str) -> list[dict]:
    """Return a list of {name, passed, detail}. Does not raise; caller decides (see assert_no_leakage)."""
    res = []

    def add(name, passed, detail=""):
        res.append({"name": name, "passed": bool(passed), "detail": str(detail)})

    ids = {k: set(v[ROW_ID]) for k, v in parts.items()}
    names = list(parts)
    overlaps = {(a, b): len(ids[a] & ids[b]) for i, a in enumerate(names) for b in names[i + 1:]}
    add("row_ids_disjoint_across_partitions", all(v == 0 for v in overlaps.values()), overlaps)

    for name, months in spec.roles().items():
        got = set(int(m) for m in parts[name][time_column].unique())
        add(f"{name}_months_match_spec", got == set(months), f"got={sorted(got)} spec={sorted(months)}")

    add("train_strictly_before_val_cal_test",
        parts["train"][time_column].max() < min(parts[k][time_column].min() for k in ("val", "cal", "test")))
    add("val_before_cal_before_test",
        parts["val"][time_column].max() < parts["cal"][time_column].min() and
        parts["cal"][time_column].max() < parts["test"][time_column].min())
    add("label_not_in_features", label not in features)
    add("time_column_not_in_features", time_column not in features)
    add("row_id_not_in_features", ROW_ID not in features)
    # An exact-duplicate application appearing in two partitions would let the
    # same record be seen in training and evaluation. Time column and row id are ignored.
    cols = [c for c in features if c in parts["train"].columns]
    hashes = {k: set(pd.util.hash_pandas_object(v[cols], index=False).tolist()) for k, v in parts.items()}
    dups = {(a, b): len(hashes[a] & hashes[b]) for i, a in enumerate(names) for b in names[i + 1:]}
    add("no_exact_duplicate_feature_vectors_across_partitions", all(v == 0 for v in dups.values()), dups)
    return res


def assert_no_leakage(checks: list[dict]) -> None:
    failed = [c for c in checks if not c["passed"]]
    if failed:
        raise AssertionError("Leakage/protocol checks failed: " + "; ".join(f"{c['name']} ({c['detail']})" for c in failed))
