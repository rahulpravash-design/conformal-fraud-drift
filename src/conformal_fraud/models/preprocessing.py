"""Preprocessing fitted on TRAINING rows only."""
from __future__ import annotations

import numpy as np
import pandas as pd


class FraudPreprocessor:
    """Fixes the category vocabulary from training data. Categories unseen in training
    become NaN (missing) at transform time. No statistics from val/cal/test are used."""

    def __init__(self, features: list[str], categorical: list[str]):
        self.features = list(features)
        self.categorical = [c for c in categorical if c in self.features]
        self.categories_: dict[str, list] = {}
        self.n_fit_rows_: int | None = None
        self.fit_row_ids_hash_: int | None = None

    def fit(self, train: pd.DataFrame) -> "FraudPreprocessor":
        for c in self.categorical:
            self.categories_[c] = sorted(train[c].dropna().astype(str).unique().tolist())
        self.n_fit_rows_ = len(train)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.n_fit_rows_ is None:
            raise RuntimeError("fit() first")
        X = df[self.features].copy()
        for c in self.categorical:
            cats = self.categories_[c]
            v = X[c].astype("string")
            v = v.where(v.isin(cats))            # unseen / missing -> NA (never extends the vocabulary)
            X[c] = pd.Categorical(v.astype(object).where(v.notna(), np.nan), categories=cats)
        return X
