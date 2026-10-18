"""SYNTHETIC TEST FIXTURE ONLY. Mimics the BAF schema (column names, month 0-7, fraud_bool) so the
pipeline mechanics can be tested without the real data. Nothing computed from it is a research result."""
import numpy as np
import pandas as pd


def make_synthetic(n_per_month: int = 6000, seed: int = 0, drift: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for m in range(8):
        n = n_per_month
        shift = 0.15 * m if drift else 0.0
        risk = rng.normal(100 + 10 * shift, 40, n)
        sim = rng.beta(2, 5, n)
        v6 = rng.normal(5000 + 400 * m * (1 if drift else 0), 800, n)
        age = rng.choice([10, 20, 30, 40, 50, 60, 70, 80, 90], size=n, p=[.02, .2, .25, .2, .13, .1, .06, .03, .01])
        ptype = rng.choice(["AA", "AB", "AC", "AD", "AE"], size=n)
        emp = rng.choice(["CA", "CB", "CC", "CD", "CE", "CF", "CG"], size=n)
        logit = -4.2 + 0.015 * (risk - 100) - 2.0 * sim + 0.0004 * (v6 - 5000) + 0.01 * (age - 40) + (ptype == "AC") * 0.8 + 0.1 * m * (1 if drift else 0)
        y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
        frames.append(pd.DataFrame({
            "fraud_bool": y, "income": rng.choice(np.arange(0.1, 1.0, 0.1), n), "name_email_similarity": sim,
            "customer_age": age, "credit_risk_score": risk, "velocity_6h": v6,
            "current_address_months_count": np.where(rng.random(n) < .01, -1, rng.integers(0, 300, n)),
            "payment_type": ptype, "employment_status": emp,
            "housing_status": rng.choice(["BA", "BB", "BC"], size=n), "source": rng.choice(["INTERNET", "TELEAPP"], size=n, p=[.98, .02]),
            "device_os": rng.choice(["windows", "linux", "macintosh", "other"], size=n), "month": m,
        }))
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "row_id", range(len(df)))
    return df
