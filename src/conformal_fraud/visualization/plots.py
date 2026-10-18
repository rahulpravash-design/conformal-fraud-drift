from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_monthly(monthly, out_dir: str | Path, variant: str, time_col: str = "month") -> list[Path]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.errorbar(monthly[time_col], monthly.prevalence * 100,
                yerr=[(monthly.prevalence - monthly.prev_lo) * 100, (monthly.prev_hi - monthly.prevalence) * 100], marker="o", capsize=3)
    ax.set_xlabel("Month"); ax.set_ylabel("Fraud prevalence (%) with 95% Wilson CI"); ax.set_title(f"{variant}: fraud prevalence by month")
    fig.tight_layout(); p = out_dir / f"audit_{variant}_prevalence.png"; fig.savefig(p, dpi=200); plt.close(fig); paths.append(p)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(monthly[time_col], monthly.n); ax.set_xlabel("Month"); ax.set_ylabel("Applications"); ax.set_title(f"{variant}: monthly volume")
    fig.tight_layout(); p = out_dir / f"audit_{variant}_volume.png"; fig.savefig(p, dpi=200); plt.close(fig); paths.append(p)
    return paths
