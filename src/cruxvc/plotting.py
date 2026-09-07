"""Publication-oriented plotting helpers without hidden styling."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_figure(fig, path: str | Path, *, dpi: int = 300) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_risk_coverage(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    coverage_column: str = "realized_test_coverage",
    risk_column: str = "accepted_risk",
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    for gate, group in frame.groupby("gate"):
        group = group.sort_values(coverage_column)
        ax.plot(group[coverage_column], group[risk_column], marker="o", label=str(gate))
    ax.set_xlabel("Acceptance coverage")
    ax.set_ylabel("Accepted-set risk")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    return save_figure(fig, path)


def plot_specification_effects(frame: pd.DataFrame, path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = frame["contrast"].astype(str)
    estimates = frame["estimate"].to_numpy(dtype=float)
    lower = frame["ci_lower"].to_numpy(dtype=float)
    upper = frame["ci_upper"].to_numpy(dtype=float)
    positions = np.arange(len(frame))
    ax.errorbar(estimates, positions, xerr=[estimates - lower, upper - estimates], fmt="o", capsize=4)
    ax.axvline(0, linewidth=1)
    ax.set_yticks(positions, labels)
    ax.set_xlabel("Noise-adjusted attribution distance (Δspec)")
    ax.grid(axis="x", alpha=0.25)
    return save_figure(fig, path)


def plot_portfolio_overlap(frame: pd.DataFrame, path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    for (left, right), group in frame.groupby(["left", "right"]):
        ax.plot(group["budget"], group["jaccard"], marker="o", label=f"{left} vs {right}")
    ax.set_xlabel("Review budget")
    ax.set_ylabel("Top-K Jaccard overlap")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    return save_figure(fig, path)
