"""Development-only RQ1 operating-characteristic simulation."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def simulate_rq1_power(
    *,
    n_cases: int,
    cross_sd: float,
    within_sd: float,
    effect_grid: Sequence[float],
    refit_grid: Sequence[int],
    repetitions: int,
    meaningful_effect: float,
    equivalence_half_width: float,
    confidence: float = 0.95,
    seed: int = 260826,
) -> pd.DataFrame:
    """Approximate paired hierarchical design using case and refit noise.

    The simulation is deliberately parameterized from Notebook 09 pilot estimates.
    It does not claim power until the actual paired-design inputs have been supplied.
    """
    rng = np.random.default_rng(seed)
    z = 1.959963984540054 if abs(confidence - 0.95) < 1e-9 else 1.959963984540054
    rows = []
    for effect in effect_grid:
        for refits in refit_grid:
            material = equivalence = inconclusive = 0
            widths = []
            estimates = []
            # Refit averaging shrinks refit noise but not case heterogeneity.
            effective_sd = np.sqrt(cross_sd**2 / max(n_cases, 1) + within_sd**2 / max(n_cases * refits, 1))
            for _ in range(repetitions):
                estimate = float(rng.normal(effect, effective_sd))
                half_width = z * effective_sd
                lower, upper = estimate - half_width, estimate + half_width
                estimates.append(estimate)
                widths.append(2 * half_width)
                if lower > meaningful_effect:
                    material += 1
                elif lower > -equivalence_half_width and upper < equivalence_half_width:
                    equivalence += 1
                else:
                    inconclusive += 1
            for decision, count in [
                ("material_sensitivity", material),
                ("practical_equivalence", equivalence),
                ("inconclusive", inconclusive),
            ]:
                probability = count / repetitions
                mcse = np.sqrt(probability * (1 - probability) / repetitions)
                rows.append(
                    {
                        "true_effect": float(effect),
                        "refits": int(refits),
                        "decision": decision,
                        "probability": float(probability),
                        "monte_carlo_se": float(mcse),
                        "mean_ci_width": float(np.mean(widths)),
                        "mean_estimate": float(np.mean(estimates)),
                        "n_cases": int(n_cases),
                        "simulation_repetitions": int(repetitions),
                    }
                )
    return pd.DataFrame(rows)
