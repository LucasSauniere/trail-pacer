import numpy as np


def pace_model(
    X:            tuple,
    base_pace:    float,
    up_sens:      float,
    down_sens:    float,
    fatigue_rate: float,
) -> np.ndarray:
    """
    Parametric pace model used both for fitting and prediction.

        pace = base_pace x grade_factor x fatigue_factor

    grade_factor
    ────────────
      uphill   (grade ≥ 0):  1 + up_sens   x grade_pct       (> 1 → slower)
      downhill (grade < 0):  1 + down_sens  x grade_pct       (< 1 if down_sens > 0 → faster)
                                                               (> 1 if down_sens < 0 → slower, technical terrain)

    fatigue_factor
    ──────────────
      1 + fatigue_rate x cum_dist_km                          (always ≥ 1)

    Parameters
    ----------
    base_pace    : flat fresh pace (min/km)
    up_sens      : ≥ 0  — slowdown per 1 % of uphill grade
    down_sens    : free — speed-up (> 0) or slowdown (< 0) per 1 % of downhill grade
    fatigue_rate : ≥ 0  — progressive slowdown per km of cumulative distance
    """
    grade_pct, cum_dist = X

    grade_factor = np.where(
        grade_pct >= 0,
        1.0 + up_sens   * grade_pct,   # uphill
        1.0 + down_sens * grade_pct,   # downhill: grade_pct < 0
    )
    grade_factor   = np.clip(grade_factor,   0.3, 8.0)
    fatigue_factor = np.clip(1.0 + fatigue_rate * cum_dist, 1.0, 4.0)

    return base_pace * grade_factor * fatigue_factor
