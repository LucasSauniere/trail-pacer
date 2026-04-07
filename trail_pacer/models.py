import numpy as np
import pandas as pd
from datetime import datetime
from .utils import Conversions as conv

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


# class PaceModel:
#     """
#     Predicts race time and per-split paces for a trail run.

#     Parameters
#     ----------
#     pace_10k_min_per_km : float
#         Flat 10 km pace in min/km (e.g. 5.0 for 5:00/km).
#     up_sens : float
#         Minutes per km added per 1 % of uphill grade.
#     down_sens : float
#         Minutes per km removed per 1 % of downhill grade.
#     fatigue_rate : float
#         Minutes per km added per km already covered.
#     """

#     def __init__(
#         self,
#         pace_10k_min_per_km: str,
#         up_sens: float = 0.05,
#         down_sens: float = 0.02,
#         fatigue_rate: float = 0.005,
#     ):
#         self.base_pace    = conv.chrono_to_min(pace_10k_min_per_km)
#         self.up_sens      = up_sens
#         self.down_sens    = down_sens
#         self.fatigue_rate = fatigue_rate

#     # ------------------------------------------------------------------
#     # Internal helpers
#     # ------------------------------------------------------------------

#     def _adjusted_pace(
#         self,
#         grade_pct: np.ndarray,
#         cum_dist_km: np.ndarray,
#         fatigue_rate: float | None = None,
#     ) -> np.ndarray:
#         """Vectorised pace (min/km) per segment given grade and fatigue."""
#         fr = fatigue_rate if fatigue_rate is not None else self.fatigue_rate

#         grade_adj = (
#             self.up_sens   * np.maximum(grade_pct, 0)   # uphill  → slower
#           + self.down_sens * np.minimum(grade_pct, 0)   # downhill → faster (grade is negative)
#         )
#         fatigue_adj = fr * cum_dist_km

#         return np.maximum(self.base_pace + grade_adj + fatigue_adj, 0.5)  # floor at 0.5 min/km

#     @staticmethod
#     def _split_distances(split_stats: pd.DataFrame) -> np.ndarray:
#         """Width of each split segment in km."""
#         return split_stats["distance_km"].diff().fillna(0).values

#     # ------------------------------------------------------------------
#     # Prediction methods
#     # ------------------------------------------------------------------

#     def predict_flat(self, split_stats: pd.DataFrame) -> float:
#         """Total time in minutes — flat pace, no grade or fatigue adjustment."""
#         return self.base_pace * split_stats["distance_km"].max()

#     def predict_no_fatigue(self, split_stats: pd.DataFrame) -> float:
#         """Total time in minutes — grade-adjusted, fatigue ignored."""
#         pace = self._adjusted_pace(
#             split_stats["grade_pct"].values,
#             np.zeros(len(split_stats)),
#             fatigue_rate=0.0,
#         )
#         return (pace * self._split_distances(split_stats)).sum()

#     def predict_with_fatigue(self, split_stats: pd.DataFrame) -> float:
#         """Total time in minutes — grade-adjusted and fatigue-adjusted."""
#         pace = self._adjusted_pace(
#             split_stats["grade_pct"].values,
#             split_stats["distance_km"].values,
#         )
#         return (pace * self._split_distances(split_stats)).sum()

#     def predict_split_times(
#         self,
#         split_stats: pd.DataFrame,
#         start_time: datetime | None = None,
#     ) -> pd.DataFrame:
#         """
#         Return a DataFrame with predicted pace and times per split.

#         Columns added:
#           predicted_pace  : adjusted pace for that split (min/km)
#           split_time_min  : time to complete that split (min)
#           cum_time_min    : cumulative race time at the end of that split
#           cum_time_str    : cumulative time formatted as H:MM:SS
#           arrival_time    : wall-clock arrival time (only if start_time given)
#         """
#         df   = split_stats.copy()
#         dist = self._split_distances(df)

#         df["predicted_pace"] = self._adjusted_pace(
#             df["grade_pct"].values,
#             df["distance_km"].values,
#         )
#         df["split_time_min"] = df["predicted_pace"] * dist
#         df["cum_time_min"]   = df["split_time_min"].cumsum()
#         df["cum_time_str"]   = df["cum_time_min"].apply(self._fmt_time)

#         if start_time is not None:
#             df["arrival_time"] = start_time + pd.to_timedelta(df["cum_time_min"], unit="m")

#         df["cum_predicted_pace"] = df["predicted_pace"].cumsum()
#         return df
    
    

#     def fit_to_total_time(
#         self,
#         split_stats: pd.DataFrame,
#         total_time_min: float,
#         fit_up_sens: bool = True,
#         fit_down_sens: bool = True,
#         fit_fatigue: bool = True,
#     ) -> dict:
#         """
#         Adjust up_sens, down_sens and/or fatigue_rate so that predict_with_fatigue()
#         matches total_time_min.

#         Parameters
#         ----------
#         split_stats : pd.DataFrame
#             Output of compute_split_stats().
#         total_time_min : float
#             The actual (or target) total race time in minutes.
#         fit_up_sens / fit_down_sens / fit_fatigue : bool
#             Which parameters to optimise; fix the others at their current values.

#         Returns
#         -------
#         dict with the fitted parameter values and the residual error.
#         """
#         from scipy.optimize import minimize
#         # Build the list of (initial_value, lower_bound, upper_bound) for active params
#         param_config = []
#         if fit_up_sens:
#             param_config.append(("up_sens",      self.up_sens,      0.0, 1.0))
#         if fit_down_sens:
#             param_config.append(("down_sens",    self.down_sens,    -1.0, 0.0))
#         if fit_fatigue:
#             param_config.append(("fatigue_rate", self.fatigue_rate, 0.0, 0.1))

#         if not param_config:
#             raise ValueError("At least one parameter must be set to fit.")

#         names  = [p[0] for p in param_config]
#         x0     = np.array([p[1] for p in param_config])
#         bounds = [(p[2], p[3]) for p in param_config]

#         def _total_time(x: np.ndarray) -> float:
#             params = dict(zip(names, x))
#             up_s   = params.get("up_sens",      self.up_sens)
#             down_s = params.get("down_sens",    self.down_sens)
#             fr     = params.get("fatigue_rate", self.fatigue_rate)

#             grade_adj = (
#                 up_s   * np.maximum(split_stats["grade_pct"].values, 0)
#             + down_s * np.minimum(split_stats["grade_pct"].values, 0)
#             )
#             fatigue_adj = fr * split_stats["distance_km"].values
#             pace = np.maximum(self.base_pace + grade_adj + fatigue_adj, 0.5)
#             dist = split_stats["distance_km"].diff().fillna(0).values
#             return (pace * dist).sum()

#         def _loss(x: np.ndarray) -> float:
#             return (_total_time(x) - total_time_min) ** 2

#         result = minimize(_loss, x0, bounds=bounds, method="L-BFGS-B")

#         # Write fitted values back onto the instance
#         fitted = dict(zip(names, result.x))
#         for attr, val in fitted.items():
#             setattr(self, attr, float(val))

#         residual = abs(_total_time(result.x) - total_time_min)
#         print(f"Fitted parameters  : {fitted}")
#         print(f"Predicted time     : {self._fmt_time(_total_time(result.x))}")
#         print(f"Target time        : {self._fmt_time(total_time_min)}")
#         print(f"Residual           : {residual:.2f} min")

#         return {**fitted, "residual_min": residual, "success": result.success}
    
#     def fit_to_split_paces(
#         self,
#         split_stats: pd.DataFrame,
#         actual_pace_min_per_km: np.ndarray,
#         fit_up_sens: bool = True,
#         fit_down_sens: bool = True,
#         fit_fatigue: bool = True,
#     ) -> dict:
#         """
#         Fit model parameters by minimising MSE between predicted
#         and actual pace across all splits.

#         Parameters
#         ----------
#         actual_pace_min_per_km : np.ndarray
#             Observed pace per split in min/km, same length as split_stats.
#         """
#         from scipy.optimize import minimize
#         param_config = []
#         if fit_up_sens:
#             param_config.append(("up_sens",      self.up_sens,      0.0,  1.0))
#         if fit_down_sens:
#             param_config.append(("down_sens",    self.down_sens,   -1.0,  0.0))
#         if fit_fatigue:
#             param_config.append(("fatigue_rate", self.fatigue_rate, 0.0,  0.1))

#         if not param_config:
#             raise ValueError("At least one parameter must be set to fit.")

#         names  = [p[0] for p in param_config]
#         x0     = np.array([p[1] for p in param_config])
#         bounds = [(p[2], p[3]) for p in param_config]

#         grades   = split_stats["grade_pct"].values
#         cum_dist = split_stats["distance_km"].values
#         target   = np.asarray(actual_pace_min_per_km)

#         def _predict_pace(x: np.ndarray) -> np.ndarray:
#             params = dict(zip(names, x))
#             up_s   = params.get("up_sens",      self.up_sens)
#             down_s = params.get("down_sens",    self.down_sens)
#             fr     = params.get("fatigue_rate", self.fatigue_rate)

#             grade_adj   = up_s * np.maximum(grades, 0) + down_s * np.minimum(grades, 0)
#             fatigue_adj = fr * cum_dist
#             return np.maximum(self.base_pace + grade_adj + fatigue_adj, 0.5)

#         def _loss(x: np.ndarray) -> float:
#             residuals = _predict_pace(x) - target
#             return np.mean(residuals ** 2)  # MSE across all splits

#         result = minimize(_loss, x0, bounds=bounds, method="L-BFGS-B")

#         fitted = dict(zip(names, result.x))
#         for attr, val in fitted.items():
#             setattr(self, attr, float(val))

#         predicted = _predict_pace(result.x)
#         mae  = np.mean(np.abs(predicted - target))
#         rmse = np.sqrt(np.mean((predicted - target) ** 2))

#         print(f"Fitted parameters : {fitted}")
#         print(f"MAE               : {mae:.3f} min/km")
#         print(f"RMSE              : {rmse:.3f} min/km")

#         return {**fitted, "mae": mae, "rmse": rmse, "success": result.success}

#     # ------------------------------------------------------------------
#     # Utilities
#     # ------------------------------------------------------------------

#     @staticmethod
#     def _fmt_time(minutes: float) -> str:
#         total_s = int(round(minutes * 60))
#         h, rem  = divmod(total_s, 3600)
#         m, s    = divmod(rem, 60)
#         return f"{h}:{m:02d}:{s:02d}"

#     def summary(self, split_stats: pd.DataFrame) -> None:
#         """Print a quick comparison of the three prediction modes."""
#         t_flat    = self.predict_flat(split_stats)
#         t_grade   = self.predict_no_fatigue(split_stats)
#         t_fatigue = self.predict_with_fatigue(split_stats)

#         print(f"Base pace       : {self.base_pace:.2f} min/km")
#         print(f"Distance        : {split_stats['distance_km'].max():.1f} km")
#         print(f"Flat estimate   : {self._fmt_time(t_flat)}")
#         print(f"Grade-adjusted  : {self._fmt_time(t_grade)}")
#         print(f"Grade + fatigue : {self._fmt_time(t_fatigue)}")

#     def __repr__(self) -> str:
#         return (
#             f"PaceModel(base_pace={self.base_pace:.2f}, "
#             f"up_sens={self.up_sens}, down_sens={self.down_sens}, "
#             f"fatigue_rate={self.fatigue_rate})"
#         )
    
class PaceModel:
    """
    Predicts race time and per-split paces for a trail run.

    Parameters
    ----------
    pace_10k_min_per_km : str
        Flat 10 km pace in min/km formatted as "M:SS" (e.g. "5:00").
    up_sens : float
        Minutes per km added per 1 % of uphill grade.
    down_sens : float
        Minutes per km reduced per 1 % of downhill grade (positive → faster downhill).
    fatigue_rate : float
        Minutes per km added per km already covered.
    """

    def __init__(
        self,
        pace_10k_min_per_km: str,
        up_sens: float = 0.05,
        down_sens: float = 0.02,
        fatigue_rate: float = 0.005,
    ):
        self.base_pace    = conv.chrono_to_min(pace_10k_min_per_km)
        self.up_sens      = up_sens
        self.down_sens    = down_sens
        self.fatigue_rate = fatigue_rate

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _adjusted_pace(
        self,
        grade_pct: np.ndarray,
        cum_dist_km: np.ndarray,
        fatigue_rate: float | None = None,
    ) -> np.ndarray:
        """Vectorised pace (min/km) per segment given grade and fatigue."""
        fr = fatigue_rate if fatigue_rate is not None else self.fatigue_rate

        grade_adj = (
            self.up_sens   * np.maximum(grade_pct, 0)
          + self.down_sens * np.minimum(grade_pct, 0)
        )
        fatigue_adj = fr * cum_dist_km

        return np.maximum(self.base_pace + grade_adj + fatigue_adj, 0.5)

    @staticmethod
    def _split_distances(split_stats: pd.DataFrame) -> np.ndarray:
        """Width of each split segment in km."""
        return split_stats["distance_km"].diff().fillna(0).values

    # ------------------------------------------------------------------
    # Prediction methods
    # ------------------------------------------------------------------

    def predict_flat(self, split_stats: pd.DataFrame) -> float:
        """Total time in minutes — flat pace, no grade or fatigue adjustment."""
        return self.base_pace * split_stats["distance_km"].max()

    def predict_no_fatigue(self, split_stats: pd.DataFrame) -> float:
        """Total time in minutes — grade-adjusted, fatigue ignored."""
        pace = self._adjusted_pace(
            split_stats["grade_pct"].values,
            np.zeros(len(split_stats)),
            fatigue_rate=0.0,
        )
        return (pace * self._split_distances(split_stats)).sum()

    def predict_with_fatigue(self, split_stats: pd.DataFrame) -> float:
        """Total time in minutes — grade-adjusted and fatigue-adjusted."""
        pace = self._adjusted_pace(
            split_stats["grade_pct"].values,
            split_stats["distance_km"].values,
        )
        return (pace * self._split_distances(split_stats)).sum()

    def predict_split_times(
        self,
        split_stats: pd.DataFrame,
        start_time: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Return a DataFrame with predicted pace and times per split.

        Columns added:
          predicted_pace  : adjusted pace for that split (min/km)
          split_time_min  : time to complete that split (min)
          cum_time_min    : cumulative race time at the end of that split
          cum_time_str    : cumulative time formatted as H:MM:SS
          arrival_time    : wall-clock arrival time (only if start_time given)
        """
        df   = split_stats.copy()
        dist = self._split_distances(df)

        df["predicted_pace"] = self._adjusted_pace(
            df["grade_pct"].values,
            df["distance_km"].values,
        )
        df["split_time_min"] = df["predicted_pace"] * dist
        df["cum_time_min"]   = df["split_time_min"].cumsum()
        df["cum_time_str"]   = df["cum_time_min"].apply(self._fmt_time)

        if start_time is not None:
            df["arrival_time"] = start_time + pd.to_timedelta(df["cum_time_min"], unit="m")

        df["cum_predicted_pace"] = df["predicted_pace"].cumsum()
        return df

    # ------------------------------------------------------------------
    # Fit methods
    # ------------------------------------------------------------------

    def fit_to_total_time(
        self,
        split_stats: pd.DataFrame,
        total_time_min: float,
        fit_base_pace: bool = False,
        fit_up_sens: bool = True,
        fit_down_sens: bool = True,
        fit_fatigue: bool = True,
    ) -> dict:
        """
        Adjust model parameters so that predict_with_fatigue() matches total_time_min.

        Parameters
        ----------
        split_stats : pd.DataFrame
            Output of process_gpx() / compute_split_stats().
        total_time_min : float
            The actual (or target) total race time in minutes.
        fit_base_pace : bool
            Whether to also optimise the base pace (default False).
            Best used when fitting to split paces; fitting a single scalar
            (total time) with base_pace free is under-determined.
        fit_up_sens / fit_down_sens / fit_fatigue : bool
            Which parameters to optimise; fix the others at their current values.

        Returns
        -------
        dict with the fitted parameter values and the residual error.
        """
        from scipy.optimize import minimize

        param_config = []
        if fit_base_pace:
            param_config.append(("base_pace",    self.base_pace,    2.0, 15.0))
        if fit_up_sens:
            param_config.append(("up_sens",      self.up_sens,      0.0,  1.0))
        if fit_down_sens:
            param_config.append(("down_sens",    self.down_sens,    0.0,  1.0))
        if fit_fatigue:
            param_config.append(("fatigue_rate", self.fatigue_rate, 0.0,  0.1))

        if not param_config:
            raise ValueError("At least one parameter must be set to fit.")

        names  = [p[0] for p in param_config]
        x0     = np.array([p[1] for p in param_config])
        bounds = [(p[2], p[3]) for p in param_config]

        def _total_time(x: np.ndarray) -> float:
            params = dict(zip(names, x))
            bp     = params.get("base_pace",    self.base_pace)
            up_s   = params.get("up_sens",      self.up_sens)
            down_s = params.get("down_sens",    self.down_sens)
            fr     = params.get("fatigue_rate", self.fatigue_rate)

            grade_adj   = (up_s   * np.maximum(split_stats["grade_pct"].values, 0)
                         + down_s * np.minimum(split_stats["grade_pct"].values, 0))
            fatigue_adj = fr * split_stats["distance_km"].values
            pace        = np.maximum(bp + grade_adj + fatigue_adj, 0.5)
            dist        = split_stats["distance_km"].diff().fillna(0).values
            return (pace * dist).sum()

        def _loss(x: np.ndarray) -> float:
            return (_total_time(x) - total_time_min) ** 2

        result = minimize(_loss, x0, bounds=bounds, method="L-BFGS-B")

        fitted = dict(zip(names, result.x))
        for attr, val in fitted.items():
            setattr(self, attr, float(val))

        residual = abs(_total_time(result.x) - total_time_min)
        print(f"Fitted parameters  : {fitted}")
        print(f"Predicted time     : {self._fmt_time(_total_time(result.x))}")
        print(f"Target time        : {self._fmt_time(total_time_min)}")
        print(f"Residual           : {residual:.2f} min")

        return {**fitted, "residual_min": residual, "success": result.success}

    def fit_to_split_paces(
        self,
        split_stats: pd.DataFrame,
        actual_pace_min_per_km: np.ndarray,
        fit_base_pace: bool = False,
        fit_up_sens: bool = True,
        fit_down_sens: bool = True,
        fit_fatigue: bool = True,
    ) -> dict:
        """
        Fit model parameters by minimising MSE between predicted
        and actual pace across all splits.

        Parameters
        ----------
        actual_pace_min_per_km : np.ndarray
            Observed pace per split in min/km, same length as split_stats.
        fit_base_pace : bool
            Whether to also optimise the base pace (default False).
            Recommended when fitting to split paces, as the data is rich
            enough to constrain the additional degree of freedom.
        """
        from scipy.optimize import minimize

        param_config = []
        if fit_base_pace:
            param_config.append(("base_pace",    self.base_pace,    2.0, 15.0))
        if fit_up_sens:
            param_config.append(("up_sens",      self.up_sens,      0.0,  1.0))
        if fit_down_sens:
            param_config.append(("down_sens",    self.down_sens,    0.0,  1.0))
        if fit_fatigue:
            param_config.append(("fatigue_rate", self.fatigue_rate, 0.0,  0.1))

        if not param_config:
            raise ValueError("At least one parameter must be set to fit.")

        names  = [p[0] for p in param_config]
        x0     = np.array([p[1] for p in param_config])
        bounds = [(p[2], p[3]) for p in param_config]

        grades   = split_stats["grade_pct"].values
        cum_dist = split_stats["distance_km"].values
        target   = np.asarray(actual_pace_min_per_km)

        def _predict_pace(x: np.ndarray) -> np.ndarray:
            params = dict(zip(names, x))
            bp     = params.get("base_pace",    self.base_pace)
            up_s   = params.get("up_sens",      self.up_sens)
            down_s = params.get("down_sens",    self.down_sens)
            fr     = params.get("fatigue_rate", self.fatigue_rate)

            grade_adj   = up_s * np.maximum(grades, 0) + down_s * np.minimum(grades, 0)
            fatigue_adj = fr * cum_dist
            return np.maximum(bp + grade_adj + fatigue_adj, 0.5)

        def _loss(x: np.ndarray) -> float:
            return np.mean((_predict_pace(x) - target) ** 2)

        result = minimize(_loss, x0, bounds=bounds, method="L-BFGS-B")

        fitted = dict(zip(names, result.x))
        for attr, val in fitted.items():
            setattr(self, attr, float(val))

        predicted = _predict_pace(result.x)
        mae  = np.mean(np.abs(predicted - target))
        rmse = np.sqrt(np.mean((predicted - target) ** 2))

        print(f"Fitted parameters : {fitted}")
        print(f"MAE               : {mae:.3f} min/km")
        print(f"RMSE              : {rmse:.3f} min/km")

        return {**fitted, "mae": mae, "rmse": rmse, "success": result.success}

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt_time(minutes: float) -> str:
        total_s = int(round(minutes * 60))
        h, rem  = divmod(total_s, 3600)
        m, s    = divmod(rem, 60)
        return f"{h}:{m:02d}:{s:02d}"

    def summary(self, split_stats: pd.DataFrame) -> None:
        """Print a quick comparison of the three prediction modes."""
        t_flat    = self.predict_flat(split_stats)
        t_grade   = self.predict_no_fatigue(split_stats)
        t_fatigue = self.predict_with_fatigue(split_stats)

        print(f"Base pace       : {self.base_pace:.2f} min/km  ({conv.min_to_chrono(self.base_pace)})")
        print(f"up_sens         : {self.up_sens}")
        print(f"down_sens       : {self.down_sens}")
        print(f"fatigue_rate    : {self.fatigue_rate}")
        print(f"Distance        : {split_stats['distance_km'].max():.1f} km")
        print(f"Flat estimate   : {self._fmt_time(t_flat)}")
        print(f"Grade-adjusted  : {self._fmt_time(t_grade)}")
        print(f"Grade + fatigue : {self._fmt_time(t_fatigue)}")

    def __repr__(self) -> str:
        return (
            f"PaceModel(base_pace={self.base_pace:.2f} ({conv.min_to_chrono(self.base_pace)}), "
            f"up_sens={self.up_sens}, down_sens={self.down_sens}, "
            f"fatigue_rate={self.fatigue_rate})"
        )