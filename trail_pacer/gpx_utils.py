from .utils import haversine_km
import gpxpy
import pandas as pd
from pathlib import Path
import numpy as np

def gpx_to_dataframe(gpx_path: Path) -> pd.DataFrame:
    """Convert every track point into a row; distances are pure horizontal (haversine)."""
    with open(gpx_path) as f:
        gpx = gpxpy.parse(f)

    rows = []
    for track in gpx.tracks:
        for seg in track.segments:
            for pt in seg.points:
                rows.append({
                    "lat":         pt.latitude,
                    "lon":         pt.longitude,
                    "elevation_m": pt.elevation or 0.0,
                })

    df = pd.DataFrame(rows)

    step_m          = haversine_km(df["lat"].values, df["lon"].values)*1000
    # df["distance_m"]    = step_km*1000
    df["distance_m"] = step_m.cumsum()

    return df

def compute_split_stats(splits: pd.DataFrame) -> pd.DataFrame:
    """
    From the interpolated splits, compute per-split stats
    (gain, loss, grade) and keep cumulative columns.
    """
    s = splits.copy()

    s["elev_diff_m"]  = s["elevation_m"].diff().fillna(0)
    s["split_gain_m"] = s["elev_diff_m"].clip(lower=0)
    s["split_loss_m"] = s["elev_diff_m"].clip(upper=0).abs()
    s["cum_gain_m"]   = s["split_gain_m"].cumsum()
    s["cum_loss_m"]   = s["split_loss_m"].cumsum()
    s["grade_pct"]    = (s["elev_diff_m"] / (s["distance_km"].diff().fillna(1) * 1000) * 100)#.clip(-50, 50)

    return s[[
        "distance_km",
        "elevation_m",
        "elev_diff_m",
        "split_gain_m",
        "split_loss_m",
        "cum_gain_m",
        "cum_loss_m",
        "grade_pct",
    ]]

def interpolate_splits(df: pd.DataFrame, step_km: float = 1.0) -> pd.DataFrame:
    """
    Interpolate df at regular distance splits.
    Returns a new DataFrame with one row per split point.
    """
    dist_km = df["distance_m"].values / 1000

    split_distances = np.arange(0, dist_km.max(), step_km)
    split_distances = np.append(split_distances, dist_km.max())  # always include the finish

    splits = pd.DataFrame({"distance_km": split_distances})

    for col in ["elevation_m", "grade_pct", "gain_m", "loss_m"]:
        if col in df.columns:
            splits[col] = np.interp(split_distances, dist_km, df[col].values)

    return splits


class GPXLoader:
    """Utility to load GPX files and convert them to DataFrames."""
    def __init__(self, gpx_path: Path, interval_km: float = 0.2):
        self.gpx_path = gpx_path
        self.interval_km = interval_km

    @staticmethod
    def load_gpx(gpx_path: Path) -> pd.DataFrame:
        return gpx_to_dataframe(gpx_path)

    def process_gpx(self) -> pd.DataFrame:
        """Load a GPX file and compute distances."""
        df = self.load_gpx(self.gpx_path)
        df = interpolate_splits(df, step_km=self.interval_km)
        df = compute_split_stats(df)
        return df
