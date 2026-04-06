
import pandas as pd
import numpy as np
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import gpxpy

def gpx_to_dataframe(gpx_path: Path) -> pd.DataFrame:
    """Convert every track point into a row with lat, lon, elevation, distance."""
    with open(gpx_path) as f:
        gpx = gpxpy.parse(f)

    rows = []
    prev_pt = None
    cumulative_m = 0.0

    for track in gpx.tracks:
        for seg in track.segments:
            for pt in seg.points:
                if prev_pt is not None:
                    dist = pt.distance_3d(prev_pt) or pt.distance_2d(prev_pt) or 0.0
                    cumulative_m += dist
                rows.append({
                    "lat":         pt.latitude,
                    "lon":         pt.longitude,
                    "elevation_m": pt.elevation or 0.0,
                    "distance_m":  cumulative_m,
                })
                prev_pt = pt

    return pd.DataFrame(rows)



def parse_gpx(filepath: Path) -> pd.DataFrame:
    """
    Parse a GPX 1.0 / 1.1 file into a DataFrame.

    Columns: lat (°), lon (°), ele (m), time (datetime64[UTC] or NaT)

    Handles:
    - GPX 1.0 and 1.1 via automatic namespace detection
    - Missing elevation (set to NaN)
    - Missing or malformed timestamps (set to NaT)
    """
    tree = ET.parse(filepath)
    root = tree.getroot()

    tag = root.tag
    ns = tag[: tag.index("}") + 1] if "{" in tag else ""

    records = []
    for pt in root.iter(f"{ns}trkpt"):
        lat = float(pt.attrib["lat"])
        lon = float(pt.attrib["lon"])

        ele_el = pt.find(f"{ns}ele")
        ele = float(ele_el.text) if ele_el is not None else np.nan

        time_el = pt.find(f"{ns}time")
        t = pd.NaT
        if time_el is not None:
            try:
                t = pd.to_datetime(time_el.text.replace("Z", "+00:00"), utc=True)
            except Exception:
                pass

        records.append({"lat": lat, "lon": lon, "ele": ele, "time": t})

    df = pd.DataFrame(records)
    has_ts = df["time"].notna().mean() > 0.9
    print(
        f"   Parsed {len(df):,} trackpoints  |  "
        f"timestamps: {'✅' if has_ts else '❌  (geometry only — prediction still works)'}"
    )
    return df


def haversine_km(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Vectorized great-circle distances between consecutive (lat, lon) pairs.
    Returns an array of length n; index 0 is always 0.0.
    """
    R = 6371.0
    la1, la2 = np.radians(lats[:-1]), np.radians(lats[1:])
    lo1, lo2 = np.radians(lons[:-1]), np.radians(lons[1:])
    a = (
        np.sin((la2 - la1) / 2) ** 2
        + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    )
    return np.concatenate([[0.0], 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))])

def fmt_time(total_min: float) -> str:
    """245.5  →  '4h 05min'"""
    h, m = int(total_min // 60), int(total_min % 60)
    return f"{h}h {m:02d}min"

def fmt_pace(dec_min: float) -> str:
    """5.75  →  '5:45'"""
    m = int(dec_min)
    s = int(round((dec_min - m) * 60))
    if s == 60:
        m += 1
        s = 0
    return f"{m}:{s:02d}"

from scipy.signal import savgol_filter

def smooth_elevation(ele: np.ndarray, window: int, poly: int) -> np.ndarray:
    """
    Savitzky-Golay filter on elevation array.
    Auto-adjusts window if the array is shorter than the requested window.
    """
    n = len(ele)
    if n < poly + 2:
        return np.nan_to_num(ele, nan=float(np.nanmean(ele)))
    w = min(window, n)
    w = w if w % 2 == 1 else w - 1   # must be odd
    w = max(w, poly + 2)
    return savgol_filter(np.nan_to_num(ele, nan=float(np.nanmean(ele))), w, poly)

def compute_segments(
    df: pd.DataFrame,
    smooth_window: int = 21,
    smooth_poly:   int = 3,
) -> pd.DataFrame:
    """
    Compute per-trackpoint geometry and (when timestamps are present) pace.

    Added columns
    -------------
    dist_km, cum_dist_km   horizontal distance
    ele_smooth             Savitzky-Golay smoothed elevation (m)
    ele_diff_m             elevation delta to next point (m)
    grade_pct              slope in %  (clipped to ±60)
    dt_seconds             time elapsed since previous point
    pace_min_km            pace in decimal min/km  (NaN if no timestamps)
    """
    df = df.copy().reset_index(drop=True)

    # Smooth elevation first — this is what grade and gain are computed from
    df["ele_smooth"] = smooth_elevation(df["ele"].values, smooth_window, smooth_poly)

    # Horizontal distances (fully vectorized)
    dists = haversine_km(df["lat"].values, df["lon"].values)
    df["dist_km"]     = dists
    df["cum_dist_km"] = np.cumsum(dists)

    # Elevation delta
    eles = df["ele_smooth"].values
    df["ele_diff_m"] = np.diff(eles, prepend=eles[0])

    # Grade (%) — clip extreme values caused by GPS noise
    with np.errstate(divide="ignore", invalid="ignore"):
        grade = np.where(
            dists > 1e-6,
            (df["ele_diff_m"].values / (dists * 1000)) * 100,
            0.0,
        )
    df["grade_pct"] = np.clip(grade, -60.0, 60.0)

    # Pace — only when ≥90 % of points carry timestamps
    if df["time"].notna().mean() >= 0.9:
        dt = df["time"].diff().dt.total_seconds().fillna(0).values
        df["dt_seconds"] = dt
        with np.errstate(divide="ignore", invalid="ignore"):
            pace = np.where(
                (dists > 1e-6) & (dt > 0),
                (dt / 60.0) / dists,
                np.nan,
            )
        df["pace_min_km"] = pace
    else:
        df["dt_seconds"]  = np.nan
        df["pace_min_km"] = np.nan

    return df



def filter_stops(
    df: pd.DataFrame,
    max_pace: float = 30.0,
    min_pace: float = 2.0,
) -> pd.DataFrame:
    """
    Remove stopped segments (aid stations, photos …) and GPS glitches.
    No-ops on geometry-only files that have no pace column.
    """
    if df["pace_min_km"].isna().all():
        return df

    before = len(df)
    df = df[
        df["pace_min_km"].notna()
        & (df["pace_min_km"] >= min_pace)
        & (df["pace_min_km"] <= max_pace)
        & (df["dist_km"]     >  1e-6)
    ].reset_index(drop=True)

    dropped = before - len(df)
    if dropped:
        print(f"   ⚠️  Removed {dropped} anomalous segments (stops / glitches)")
    return df


def resample_fixed_interval(
    df: pd.DataFrame,
    interval_km: float = 0.2,
) -> pd.DataFrame:
    """
    Interpolate all numeric signal columns onto a regular distance grid.

    Works for both reference runs (with pace) and target GPX (geometry only).
    Recomputes ele_diff_m from the resampled ele_smooth for accurate gain/loss.
    """

    # Add this as a default before the conditional that fills it
    df["pace_min_km"] = np.nan
    
    src  = df["cum_dist_km"].values
    end  = src[-1]
    grid = np.arange(interval_km, end, interval_km)

    if len(grid) < 2:
        raise ValueError(f"Route too short ({end:.2f} km) for interval {interval_km} km")

    out = {"cum_dist_km": grid, "dist_km": float(interval_km)}
    for col in ("ele_smooth", "grade_pct", "pace_min_km"):
        if col in df.columns and df[col].notna().any():
            out[col] = np.interp(grid, src, df[col].values)

    resampled = pd.DataFrame(out)

    # Recompute elevation delta from resampled elevation
    if "ele_smooth" in resampled.columns:
        resampled["ele_diff_m"] = resampled["ele_smooth"].diff().fillna(0.0)

    return resampled


def preprocess_gpx(
    gpx_path:     Path,
    interval_km:  float = 0.2,
    smooth_window:int   = 21,
    smooth_poly:  int   = 3,
    max_pace:     float = 30.0,
    min_pace:     float = 2.0,
) -> pd.DataFrame:
    """Full preprocessing pipeline for one GPX file."""
    raw = parse_gpx(gpx_path)
    seg = compute_segments(raw, smooth_window, smooth_poly)
    seg = filter_stops(seg, max_pace, min_pace)
    seg = resample_fixed_interval(seg, interval_km)
    return seg
