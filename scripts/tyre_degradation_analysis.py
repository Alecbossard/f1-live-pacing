"""
Tyre degradation analysis on top of DuckDB-stored F1 race data.

Goals:
- read laps data for one race from DuckDB
- build stints per driver using the 'Stint' column
- fit a simple linear model of lap_time_s vs. lap_index_in_stint
  -> slope = estimated tyre degradation [seconds per lap]
- aggregate results per driver and print a small ranking

This is not perfect race engineering, but it clearly shows:
- data engineering (DuckDB, SQL)
- data analysis (grouping, features)
- simple ML / modelling (linear regression with numpy.polyfit)
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def load_laps_for_race(
    season_year: int,
    grand_prix_name: str,
    db_path: str = "data/f1_races.duckdb",
) -> pd.DataFrame:
    """
    Load laps for one race from DuckDB.

    Assumptions:
    - table 'laps' exists and was filled by the ingestion script
    - table contains columns:
      season_year, grand_prix_name, Driver, Team, Stint, LapNumber, LapTime, Compound

    :param season_year: year of the season (e.g. 2024)
    :param grand_prix_name: GP name used during ingestion (e.g. "Bahrain")
    :param db_path: path to the DuckDB file
    :return: DataFrame with one row per lap
    """
    db_file = Path(db_path)
    if not db_file.exists():
        raise FileNotFoundError(f"DuckDB database not found at: {db_file.resolve()}")

    conn = duckdb.connect(str(db_file))

    query = """
        SELECT
            season_year,
            grand_prix_name,
            Driver       AS driver_code,
            Team         AS team,
            Stint        AS stint,
            LapNumber    AS lap_number,
            LapTime,
            Compound     AS tyre_compound
        FROM laps
        WHERE season_year = ? AND grand_prix_name = ?
    """

    laps_df = conn.execute(query, [season_year, grand_prix_name]).df()
    conn.close()

    if laps_df.empty:
        raise ValueError(
            f"No laps found in DuckDB for year={season_year}, gp='{grand_prix_name}'.\n"
            "Did you run the ingestion script for this race?"
        )

    # Convert LapTime to seconds
    # If it's already timedelta64, dt.total_seconds will work.
    if not pd.api.types.is_timedelta64_dtype(laps_df["LapTime"]):
        laps_df["LapTime"] = pd.to_timedelta(laps_df["LapTime"])

    laps_df["lap_time_s"] = laps_df["LapTime"].dt.total_seconds()

    # Clean up NaNs in stint
    laps_df = laps_df.dropna(subset=["stint", "lap_time_s"])
    laps_df["stint"] = laps_df["stint"].astype(int)

    # Sort for nicer grouping
    laps_df = laps_df.sort_values(["driver_code", "stint", "lap_number"]).reset_index(drop=True)

    print(
        f"[INFO] Loaded {len(laps_df)} laps for {grand_prix_name} {season_year} "
        f"from {db_file.resolve()}"
    )

    return laps_df


def compute_stint_degradation(
    laps_df: pd.DataFrame,
    min_laps_per_stint: int = 5,
) -> pd.DataFrame:
    """
    Compute tyre degradation per stint using a simple linear model.

    For each driver + stint:
    - build an x axis = [0, 1, 2, ..., n-1] (lap index inside stint)
    - y = lap_time_s
    - fit y = slope * x + intercept using numpy.polyfit
    - slope is interpreted as degradation [seconds per lap]

    :param laps_df: DataFrame with laps (one row per lap)
    :param min_laps_per_stint: minimum number of laps required to fit a model
    :return: DataFrame with one row per driver+stint+compound
    """
    results = []

    group_cols = ["driver_code", "team", "stint", "tyre_compound"]
    for (driver_code, team, stint, tyre_compound), group in laps_df.groupby(group_cols):
        group = group.sort_values("lap_number")

        if len(group) < min_laps_per_stint:
            # not enough laps to estimate a slope reliably
            continue

        # build x = index inside stint
        x = np.arange(len(group), dtype=float)
        y = group["lap_time_s"].values.astype(float)

        # simple linear regression: y = slope * x + intercept
        slope, intercept = np.polyfit(x, y, 1)

        result = {
            "driver_code": driver_code,
            "team": team,
            "stint": stint,
            "tyre_compound": tyre_compound,
            "laps_in_stint": len(group),
            "slope_s_per_lap": slope,
            "base_lap_time_s": float(y[0]),
        }
        results.append(result)

    stint_deg_df = pd.DataFrame(results)

    if stint_deg_df.empty:
        print("[WARN] No stints with enough laps to compute degradation.")
        return stint_deg_df

    # Sort by slope: lower slope = better tyre management
    stint_deg_df = stint_deg_df.sort_values("slope_s_per_lap").reset_index(drop=True)

    return stint_deg_df


def aggregate_driver_degradation(stint_deg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate stint degradation metrics per driver.

    For each driver:
    - compute mean slope across all stints
    - count number of stints used in the average

    :param stint_deg_df: DataFrame with stint-level degradation
    :return: DataFrame with one row per driver
    """
    if stint_deg_df.empty:
        return stint_deg_df

    agg_df = (
        stint_deg_df.groupby(["driver_code", "team"])
        .agg(
            avg_slope_s_per_lap=("slope_s_per_lap", "mean"),
            stints_count=("slope_s_per_lap", "count"),
        )
        .reset_index()
        .sort_values("avg_slope_s_per_lap")
        .reset_index(drop=True)
    )

    return agg_df


def print_stint_table(stint_deg_df: pd.DataFrame, season_year: int, grand_prix_name: str):
    """
    Print stint-level tyre degradation metrics.
    """
    if stint_deg_df.empty:
        print("[INFO] No stint degradation to display.")
        return

    print(f"\n========== Tyre degradation per stint - {grand_prix_name} {season_year} ==========\n")

    display_cols = [
        "driver_code",
        "team",
        "stint",
        "tyre_compound",
        "laps_in_stint",
        "slope_s_per_lap",
        "base_lap_time_s",
    ]

    print(
        stint_deg_df[display_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


def print_driver_table(driver_deg_df: pd.DataFrame, season_year: int, grand_prix_name: str):
    """
    Print driver-level aggregated tyre degradation metrics.
    """
    if driver_deg_df.empty:
        print("[INFO] No driver degradation to display.")
        return

    print(f"\n========== Average tyre degradation per driver - {grand_prix_name} {season_year} ==========\n")

    display_cols = [
        "driver_code",
        "team",
        "stints_count",
        "avg_slope_s_per_lap",
    ]

    print(
        driver_deg_df[display_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(
        "\nNote:\n"
        "- slope is in [seconds per lap]; lower = better tyre management.\n"
        "- values can be noisy for drivers with few long stints."
    )


def main():
    # Parameters of the race to analyze
    season_year = 2024
    grand_prix_name = "Bahrain"  # must match what you used in the ingestion script

    # 1) Load laps from DuckDB
    laps_df = load_laps_for_race(season_year, grand_prix_name)

    # 2) Compute stint-level degradation
    stint_deg_df = compute_stint_degradation(laps_df, min_laps_per_stint=5)

    # 3) Aggregate per driver
    driver_deg_df = aggregate_driver_degradation(stint_deg_df)

    # 4) Print results
    print_stint_table(stint_deg_df, season_year, grand_prix_name)
    print_driver_table(driver_deg_df, season_year, grand_prix_name)


if __name__ == "__main__":
    main()
