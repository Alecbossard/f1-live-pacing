"""
Season-level tyre degradation analysis on top of DuckDB-stored F1 data.

Goals:
- load all laps for a season from DuckDB
- build stints for each driver in each race
- fit a simple linear model lap_time_s vs. lap_index_in_stint
  -> slope = estimated tyre degradation [seconds per lap]
- aggregate:
  1) per race (for each GP)
  2) for the full season (per driver)

This lets you say things like:
- "Who has the best tyre management over the 2024 season?"
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def load_laps_for_season(
    season_year: int,
    db_path: str = "data/f1_races.duckdb",
) -> pd.DataFrame:
    """
    Load all laps for one season from DuckDB.

    Assumptions:
    - table 'laps' exists and contains columns:
      season_year, grand_prix_name, round_number,
      Driver, Team, Stint, LapNumber, LapTime, Compound

    :param season_year: year of the season (e.g. 2024)
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
            round_number,
            Driver       AS driver_code,
            Team         AS team,
            Stint        AS stint,
            LapNumber    AS lap_number,
            LapTime,
            Compound     AS tyre_compound
        FROM laps
        WHERE season_year = ?
    """

    laps_df = conn.execute(query, [season_year]).df()
    conn.close()

    if laps_df.empty:
        raise ValueError(
            f"No laps found in DuckDB for season_year={season_year}.\n"
            "Did you run the season ingestion script?"
        )

    # Convert LapTime to seconds
    if not pd.api.types.is_timedelta64_dtype(laps_df["LapTime"]):
        laps_df["LapTime"] = pd.to_timedelta(laps_df["LapTime"])

    laps_df["lap_time_s"] = laps_df["LapTime"].dt.total_seconds()

    laps_df = laps_df.dropna(subset=["stint", "lap_time_s"])
    laps_df["stint"] = laps_df["stint"].astype(int)

    laps_df = laps_df.sort_values(
        ["round_number", "driver_code", "stint", "lap_number"]
    ).reset_index(drop=True)

    print(
        f"[INFO] Loaded {len(laps_df)} laps for season {season_year} "
        f"from {db_file.resolve()}"
    )

    return laps_df


def compute_stint_degradation_for_season(
    laps_df: pd.DataFrame,
    min_laps_per_stint: int = 5,
) -> pd.DataFrame:
    """
    Compute tyre degradation per stint across the whole season.

    Grouping keys:
    - season_year
    - grand_prix_name
    - round_number
    - driver_code
    - team
    - stint
    - tyre_compound

    For each group:
    - build x = [0, 1, 2, ..., n-1] (index inside stint)
    - y = lap_time_s
    - fit y = slope * x + intercept (numpy.polyfit)
    - slope = degradation [seconds per lap]

    :param laps_df: DataFrame with laps (one row per lap)
    :param min_laps_per_stint: minimum number of laps required to fit a model
    :return: DataFrame with one row per (race, driver, stint, compound)
    """
    results = []

    group_cols = [
        "season_year",
        "grand_prix_name",
        "round_number",
        "driver_code",
        "team",
        "stint",
        "tyre_compound",
    ]

    for key, group in laps_df.groupby(group_cols):
        (season_year,
         grand_prix_name,
         round_number,
         driver_code,
         team,
         stint,
         tyre_compound) = key

        group = group.sort_values("lap_number")

        if len(group) < min_laps_per_stint:
            continue

        x = np.arange(len(group), dtype=float)
        y = group["lap_time_s"].values.astype(float)

        slope, intercept = np.polyfit(x, y, 1)

        results.append(
            {
                "season_year": int(season_year),
                "grand_prix_name": str(grand_prix_name),
                "round_number": int(round_number),
                "driver_code": str(driver_code),
                "team": str(team),
                "stint": int(stint),
                "tyre_compound": str(tyre_compound),
                "laps_in_stint": len(group),
                "slope_s_per_lap": float(slope),
                "base_lap_time_s": float(y[0]),
            }
        )

    stint_deg_df = pd.DataFrame(results)

    if stint_deg_df.empty:
        print("[WARN] No stints with enough laps to compute degradation.")
        return stint_deg_df

    stint_deg_df = stint_deg_df.sort_values(
        ["season_year", "round_number", "slope_s_per_lap"]
    ).reset_index(drop=True)

    return stint_deg_df


def aggregate_driver_degradation_by_race(stint_deg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate stint degradation per driver per race.

    For each (season, race, driver):
    - mean slope across all stints
    - number of stints

    :param stint_deg_df: DataFrame with stint-level degradation
    :return: DataFrame with one row per driver per race
    """
    if stint_deg_df.empty:
        return stint_deg_df

    agg_df = (
        stint_deg_df.groupby(
            ["season_year", "grand_prix_name", "round_number", "driver_code", "team"]
        )
        .agg(
            avg_slope_s_per_lap=("slope_s_per_lap", "mean"),
            stints_count=("slope_s_per_lap", "count"),
        )
        .reset_index()
        .sort_values(["season_year", "round_number", "avg_slope_s_per_lap"])
        .reset_index(drop=True)
    )

    return agg_df


def aggregate_driver_degradation_full_season(stint_deg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate stint degradation per driver over the full season.

    For each (season, driver):
    - mean slope across all stints in all races
    - number of stints

    :param stint_deg_df: DataFrame with stint-level degradation
    :return: DataFrame with one row per driver for the full season
    """
    if stint_deg_df.empty:
        return stint_deg_df

    agg_df = (
        stint_deg_df.groupby(["season_year", "driver_code", "team"])
        .agg(
            avg_slope_s_per_lap=("slope_s_per_lap", "mean"),
            stints_count=("slope_s_per_lap", "count"),
        )
        .reset_index()
        .sort_values(["season_year", "avg_slope_s_per_lap"])
        .reset_index(drop=True)
    )

    return agg_df


def print_season_ranking(
    driver_season_df: pd.DataFrame,
    season_year: int,
    top_n: int = 10,
):
    """
    Print a season-level ranking of tyre degradation.

    Lower avg_slope_s_per_lap = better tyre management.
    """
    if driver_season_df.empty:
        print("[INFO] No season-level degradation data to display.")
        return

    df = driver_season_df[driver_season_df["season_year"] == season_year].copy()
    if df.empty:
        print(f"[INFO] No data for season {season_year}.")
        return

    df = df.sort_values("avg_slope_s_per_lap").reset_index(drop=True)

    print(f"\n========== Season {season_year} tyre degradation ranking ==========\n")

    df_display = df.head(top_n)[
        ["driver_code", "team", "stints_count", "avg_slope_s_per_lap"]
    ]

    print(
        df_display.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(
        "\nNote:\n"
        "- avg_slope_s_per_lap is in [seconds per lap].\n"
        "- lower values mean better tyre management over the whole season.\n"
        "- stints_count = number of stints used to estimate the average."
    )


def print_last_race_ranking(
    driver_race_df: pd.DataFrame,
    season_year: int,
):
    """
    Print tyre degradation ranking for the last race of the season in the DB.
    """
    if driver_race_df.empty:
        print("[INFO] No race-level degradation data to display.")
        return

    df = driver_race_df[driver_race_df["season_year"] == season_year].copy()
    if df.empty:
        print(f"[INFO] No race-level data for season {season_year}.")
        return

    last_round = int(df["round_number"].max())
    df_last = df[df["round_number"] == last_round].copy()

    if df_last.empty:
        print(f"[INFO] No data for the last round ({last_round}).")
        return

    gp_name = df_last["grand_prix_name"].iloc[0]

    df_last = df_last.sort_values("avg_slope_s_per_lap").reset_index(drop=True)

    print(
        f"\n========== Tyre degradation ranking - "
        f"{gp_name} (round {last_round}, {season_year}) ==========\n"
    )

    df_display = df_last[
        ["driver_code", "team", "stints_count", "avg_slope_s_per_lap"]
    ]

    print(
        df_display.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


def main():
    # Choose the season to analyze
    season_year = 2024

    # 1) Load all laps for this season from DuckDB
    laps_df = load_laps_for_season(season_year, db_path="data/f1_races.duckdb")

    # 2) Compute stint-level degradation across the whole season
    stint_deg_df = compute_stint_degradation_for_season(
        laps_df,
        min_laps_per_stint=5,
    )

    # 3) Aggregate:
    driver_race_df = aggregate_driver_degradation_by_race(stint_deg_df)
    driver_season_df = aggregate_driver_degradation_full_season(stint_deg_df)

    # 4) Print season ranking
    print_season_ranking(driver_season_df, season_year, top_n=10)

    # 5) Print ranking for the last race in the DB
    print_last_race_ranking(driver_race_df, season_year)


if __name__ == "__main__":
    main()
