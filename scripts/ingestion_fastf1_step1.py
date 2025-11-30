"""
Simple ingestion script for F1 race data using FastF1.

Goal:
- connect to the FastF1 data source
- retrieve race laps for one Grand Prix
- compute a small statistic: best lap per driver

This will be the base for:
- saving data into a database
- exposing results via an API / dashboard
"""

from pathlib import Path

import fastf1
import pandas as pd


def init_cache():
    """
    Initialize the FastF1 cache in the folder data/cache_f1.
    """
    cache_folder = Path("data") / "cache_f1"
    cache_folder.mkdir(parents=True, exist_ok=True)

    # Enable FastF1 cache
    fastf1.Cache.enable_cache(cache_folder)
    print(f"[INFO] FastF1 cache initialized in: {cache_folder.resolve()}")


def load_race_session(year: int, gp_name: str):
    """
    Load a race session ('R') for a given year and Grand Prix name.

    :param year: season year (e.g. 2024)
    :param gp_name: Grand Prix name as expected by FastF1,
                    e.g. 'Bahrain', 'Saudi Arabia', 'Australia', ...
    :return: FastF1 session object
    """
    print(f"[INFO] Loading session: year={year}, gp='{gp_name}', type='R' (race)")
    session = fastf1.get_session(year, gp_name, "R")

    # load() fetches timing, laps, telemetry, etc.
    session.load()
    print(f"[INFO] Session loaded: {session.event['EventName']} ({session.event['EventDate']})")

    return session


def extract_laps(session) -> pd.DataFrame:
    """
    Extract all laps of the session as a Pandas DataFrame.

    :param session: FastF1 session already loaded
    :return: DataFrame with all laps
    """
    laps_df = session.laps
    print(f"[INFO] Total number of laps in the race: {len(laps_df)}")

    return laps_df


def compute_best_laps_per_driver(laps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the best lap (minimum lap time) for each driver.

    :param laps_df: DataFrame containing laps (session.laps)
    :return: DataFrame with one row per driver and their best lap
    """
    # Remove laps with invalid lap times (NaN)
    valid_laps_df = laps_df.dropna(subset=["LapTime"])

    # Group by driver and take the lap with the minimum LapTime
    best_laps_df = (
        valid_laps_df.loc[valid_laps_df.groupby("Driver")["LapTime"].idxmin()]
        .copy()
    )

    # Keep a few interesting columns
    useful_columns = [
        "Driver",        # driver code (e.g. VER, LEC, HAM)
        "DriverNumber",  # car number
        "LapNumber",     # lap number
        "LapTime",       # lap time (Timedelta)
        "Compound",      # tyre compound
        "TyreLife",      # number of laps on this tyre set
        "Team",          # team name
    ]

    best_laps_df = best_laps_df[useful_columns].sort_values("LapTime")

    # Rename columns to more readable English names
    best_laps_df = best_laps_df.rename(
        columns={
            "Driver": "driver_code",
            "DriverNumber": "car_number",
            "LapNumber": "lap_number",
            "LapTime": "lap_time",
            "Compound": "tyre_compound",
            "TyreLife": "tyre_age",
            "Team": "team",
        }
    )

    return best_laps_df.reset_index(drop=True)


def print_best_lap_ranking(best_laps_df: pd.DataFrame):
    """
    Print a simple table of best laps per driver.
    """
    print("\n========== Best laps per driver ==========\n")

    display_df = best_laps_df.copy()

    # Convert lap time to seconds for a compact display
    display_df["lap_time_s"] = display_df["lap_time"].dt.total_seconds()

    display_columns = [
        "driver_code",
        "car_number",
        "team",
        "lap_number",
        "tyre_compound",
        "tyre_age",
        "lap_time_s",
    ]

    print(
        display_df[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )


def main():
    # 1) Initialize the cache
    init_cache()

    # 2) Parameters of the race to analyze
    # You can change these values for other GPs / seasons
    season_year = 2024
    grand_prix_name = "Bahrain"   # examples: "Bahrain", "Saudi Arabia", "Australia", "Japan", ...

    # 3) Load the race session
    race_session = load_race_session(season_year, grand_prix_name)

    # 4) Extract all laps
    laps_df = extract_laps(race_session)

    # 5) Compute best laps per driver
    best_laps_df = compute_best_laps_per_driver(laps_df)

    # 6) Print a simple ranking
    print_best_lap_ranking(best_laps_df)


if __name__ == "__main__":
    main()
