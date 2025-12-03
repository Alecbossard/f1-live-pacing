"""
Ingestion script for F1 race data using FastF1 + DuckDB.

Goals:
- connect to the FastF1 data source
- retrieve race laps for one Grand Prix
- compute a small statistic: best lap per driver
- save data into a local DuckDB database
- run a simple SQL query on top of that data

This is a good "data engineering + analytics" building block
for a real portfolio project.
"""

from pathlib import Path

import fastf1
import pandas as pd
import duckdb


def init_cache():
    """
    Initialize the FastF1 cache in the folder data/cache_f1.
    """
    cache_folder = Path("data") / "cache_f1"
    cache_folder.mkdir(parents=True, exist_ok=True)

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


def extract_laps(session, season_year: int, grand_prix_name: str) -> pd.DataFrame:
    """
    Extract all laps of the session as a Pandas DataFrame
    and add some metadata columns.

    :param session: FastF1 session already loaded
    :param season_year: year of the season
    :param grand_prix_name: GP name used as identifier
    :return: DataFrame with all laps
    """
    laps_df = session.laps.copy()
    print(f"[INFO] Total number of laps in the race: {len(laps_df)}")

    # Add simple metadata so we know where the laps come from
    laps_df["season_year"] = season_year
    laps_df["grand_prix_name"] = grand_prix_name

    return laps_df


def compute_best_laps_per_driver(laps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the best lap (minimum lap time) for each driver.

    :param laps_df: DataFrame containing laps (session.laps with metadata)
    :return: DataFrame with one row per driver and their best lap
    """
    # Remove laps with invalid lap times (NaN)
    valid_laps_df = laps_df.dropna(subset=["LapTime"])

    # Group by driver and take the lap with the minimum LapTime
    best_laps_df = valid_laps_df.loc[
        valid_laps_df.groupby("Driver")["LapTime"].idxmin()
    ].copy()

    # Keep a few interesting columns
    useful_columns = [
        "season_year",
        "grand_prix_name",
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

    # Add lap time in seconds for simpler SQL queries later
    best_laps_df["lap_time_s"] = best_laps_df["lap_time"].dt.total_seconds()

    return best_laps_df.reset_index(drop=True)


def print_best_lap_ranking(best_laps_df: pd.DataFrame):
    """
    Print a simple table of best laps per driver.
    """
    print("\n========== Best laps per driver (from DataFrame) ==========\n")

    display_df = best_laps_df.copy()

    display_columns = [
        "season_year",
        "grand_prix_name",
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


def save_to_duckdb(
    laps_df: pd.DataFrame,
    best_laps_df: pd.DataFrame,
    db_path: str = "data/f1_races.duckdb",
):
    """
    Save laps and best laps into a DuckDB file.

    For simplicity:
    - we create / append to two tables: laps, best_laps
    - we do not handle duplicates yet (same race ingested twice)

    :param laps_df: DataFrame with all laps
    :param best_laps_df: DataFrame with best lap per driver
    :param db_path: path to the DuckDB database file
    """
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Saving data into DuckDB at: {db_file.resolve()}")

    conn = duckdb.connect(str(db_file))

    # Register DataFrames as temporary views
    conn.register("laps_df_view", laps_df)
    conn.register("best_laps_df_view", best_laps_df)

    # Create tables if they do not exist, then insert rows
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS laps AS
        SELECT * FROM laps_df_view
        """
    )
    conn.execute(
        """
        INSERT INTO laps
        SELECT * FROM laps_df_view
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS best_laps AS
        SELECT * FROM best_laps_df_view
        """
    )
    conn.execute(
        """
        INSERT INTO best_laps
        SELECT * FROM best_laps_df_view
        """
    )

    conn.close()
    print("[INFO] Data saved into DuckDB (tables: laps, best_laps).")


def run_example_query(db_path: str = "data/f1_races.duckdb"):
    """
    Run a small example SQL query on the DuckDB file.

    Example:
    - list best lap times (in seconds) per driver for each GP,
      ordered by lap time
    """
    print("\n========== Example SQL query on DuckDB ==========\n")

    db_file = Path(db_path)
    if not db_file.exists():
        print(f"[WARN] Database file not found at: {db_file.resolve()}")
        return

    conn = duckdb.connect(str(db_file))

    query = """
        SELECT
            season_year,
            grand_prix_name,
            driver_code,
            team,
            lap_number,
            tyre_compound,
            lap_time_s
        FROM best_laps
        ORDER BY season_year DESC, grand_prix_name, lap_time_s
        LIMIT 20
    """

    result_df = conn.execute(query).df()
    conn.close()

    if result_df.empty:
        print("[INFO] Query returned no rows.")
        return

    print(
        result_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )


def main():
    # 1) Initialize the FastF1 cache
    init_cache()

    # 2) Parameters of the race to analyze
    # You can change these values for other GPs / seasons
    season_year = 2024
    grand_prix_name = "Bahrain"   # e.g. "Bahrain", "Saudi Arabia", "Australia", "Japan", ...

    # 3) Load the race session
    race_session = load_race_session(season_year, grand_prix_name)

    # 4) Extract all laps and add metadata
    laps_df = extract_laps(race_session, season_year, grand_prix_name)

    # 5) Compute best laps per driver
    best_laps_df = compute_best_laps_per_driver(laps_df)

    # 6) Print a simple ranking from the DataFrame
    print_best_lap_ranking(best_laps_df)

    # 7) Save both DataFrames into DuckDB
    save_to_duckdb(laps_df, best_laps_df)

    # 8) Run an example SQL query on DuckDB
    run_example_query()


if __name__ == "__main__":
    main()
