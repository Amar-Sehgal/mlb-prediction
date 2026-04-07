"""
Fetch historical MLB data using pybaseball.

Downloads:
- Game-level results (schedule_and_record from Baseball Reference)
- Team batting stats (season-level from FanGraphs)
- Team pitching stats (season-level from FanGraphs)
- Individual pitcher stats for starting pitcher features
"""
import os
import logging
import time

import pandas as pd
import pybaseball as pb

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
SEASONS = [2022, 2023, 2024, 2025]

# pybaseball caching
pb.cache.enable()

# MLB team abbreviation mapping (Baseball Reference -> standard)
BR_TEAM_MAP = {
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHC": "CHC", "CHW": "CHW", "CIN": "CIN", "CLE": "CLE",
    "COL": "COL", "DET": "DET", "HOU": "HOU", "KCR": "KCR",
    "LAA": "LAA", "LAD": "LAD", "MIA": "MIA", "MIL": "MIL",
    "MIN": "MIN", "NYM": "NYM", "NYY": "NYY", "OAK": "OAK",
    "PHI": "PHI", "PIT": "PIT", "SDP": "SDP", "SEA": "SEA",
    "SFG": "SFG", "STL": "STL", "TBR": "TBR", "TEX": "TEX",
    "TOR": "TOR", "WSN": "WSN",
    # Common alternate abbreviations
    "KCA": "KCR", "KAN": "KCR",
    "TBA": "TBR", "TAM": "TBR",
    "SFN": "SFG", "SF": "SFG",
    "SDN": "SDP", "SD": "SDP",
    "NYA": "NYY", "NY": "NYY",
    "NYN": "NYM",
    "LAN": "LAD", "LA": "LAD",
    "ANA": "LAA",
    "CHA": "CHW", "CHN": "CHC",
    "WAS": "WSN",
    "SLN": "STL",
    "PHA": "OAK",
}


def fetch_schedule(year: int) -> pd.DataFrame:
    """Fetch full season schedule with results from Baseball Reference."""
    logger.info(f"Fetching {year} schedule...")

    all_games = []
    teams_done = set()

    # Get schedule for each team, deduplicate
    for team in ["ARI", "ATL", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE",
                 "COL", "DET", "HOU", "KCR", "LAA", "LAD", "MIA", "MIL",
                 "MIN", "NYM", "NYY", "OAK", "PHI", "PIT", "SDP", "SEA",
                 "SFG", "STL", "TBR", "TEX", "TOR", "WSN"]:
        try:
            sched = pb.schedule_and_record(year, team)
            sched["team"] = team
            all_games.append(sched)
            teams_done.add(team)
            time.sleep(1)  # Rate limit
        except Exception as e:
            logger.warning(f"Failed to fetch {team} {year}: {e}")

    if not all_games:
        raise RuntimeError(f"No schedule data fetched for {year}")

    df = pd.concat(all_games, ignore_index=True)

    # Standardize columns
    df = df.rename(columns={
        "Date": "date",
        "Tm": "team_abbr",
        "Opp": "opp_abbr",
        "R": "runs_scored",
        "RA": "runs_allowed",
        "W/L": "result",
        "Home_Away": "home_away",
        "Win": "winning_pitcher",
        "Loss": "losing_pitcher",
        "Save": "save_pitcher",
    })

    return df


def parse_games_from_schedule(schedule_df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Convert team-perspective schedule into game-level DataFrame.
    One row per game with home/away teams and scores.
    """
    games = []
    seen_games = set()

    for _, row in schedule_df.iterrows():
        team = row.get("team", "")
        opp = str(row.get("opp_abbr", row.get("Opp", "")))
        # Remove @ prefix for away games
        opp_clean = opp.lstrip("@").strip()
        opp_clean = BR_TEAM_MAP.get(opp_clean, opp_clean)

        date = row.get("date", row.get("Date", ""))
        runs = row.get("runs_scored", row.get("R", None))
        runs_against = row.get("runs_allowed", row.get("RA", None))

        if pd.isna(runs) or pd.isna(runs_against):
            continue

        runs = int(runs)
        runs_against = int(runs_against)

        # Determine home/away
        home_away = str(row.get("home_away", row.get("Home_Away", "")))
        is_home = home_away != "@"

        if is_home:
            home_team, away_team = team, opp_clean
            home_runs, away_runs = runs, runs_against
        else:
            home_team, away_team = opp_clean, team
            home_runs, away_runs = runs_against, runs

        # Deduplicate (same game appears once per team)
        game_key = f"{date}_{min(home_team, away_team)}_{max(home_team, away_team)}"
        if game_key in seen_games:
            continue
        seen_games.add(game_key)

        games.append({
            "game_id": game_key,
            "date": date,
            "season": year,
            "home_team": home_team,
            "away_team": away_team,
            "home_runs": home_runs,
            "away_runs": away_runs,
            "home_win": int(home_runs > away_runs),
        })

    df = pd.DataFrame(games)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def fetch_team_batting(year: int) -> pd.DataFrame:
    """Fetch team-level batting stats from FanGraphs."""
    logger.info(f"Fetching {year} team batting stats...")
    try:
        df = pb.team_batting(year)
        df["season"] = year
        return df
    except Exception as e:
        logger.warning(f"Failed to fetch team batting {year}: {e}")
        return pd.DataFrame()


def fetch_team_pitching(year: int) -> pd.DataFrame:
    """Fetch team-level pitching stats from FanGraphs."""
    logger.info(f"Fetching {year} team pitching stats...")
    try:
        df = pb.team_pitching(year)
        df["season"] = year
        return df
    except Exception as e:
        logger.warning(f"Failed to fetch team pitching {year}: {e}")
        return pd.DataFrame()


def fetch_pitcher_stats(year: int) -> pd.DataFrame:
    """Fetch individual pitcher stats from FanGraphs for SP identification."""
    logger.info(f"Fetching {year} pitcher stats...")
    try:
        df = pb.pitching_stats(year, qual=20)  # min 20 IP
        df["season"] = year
        return df
    except Exception as e:
        logger.warning(f"Failed to fetch pitcher stats {year}: {e}")
        return pd.DataFrame()


def fetch_all(seasons: list[int] | None = None):
    """Fetch all data for specified seasons and save to data/raw/."""
    seasons = seasons or SEASONS
    os.makedirs(DATA_DIR, exist_ok=True)

    all_games = []
    all_batting = []
    all_pitching = []
    all_pitchers = []

    for year in seasons:
        logger.info(f"=== Fetching {year} season ===")

        # Schedule/results
        sched = fetch_schedule(year)
        games = parse_games_from_schedule(sched, year)
        all_games.append(games)
        logger.info(f"  {len(games)} games parsed for {year}")

        # Team stats
        batting = fetch_team_batting(year)
        if len(batting) > 0:
            all_batting.append(batting)

        pitching = fetch_team_pitching(year)
        if len(pitching) > 0:
            all_pitching.append(pitching)

        # Pitcher stats
        pitchers = fetch_pitcher_stats(year)
        if len(pitchers) > 0:
            all_pitchers.append(pitchers)

        time.sleep(2)  # Be nice to the APIs

    # Save combined data
    games_df = pd.concat(all_games, ignore_index=True)
    games_df.to_csv(os.path.join(DATA_DIR, "games.csv"), index=False)
    logger.info(f"Saved {len(games_df)} total games")

    if all_batting:
        batting_df = pd.concat(all_batting, ignore_index=True)
        batting_df.to_csv(os.path.join(DATA_DIR, "team_batting.csv"), index=False)
        logger.info(f"Saved team batting ({len(batting_df)} rows)")

    if all_pitching:
        pitching_df = pd.concat(all_pitching, ignore_index=True)
        pitching_df.to_csv(os.path.join(DATA_DIR, "team_pitching.csv"), index=False)
        logger.info(f"Saved team pitching ({len(pitching_df)} rows)")

    if all_pitchers:
        pitchers_df = pd.concat(all_pitchers, ignore_index=True)
        pitchers_df.to_csv(os.path.join(DATA_DIR, "pitcher_stats.csv"), index=False)
        logger.info(f"Saved pitcher stats ({len(pitchers_df)} rows)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    fetch_all()
