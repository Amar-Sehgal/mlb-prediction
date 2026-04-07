"""
Fetch historical MLB data.

Sources:
- Game results: Retrosheet game logs (direct download)
- Team batting/pitching stats: FanGraphs via pybaseball
- Individual pitcher stats: FanGraphs via pybaseball
"""
import io
import os
import logging
import time
import zipfile

import pandas as pd
import requests
import pybaseball as pb

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
SEASONS = [2022, 2023, 2024, 2025]

# pybaseball caching
pb.cache.enable()

# Retrosheet team code -> standard abbreviation
RETRO_TEAM_MAP = {
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHA": "CHW", "CHN": "CHC", "CIN": "CIN", "CLE": "CLE",
    "COL": "COL", "DET": "DET", "HOU": "HOU", "KCA": "KCR",
    "LAN": "LAD", "ANA": "LAA", "MIA": "MIA", "MIL": "MIL",
    "MIN": "MIN", "NYA": "NYY", "NYN": "NYM", "OAK": "OAK",
    "PHI": "PHI", "PIT": "PIT", "SDN": "SDP", "SEA": "SEA",
    "SFN": "SFG", "SLN": "STL", "TBA": "TBR", "TEX": "TEX",
    "TOR": "TOR", "WAS": "WSN",
}

# Retrosheet game log column names (161 columns)
# See https://www.retrosheet.org/gamelogs/glfields.txt
GAMELOG_COLS = [
    "date", "game_num", "day_of_week",
    "visiting_team", "visiting_league", "visiting_game_num",
    "home_team", "home_league", "home_game_num",
    "visiting_score", "home_score",
    "game_length_outs", "day_night", "completion_info",
    "forfeit_info", "protest_info", "park_id",
    "attendance", "game_duration_minutes", "visiting_line_score", "home_line_score",
    # Visiting batting stats (cols 21-49)
    "v_at_bats", "v_hits", "v_doubles", "v_triples", "v_homeruns",
    "v_rbi", "v_sac_hits", "v_sac_flies", "v_dp", "v_walks",
    "v_intentional_walks", "v_strikeouts", "v_stolen_bases",
    "v_caught_stealing", "v_gidp", "v_catcher_interference",
    "v_left_on_base", "v_pitchers_used", "v_individual_earned_runs",
    "v_team_earned_runs", "v_wild_pitches", "v_balks",
    "v_putouts", "v_assists", "v_errors", "v_passed_balls",
    "v_double_plays", "v_triple_plays",
    # Home batting stats (cols 49-77)
    "h_at_bats", "h_hits", "h_doubles", "h_triples", "h_homeruns",
    "h_rbi", "h_sac_hits", "h_sac_flies", "h_dp", "h_walks",
    "h_intentional_walks", "h_strikeouts", "h_stolen_bases",
    "h_caught_stealing", "h_gidp", "h_catcher_interference",
    "h_left_on_base", "h_pitchers_used", "h_individual_earned_runs",
    "h_team_earned_runs", "h_wild_pitches", "h_balks",
    "h_putouts", "h_assists", "h_errors", "h_passed_balls",
    "h_double_plays", "h_triple_plays",
    # Umpires
    "hp_umpire_id", "hp_umpire_name",
    "1b_umpire_id", "1b_umpire_name",
    "2b_umpire_id", "2b_umpire_name",
    "3b_umpire_id", "3b_umpire_name",
    "lf_umpire_id", "lf_umpire_name",
    "rf_umpire_id", "rf_umpire_name",
    # Managers
    "v_manager_id", "v_manager_name",
    "h_manager_id", "h_manager_name",
    # Starting pitchers
    "v_starting_pitcher_id", "v_starting_pitcher_name",
    "h_starting_pitcher_id", "h_starting_pitcher_name",
    # Visiting starting lineup (9 batters: id, name, pos each = 27 cols)
    "v_batter1_id", "v_batter1_name", "v_batter1_pos",
    "v_batter2_id", "v_batter2_name", "v_batter2_pos",
    "v_batter3_id", "v_batter3_name", "v_batter3_pos",
    "v_batter4_id", "v_batter4_name", "v_batter4_pos",
    "v_batter5_id", "v_batter5_name", "v_batter5_pos",
    "v_batter6_id", "v_batter6_name", "v_batter6_pos",
    "v_batter7_id", "v_batter7_name", "v_batter7_pos",
    "v_batter8_id", "v_batter8_name", "v_batter8_pos",
    "v_batter9_id", "v_batter9_name", "v_batter9_pos",
    # Home starting lineup (9 batters: id, name, pos each = 27 cols)
    "h_batter1_id", "h_batter1_name", "h_batter1_pos",
    "h_batter2_id", "h_batter2_name", "h_batter2_pos",
    "h_batter3_id", "h_batter3_name", "h_batter3_pos",
    "h_batter4_id", "h_batter4_name", "h_batter4_pos",
    "h_batter5_id", "h_batter5_name", "h_batter5_pos",
    "h_batter6_id", "h_batter6_name", "h_batter6_pos",
    "h_batter7_id", "h_batter7_name", "h_batter7_pos",
    "h_batter8_id", "h_batter8_name", "h_batter8_pos",
    "h_batter9_id", "h_batter9_name", "h_batter9_pos",
    # Additional info
    "additional_info", "acquisition_info",
]


def fetch_retrosheet_games(year: int) -> pd.DataFrame:
    """Download and parse Retrosheet game logs for a season."""
    logger.info(f"Fetching {year} Retrosheet game logs...")
    url = f"https://www.retrosheet.org/gamelogs/gl{year}.zip"

    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to download Retrosheet {year}: HTTP {r.status_code}")

    z = zipfile.ZipFile(io.BytesIO(r.content))
    fname = z.namelist()[0]

    with z.open(fname) as f:
        df = pd.read_csv(f, header=None)

    # Use positional column access (retrosheet game log format)
    # Col indices: 0=date, 1=game_num, 2=dow, 3=vis_team, 4=vis_league, 5=vis_game#
    #   6=home_team, 7=home_league, 8=home_game#, 9=vis_score, 10=home_score
    #   12=day_night, 16=park_id
    #   21=v_ab, 22=v_hits, 23=v_2b, 24=v_3b, 25=v_hr, 30=v_bb, 32=v_so
    #   44=v_errors, 38=v_pitchers_used, 40=v_team_er
    #   49=h_ab, 50=h_hits, 53=h_hr, 58=h_bb, 60=h_so, 72=h_errors
    #   66=h_pitchers_used, 68=h_team_er
    #   101=v_sp_id, 102=v_sp_name, 103=h_sp_id, 104=h_sp_name
    home_team = df[6].map(RETRO_TEAM_MAP).fillna(df[6])
    away_team = df[3].map(RETRO_TEAM_MAP).fillna(df[3])

    games = pd.DataFrame({
        "game_id": df[0].astype(str) + "_" + away_team.astype(str) + "_" + home_team.astype(str) + "_" + df[1].astype(str),
        "date": pd.to_datetime(df[0].astype(str), format="%Y%m%d"),
        "season": year,
        "home_team": home_team,
        "away_team": away_team,
        "home_runs": df[10],
        "away_runs": df[9],
        "home_win": (df[10] > df[9]).astype(int),
        "day_night": df[12],
        "park_id": df[16],
        "home_hits": df[50],
        "away_hits": df[22],
        "home_errors": df[72],
        "away_errors": df[44],
        "home_walks": df[58],
        "away_walks": df[30],
        "home_strikeouts": df[60],
        "away_strikeouts": df[32],
        "home_homeruns": df[53],
        "away_homeruns": df[25],
        "home_pitchers_used": df[66],
        "away_pitchers_used": df[38],
        "home_earned_runs": df[68],
        "away_earned_runs": df[40],
        "home_starter": df[104],
        "away_starter": df[102],
    })

    games = games.sort_values("date").reset_index(drop=True)
    logger.info(f"  {len(games)} games for {year}")
    return games


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

        # Game results from Retrosheet
        try:
            games = fetch_retrosheet_games(year)
            all_games.append(games)
        except Exception as e:
            logger.warning(f"Failed to fetch Retrosheet games for {year}: {e}")
            continue

        # Team stats from FanGraphs
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

        time.sleep(2)  # Be nice to FanGraphs

    # Save combined data
    if all_games:
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
