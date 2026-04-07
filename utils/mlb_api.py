"""
Fetch game results from the MLB Stats API for the current season.
Used when Retrosheet data isn't available yet (in-season).
"""
import logging
from datetime import datetime, timedelta

import pandas as pd
import requests

logger = logging.getLogger(__name__)

MLB_API = "https://statsapi.mlb.com/api/v1"

# MLB team ID -> standard abbreviation
MLB_ID_TO_ABBR = {
    109: "ARI", 144: "ATL", 110: "BAL", 111: "BOS", 112: "CHC",
    145: "CHW", 113: "CIN", 114: "CLE", 115: "COL", 116: "DET",
    117: "HOU", 118: "KCR", 108: "LAA", 119: "LAD", 146: "MIA",
    158: "MIL", 142: "MIN", 121: "NYM", 147: "NYY", 133: "OAK",
    143: "PHI", 134: "PIT", 135: "SDP", 136: "SEA", 137: "SFG",
    138: "STL", 139: "TBR", 140: "TEX", 141: "TOR", 120: "WSN",
}


def fetch_completed_games(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch completed regular season game results from MLB Stats API.
    Returns DataFrame in same format as retrosheet game data.
    """
    url = (f"{MLB_API}/schedule?sportId=1&startDate={start_date}"
           f"&endDate={end_date}&gameType=R"
           f"&hydrate=probablePitcher,linescore")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()

    games = []
    for date_block in data.get("dates", []):
        game_date = date_block["date"]
        game_num = {}  # Track doubleheaders per matchup

        for game in date_block.get("games", []):
            status = game.get("status", {}).get("abstractGameState", "")
            if status != "Final":
                continue

            away_info = game["teams"]["away"]
            home_info = game["teams"]["home"]
            away_id = away_info["team"]["id"]
            home_id = home_info["team"]["id"]
            away_team = MLB_ID_TO_ABBR.get(away_id, f"UNK_{away_id}")
            home_team = MLB_ID_TO_ABBR.get(home_id, f"UNK_{home_id}")

            away_score = away_info.get("score", 0)
            home_score = home_info.get("score", 0)

            # Track doubleheaders
            matchup_key = f"{away_team}_{home_team}"
            game_num[matchup_key] = game_num.get(matchup_key, 0) + 1

            # Get linescore stats if available
            linescore = game.get("linescore", {})
            home_hits = linescore.get("teams", {}).get("home", {}).get("hits", None)
            away_hits = linescore.get("teams", {}).get("away", {}).get("hits", None)
            home_errors = linescore.get("teams", {}).get("home", {}).get("errors", None)
            away_errors = linescore.get("teams", {}).get("away", {}).get("errors", None)

            # Get probable pitchers
            home_pitcher = ""
            away_pitcher = ""
            try:
                hp = home_info.get("probablePitcher", {})
                ap = away_info.get("probablePitcher", {})
                home_pitcher = hp.get("fullName", "")
                away_pitcher = ap.get("fullName", "")
            except Exception:
                pass

            # Day/night
            day_night = "D" if game.get("dayNight") == "day" else "N"

            date_clean = game_date.replace("-", "")
            game_id = f"{date_clean}_{away_team}_{home_team}_{game_num[matchup_key]}"

            games.append({
                "game_id": game_id,
                "date": pd.Timestamp(game_date),
                "season": int(game_date[:4]),
                "home_team": home_team,
                "away_team": away_team,
                "home_runs": home_score,
                "away_runs": away_score,
                "home_win": int(home_score > away_score),
                "day_night": day_night,
                "park_id": "",
                "home_hits": home_hits,
                "away_hits": away_hits,
                "home_errors": home_errors,
                "away_errors": away_errors,
                "home_walks": None,
                "away_walks": None,
                "home_strikeouts": None,
                "away_strikeouts": None,
                "home_homeruns": None,
                "away_homeruns": None,
                "home_pitchers_used": None,
                "away_pitchers_used": None,
                "home_earned_runs": None,
                "away_earned_runs": None,
                "home_starter": home_pitcher,
                "away_starter": away_pitcher,
            })

    df = pd.DataFrame(games)
    if len(df) > 0:
        df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_current_season_games() -> pd.DataFrame:
    """Fetch all completed games for the current season."""
    year = datetime.now().year
    start = f"{year}-03-01"
    end = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"Fetching {year} games from MLB API ({start} to {end})...")
    df = fetch_completed_games(start, end)
    logger.info(f"  {len(df)} completed games")
    return df


def fetch_todays_schedule(date_str: str | None = None) -> pd.DataFrame:
    """Fetch today's scheduled games with probable pitchers."""
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    url = (f"{MLB_API}/schedule?sportId=1&date={date_str}"
           f"&gameType=R&hydrate=probablePitcher")
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()

    games = []
    for date_block in data.get("dates", []):
        for game in date_block.get("games", []):
            status = game.get("status", {}).get("abstractGameState", "")
            if status == "Final":
                continue

            away_info = game["teams"]["away"]
            home_info = game["teams"]["home"]
            away_id = away_info["team"]["id"]
            home_id = home_info["team"]["id"]

            away_pitcher = ""
            home_pitcher = ""
            try:
                away_pitcher = away_info.get("probablePitcher", {}).get("fullName", "TBD")
                home_pitcher = home_info.get("probablePitcher", {}).get("fullName", "TBD")
            except Exception:
                pass

            day_night = "D" if game.get("dayNight") == "day" else "N"

            games.append({
                "home_team": MLB_ID_TO_ABBR.get(home_id, f"UNK_{home_id}"),
                "away_team": MLB_ID_TO_ABBR.get(away_id, f"UNK_{away_id}"),
                "home_starter": home_pitcher,
                "away_starter": away_pitcher,
                "game_time": game.get("gameDate", ""),
                "day_night": day_night,
            })

    return pd.DataFrame(games)
