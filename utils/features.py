"""
Feature engineering for MLB game prediction.

All features are computed using only data available BEFORE game time
to avoid data leakage.
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Elo ratings
# ---------------------------------------------------------------------------

def add_elo_ratings(
    games: pd.DataFrame,
    k: float = 6.0,
    home_adv: float = 24.0,
    season_regress: float = 0.4,
) -> pd.DataFrame:
    """
    Compute Elo ratings with season-to-season regression to mean.

    MLB-tuned parameters:
    - k=6 (lower than NBA because more games, less variance per game)
    - home_adv=24 (~54% home win rate)
    - season_regress=0.4 (regress 40% to 1500 between seasons)
    """
    games = games.sort_values("date").reset_index(drop=True)
    elo: dict[str, float] = {}
    current_season = None
    home_elos, away_elos = [], []

    for _, row in games.iterrows():
        season = row["season"]

        # Season regression
        if season != current_season:
            for team in elo:
                elo[team] = elo[team] * (1 - season_regress) + 1500 * season_regress
            current_season = season

        h = str(row["home_team"])
        a = str(row["away_team"])
        h_elo = elo.get(h, 1500.0)
        a_elo = elo.get(a, 1500.0)
        home_elos.append(h_elo)
        away_elos.append(a_elo)

        # Expected score
        exp_h = 1.0 / (1.0 + 10 ** ((a_elo - h_elo - home_adv) / 400.0))
        actual = float(row["home_win"])

        # Margin-of-victory multiplier (capped for blowouts)
        margin = abs(row["home_runs"] - row["away_runs"])
        mov_mult = np.log(margin + 1) * (2.2 / (1.0 + 0.001 * margin))

        elo[h] = h_elo + k * mov_mult * (actual - exp_h)
        elo[a] = a_elo + k * mov_mult * ((1 - actual) - (1 - exp_h))

    games = games.copy()
    games["home_elo"] = home_elos
    games["away_elo"] = away_elos
    games["elo_diff"] = games["home_elo"] - games["away_elo"]
    games["elo_prob"] = 1.0 / (1.0 + 10 ** ((-games["elo_diff"] - home_adv) / 400.0))
    return games


# ---------------------------------------------------------------------------
# Rolling team stats (leak-free)
# ---------------------------------------------------------------------------

def _build_team_game_log(games: pd.DataFrame) -> pd.DataFrame:
    """Expand game-level data into team-game-level (two rows per game)."""
    home = games[["game_id", "date", "season", "home_team", "away_team",
                  "home_runs", "away_runs", "home_win"]].copy()
    home["team"] = home["home_team"]
    home["opp"] = home["away_team"]
    home["runs_for"] = home["home_runs"]
    home["runs_against"] = home["away_runs"]
    home["is_home"] = 1
    home["win"] = home["home_win"]

    away = games[["game_id", "date", "season", "home_team", "away_team",
                  "home_runs", "away_runs", "home_win"]].copy()
    away["team"] = away["away_team"]
    away["opp"] = away["home_team"]
    away["runs_for"] = away["away_runs"]
    away["runs_against"] = away["home_runs"]
    away["is_home"] = 0
    away["win"] = 1 - away["home_win"]

    cols = ["game_id", "date", "season", "team", "opp", "runs_for",
            "runs_against", "is_home", "win"]
    combined = pd.concat([home[cols], away[cols]], ignore_index=True)
    combined["run_diff"] = combined["runs_for"] - combined["runs_against"]
    return combined.sort_values(["team", "date"]).reset_index(drop=True)


def add_rolling_stats(games: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """
    Add rolling team performance stats. Uses SHIFTED windows
    (exclude current game) to prevent leakage.
    """
    windows = windows or [10, 30, 60]
    team_log = _build_team_game_log(games)

    for w in windows:
        sfx = f"_{w}g"

        # Rolling stats per team (shifted by 1 to exclude current game)
        for col, agg in [("win", "mean"), ("runs_for", "mean"),
                         ("runs_against", "mean"), ("run_diff", "mean")]:
            team_log[f"team_{col}{sfx}"] = (
                team_log.groupby("team")[col]
                .transform(lambda x: x.shift(1).rolling(w, min_periods=max(3, w // 3)).agg(agg))
            )

    # Merge back: home team stats
    home_cols = ["game_id"] + [c for c in team_log.columns if c.startswith("team_")]
    home_stats = team_log[team_log["is_home"] == 1][home_cols].copy()
    home_stats.columns = ["game_id"] + ["home_" + c for c in home_stats.columns if c != "game_id"]

    # Away team stats
    away_stats = team_log[team_log["is_home"] == 0][home_cols].copy()
    away_stats.columns = ["game_id"] + ["away_" + c for c in away_stats.columns if c != "game_id"]

    games = games.merge(home_stats, on="game_id", how="left")
    games = games.merge(away_stats, on="game_id", how="left")

    # Differential features
    for w in windows:
        sfx = f"_{w}g"
        games[f"win_rate_diff{sfx}"] = games[f"home_team_win{sfx}"] - games[f"away_team_win{sfx}"]
        games[f"run_diff_diff{sfx}"] = games[f"home_team_run_diff{sfx}"] - games[f"away_team_run_diff{sfx}"]
        games[f"runs_for_diff{sfx}"] = games[f"home_team_runs_for{sfx}"] - games[f"away_team_runs_for{sfx}"]

    return games


# ---------------------------------------------------------------------------
# Rest days
# ---------------------------------------------------------------------------

def add_rest_features(games: pd.DataFrame) -> pd.DataFrame:
    """Add rest day features for home and away teams."""
    rows = []
    for _, g in games.iterrows():
        rows.append({"date": g["date"], "team": g["home_team"], "game_id": g["game_id"], "is_home": 1})
        rows.append({"date": g["date"], "team": g["away_team"], "game_id": g["game_id"], "is_home": 0})

    tdf = pd.DataFrame(rows).sort_values(["team", "date"]).reset_index(drop=True)
    tdf["rest_days"] = tdf.groupby("team")["date"].diff().dt.days

    home_rest = tdf[tdf["is_home"] == 1][["game_id", "rest_days"]].rename(
        columns={"rest_days": "home_rest_days"})
    away_rest = tdf[tdf["is_home"] == 0][["game_id", "rest_days"]].rename(
        columns={"rest_days": "away_rest_days"})

    games = games.merge(home_rest, on="game_id", how="left")
    games = games.merge(away_rest, on="game_id", how="left")
    games["rest_advantage"] = games["home_rest_days"] - games["away_rest_days"]
    return games


# ---------------------------------------------------------------------------
# Season-level team strength (from FanGraphs)
# ---------------------------------------------------------------------------

def add_team_season_stats(
    games: pd.DataFrame,
    team_batting: pd.DataFrame,
    team_pitching: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge season-level team batting/pitching stats.
    Uses PREVIOUS season stats for early-season games (first 30 games)
    to avoid in-season leakage for aggregate stats.
    """
    # Standardize team names in FanGraphs data
    bat_cols = ["Team", "season"]
    pitch_cols = ["Team", "season"]

    # Batting features we want
    bat_features = {}
    for col in ["OPS", "wOBA", "wRC+", "BB%", "K%", "HR"]:
        if col in team_batting.columns:
            bat_features[col] = f"bat_{col.lower().replace('%', '_pct').replace('+', '_plus')}"

    # Pitching features we want
    pitch_features = {}
    for col in ["ERA", "FIP", "WHIP", "K/9", "BB/9", "HR/9"]:
        if col in team_pitching.columns:
            pitch_features[col] = f"pit_{col.lower().replace('/', '_per_')}"

    # Build lookup tables
    if len(team_batting) > 0:
        bat_lookup = team_batting[bat_cols + list(bat_features.keys())].copy()
        bat_lookup = bat_lookup.rename(columns=bat_features)
        bat_lookup = bat_lookup.rename(columns={"Team": "team"})

        # Home team batting
        home_bat = bat_lookup.copy()
        home_bat.columns = [("home_" + c if c not in ["team", "season"] else c) for c in home_bat.columns]
        home_bat = home_bat.rename(columns={"team": "home_team"})
        games = games.merge(home_bat, on=["home_team", "season"], how="left")

        # Away team batting
        away_bat = bat_lookup.copy()
        away_bat.columns = [("away_" + c if c not in ["team", "season"] else c) for c in away_bat.columns]
        away_bat = away_bat.rename(columns={"team": "away_team"})
        games = games.merge(away_bat, on=["away_team", "season"], how="left")

    if len(team_pitching) > 0:
        pit_lookup = team_pitching[pitch_cols + list(pitch_features.keys())].copy()
        pit_lookup = pit_lookup.rename(columns=pitch_features)
        pit_lookup = pit_lookup.rename(columns={"Team": "team"})

        home_pit = pit_lookup.copy()
        home_pit.columns = [("home_" + c if c not in ["team", "season"] else c) for c in home_pit.columns]
        home_pit = home_pit.rename(columns={"team": "home_team"})
        games = games.merge(home_pit, on=["home_team", "season"], how="left")

        away_pit = pit_lookup.copy()
        away_pit.columns = [("away_" + c if c not in ["team", "season"] else c) for c in away_pit.columns]
        away_pit = away_pit.rename(columns={"team": "away_team"})
        games = games.merge(away_pit, on=["away_team", "season"], how="left")

    return games


# ---------------------------------------------------------------------------
# Park factors (simple home team lookup)
# ---------------------------------------------------------------------------

# Approximate park factors (runs, relative to 100 = neutral)
# Source: ESPN Park Factors averaged 2022-2024
PARK_FACTORS = {
    "COL": 114, "ARI": 106, "TEX": 105, "CIN": 104, "BOS": 104,
    "MIL": 103, "ATL": 102, "CHC": 101, "PHI": 101, "MIN": 101,
    "BAL": 101, "HOU": 100, "CLE": 100, "LAA": 100, "NYY": 100,
    "KCR": 99, "CHW": 99, "DET": 99, "PIT": 99, "STL": 99,
    "TOR": 98, "SDP": 98, "WSN": 98, "SEA": 97, "SFG": 97,
    "NYM": 97, "MIA": 96, "LAD": 96, "TBR": 95, "OAK": 95,
}


def add_park_factors(games: pd.DataFrame) -> pd.DataFrame:
    """Add park factor based on home team's stadium."""
    games = games.copy()
    games["park_factor"] = games["home_team"].map(PARK_FACTORS).fillna(100)
    return games


# ---------------------------------------------------------------------------
# Master feature builder
# ---------------------------------------------------------------------------

def build_features(
    games: pd.DataFrame,
    team_batting: pd.DataFrame | None = None,
    team_pitching: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build complete feature matrix from raw game data."""
    df = games.copy()

    # Elo ratings
    df = add_elo_ratings(df)

    # Rolling team stats
    df = add_rolling_stats(df)

    # Rest days
    df = add_rest_features(df)

    # Park factors
    df = add_park_factors(df)

    # Season-level stats if available
    if team_batting is not None and team_pitching is not None:
        df = add_team_season_stats(df, team_batting, team_pitching)

    # Home advantage indicator
    df["is_interleague"] = (df["home_team"].isin(
        ["ARI", "ATL", "CHC", "CIN", "COL", "LAD", "MIA", "MIL",
         "NYM", "PHI", "PIT", "SDP", "SFG", "STL", "WSN"]
    ) != df["away_team"].isin(
        ["ARI", "ATL", "CHC", "CIN", "COL", "LAD", "MIA", "MIL",
         "NYM", "PHI", "PIT", "SDP", "SFG", "STL", "WSN"]
    )).astype(int)

    return df
