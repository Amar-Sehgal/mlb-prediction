"""
Feature engineering for MLB game prediction.

All features are computed using only data available BEFORE game time
to avoid data leakage. Features are organized into named groups for
ablation testing.

Feature groups:
  - elo: Elo ratings and win probability
  - rolling_basic: Rolling win rate, runs for/against, run diff
  - rolling_extended: Rolling hits, HRs, walks, strikeouts, errors
  - rolling_pitching: Rolling earned runs, pitchers used (bullpen proxy)
  - rest: Rest days and back-to-back indicators
  - park: Park factors
  - season_batting: Season-level team batting stats (FanGraphs)
  - season_pitching: Season-level team pitching stats (FanGraphs)
  - starter: Starting pitcher quality (matched from FanGraphs)
  - pythagorean: Pythagorean win expectation from runs scored/allowed
  - streaks: Win/loss streak lengths
  - h2h: Head-to-head record between the two teams
  - day_night: Day vs night game indicator
  - home_away_splits: Separate home/away rolling performance
  - momentum: Weighted recent form (exponential decay)
  - season_progress: Month of season, games played
  - interleague: Interleague game indicator
"""
import numpy as np
import pandas as pd


# ===========================================================================
# GROUP: elo
# ===========================================================================

def add_elo_ratings(
    games: pd.DataFrame,
    k: float = 6.0,
    home_adv: float = 24.0,
    season_regress: float = 0.4,
) -> pd.DataFrame:
    """Elo ratings with margin-of-victory and season regression."""
    games = games.sort_values("date").reset_index(drop=True)
    elo: dict[str, float] = {}
    current_season = None
    home_elos, away_elos = [], []

    for _, row in games.iterrows():
        season = row["season"]
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

        exp_h = 1.0 / (1.0 + 10 ** ((a_elo - h_elo - home_adv) / 400.0))

        # Skip Elo update for prediction rows (no result yet)
        if not pd.isna(row["home_win"]):
            actual = float(row["home_win"])
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


# ===========================================================================
# Shared: team game log builder
# ===========================================================================

def _build_team_game_log(games: pd.DataFrame) -> pd.DataFrame:
    """Expand game-level data into team-game-level (two rows per game)."""
    # Collect all per-side stat columns available in the game data
    stat_map_home = {}
    stat_map_away = {}
    for stat in ["hits", "errors", "walks", "strikeouts", "homeruns",
                 "pitchers_used", "earned_runs"]:
        hcol = f"home_{stat}"
        acol = f"away_{stat}"
        if hcol in games.columns and acol in games.columns:
            stat_map_home[stat] = hcol
            stat_map_away[stat] = acol

    base_cols = ["game_id", "date", "season", "home_team", "away_team",
                 "home_runs", "away_runs", "home_win"]

    home = games[base_cols].copy()
    home["team"] = home["home_team"]
    home["opp"] = home["away_team"]
    home["runs_for"] = home["home_runs"]
    home["runs_against"] = home["away_runs"]
    home["is_home"] = 1
    home["win"] = home["home_win"]
    for stat, col in stat_map_home.items():
        home[stat] = games[col]

    away = games[base_cols].copy()
    away["team"] = away["away_team"]
    away["opp"] = away["home_team"]
    away["runs_for"] = away["away_runs"]
    away["runs_against"] = away["home_runs"]
    away["is_home"] = 0
    away["win"] = 1 - away["home_win"]
    for stat, col in stat_map_away.items():
        away[stat] = games[col]

    keep = ["game_id", "date", "season", "team", "opp", "runs_for",
            "runs_against", "is_home", "win"] + list(stat_map_home.keys())
    combined = pd.concat([home[keep], away[keep]], ignore_index=True)
    combined["run_diff"] = combined["runs_for"] - combined["runs_against"]
    return combined.sort_values(["team", "date"]).reset_index(drop=True)


def _shifted_rolling(series: pd.Series, window: int, agg: str = "mean") -> pd.Series:
    """Rolling stat shifted by 1 to exclude current game."""
    return series.shift(1).rolling(window, min_periods=max(3, window // 3)).agg(agg)


def _merge_team_stats(games: pd.DataFrame, team_log: pd.DataFrame,
                      stat_cols: list[str]) -> pd.DataFrame:
    """Merge team-level stats back to game-level for home and away."""
    merge_cols = ["game_id"] + stat_cols

    home_stats = team_log[team_log["is_home"] == 1][merge_cols].copy()
    home_stats.columns = ["game_id"] + ["home_" + c for c in stat_cols]

    away_stats = team_log[team_log["is_home"] == 0][merge_cols].copy()
    away_stats.columns = ["game_id"] + ["away_" + c for c in stat_cols]

    games = games.merge(home_stats, on="game_id", how="left")
    games = games.merge(away_stats, on="game_id", how="left")
    return games


# ===========================================================================
# GROUP: rolling_basic
# ===========================================================================

def add_rolling_basic(games: pd.DataFrame,
                      windows: list[int] | None = None) -> pd.DataFrame:
    """Rolling win rate, runs for/against, run differential."""
    windows = windows or [10, 30, 60]
    team_log = _build_team_game_log(games)
    new_cols = []

    for w in windows:
        sfx = f"_{w}g"
        for col, agg in [("win", "mean"), ("runs_for", "mean"),
                         ("runs_against", "mean"), ("run_diff", "mean")]:
            cname = f"team_{col}{sfx}"
            team_log[cname] = team_log.groupby("team")[col].transform(
                lambda x: _shifted_rolling(x, w, agg))
            new_cols.append(cname)

    games = _merge_team_stats(games, team_log, new_cols)

    # Differentials
    for w in windows:
        sfx = f"_{w}g"
        games[f"win_rate_diff{sfx}"] = games[f"home_team_win{sfx}"] - games[f"away_team_win{sfx}"]
        games[f"run_diff_diff{sfx}"] = games[f"home_team_run_diff{sfx}"] - games[f"away_team_run_diff{sfx}"]
        games[f"runs_for_diff{sfx}"] = games[f"home_team_runs_for{sfx}"] - games[f"away_team_runs_for{sfx}"]

    return games


# ===========================================================================
# GROUP: rolling_extended (hits, HRs, walks, Ks, errors from retrosheet)
# ===========================================================================

def add_rolling_extended(games: pd.DataFrame,
                         windows: list[int] | None = None) -> pd.DataFrame:
    """Rolling batting stats from game-level retrosheet data."""
    windows = windows or [15, 30]
    team_log = _build_team_game_log(games)
    new_cols = []

    extended_stats = ["hits", "homeruns", "walks", "strikeouts", "errors"]
    available = [s for s in extended_stats if s in team_log.columns]

    for w in windows:
        sfx = f"_{w}g"
        for stat in available:
            cname = f"team_{stat}{sfx}"
            team_log[cname] = team_log.groupby("team")[stat].transform(
                lambda x: _shifted_rolling(x, w, "mean"))
            new_cols.append(cname)

    games = _merge_team_stats(games, team_log, new_cols)

    # Differentials
    for w in windows:
        sfx = f"_{w}g"
        for stat in available:
            games[f"{stat}_diff{sfx}"] = (
                games[f"home_team_{stat}{sfx}"] - games[f"away_team_{stat}{sfx}"]
            )

    return games


# ===========================================================================
# GROUP: rolling_pitching (earned runs, pitchers used as bullpen proxy)
# ===========================================================================

def add_rolling_pitching(games: pd.DataFrame,
                         windows: list[int] | None = None) -> pd.DataFrame:
    """Rolling pitching stats: earned run rate, bullpen usage."""
    windows = windows or [15, 30]
    team_log = _build_team_game_log(games)
    new_cols = []

    pitch_stats = ["earned_runs", "pitchers_used"]
    available = [s for s in pitch_stats if s in team_log.columns]

    for w in windows:
        sfx = f"_{w}g"
        for stat in available:
            cname = f"team_{stat}{sfx}"
            team_log[cname] = team_log.groupby("team")[stat].transform(
                lambda x: _shifted_rolling(x, w, "mean"))
            new_cols.append(cname)

    games = _merge_team_stats(games, team_log, new_cols)

    # Differentials
    for w in windows:
        sfx = f"_{w}g"
        for stat in available:
            games[f"{stat}_diff{sfx}"] = (
                games[f"home_team_{stat}{sfx}"] - games[f"away_team_{stat}{sfx}"]
            )

    return games


# ===========================================================================
# GROUP: rest
# ===========================================================================

def add_rest_features(games: pd.DataFrame) -> pd.DataFrame:
    """Rest days, back-to-back indicators."""
    rows = []
    for _, g in games.iterrows():
        rows.append({"date": g["date"], "team": g["home_team"],
                     "game_id": g["game_id"], "is_home": 1})
        rows.append({"date": g["date"], "team": g["away_team"],
                     "game_id": g["game_id"], "is_home": 0})

    tdf = pd.DataFrame(rows).sort_values(["team", "date"]).reset_index(drop=True)
    tdf["rest_days"] = tdf.groupby("team")["date"].diff().dt.days

    home_rest = tdf[tdf["is_home"] == 1][["game_id", "rest_days"]].rename(
        columns={"rest_days": "home_rest_days"})
    away_rest = tdf[tdf["is_home"] == 0][["game_id", "rest_days"]].rename(
        columns={"rest_days": "away_rest_days"})

    games = games.merge(home_rest, on="game_id", how="left")
    games = games.merge(away_rest, on="game_id", how="left")
    games["rest_advantage"] = games["home_rest_days"] - games["away_rest_days"]
    games["home_b2b"] = (games["home_rest_days"] == 0).astype(int)
    games["away_b2b"] = (games["away_rest_days"] == 0).astype(int)
    games["b2b_diff"] = games["away_b2b"] - games["home_b2b"]
    return games


# ===========================================================================
# GROUP: park
# ===========================================================================

PARK_FACTORS = {
    "COL": 114, "ARI": 106, "TEX": 105, "CIN": 104, "BOS": 104,
    "MIL": 103, "ATL": 102, "CHC": 101, "PHI": 101, "MIN": 101,
    "BAL": 101, "HOU": 100, "CLE": 100, "LAA": 100, "NYY": 100,
    "KCR": 99, "CHW": 99, "DET": 99, "PIT": 99, "STL": 99,
    "TOR": 98, "SDP": 98, "WSN": 98, "SEA": 97, "SFG": 97,
    "NYM": 97, "MIA": 96, "LAD": 96, "TBR": 95, "OAK": 95,
}


def add_park_factors(games: pd.DataFrame) -> pd.DataFrame:
    """Park factor for the home stadium."""
    games = games.copy()
    games["park_factor"] = games["home_team"].map(PARK_FACTORS).fillna(100)
    return games


# ===========================================================================
# GROUP: season_batting
# ===========================================================================

def add_season_batting(games: pd.DataFrame,
                       team_batting: pd.DataFrame) -> pd.DataFrame:
    """Season-level team batting stats from FanGraphs."""
    if team_batting is None or len(team_batting) == 0:
        return games

    col_map = {}
    for col in ["OPS", "wOBA", "wRC+", "BB%", "K%", "ISO", "BABIP",
                "Hard%", "HR/FB", "Barrel%", "xwOBA"]:
        if col in team_batting.columns:
            safe = col.lower().replace("%", "_pct").replace("+", "_plus").replace("/", "_per_")
            col_map[col] = f"bat_{safe}"

    lookup = team_batting[["Team", "season"] + list(col_map.keys())].copy()
    lookup = lookup.rename(columns=col_map)
    lookup = lookup.rename(columns={"Team": "team"})

    for side, team_col in [("home", "home_team"), ("away", "away_team")]:
        side_df = lookup.copy()
        side_df.columns = [
            (f"{side}_{c}" if c not in ["team", "season"] else c)
            for c in side_df.columns
        ]
        side_df = side_df.rename(columns={"team": team_col})
        games = games.merge(side_df, on=[team_col, "season"], how="left")

    # Differentials for key stats
    for stat in col_map.values():
        hcol = f"home_{stat}"
        acol = f"away_{stat}"
        if hcol in games.columns and acol in games.columns:
            games[f"{stat}_diff"] = games[hcol] - games[acol]

    return games


# ===========================================================================
# GROUP: season_pitching
# ===========================================================================

def add_season_pitching(games: pd.DataFrame,
                        team_pitching: pd.DataFrame) -> pd.DataFrame:
    """Season-level team pitching stats from FanGraphs."""
    if team_pitching is None or len(team_pitching) == 0:
        return games

    col_map = {}
    for col in ["ERA", "FIP", "xFIP", "SIERA", "WHIP", "K/9", "BB/9",
                "HR/9", "K-BB%", "LOB%", "GB%", "Hard%", "Barrel%",
                "Stuff+", "Location+", "Pitching+"]:
        if col in team_pitching.columns:
            safe = col.lower().replace("%", "_pct").replace("+", "_plus").replace("/", "_per_").replace("-", "_")
            col_map[col] = f"pit_{safe}"

    lookup = team_pitching[["Team", "season"] + list(col_map.keys())].copy()
    lookup = lookup.rename(columns=col_map)
    lookup = lookup.rename(columns={"Team": "team"})

    for side, team_col in [("home", "home_team"), ("away", "away_team")]:
        side_df = lookup.copy()
        side_df.columns = [
            (f"{side}_{c}" if c not in ["team", "season"] else c)
            for c in side_df.columns
        ]
        side_df = side_df.rename(columns={"team": team_col})
        games = games.merge(side_df, on=[team_col, "season"], how="left")

    # Differentials (note: for pitching, lower is better for ERA/FIP/WHIP,
    # so we do away - home so positive = home advantage)
    lower_better = ["era", "fip", "xfip", "siera", "whip", "bb_per_9",
                    "hr_per_9", "hard_pct", "barrel_pct"]
    higher_better = ["k_per_9", "k_bb_pct", "lob_pct", "gb_pct",
                     "stuff_plus", "location_plus", "pitching_plus"]

    for stat in col_map.values():
        hcol = f"home_{stat}"
        acol = f"away_{stat}"
        if hcol in games.columns and acol in games.columns:
            short = stat.replace("pit_", "")
            if short in lower_better:
                games[f"{stat}_diff"] = games[acol] - games[hcol]  # reversed
            else:
                games[f"{stat}_diff"] = games[hcol] - games[acol]

    return games


# ===========================================================================
# GROUP: starter (individual starting pitcher quality)
# ===========================================================================

def add_starter_features(games: pd.DataFrame,
                         pitcher_stats: pd.DataFrame) -> pd.DataFrame:
    """Match starting pitcher names to FanGraphs season stats."""
    if pitcher_stats is None or len(pitcher_stats) == 0:
        return games

    sp_cols = {}
    for col in ["ERA", "FIP", "xFIP", "SIERA", "WHIP", "K/9", "BB/9",
                "K-BB%", "WAR", "IP", "GS", "HR/9", "BABIP", "LOB%",
                "Hard%", "Barrel%", "Stuff+", "Location+", "Pitching+"]:
        if col in pitcher_stats.columns:
            safe = col.lower().replace("%", "_pct").replace("+", "_plus").replace("/", "_per_").replace("-", "_")
            sp_cols[col] = f"sp_{safe}"

    # Build pitcher lookup: name + season -> stats
    # Use per-season stats. For pitchers with same name, take one with most GS
    plookup = pitcher_stats[["Name", "season"] + list(sp_cols.keys())].copy()
    plookup = plookup.rename(columns=sp_cols)
    if "sp_gs" in plookup.columns:
        plookup = plookup.sort_values("sp_gs", ascending=False).drop_duplicates(
            subset=["Name", "season"], keep="first")
    else:
        plookup = plookup.drop_duplicates(subset=["Name", "season"], keep="first")

    # Merge for home starter
    home_sp = plookup.copy()
    home_sp.columns = [
        (f"home_{c}" if c not in ["Name", "season"] else c)
        for c in home_sp.columns
    ]
    home_sp = home_sp.rename(columns={"Name": "home_starter"})
    games = games.merge(home_sp, on=["home_starter", "season"], how="left")

    # Merge for away starter
    away_sp = plookup.copy()
    away_sp.columns = [
        (f"away_{c}" if c not in ["Name", "season"] else c)
        for c in away_sp.columns
    ]
    away_sp = away_sp.rename(columns={"Name": "away_starter"})
    games = games.merge(away_sp, on=["away_starter", "season"], how="left")

    # Starter differentials (lower is better for ERA/FIP/WHIP etc)
    lower_better = ["era", "fip", "xfip", "siera", "whip", "bb_per_9",
                    "hr_per_9", "babip", "hard_pct", "barrel_pct"]
    for stat in sp_cols.values():
        hcol = f"home_{stat}"
        acol = f"away_{stat}"
        if hcol in games.columns and acol in games.columns:
            short = stat.replace("sp_", "")
            if short in lower_better:
                games[f"{stat}_diff"] = games[acol] - games[hcol]
            else:
                games[f"{stat}_diff"] = games[hcol] - games[acol]

    # Starter quality flag: is starter known?
    games["home_starter_known"] = games.get("home_sp_era", pd.Series(dtype=float)).notna().astype(int)
    games["away_starter_known"] = games.get("away_sp_era", pd.Series(dtype=float)).notna().astype(int)

    return games


# ===========================================================================
# GROUP: pythagorean (expected win% from runs scored/allowed)
# ===========================================================================

def add_pythagorean(games: pd.DataFrame,
                    exponent: float = 1.83) -> pd.DataFrame:
    """Pythagorean win expectation using rolling runs scored/allowed."""
    team_log = _build_team_game_log(games)
    window = 40
    new_cols = []

    for stat in ["runs_for", "runs_against"]:
        cname = f"team_{stat}_pyth"
        team_log[cname] = team_log.groupby("team")[stat].transform(
            lambda x: _shifted_rolling(x, window, "sum"))
        new_cols.append(cname)

    # Pythagorean win expectation
    rf = team_log["team_runs_for_pyth"]
    ra = team_log["team_runs_against_pyth"]
    team_log["team_pyth_wpct"] = rf ** exponent / (rf ** exponent + ra ** exponent)
    new_cols.append("team_pyth_wpct")

    games = _merge_team_stats(games, team_log, new_cols)
    games["pyth_diff"] = games["home_team_pyth_wpct"] - games["away_team_pyth_wpct"]
    return games


# ===========================================================================
# GROUP: streaks (current win/loss streak)
# ===========================================================================

def add_streaks(games: pd.DataFrame) -> pd.DataFrame:
    """Current win and loss streak length for each team."""
    team_log = _build_team_game_log(games)

    # Compute streak: count consecutive same results before current game
    def _streak(wins: pd.Series) -> pd.Series:
        shifted = wins.shift(1)
        streaks = []
        current_streak = 0
        last_val = None
        for val in shifted:
            if pd.isna(val):
                streaks.append(0)
                continue
            if val == last_val:
                current_streak += 1
            else:
                current_streak = 1
                last_val = val
            # Positive for wins, negative for losses
            streaks.append(current_streak if val == 1 else -current_streak)
        return pd.Series(streaks, index=wins.index)

    team_log["team_streak"] = team_log.groupby("team")["win"].transform(_streak)

    games = _merge_team_stats(games, team_log, ["team_streak"])
    games["streak_diff"] = games["home_team_streak"] - games["away_team_streak"]
    return games


# ===========================================================================
# GROUP: h2h (head-to-head record this season)
# ===========================================================================

def add_h2h(games: pd.DataFrame) -> pd.DataFrame:
    """Rolling head-to-head win rate between the two teams this season."""
    games = games.sort_values("date").reset_index(drop=True)
    h2h_records: dict[tuple, list] = {}  # (team, opp, season) -> [results]
    h2h_wpcts = []

    for _, row in games.iterrows():
        h = row["home_team"]
        a = row["away_team"]
        s = row["season"]

        key_h = (h, a, s)
        key_a = (a, h, s)

        # Get prior H2H record for home team vs away team
        prior_h = h2h_records.get(key_h, [])
        prior_a = h2h_records.get(key_a, [])

        if len(prior_h) + len(prior_a) > 0:
            h_wins = sum(prior_h) + len(prior_a) - sum(prior_a)
            total = len(prior_h) + len(prior_a)
            h2h_wpcts.append(h_wins / total)
        else:
            h2h_wpcts.append(0.5)  # no prior matchups

        # Record result (skip if no result yet, e.g. prediction rows)
        if not pd.isna(row["home_win"]):
            h2h_records.setdefault(key_h, []).append(int(row["home_win"]))
            h2h_records.setdefault(key_a, []).append(1 - int(row["home_win"]))

    games = games.copy()
    games["h2h_wpct"] = h2h_wpcts
    games["h2h_adv"] = games["h2h_wpct"] - 0.5
    return games


# ===========================================================================
# GROUP: day_night
# ===========================================================================

def add_day_night(games: pd.DataFrame) -> pd.DataFrame:
    """Day (D) vs Night (N) game indicator."""
    games = games.copy()
    if "day_night" in games.columns:
        games["is_night_game"] = (games["day_night"] == "N").astype(int)
        games["is_day_game"] = (games["day_night"] == "D").astype(int)
    return games


# ===========================================================================
# GROUP: home_away_splits (team performance at home vs away)
# ===========================================================================

def add_home_away_splits(games: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Rolling performance split by home/away context."""
    team_log = _build_team_game_log(games)
    new_cols = []

    for context, ctx_val in [("athome", 1), ("away", 0)]:
        for stat in ["win", "run_diff"]:
            cname = f"team_{stat}_{context}_{window}g"
            # Only use games played in that context
            team_log[cname] = np.nan
            mask = team_log["is_home"] == ctx_val
            team_log.loc[mask, cname] = (
                team_log.loc[mask].groupby("team")[stat]
                .transform(lambda x: _shifted_rolling(x, window, "mean"))
            )
            # Forward-fill within team so we have values for away games too
            team_log[cname] = team_log.groupby("team")[cname].transform(
                lambda x: x.ffill())
            new_cols.append(cname)

    games = _merge_team_stats(games, team_log, new_cols)

    # Home team's home record vs away team's away record
    sfx = f"_{window}g"
    if f"home_team_win_athome{sfx}" in games.columns and f"away_team_win_away{sfx}" in games.columns:
        games[f"context_win_diff{sfx}"] = (
            games[f"home_team_win_athome{sfx}"] - games[f"away_team_win_away{sfx}"]
        )
    if f"home_team_run_diff_athome{sfx}" in games.columns and f"away_team_run_diff_away{sfx}" in games.columns:
        games[f"context_rdiff_diff{sfx}"] = (
            games[f"home_team_run_diff_athome{sfx}"] - games[f"away_team_run_diff_away{sfx}"]
        )

    return games


# ===========================================================================
# GROUP: momentum (exponentially weighted recent form)
# ===========================================================================

def add_momentum(games: pd.DataFrame, span: int = 10) -> pd.DataFrame:
    """Exponentially weighted moving average of win/run_diff."""
    team_log = _build_team_game_log(games)
    new_cols = []

    for stat in ["win", "run_diff"]:
        cname = f"team_ema_{stat}_{span}"
        team_log[cname] = team_log.groupby("team")[stat].transform(
            lambda x: x.shift(1).ewm(span=span, min_periods=3).mean()
        )
        new_cols.append(cname)

    games = _merge_team_stats(games, team_log, new_cols)
    sfx = f"_{span}"
    games[f"ema_win_diff{sfx}"] = (
        games[f"home_team_ema_win{sfx}"] - games[f"away_team_ema_win{sfx}"]
    )
    games[f"ema_rdiff_diff{sfx}"] = (
        games[f"home_team_ema_run_diff{sfx}"] - games[f"away_team_ema_run_diff{sfx}"]
    )
    return games


# ===========================================================================
# GROUP: season_progress
# ===========================================================================

def add_season_progress(games: pd.DataFrame) -> pd.DataFrame:
    """Month of season, days since opening day."""
    games = games.copy()
    games["month"] = games["date"].dt.month
    # Games played this season (proxy for how reliable rolling stats are)
    games["season_day"] = games.groupby("season")["date"].transform(
        lambda x: (x - x.min()).dt.days
    )
    # Early season flag (first 3 weeks)
    games["early_season"] = (games["season_day"] < 21).astype(int)
    return games


# ===========================================================================
# GROUP: interleague
# ===========================================================================

NL_TEAMS = {"ARI", "ATL", "CHC", "CIN", "COL", "LAD", "MIA", "MIL",
            "NYM", "PHI", "PIT", "SDP", "SFG", "STL", "WSN"}


def add_interleague(games: pd.DataFrame) -> pd.DataFrame:
    """Interleague game indicator."""
    games = games.copy()
    games["is_interleague"] = (
        games["home_team"].isin(NL_TEAMS) != games["away_team"].isin(NL_TEAMS)
    ).astype(int)
    return games


# ===========================================================================
# FEATURE GROUP REGISTRY
# ===========================================================================

FEATURE_GROUPS = {
    "elo": {
        "fn": "add_elo_ratings",
        "cols": ["elo_diff", "elo_prob"],
    },
    "rolling_basic": {
        "fn": "add_rolling_basic",
        "cols": [
            "win_rate_diff_10g", "win_rate_diff_30g", "win_rate_diff_60g",
            "run_diff_diff_10g", "run_diff_diff_30g", "run_diff_diff_60g",
            "runs_for_diff_10g", "runs_for_diff_30g", "runs_for_diff_60g",
            "home_team_win_10g", "away_team_win_10g",
            "home_team_win_30g", "away_team_win_30g",
            "home_team_runs_for_10g", "away_team_runs_for_10g",
            "home_team_runs_against_10g", "away_team_runs_against_10g",
        ],
    },
    "rolling_extended": {
        "fn": "add_rolling_extended",
        "cols": [
            "hits_diff_15g", "hits_diff_30g",
            "homeruns_diff_15g", "homeruns_diff_30g",
            "walks_diff_15g", "walks_diff_30g",
            "strikeouts_diff_15g", "strikeouts_diff_30g",
            "errors_diff_15g", "errors_diff_30g",
        ],
    },
    "rolling_pitching": {
        "fn": "add_rolling_pitching",
        "cols": [
            "earned_runs_diff_15g", "earned_runs_diff_30g",
            "pitchers_used_diff_15g", "pitchers_used_diff_30g",
            "home_team_earned_runs_15g", "away_team_earned_runs_15g",
            "home_team_earned_runs_30g", "away_team_earned_runs_30g",
        ],
    },
    "rest": {
        "fn": "add_rest_features",
        "cols": ["home_rest_days", "away_rest_days", "rest_advantage",
                 "home_b2b", "away_b2b", "b2b_diff"],
    },
    "park": {
        "fn": "add_park_factors",
        "cols": ["park_factor"],
    },
    "season_batting": {
        "fn": "add_season_batting",
        "cols": None,  # dynamic - filled at runtime
    },
    "season_pitching": {
        "fn": "add_season_pitching",
        "cols": None,
    },
    "starter": {
        "fn": "add_starter_features",
        "cols": None,
    },
    "pythagorean": {
        "fn": "add_pythagorean",
        "cols": ["pyth_diff", "home_team_pyth_wpct", "away_team_pyth_wpct"],
    },
    "streaks": {
        "fn": "add_streaks",
        "cols": ["home_team_streak", "away_team_streak", "streak_diff"],
    },
    "h2h": {
        "fn": "add_h2h",
        "cols": ["h2h_wpct", "h2h_adv"],
    },
    "day_night": {
        "fn": "add_day_night",
        "cols": ["is_night_game", "is_day_game"],
    },
    "home_away_splits": {
        "fn": "add_home_away_splits",
        "cols": ["context_win_diff_30g", "context_rdiff_diff_30g"],
    },
    "momentum": {
        "fn": "add_momentum",
        "cols": ["ema_win_diff_10", "ema_rdiff_diff_10"],
    },
    "season_progress": {
        "fn": "add_season_progress",
        "cols": ["month", "season_day", "early_season"],
    },
    "interleague": {
        "fn": "add_interleague",
        "cols": ["is_interleague"],
    },
}


# ===========================================================================
# Master feature builder
# ===========================================================================

def build_features(
    games: pd.DataFrame,
    team_batting: pd.DataFrame | None = None,
    team_pitching: pd.DataFrame | None = None,
    pitcher_stats: pd.DataFrame | None = None,
    groups: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build complete feature matrix from raw game data.

    Args:
        groups: List of feature group names to include.
                If None, includes all groups.
    """
    all_groups = list(FEATURE_GROUPS.keys())
    groups = groups or all_groups

    df = games.copy()
    df["date"] = pd.to_datetime(df["date"])

    for group in groups:
        if group not in FEATURE_GROUPS:
            continue

        if group == "elo":
            df = add_elo_ratings(df)
        elif group == "rolling_basic":
            df = add_rolling_basic(df)
        elif group == "rolling_extended":
            df = add_rolling_extended(df)
        elif group == "rolling_pitching":
            df = add_rolling_pitching(df)
        elif group == "rest":
            df = add_rest_features(df)
        elif group == "park":
            df = add_park_factors(df)
        elif group == "season_batting":
            df = add_season_batting(df, team_batting)
        elif group == "season_pitching":
            df = add_season_pitching(df, team_pitching)
        elif group == "starter":
            df = add_starter_features(df, pitcher_stats)
        elif group == "pythagorean":
            df = add_pythagorean(df)
        elif group == "streaks":
            df = add_streaks(df)
        elif group == "h2h":
            df = add_h2h(df)
        elif group == "day_night":
            df = add_day_night(df)
        elif group == "home_away_splits":
            df = add_home_away_splits(df)
        elif group == "momentum":
            df = add_momentum(df)
        elif group == "season_progress":
            df = add_season_progress(df)
        elif group == "interleague":
            df = add_interleague(df)

    return df


def get_feature_columns(df: pd.DataFrame, groups: list[str] | None = None) -> list[str]:
    """Get all numeric feature columns for modeling, optionally filtered by group."""
    # Columns that are NOT features (metadata / target)
    exclude = {"game_id", "date", "season", "home_team", "away_team",
               "home_runs", "away_runs", "home_win", "day_night", "park_id",
               "home_starter", "away_starter",
               "home_hits", "away_hits", "home_errors", "away_errors",
               "home_walks", "away_walks", "home_strikeouts", "away_strikeouts",
               "home_homeruns", "away_homeruns", "home_pitchers_used",
               "away_pitchers_used", "home_earned_runs", "away_earned_runs",
               "home_elo", "away_elo"}

    candidates = [c for c in df.columns if c not in exclude
                  and df[c].dtype in ("float64", "int64", "float32", "int32")]
    return candidates
