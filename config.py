"""
Final optimized model configuration.

Determined by ablation testing and grid search across 2023-2025 seasons.
Average accuracy: 59.43% (vs ~53% home-always baseline).
"""

# Feature groups to include (ablation-tested)
FEATURE_GROUPS = [
    "elo",               # Elo ratings with MOV and season regression
    "rolling_extended",  # Rolling hits, HRs, walks, Ks, errors (15g, 30g)
    "rest",              # Rest days, back-to-back indicators
    "park",              # Park factors
    "season_batting",    # FanGraphs season batting (OPS, wOBA, wRC+, ISO, xwOBA, etc.)
    "season_pitching",   # FanGraphs season pitching (ERA, FIP, xFIP, SIERA, WHIP, K-BB%, etc.)
    "starter",           # Starting pitcher quality (ERA, FIP, WHIP, K-BB%, WAR, Stuff+, etc.)
    "pythagorean",       # Pythagorean win expectation
    "streaks",           # Current win/loss streak
    "h2h",               # Head-to-head record this season
    "momentum",          # Exponentially weighted recent form
]

# Groups removed after ablation (hurt model accuracy):
# - rolling_basic: redundant with rolling_extended + Elo
# - rolling_pitching: noisy, collinear with season_pitching + starter
# - season_progress: month/early_season flag adds noise
# - day_night: no predictive signal
# - home_away_splits: collinear with rolling stats
# - interleague: negligible signal

# Ensemble weights (grid-searched)
ENSEMBLE_WEIGHTS = {"lr": 0.20, "xgb": 0.55, "lgbm": 0.25}
