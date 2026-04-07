"""
Optimize the ensemble: feature selection, weight tuning, hyperparameter search.

Usage:
    python3 strategies/optimize.py
"""
import os
import sys
import logging
import itertools

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.features import build_features, get_feature_columns
from models.ensemble import evaluate, train_and_predict

logger = logging.getLogger(__name__)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
TEST_SEASONS = [2023, 2024, 2025]


def load_data():
    games = pd.read_csv(os.path.join(DATA_DIR, "games.csv"), parse_dates=["date"])
    batting = pd.read_csv(os.path.join(DATA_DIR, "team_batting.csv"))
    pitching = pd.read_csv(os.path.join(DATA_DIR, "team_pitching.csv"))
    pitcher_stats = pd.read_csv(os.path.join(DATA_DIR, "pitcher_stats.csv"))
    return games, batting, pitching, pitcher_stats


def eval_config(df, feature_cols, test_seasons, weights=None):
    """Evaluate a configuration across test seasons."""
    accs, lls, briers = [], [], []
    for year in test_seasons:
        train = df[df["season"] < year]
        test = df[df["season"] == year]
        if len(train) == 0 or len(test) == 0:
            continue
        result = evaluate(train, test, feature_cols=feature_cols, weights=weights)
        accs.append(result["accuracy"])
        lls.append(result["log_loss"])
        briers.append(result["brier_score"])
    return {
        "accuracy": np.mean(accs),
        "log_loss": np.mean(lls),
        "brier_score": np.mean(briers),
        "n_features": len(feature_cols),
    }


def run_optimization():
    games, batting, pitching, pitcher_stats = load_data()

    # -----------------------------------------------------------------------
    # Step 1: Test curated feature sets
    # -----------------------------------------------------------------------
    print("=" * 80)
    print("STEP 1: Feature Set Comparison")
    print("=" * 80)

    configs = {
        "all": [
            "elo", "rolling_basic", "rolling_extended", "rolling_pitching",
            "rest", "park", "season_batting", "season_pitching", "starter",
            "pythagorean", "streaks", "h2h", "day_night", "home_away_splits",
            "momentum", "season_progress", "interleague",
        ],
        "keep_only": [
            "elo", "rolling_extended", "rest", "park", "season_batting",
            "season_pitching", "starter", "pythagorean", "streaks", "h2h",
            "momentum",
        ],
        "keep+neutral": [
            "elo", "rolling_basic", "rolling_extended", "rest", "park",
            "season_batting", "season_pitching", "starter", "pythagorean",
            "streaks", "h2h", "momentum", "interleague",
        ],
        "minimal": [
            "elo", "starter", "season_batting", "rest", "rolling_extended",
        ],
    }

    best_config = None
    best_acc = 0

    for name, groups in configs.items():
        df = build_features(games, batting, pitching, pitcher_stats, groups=groups)
        feature_cols = get_feature_columns(df)
        result = eval_config(df, feature_cols, TEST_SEASONS)
        marker = ""
        if result["accuracy"] > best_acc:
            best_acc = result["accuracy"]
            best_config = name
            marker = " <-- BEST"
        print(f"{name:<20} Acc={result['accuracy']:.4f}  LL={result['log_loss']:.4f}  "
              f"Brier={result['brier_score']:.4f}  Features={result['n_features']}{marker}")

    print(f"\nBest config: {best_config}")

    # -----------------------------------------------------------------------
    # Step 2: Ensemble weight tuning
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("STEP 2: Ensemble Weight Tuning")
    print(f"{'=' * 80}")

    # Use the best feature config
    best_groups = configs[best_config]
    df = build_features(games, batting, pitching, pitcher_stats, groups=best_groups)
    feature_cols = get_feature_columns(df)

    weight_configs = [
        {"lr": 0.2, "xgb": 0.4, "lgbm": 0.4},
        {"lr": 0.1, "xgb": 0.45, "lgbm": 0.45},
        {"lr": 0.3, "xgb": 0.35, "lgbm": 0.35},
        {"lr": 0.0, "xgb": 0.5, "lgbm": 0.5},
        {"lr": 0.15, "xgb": 0.5, "lgbm": 0.35},
        {"lr": 0.15, "xgb": 0.35, "lgbm": 0.5},
        {"lr": 0.25, "xgb": 0.25, "lgbm": 0.5},
        {"lr": 0.25, "xgb": 0.5, "lgbm": 0.25},
        {"lr": 0.33, "xgb": 0.33, "lgbm": 0.34},
    ]

    best_weights = None
    best_wacc = 0

    for w in weight_configs:
        result = eval_config(df, feature_cols, TEST_SEASONS, weights=w)
        marker = ""
        if result["accuracy"] > best_wacc:
            best_wacc = result["accuracy"]
            best_weights = w
            marker = " <-- BEST"
        print(f"LR={w['lr']:.2f} XGB={w['xgb']:.2f} LGBM={w['lgbm']:.2f}  "
              f"Acc={result['accuracy']:.4f}  LL={result['log_loss']:.4f}{marker}")

    print(f"\nBest weights: {best_weights}")

    # -----------------------------------------------------------------------
    # Step 3: Per-season detail with best config
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print(f"STEP 3: Per-Season Results (config={best_config}, weights={best_weights})")
    print(f"{'=' * 80}")

    for year in TEST_SEASONS:
        train = df[df["season"] < year]
        test = df[df["season"] == year]
        result = evaluate(train, test, feature_cols=feature_cols, weights=best_weights)
        baseline = test["home_win"].mean()
        print(f"\n{year}: Acc={result['accuracy']:.4f} (baseline={baseline:.4f}, "
              f"lift={result['accuracy'] - baseline:+.4f})  "
              f"LL={result['log_loss']:.4f}  Brier={result['brier_score']:.4f}")
        print(f"  Top features: {list(result['top_features'].keys())[:10]}")

    # -----------------------------------------------------------------------
    # Step 4: Feature importance summary
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("STEP 4: Final Feature Importance (trained on 2022-2024, tested on 2025)")
    print(f"{'=' * 80}")

    train = df[df["season"] < 2025]
    test = df[df["season"] == 2025]
    result = evaluate(train, test, feature_cols=feature_cols, weights=best_weights)
    for feat, imp in sorted(result["top_features"].items(), key=lambda x: -x[1]):
        print(f"  {feat:<45} {imp:.4f}")

    return best_config, best_weights, configs[best_config]


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    run_optimization()
