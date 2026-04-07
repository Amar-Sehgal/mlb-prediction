"""
Generate predictions for today's MLB games.

Trains the ensemble on all historical data (including current season
games fetched from MLB Stats API), then predicts today's matchups.

Usage:
    python3 predict_today.py                    # predict today's games
    python3 predict_today.py --date 2025-04-10  # predict specific date
"""
import argparse
import os
import sys
import logging
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from config import FEATURE_GROUPS, ENSEMBLE_WEIGHTS
from utils.features import build_features, get_feature_columns
from utils.mlb_api import fetch_current_season_games, fetch_todays_schedule
from models.ensemble import train_and_predict

logger = logging.getLogger(__name__)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")


def predict(date_str: str | None = None):
    """Generate predictions for a given date."""
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    pred_date = pd.Timestamp(date_str)
    pred_year = pred_date.year
    logger.info(f"Predicting games for {date_str}")

    # Load historical data (retrosheet + FanGraphs)
    games = pd.read_csv(os.path.join(DATA_DIR, "games.csv"), parse_dates=["date"])
    batting = pd.read_csv(os.path.join(DATA_DIR, "team_batting.csv"))
    pitching = pd.read_csv(os.path.join(DATA_DIR, "team_pitching.csv"))
    pitcher_stats = pd.read_csv(os.path.join(DATA_DIR, "pitcher_stats.csv"))

    # If predicting for a season not in retrosheet, fetch from MLB API
    max_retro_season = games["season"].max()
    if pred_year > max_retro_season:
        logger.info(f"Fetching {pred_year} games from MLB Stats API...")
        current_games = fetch_current_season_games()
        if len(current_games) > 0:
            # Only keep games before prediction date
            current_games = current_games[current_games["date"] < pred_date]
            logger.info(f"  {len(current_games)} prior games in {pred_year}")
            games = pd.concat([games, current_games], ignore_index=True)

    # Fetch today's schedule
    today_games = fetch_todays_schedule(date_str)
    if len(today_games) == 0:
        print(f"No games found for {date_str}")
        return None

    print(f"\nFound {len(today_games)} games for {date_str}")

    # Create prediction rows
    pred_rows = []
    for _, g in today_games.iterrows():
        game_id = f"pred_{date_str}_{g['away_team']}_{g['home_team']}"
        pred_rows.append({
            "game_id": game_id,
            "date": pred_date,
            "season": pred_year,
            "home_team": g["home_team"],
            "away_team": g["away_team"],
            "home_runs": np.nan,
            "away_runs": np.nan,
            "home_win": np.nan,
            "day_night": g.get("day_night", "N"),
            "park_id": "",
            "home_hits": np.nan, "away_hits": np.nan,
            "home_errors": np.nan, "away_errors": np.nan,
            "home_walks": np.nan, "away_walks": np.nan,
            "home_strikeouts": np.nan, "away_strikeouts": np.nan,
            "home_homeruns": np.nan, "away_homeruns": np.nan,
            "home_pitchers_used": np.nan, "away_pitchers_used": np.nan,
            "home_earned_runs": np.nan, "away_earned_runs": np.nan,
            "home_starter": g.get("home_starter", ""),
            "away_starter": g.get("away_starter", ""),
        })

    pred_df = pd.DataFrame(pred_rows)
    combined = pd.concat([games, pred_df], ignore_index=True)

    # Build features on combined data
    df = build_features(combined, batting, pitching, pitcher_stats,
                        groups=FEATURE_GROUPS)

    # Split
    train = df[df["home_win"].notna()].copy()
    today = df[df["game_id"].str.startswith("pred_")].copy()

    if len(today) == 0:
        print("No predictable games found")
        return None

    feature_cols = get_feature_columns(df)
    today["home_win"] = 0  # Dummy for prediction

    # Predict
    preds, models = train_and_predict(
        train, today,
        feature_cols=feature_cols,
        weights=ENSEMBLE_WEIGHTS,
    )

    # Display
    print(f"\n{'='*82}")
    print(f"MLB Predictions for {date_str}")
    print(f"{'='*82}")
    print(f"{'Away':<6} {'@':^3} {'Home':<6} {'Home%':>7} {'Away%':>7} "
          f"{'Pick':>6} {'Conf':>6}  {'Starters'}")
    print(f"{'-'*82}")

    for _, row in preds.iterrows():
        prob = row["pred_prob"]
        away_prob = 1 - prob
        pick = row["home_team"] if prob >= 0.5 else row["away_team"]
        conf = max(prob, away_prob)

        if conf >= 0.60:
            tier = "HIGH"
        elif conf >= 0.55:
            tier = "MED"
        else:
            tier = "LOW"

        match = today_games[
            (today_games["home_team"] == row["home_team"]) &
            (today_games["away_team"] == row["away_team"])
        ]
        home_sp = away_sp = "TBD"
        if len(match) > 0:
            home_sp = match.iloc[0].get("home_starter", "TBD") or "TBD"
            away_sp = match.iloc[0].get("away_starter", "TBD") or "TBD"

        print(f"{row['away_team']:<6} {'@':^3} {row['home_team']:<6} "
              f"{prob:>6.1%} {away_prob:>6.1%} "
              f"{pick:>6} {tier:>6}  {away_sp} vs {home_sp}")

    print(f"{'='*82}")

    # Summary
    high_conf = preds[preds["pred_prob"].apply(lambda p: max(p, 1-p)) >= 0.55]
    print(f"\n{len(high_conf)}/{len(preds)} games with 55%+ confidence")

    agree = preds[
        ((preds["pred_prob_lr"] >= 0.5) == (preds["pred_prob_xgb"] >= 0.5)) &
        ((preds["pred_prob_xgb"] >= 0.5) == (preds["pred_prob_lgbm"] >= 0.5))
    ]
    print(f"{len(agree)}/{len(preds)} games with all 3 sub-models agreeing")

    return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLB game predictions")
    parser.add_argument("--date", type=str, default=None,
                        help="Date to predict (YYYY-MM-DD, default: today)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    predict(args.date)
