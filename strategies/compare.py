"""
Compare all model strategies on a given test season.

Usage:
    python3 strategies/compare.py [--test-year 2025]
"""
import argparse
import os
import sys
import logging

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.features import build_features
from models import elo, logistic, xgboost_model, lgbm_model, ensemble

logger = logging.getLogger(__name__)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw data files."""
    games = pd.read_csv(os.path.join(DATA_DIR, "games.csv"), parse_dates=["date"])
    batting = pd.read_csv(os.path.join(DATA_DIR, "team_batting.csv"))
    pitching = pd.read_csv(os.path.join(DATA_DIR, "team_pitching.csv"))
    return games, batting, pitching


def run_comparison(test_year: int = 2025):
    """Run all models and print comparison table."""
    games, batting, pitching = load_data()

    logger.info(f"Building features for {len(games)} games...")
    df = build_features(games, batting, pitching)

    # Split: train on all years before test_year, test on test_year
    train = df[df["season"] < test_year].copy()
    test = df[df["season"] == test_year].copy()

    logger.info(f"Train: {len(train)} games ({train['season'].unique()})")
    logger.info(f"Test: {len(test)} games ({test_year})")

    # Evaluate all models
    results = []

    # Elo (no training)
    elo_result = elo.evaluate(test)
    results.append(elo_result)
    logger.info(f"Elo: {elo_result['accuracy']:.4f} accuracy")

    # Logistic Regression
    lr_result = logistic.evaluate(train, test)
    results.append(lr_result)
    logger.info(f"Logistic: {lr_result['accuracy']:.4f} accuracy")

    # XGBoost
    xgb_result = xgboost_model.evaluate(train, test)
    results.append(xgb_result)
    logger.info(f"XGBoost: {xgb_result['accuracy']:.4f} accuracy")

    # LightGBM
    lgbm_result = lgbm_model.evaluate(train, test)
    results.append(lgbm_result)
    logger.info(f"LightGBM: {lgbm_result['accuracy']:.4f} accuracy")

    # Ensemble
    ens_result = ensemble.evaluate(train, test)
    results.append(ens_result)
    logger.info(f"Ensemble: {ens_result['accuracy']:.4f} accuracy")

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"MLB Game Prediction - Strategy Comparison ({test_year})")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'Accuracy':>10} {'Log Loss':>10} {'Brier':>10} {'N Games':>10}")
    print(f"{'-'*70}")

    baseline_acc = test["home_win"].mean()
    print(f"{'Always Home':<25} {baseline_acc:>10.4f} {'N/A':>10} {'N/A':>10} {len(test):>10}")

    for r in sorted(results, key=lambda x: -x["accuracy"]):
        print(f"{r['model']:<25} {r['accuracy']:>10.4f} {r['log_loss']:>10.4f} "
              f"{r['brier_score']:>10.4f} {r['n_games']:>10}")

    print(f"{'='*70}")

    # Feature importance from best tree model
    if "top_features" in xgb_result:
        print(f"\nXGBoost Top Features:")
        for feat, imp in sorted(xgb_result["top_features"].items(), key=lambda x: -x[1]):
            print(f"  {feat:<35} {imp:.4f}")

    if "coefs" in lr_result:
        print(f"\nLogistic Regression Coefficients:")
        for feat, coef in sorted(lr_result["coefs"].items(), key=lambda x: -abs(x[1])):
            print(f"  {feat:<35} {coef:+.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-year", type=int, default=2025)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run_comparison(args.test_year)
