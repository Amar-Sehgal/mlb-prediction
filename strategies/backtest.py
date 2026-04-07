"""
Multi-season backtest: train on prior seasons, test on each target season.

Usage:
    python3 strategies/backtest.py [--seasons 2023 2024 2025]
"""
import argparse
import os
import sys
import logging

import pandas as pd
import numpy as np

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


def backtest_season(df: pd.DataFrame, test_year: int) -> list[dict]:
    """Run all models for a single test season."""
    train = df[df["season"] < test_year].copy()
    test = df[df["season"] == test_year].copy()

    if len(train) == 0 or len(test) == 0:
        logger.warning(f"Skipping {test_year}: train={len(train)}, test={len(test)}")
        return []

    results = []
    models = [
        ("Elo", lambda tr, te: elo.evaluate(te)),
        ("Logistic", lambda tr, te: logistic.evaluate(tr, te)),
        ("XGBoost", lambda tr, te: xgboost_model.evaluate(tr, te)),
        ("LightGBM", lambda tr, te: lgbm_model.evaluate(tr, te)),
        ("Ensemble", lambda tr, te: ensemble.evaluate(tr, te)),
    ]

    baseline_acc = test["home_win"].mean()

    for name, eval_fn in models:
        try:
            result = eval_fn(train, test)
            result["season"] = test_year
            result["train_seasons"] = list(train["season"].unique())
            result["baseline_accuracy"] = baseline_acc
            result["lift_vs_baseline"] = result["accuracy"] - baseline_acc
            results.append(result)
        except Exception as e:
            logger.warning(f"  {name} failed on {test_year}: {e}")

    return results


def run_backtest(test_seasons: list[int]):
    """Run full backtest across multiple seasons."""
    games, batting, pitching = load_data()

    logger.info(f"Building features for {len(games)} games...")
    df = build_features(games, batting, pitching)

    all_results = []
    for year in test_seasons:
        logger.info(f"\n=== Backtesting {year} ===")
        results = backtest_season(df, year)
        all_results.extend(results)

        for r in results:
            logger.info(f"  {r['model']:<25} {r['accuracy']:.4f} (baseline: {r['baseline_accuracy']:.4f})")

    # Summary table
    print(f"\n{'='*80}")
    print(f"MLB Game Prediction - Multi-Season Backtest")
    print(f"{'='*80}")

    # Per-season results
    for year in test_seasons:
        year_results = [r for r in all_results if r["season"] == year]
        if not year_results:
            continue

        baseline = year_results[0]["baseline_accuracy"]
        print(f"\n--- {year} Season (baseline: {baseline:.4f}) ---")
        print(f"{'Model':<25} {'Accuracy':>10} {'Log Loss':>10} {'Brier':>10} {'Lift':>10}")
        print(f"{'-'*65}")

        for r in sorted(year_results, key=lambda x: -x["accuracy"]):
            print(f"{r['model']:<25} {r['accuracy']:>10.4f} {r['log_loss']:>10.4f} "
                  f"{r['brier_score']:>10.4f} {r['lift_vs_baseline']:>+10.4f}")

    # Average across seasons
    print(f"\n{'='*80}")
    print(f"Average Across All Test Seasons")
    print(f"{'='*80}")
    print(f"{'Model':<25} {'Avg Acc':>10} {'Avg LogLoss':>12} {'Avg Brier':>10} {'Avg Lift':>10}")
    print(f"{'-'*67}")

    model_names = sorted(set(r["model"] for r in all_results))
    for model_name in model_names:
        model_results = [r for r in all_results if r["model"] == model_name]
        if not model_results:
            continue
        avg_acc = np.mean([r["accuracy"] for r in model_results])
        avg_ll = np.mean([r["log_loss"] for r in model_results])
        avg_brier = np.mean([r["brier_score"] for r in model_results])
        avg_lift = np.mean([r["lift_vs_baseline"] for r in model_results])
        print(f"{model_name:<25} {avg_acc:>10.4f} {avg_ll:>12.4f} {avg_brier:>10.4f} {avg_lift:>+10.4f}")

    print(f"{'='*80}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, default=[2023, 2024, 2025])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run_backtest(args.seasons)
