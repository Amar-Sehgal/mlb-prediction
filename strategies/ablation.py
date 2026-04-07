"""
Feature group ablation testing.

Tests each feature group's marginal contribution to the ensemble model
across multiple seasons. Two modes:
  1. Add-one: Start with base (elo), add each group one at a time
  2. Drop-one: Start with all groups, drop each one at a time

Usage:
    python3 strategies/ablation.py
"""
import os
import sys
import logging
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.features import build_features, get_feature_columns, FEATURE_GROUPS
from models.ensemble import evaluate

logger = logging.getLogger(__name__)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

TEST_SEASONS = [2023, 2024, 2025]
ALL_GROUPS = list(FEATURE_GROUPS.keys())
BASE_GROUPS = ["elo"]  # always included


def load_data():
    games = pd.read_csv(os.path.join(DATA_DIR, "games.csv"), parse_dates=["date"])
    batting = pd.read_csv(os.path.join(DATA_DIR, "team_batting.csv"))
    pitching = pd.read_csv(os.path.join(DATA_DIR, "team_pitching.csv"))
    pitcher_stats = pd.read_csv(os.path.join(DATA_DIR, "pitcher_stats.csv"))
    return games, batting, pitching, pitcher_stats


def eval_groups(games, batting, pitching, pitcher_stats,
                groups: list[str], test_seasons: list[int]) -> dict:
    """Build features for given groups and evaluate across seasons."""
    df = build_features(games, batting, pitching, pitcher_stats, groups=groups)
    feature_cols = get_feature_columns(df, groups)

    accs, lls, briers = [], [], []
    for year in test_seasons:
        train = df[df["season"] < year]
        test = df[df["season"] == year]
        if len(train) == 0 or len(test) == 0:
            continue
        try:
            result = evaluate(train, test, feature_cols=feature_cols)
            accs.append(result["accuracy"])
            lls.append(result["log_loss"])
            briers.append(result["brier_score"])
        except Exception as e:
            logger.warning(f"  Failed for {groups}: {e}")

    if not accs:
        return {"accuracy": 0, "log_loss": 1, "brier_score": 0.5}

    return {
        "accuracy": np.mean(accs),
        "log_loss": np.mean(lls),
        "brier_score": np.mean(briers),
        "n_features": result.get("n_features", 0),
    }


def run_ablation():
    games, batting, pitching, pitcher_stats = load_data()

    print("=" * 80)
    print("PHASE 1: BASELINE (all features)")
    print("=" * 80)

    all_result = eval_groups(games, batting, pitching, pitcher_stats,
                             ALL_GROUPS, TEST_SEASONS)
    print(f"All groups ({len(ALL_GROUPS)}): Acc={all_result['accuracy']:.4f}  "
          f"LL={all_result['log_loss']:.4f}  Brier={all_result['brier_score']:.4f}  "
          f"Features={all_result.get('n_features', '?')}")

    # -----------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("PHASE 2: DROP-ONE ABLATION")
    print("Each row shows what happens when we REMOVE that group.")
    print(f"{'=' * 80}")
    print(f"{'Group':<22} {'Accuracy':>10} {'Log Loss':>10} {'Brier':>10} "
          f"{'Acc Delta':>10} {'LL Delta':>10}")
    print("-" * 72)

    drop_results = {}
    for group in ALL_GROUPS:
        groups_without = [g for g in ALL_GROUPS if g != group]
        result = eval_groups(games, batting, pitching, pitcher_stats,
                             groups_without, TEST_SEASONS)
        drop_results[group] = result
        acc_delta = result["accuracy"] - all_result["accuracy"]
        ll_delta = result["log_loss"] - all_result["log_loss"]
        # Positive acc_delta = removing group HELPED (group was hurting)
        # Negative acc_delta = removing group HURT (group was helping)
        marker = " <-- hurts!" if acc_delta > 0.001 else ""
        print(f"{group:<22} {result['accuracy']:>10.4f} {result['log_loss']:>10.4f} "
              f"{result['brier_score']:>10.4f} {acc_delta:>+10.4f} {ll_delta:>+10.4f}{marker}")

    # -----------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("PHASE 3: ADD-ONE ABLATION")
    print("Starting from Elo only, add each group individually.")
    print(f"{'=' * 80}")

    base_result = eval_groups(games, batting, pitching, pitcher_stats,
                              BASE_GROUPS, TEST_SEASONS)
    print(f"{'Base (elo only)':<22} Acc={base_result['accuracy']:.4f}  "
          f"LL={base_result['log_loss']:.4f}")
    print()
    print(f"{'Group Added':<22} {'Accuracy':>10} {'Log Loss':>10} {'Brier':>10} "
          f"{'Acc Lift':>10} {'LL Lift':>10}")
    print("-" * 72)

    add_results = {}
    for group in ALL_GROUPS:
        if group in BASE_GROUPS:
            continue
        groups_with = BASE_GROUPS + [group]
        result = eval_groups(games, batting, pitching, pitcher_stats,
                             groups_with, TEST_SEASONS)
        add_results[group] = result
        acc_lift = result["accuracy"] - base_result["accuracy"]
        ll_lift = result["log_loss"] - base_result["log_loss"]
        marker = " ***" if acc_lift > 0.003 else ""
        print(f"+ {group:<20} {result['accuracy']:>10.4f} {result['log_loss']:>10.4f} "
              f"{result['brier_score']:>10.4f} {acc_lift:>+10.4f} {ll_lift:>+10.4f}{marker}")

    # -----------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("SUMMARY: Feature Group Rankings")
    print(f"{'=' * 80}")

    # Rank by drop-one impact (most negative acc_delta = most valuable)
    rankings = []
    for group in ALL_GROUPS:
        if group in drop_results:
            acc_delta = drop_results[group]["accuracy"] - all_result["accuracy"]
            ll_delta = drop_results[group]["log_loss"] - all_result["log_loss"]
            rankings.append((group, acc_delta, ll_delta))

    rankings.sort(key=lambda x: x[1])  # Most negative first = most valuable
    print(f"\n{'Group':<22} {'Drop Acc Impact':>15} {'Drop LL Impact':>15} {'Verdict':>12}")
    print("-" * 64)
    for group, acc_d, ll_d in rankings:
        if acc_d < -0.001:
            verdict = "KEEP"
        elif acc_d > 0.001:
            verdict = "REMOVE"
        else:
            verdict = "NEUTRAL"
        print(f"{group:<22} {acc_d:>+15.4f} {ll_d:>+15.4f} {verdict:>12}")

    return all_result, drop_results, add_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    run_ablation()
