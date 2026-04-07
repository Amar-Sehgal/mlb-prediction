"""
XGBoost model for MLB game prediction.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss


FEATURE_COLS = [
    # Elo
    "elo_diff", "elo_prob",
    # Rolling win rates
    "win_rate_diff_10g", "win_rate_diff_30g", "win_rate_diff_60g",
    # Rolling run differentials
    "run_diff_diff_10g", "run_diff_diff_30g", "run_diff_diff_60g",
    # Rolling runs scored
    "runs_for_diff_10g", "runs_for_diff_30g", "runs_for_diff_60g",
    # Individual rolling stats
    "home_team_runs_for_10g", "away_team_runs_for_10g",
    "home_team_runs_against_10g", "away_team_runs_against_10g",
    "home_team_win_30g", "away_team_win_30g",
    # Rest
    "home_rest_days", "away_rest_days", "rest_advantage",
    # Park
    "park_factor",
    # Interleague
    "is_interleague",
]

# Season-level stat columns (added if available)
SEASON_STAT_COLS = [
    "home_bat_ops", "away_bat_ops",
    "home_bat_woba", "away_bat_woba",
    "home_bat_wrc_plus", "away_bat_wrc_plus",
    "home_pit_era", "away_pit_era",
    "home_pit_fip", "away_pit_fip",
    "home_pit_whip", "away_pit_whip",
    "home_pit_k_per_9", "away_pit_k_per_9",
]


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return feature columns that exist in the DataFrame."""
    all_cols = FEATURE_COLS + SEASON_STAT_COLS
    return [c for c in all_cols if c in df.columns]


def train_and_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str] | None = None,
    params: dict | None = None,
) -> tuple[pd.DataFrame, xgb.XGBClassifier]:
    """Train XGBoost on train set, predict on test set."""
    feature_cols = feature_cols or get_feature_cols(train)

    train_clean = train.dropna(subset=["home_win"])
    test_clean = test.copy()

    X_train = train_clean[feature_cols].values
    y_train = train_clean["home_win"].values
    X_test = test_clean[feature_cols].values

    default_params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "verbosity": 0,
    }
    if params:
        default_params.update(params)

    model = xgb.XGBClassifier(**default_params)
    model.fit(X_train, y_train)

    test_out = test.copy()
    test_out["pred_prob"] = model.predict_proba(X_test)[:, 1]
    test_out["pred_win"] = (test_out["pred_prob"] >= 0.5).astype(int)

    return test_out, model


def evaluate(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Train and evaluate XGBoost."""
    preds, model = train_and_predict(train, test)
    mask = preds["home_win"].notna() & preds["pred_prob"].notna()
    preds = preds[mask]

    feature_cols = get_feature_cols(train)
    importances = dict(zip(feature_cols, model.feature_importances_))
    top_features = dict(sorted(importances.items(), key=lambda x: -x[1])[:10])

    return {
        "model": "XGBoost",
        "n_games": len(preds),
        "accuracy": accuracy_score(preds["home_win"], preds["pred_win"]),
        "log_loss": log_loss(preds["home_win"], preds["pred_prob"]),
        "brier_score": brier_score_loss(preds["home_win"], preds["pred_prob"]),
        "top_features": top_features,
    }
