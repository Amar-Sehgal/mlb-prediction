"""
LightGBM model for MLB game prediction.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

from models.xgboost_model import get_feature_cols, FEATURE_COLS, SEASON_STAT_COLS


def train_and_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str] | None = None,
    params: dict | None = None,
) -> tuple[pd.DataFrame, lgb.LGBMClassifier]:
    """Train LightGBM on train set, predict on test set."""
    feature_cols = feature_cols or get_feature_cols(train)

    train_clean = train.dropna(subset=["home_win"])
    test_clean = test.copy()

    X_train = train_clean[feature_cols].values
    y_train = train_clean["home_win"].values
    X_test = test_clean[feature_cols].values

    default_params = {
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "binary",
        "metric": "binary_logloss",
        "random_state": 42,
        "verbose": -1,
    }
    if params:
        default_params.update(params)

    model = lgb.LGBMClassifier(**default_params)
    model.fit(X_train, y_train)

    test_out = test.copy()
    test_out["pred_prob"] = model.predict_proba(X_test)[:, 1]
    test_out["pred_win"] = (test_out["pred_prob"] >= 0.5).astype(int)

    return test_out, model


def evaluate(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Train and evaluate LightGBM."""
    preds, model = train_and_predict(train, test)
    mask = preds["home_win"].notna() & preds["pred_prob"].notna()
    preds = preds[mask]

    feature_cols = get_feature_cols(train)
    importances = dict(zip(feature_cols, model.feature_importances_))
    top_features = dict(sorted(importances.items(), key=lambda x: -x[1])[:10])

    return {
        "model": "LightGBM",
        "n_games": len(preds),
        "accuracy": accuracy_score(preds["home_win"], preds["pred_win"]),
        "log_loss": log_loss(preds["home_win"], preds["pred_prob"]),
        "brier_score": brier_score_loss(preds["home_win"], preds["pred_prob"]),
        "top_features": top_features,
    }
