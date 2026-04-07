"""
Ensemble model: weighted average of Logistic Regression, XGBoost, LightGBM.
Dynamically uses all available numeric feature columns.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
import xgboost as xgb
import lightgbm as lgb

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.features import get_feature_columns


def _get_clean_data(df: pd.DataFrame, feature_cols: list[str]):
    """Drop rows where any feature or target is NaN."""
    cols = feature_cols + ["home_win"]
    mask = df[cols].notna().all(axis=1)
    return df[mask], feature_cols


def train_and_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str] | None = None,
    weights: dict[str, float] | None = None,
    lr_params: dict | None = None,
    xgb_params: dict | None = None,
    lgb_params: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Train LR + XGBoost + LightGBM, return weighted average predictions.
    """
    weights = weights or {"lr": 0.2, "xgb": 0.4, "lgbm": 0.4}

    if feature_cols is None:
        feature_cols = get_feature_columns(train)

    # For tree models: use raw data (they handle NaN natively)
    X_train_raw = train[feature_cols].values
    y_train_raw = train["home_win"].values
    X_test_raw = test[feature_cols].values

    # For LR: fill NaN with training median
    train_medians = train[feature_cols].median()
    X_train_filled = train[feature_cols].fillna(train_medians).values
    y_train_filled = train["home_win"].values
    X_test_filled = test[feature_cols].fillna(train_medians).values

    # --- Logistic Regression ---
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_filled)
    X_test_sc = scaler.transform(X_test_filled)

    lr_default = {"max_iter": 1000, "C": 1.0}
    if lr_params:
        lr_default.update(lr_params)
    lr_model = LogisticRegression(**lr_default)
    lr_model.fit(X_train_sc, y_train_filled)
    lr_prob = lr_model.predict_proba(X_test_sc)[:, 1]

    # --- XGBoost ---
    xgb_default = {
        "n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 10,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "objective": "binary:logistic", "eval_metric": "logloss",
        "random_state": 42, "verbosity": 0,
    }
    if xgb_params:
        xgb_default.update(xgb_params)
    xgb_model = xgb.XGBClassifier(**xgb_default)
    xgb_model.fit(X_train_raw, y_train_raw)
    xgb_prob = xgb_model.predict_proba(X_test_raw)[:, 1]

    # --- LightGBM ---
    lgb_default = {
        "n_estimators": 400, "max_depth": 4, "learning_rate": 0.03,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 20,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "objective": "binary", "metric": "binary_logloss",
        "random_state": 42, "verbose": -1,
    }
    if lgb_params:
        lgb_default.update(lgb_params)
    lgb_model = lgb.LGBMClassifier(**lgb_default)
    lgb_model.fit(X_train_raw, y_train_raw)
    lgb_prob = lgb_model.predict_proba(X_test_raw)[:, 1]

    # --- Ensemble ---
    test_out = test.copy()
    test_out["pred_prob_lr"] = lr_prob
    test_out["pred_prob_xgb"] = xgb_prob
    test_out["pred_prob_lgbm"] = lgb_prob

    test_out["pred_prob"] = (
        weights["lr"] * lr_prob +
        weights["xgb"] * xgb_prob +
        weights["lgbm"] * lgb_prob
    )
    test_out["pred_win"] = (test_out["pred_prob"] >= 0.5).astype(int)

    # Feature importance from XGBoost
    importances = dict(zip(feature_cols, xgb_model.feature_importances_))

    models = {
        "lr": lr_model, "xgb": xgb_model, "lgbm": lgb_model,
        "scaler": scaler, "feature_cols": feature_cols,
        "importances": importances,
    }

    return test_out, models


def evaluate(train: pd.DataFrame, test: pd.DataFrame,
             feature_cols: list[str] | None = None,
             weights: dict | None = None) -> dict:
    """Train and evaluate ensemble."""
    preds, models = train_and_predict(train, test, feature_cols=feature_cols,
                                      weights=weights)
    mask = preds["home_win"].notna() & preds["pred_prob"].notna()
    preds = preds[mask]

    importances = models["importances"]
    top_features = dict(sorted(importances.items(), key=lambda x: -x[1])[:15])

    return {
        "model": "Ensemble",
        "n_games": len(preds),
        "accuracy": accuracy_score(preds["home_win"], preds["pred_win"]),
        "log_loss": log_loss(preds["home_win"], preds["pred_prob"]),
        "brier_score": brier_score_loss(preds["home_win"], preds["pred_prob"]),
        "n_features": len(models["feature_cols"]),
        "top_features": top_features,
    }
