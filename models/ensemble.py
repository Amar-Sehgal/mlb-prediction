"""
Ensemble model: weighted average of multiple model predictions.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

from models import logistic, xgboost_model, lgbm_model


def train_and_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Train all sub-models and combine predictions with weights.
    Default weights: LR=0.2, XGB=0.4, LGBM=0.4
    """
    weights = weights or {"lr": 0.2, "xgb": 0.4, "lgbm": 0.4}

    # Train each model
    lr_preds, _ = logistic.train_and_predict(train, test)
    xgb_preds, _ = xgboost_model.train_and_predict(train, test)
    lgb_preds, _ = lgbm_model.train_and_predict(train, test)

    test_out = test.copy()
    test_out["pred_prob_lr"] = lr_preds["pred_prob"]
    test_out["pred_prob_xgb"] = xgb_preds["pred_prob"]
    test_out["pred_prob_lgbm"] = lgb_preds["pred_prob"]

    # Weighted average
    test_out["pred_prob"] = (
        weights["lr"] * test_out["pred_prob_lr"].fillna(0.5) +
        weights["xgb"] * test_out["pred_prob_xgb"].fillna(0.5) +
        weights["lgbm"] * test_out["pred_prob_lgbm"].fillna(0.5)
    )
    test_out["pred_win"] = (test_out["pred_prob"] >= 0.5).astype(int)

    return test_out


def evaluate(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Train and evaluate ensemble."""
    preds = train_and_predict(train, test)
    mask = preds["home_win"].notna() & preds["pred_prob"].notna()
    preds = preds[mask]

    return {
        "model": "Ensemble (LR+XGB+LGBM)",
        "n_games": len(preds),
        "accuracy": accuracy_score(preds["home_win"], preds["pred_win"]),
        "log_loss": log_loss(preds["home_win"], preds["pred_prob"]),
        "brier_score": brier_score_loss(preds["home_win"], preds["pred_prob"]),
    }
