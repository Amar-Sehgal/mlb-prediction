"""
Elo-only baseline model for MLB game prediction.
Predicts home win probability from Elo difference.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss


def predict(games: pd.DataFrame) -> pd.DataFrame:
    """Predict using Elo probability (no training needed)."""
    df = games.copy()
    df["pred_prob"] = df["elo_prob"]
    df["pred_win"] = (df["pred_prob"] >= 0.5).astype(int)
    return df


def evaluate(games: pd.DataFrame) -> dict:
    """Evaluate Elo predictions on games with results."""
    df = predict(games)
    mask = df["home_win"].notna() & df["pred_prob"].notna()
    df = df[mask]

    return {
        "model": "Elo",
        "n_games": len(df),
        "accuracy": accuracy_score(df["home_win"], df["pred_win"]),
        "log_loss": log_loss(df["home_win"], df["pred_prob"]),
        "brier_score": brier_score_loss(df["home_win"], df["pred_prob"]),
        "home_win_rate": df["home_win"].mean(),
        "pred_home_rate": df["pred_win"].mean(),
    }
