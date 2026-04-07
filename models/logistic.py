"""
Logistic regression model for MLB game prediction.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss


FEATURE_COLS = [
    "elo_diff", "elo_prob",
    "win_rate_diff_10g", "win_rate_diff_30g", "win_rate_diff_60g",
    "run_diff_diff_10g", "run_diff_diff_30g", "run_diff_diff_60g",
    "rest_advantage",
    "park_factor",
]


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return feature columns that exist in the DataFrame."""
    return [c for c in FEATURE_COLS if c in df.columns]


def train_and_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, LogisticRegression]:
    """Train logistic regression on train set, predict on test set."""
    feature_cols = feature_cols or get_feature_cols(train)

    # Drop rows with NaN features
    train_clean = train.dropna(subset=feature_cols + ["home_win"])
    test_clean = test.dropna(subset=feature_cols)

    X_train = train_clean[feature_cols].values
    y_train = train_clean["home_win"].values
    X_test = test_clean[feature_cols].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_train_scaled, y_train)

    test_out = test.copy()
    test_out["pred_prob"] = np.nan
    test_out.loc[test_clean.index, "pred_prob"] = model.predict_proba(X_test_scaled)[:, 1]
    test_out["pred_win"] = (test_out["pred_prob"] >= 0.5).astype(int)

    return test_out, model


def evaluate(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Train and evaluate logistic regression."""
    preds, model = train_and_predict(train, test)
    mask = preds["home_win"].notna() & preds["pred_prob"].notna()
    preds = preds[mask]

    feature_cols = get_feature_cols(train)

    return {
        "model": "Logistic Regression",
        "n_games": len(preds),
        "accuracy": accuracy_score(preds["home_win"], preds["pred_win"]),
        "log_loss": log_loss(preds["home_win"], preds["pred_prob"]),
        "brier_score": brier_score_loss(preds["home_win"], preds["pred_prob"]),
        "features": feature_cols,
        "coefs": dict(zip(feature_cols, model.coef_[0])),
    }
