"""
backtest.py
------------
Walk-forward backtest of a classifier that predicts the SIGN of the forward
N-day return, using jump-based and technical features.

Why walk-forward and not a random train/test split?
Financial time series are autocorrelated and non-stationary. A random split
lets the model "see the future" indirectly (e.g. training on day 500 while
testing on day 300), which inflates apparent accuracy in a way that will not
hold up on genuinely new data. Walk-forward validation only ever trains on
data strictly before the test window, which is the only fair simulation of
how the model would actually be used in real time.

We compare the model against two baselines:
  1. Naive "always predict up" (captures the market's natural upward drift)
  2. Random coin flip
A model is only interesting if it beats #1, not just #2.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def make_target(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """Binary target: 1 if forward `horizon`-day return is positive, else 0."""
    fwd_return = df["Close"].shift(-horizon) / df["Close"] - 1
    return (fwd_return > 0).astype(int), fwd_return


def walk_forward_backtest(df: pd.DataFrame, feature_cols: list, horizon: int = 5,
                           train_window: int = 500, test_window: int = 60,
                           model_type: str = "logistic") -> pd.DataFrame:
    """
    Roll a training window of `train_window` days forward through the data,
    testing on the next `test_window` days each time, then advancing.

    Returns a DataFrame with one row per test period, containing accuracy,
    precision, recall, and the two baseline accuracies for comparison.
    """
    target, fwd_return = make_target(df, horizon)
    data = df[feature_cols].copy()
    data["target"] = target
    data["fwd_return"] = fwd_return
    data = data.dropna()

    n = len(data)
    results = []
    start = 0

    while start + train_window + test_window <= n:
        train = data.iloc[start: start + train_window]
        test = data.iloc[start + train_window: start + train_window + test_window]

        X_train, y_train = train[feature_cols].values, train["target"].values
        X_test, y_test = test[feature_cols].values, test["target"].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        if model_type == "xgboost" and HAS_XGB:
            model = XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                verbosity=0
            )
            model.fit(X_train, y_train)  # tree models don't need scaling
            preds = model.predict(X_test)
        else:
            model = LogisticRegression(max_iter=1000, C=0.5)
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)

        # baselines
        naive_up_acc = accuracy_score(y_test, np.ones_like(y_test))
        rng = np.random.default_rng(0)
        random_preds = rng.integers(0, 2, size=len(y_test))
        random_acc = accuracy_score(y_test, random_preds)

        model_acc = accuracy_score(y_test, preds)
        # guard precision/recall against undefined case (no positive predictions)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)

        # strategy return: go long when model predicts up, flat otherwise (long-only, no shorting assumed)
        strategy_returns = test["fwd_return"].values * (preds == 1)
        buy_hold_returns = test["fwd_return"].values

        results.append({
            "test_start": test.index[0],
            "test_end": test.index[-1],
            "n_test": len(test),
            "model_accuracy": round(model_acc, 4),
            "naive_up_accuracy": round(naive_up_acc, 4),
            "random_accuracy": round(random_acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "strategy_total_return_pct": round(100 * np.nansum(strategy_returns), 3),
            "buyhold_total_return_pct": round(100 * np.nansum(buy_hold_returns), 3),
        })

        start += test_window  # advance the window

    return pd.DataFrame(results)


def summarize_backtest(results_df: pd.DataFrame) -> dict:
    """Aggregate walk-forward fold results into headline numbers."""
    if results_df.empty:
        return {"error": "No folds were run — dataset likely too short for the given window sizes."}

    return {
        "num_folds": len(results_df),
        "avg_model_accuracy": round(results_df["model_accuracy"].mean(), 4),
        "avg_naive_up_accuracy": round(results_df["naive_up_accuracy"].mean(), 4),
        "avg_random_accuracy": round(results_df["random_accuracy"].mean(), 4),
        "model_beats_naive_pct_of_folds": round(
            100 * (results_df["model_accuracy"] > results_df["naive_up_accuracy"]).mean(), 1
        ),
        "avg_precision": round(results_df["precision"].mean(), 4),
        "avg_recall": round(results_df["recall"].mean(), 4),
        "cumulative_strategy_return_pct": round(results_df["strategy_total_return_pct"].sum(), 2),
        "cumulative_buyhold_return_pct": round(results_df["buyhold_total_return_pct"].sum(), 2),
    }


if __name__ == "__main__":
    from data_loader import generate_synthetic
    from jump_detector import detect_jumps
    from features import build_features, FEATURE_COLUMNS

    df = generate_synthetic(n_days=3000, seed=11)
    df = detect_jumps(df)
    df = build_features(df)

    results = walk_forward_backtest(df, FEATURE_COLUMNS, horizon=5,
                                     train_window=500, test_window=60,
                                     model_type="logistic")
    pd.set_option("display.width", 160)
    print(results.to_string(index=False))
    print("\n--- Summary ---")
    for k, v in summarize_backtest(results).items():
        print(f"  {k}: {v}")
