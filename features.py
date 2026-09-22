"""
features.py
------------
Builds a feature matrix for the predictive model. All features are computed
using only information available AS OF the close of day t (no lookahead), so
they can be legitimately used to predict day t+1...t+N returns.
"""

import numpy as np
import pandas as pd


def build_features(df_with_jumps: pd.DataFrame) -> pd.DataFrame:
    """
    Construct predictive features from the jump-annotated OHLCV DataFrame.
    Every feature here is knowable at the close of day t.
    """
    out = df_with_jumps.copy()

    # --- Jump-specific features ---
    out["JumpZScore"] = out["ZScore"]
    out["IsJumpInt"] = out["IsJump"].astype(int)
    out["JumpDir"] = out["JumpDirection"]

    # Days since the most recent jump (captures "still reacting to a jump" state)
    jump_idx = np.where(out["IsJump"].values)[0]
    days_since = np.full(len(out), np.nan)
    last_jump = -np.inf
    for i in range(len(out)):
        if out["IsJump"].iloc[i]:
            last_jump = i
        days_since[i] = i - last_jump if last_jump != -np.inf else np.nan
    out["DaysSinceLastJump"] = days_since

    # Rolling count of jumps in trailing windows (jump clustering / volatility regime)
    out["JumpCount_20d"] = out["IsJumpInt"].rolling(20, min_periods=1).sum()
    out["JumpCount_60d"] = out["IsJumpInt"].rolling(60, min_periods=1).sum()

    # --- Volume features (jumps are often accompanied by volume spikes) ---
    out["VolumeZScore"] = (
        (out["Volume"] - out["Volume"].rolling(20).mean())
        / out["Volume"].rolling(20).std()
    )

    # --- Momentum / trend context ---
    out["Return_5d"] = out["Close"].pct_change(5)
    out["Return_10d"] = out["Close"].pct_change(10)
    out["Return_20d"] = out["Close"].pct_change(20)

    # --- Volatility regime ---
    out["Volatility_20d"] = out["LogReturn"].rolling(20).std()
    out["Volatility_60d"] = out["LogReturn"].rolling(60).std()
    out["VolRegimeRatio"] = out["Volatility_20d"] / out["Volatility_60d"]

    # --- Distance from moving averages ---
    out["SMA_20"] = out["Close"].rolling(20).mean()
    out["SMA_50"] = out["Close"].rolling(50).mean()
    out["DistFromSMA20"] = out["Close"] / out["SMA_20"] - 1
    out["DistFromSMA50"] = out["Close"] / out["SMA_50"] - 1

    return out


FEATURE_COLUMNS = [
    "JumpZScore", "IsJumpInt", "JumpDir", "DaysSinceLastJump",
    "JumpCount_20d", "JumpCount_60d", "VolumeZScore",
    "Return_5d", "Return_10d", "Return_20d",
    "Volatility_20d", "Volatility_60d", "VolRegimeRatio",
    "DistFromSMA20", "DistFromSMA50",
]


if __name__ == "__main__":
    from data_loader import generate_synthetic
    from jump_detector import detect_jumps

    df = generate_synthetic()
    df = detect_jumps(df)
    df = build_features(df)
    print(df[FEATURE_COLUMNS].dropna().head())
    print(f"\n{df[FEATURE_COLUMNS].dropna().shape[0]} usable rows after dropping warmup NaNs")
