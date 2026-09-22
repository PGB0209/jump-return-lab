"""
jump_detector.py
-----------------
Identifies "jump" days in a price series: days where the return is unusually
large relative to the stock's *recent* volatility, not just a fixed % cutoff.

Why volatility-adjusted rather than a fixed threshold (e.g. "|return| > 5%")?
A fixed threshold treats a 5% move in a normally-2%-a-day-volatile stock the
same as a 5% move in a normally-0.5%-a-day-volatile stock — but the second is
a far more unusual (and arguably more informative) event. Standardizing by
rolling volatility makes jumps comparable across different stocks and across
different volatility regimes of the same stock over time.
"""

import numpy as np
import pandas as pd


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add log return and simple return columns to the OHLCV DataFrame."""
    out = df.copy()
    out["LogReturn"] = np.log(out["Close"] / out["Close"].shift(1))
    out["SimpleReturn"] = out["Close"].pct_change()
    return out


def detect_jumps(df: pd.DataFrame, vol_window: int = 20, z_threshold: float = 2.5,
                  min_history: int = 20) -> pd.DataFrame:
    """
    Flag jump days using a rolling z-score of log returns.

    z_t = (r_t - mean(r, window)) / std(r, window)

    A day is flagged as a jump if |z_t| > z_threshold. The rolling stats are
    computed using data strictly BEFORE day t (shifted by 1) to avoid
    lookahead bias: today's classification can't depend on today's own return.

    Parameters
    ----------
    vol_window   : trailing window (days) used to estimate local volatility
    z_threshold  : number of rolling standard deviations that counts as a jump
    min_history  : minimum days of history required before flagging starts

    Returns
    -------
    DataFrame with added columns: LogReturn, RollingMean, RollingStd, ZScore,
    IsJump (bool), JumpDirection (+1 up jump, -1 down jump, 0 none)
    """
    out = compute_returns(df)

    # shift(1) so day t's threshold is based on info available *before* day t's return
    roll_mean = out["LogReturn"].shift(1).rolling(vol_window, min_periods=vol_window).mean()
    roll_std = out["LogReturn"].shift(1).rolling(vol_window, min_periods=vol_window).std()

    out["RollingMean"] = roll_mean
    out["RollingStd"] = roll_std
    out["ZScore"] = (out["LogReturn"] - roll_mean) / roll_std

    out["IsJump"] = out["ZScore"].abs() > z_threshold
    out.loc[out.index[:min_history], "IsJump"] = False  # guard against unstable early window

    out["JumpDirection"] = 0
    out.loc[out["IsJump"] & (out["ZScore"] > 0), "JumpDirection"] = 1
    out.loc[out["IsJump"] & (out["ZScore"] < 0), "JumpDirection"] = -1

    return out


def jump_summary(df_with_jumps: pd.DataFrame) -> dict:
    """Quick summary stats about detected jumps, useful as a sanity check."""
    jumps = df_with_jumps[df_with_jumps["IsJump"]]
    total_days = df_with_jumps["ZScore"].notna().sum()
    return {
        "total_trading_days_evaluated": int(total_days),
        "num_jumps": int(len(jumps)),
        "jump_rate_pct": round(100 * len(jumps) / total_days, 2) if total_days else None,
        "num_up_jumps": int((jumps["JumpDirection"] == 1).sum()),
        "num_down_jumps": int((jumps["JumpDirection"] == -1).sum()),
        "avg_up_jump_return_pct": round(100 * jumps.loc[jumps["JumpDirection"] == 1, "SimpleReturn"].mean(), 2) if (jumps["JumpDirection"] == 1).any() else None,
        "avg_down_jump_return_pct": round(100 * jumps.loc[jumps["JumpDirection"] == -1, "SimpleReturn"].mean(), 2) if (jumps["JumpDirection"] == -1).any() else None,
    }


if __name__ == "__main__":
    from data_loader import generate_synthetic

    df = generate_synthetic()
    result = detect_jumps(df)
    summary = jump_summary(result)
    print("Jump detection summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    true_jumps = df.attrs["true_jump_mask"].sum()
    print(f"\n(Ground truth: {true_jumps} jumps were embedded synthetically)")
