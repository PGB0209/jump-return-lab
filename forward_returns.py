"""
forward_returns.py
-------------------
The core question this tool asks: "After a jump of a given direction, what
happens to returns over the following N days — and is that different from
what happens on a random (non-jump) day?"

This is deliberately framed as a hypothesis test, not just a mean comparison,
because a few percentage points of difference in average forward return means
nothing if the samples are noisy and the sample size is small (jumps are rare
events by construction).
"""

import numpy as np
import pandas as pd
from scipy import stats


def compute_forward_returns(df: pd.DataFrame, horizons=(1, 3, 5, 10, 20)) -> pd.DataFrame:
    """
    Add forward simple-return columns for each horizon in `horizons`.
    Fwd_5 on day t = (Close[t+5] / Close[t]) - 1
    """
    out = df.copy()
    for h in horizons:
        out[f"Fwd_{h}"] = out["Close"].shift(-h) / out["Close"] - 1
    return out


def analyze_jump_outcomes(df_with_jumps: pd.DataFrame, horizons=(1, 3, 5, 10, 20)) -> pd.DataFrame:
    """
    Compare forward returns after UP jumps, DOWN jumps, and NON-jump days.
    Runs a two-sample t-test (jump days vs non-jump days) for each horizon
    and direction, reporting the mean, sample size, and p-value.

    Returns a tidy DataFrame, one row per (direction, horizon) combination.
    """
    df = compute_forward_returns(df_with_jumps, horizons)
    rows = []

    non_jump = df[~df["IsJump"]]
    up_jump = df[df["JumpDirection"] == 1]
    down_jump = df[df["JumpDirection"] == -1]

    groups = {"UP_JUMP": up_jump, "DOWN_JUMP": down_jump}

    for h in horizons:
        col = f"Fwd_{h}"
        baseline = non_jump[col].dropna()

        for label, group in groups.items():
            sample = group[col].dropna()
            if len(sample) < 5:
                rows.append({
                    "direction": label, "horizon_days": h, "n": len(sample),
                    "mean_fwd_return_pct": np.nan, "baseline_mean_pct": np.nan,
                    "t_stat": np.nan, "p_value": np.nan,
                    "note": "insufficient sample size (<5 jumps)"
                })
                continue

            t_stat, p_val = stats.ttest_ind(sample, baseline, equal_var=False)
            rows.append({
                "direction": label,
                "horizon_days": h,
                "n": len(sample),
                "mean_fwd_return_pct": round(100 * sample.mean(), 3),
                "baseline_mean_pct": round(100 * baseline.mean(), 3),
                "diff_pct": round(100 * (sample.mean() - baseline.mean()), 3),
                "t_stat": round(t_stat, 3),
                "p_value": round(p_val, 4),
                "significant_at_5pct": bool(p_val < 0.05),
                "note": ""
            })

    return pd.DataFrame(rows)


def interpret_results(results_df: pd.DataFrame) -> str:
    """Generate a plain-English interpretation of the statistical results."""
    lines = []
    sig = results_df[results_df.get("significant_at_5pct", pd.Series(dtype=bool)) == True]

    if sig.empty:
        lines.append(
            "No horizon/direction combination showed a statistically significant "
            "difference from baseline (non-jump) forward returns at the 5% level. "
            "This means the jump signal, as defined, does not show reliable predictive "
            "power over random chance in this dataset — a common and important finding, "
            "not a failure of the analysis."
        )
    else:
        for _, row in sig.iterrows():
            direction = "an upward" if row["direction"] == "UP_JUMP" else "a downward"
            momentum_or_reversal = "continuation (momentum)" if (
                (row["direction"] == "UP_JUMP" and row["diff_pct"] > 0) or
                (row["direction"] == "DOWN_JUMP" and row["diff_pct"] < 0)
            ) else "reversal"
            lines.append(
                f"After {direction} jump, the {row['horizon_days']}-day forward return "
                f"averaged {row['mean_fwd_return_pct']}% vs a baseline of {row['baseline_mean_pct']}% "
                f"(p={row['p_value']}), suggesting a {momentum_or_reversal} pattern at this horizon."
            )
    return "\n".join(lines)


if __name__ == "__main__":
    from data_loader import generate_synthetic
    from jump_detector import detect_jumps

    df = generate_synthetic(n_days=3000, seed=7)
    df = detect_jumps(df)
    results = analyze_jump_outcomes(df)
    pd.set_option("display.width", 140)
    print(results.to_string(index=False))
    print("\n--- Interpretation ---")
    print(interpret_results(results))
