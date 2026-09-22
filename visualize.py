"""
visualize.py
-------------
Produces a small set of diagnostic plots:
  1. Price chart with jumps marked
  2. Distribution of forward returns after up-jumps vs down-jumps vs baseline
  3. Walk-forward accuracy: model vs naive-up baseline over time
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_price_with_jumps(df_with_jumps: pd.DataFrame, save_path: str, ticker: str = ""):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_with_jumps.index, df_with_jumps["Close"], color="#2b6cb0", linewidth=1, label="Close")

    up = df_with_jumps[df_with_jumps["JumpDirection"] == 1]
    down = df_with_jumps[df_with_jumps["JumpDirection"] == -1]
    ax.scatter(up.index, up["Close"], color="#2f855a", marker="^", s=50, zorder=5, label="Up jump")
    ax.scatter(down.index, down["Close"], color="#c53030", marker="v", s=50, zorder=5, label="Down jump")

    ax.set_title(f"{ticker} Price with Detected Jumps".strip())
    ax.set_ylabel("Price")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_forward_return_distributions(df_with_jumps: pd.DataFrame, save_path: str, horizon: int = 5):
    fwd_col = f"Fwd_{horizon}"
    if fwd_col not in df_with_jumps.columns:
        df_with_jumps = df_with_jumps.copy()
        df_with_jumps[fwd_col] = df_with_jumps["Close"].shift(-horizon) / df_with_jumps["Close"] - 1

    baseline = df_with_jumps.loc[~df_with_jumps["IsJump"], fwd_col].dropna() * 100
    up = df_with_jumps.loc[df_with_jumps["JumpDirection"] == 1, fwd_col].dropna() * 100
    down = df_with_jumps.loc[df_with_jumps["JumpDirection"] == -1, fwd_col].dropna() * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(
        min(baseline.min(), up.min() if len(up) else 0, down.min() if len(down) else 0),
        max(baseline.max(), up.max() if len(up) else 0, down.max() if len(down) else 0),
        40
    )
    ax.hist(baseline, bins=bins, alpha=0.4, density=True, label=f"Non-jump days (n={len(baseline)})", color="#718096")
    if len(up) >= 5:
        ax.hist(up, bins=bins, alpha=0.55, density=True, label=f"After up jump (n={len(up)})", color="#2f855a")
    if len(down) >= 5:
        ax.hist(down, bins=bins, alpha=0.55, density=True, label=f"After down jump (n={len(down)})", color="#c53030")

    ax.set_title(f"{horizon}-Day Forward Return Distribution")
    ax.set_xlabel("Forward return (%)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_backtest_accuracy(backtest_results: pd.DataFrame, save_path: str):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(backtest_results["test_start"], backtest_results["model_accuracy"],
            marker="o", label="Model accuracy", color="#2b6cb0")
    ax.plot(backtest_results["test_start"], backtest_results["naive_up_accuracy"],
            marker="o", label="Naive 'always up' baseline", color="#dd6b20", linestyle="--")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="Coin flip (50%)")

    ax.set_title("Walk-Forward Accuracy: Model vs Baseline (per test fold)")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    from data_loader import generate_synthetic
    from jump_detector import detect_jumps
    from features import build_features, FEATURE_COLUMNS
    from backtest import walk_forward_backtest

    df = generate_synthetic(n_days=1500, seed=3)
    df = detect_jumps(df)
    df_feat = build_features(df)

    plot_price_with_jumps(df, "/tmp/test_price.png", ticker="SYNTH")
    plot_forward_return_distributions(df, "/tmp/test_dist.png", horizon=5)

    bt = walk_forward_backtest(df_feat, FEATURE_COLUMNS, horizon=5, train_window=400, test_window=60)
    plot_backtest_accuracy(bt, "/tmp/test_accuracy.png")
    print("Plots generated successfully in /tmp/")
