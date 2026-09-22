"""
main.py
--------
Command-line entry point for the jump-return analysis toolkit.

USAGE
-----
  # Fetch live data for a ticker and run full analysis:
  python main.py --ticker AAPL --start 2015-01-01

  # Use a local CSV instead of live data:
  python main.py --csv my_stock_data.csv --ticker MYSTOCK

  # Run on synthetic data (no network needed) to see how it works:
  python main.py --synthetic --ticker SYNTH

OUTPUT
------
Creates a folder `output_<TICKER>/` containing:
  - jump_summary.txt          : jump detection stats
  - forward_return_stats.csv  : full statistical test results by horizon/direction
  - backtest_results.csv      : per-fold walk-forward backtest results
  - backtest_summary.txt      : headline backtest numbers
  - price_with_jumps.png
  - forward_return_dist.png
  - backtest_accuracy.png
"""

import argparse
import os
import sys
import pandas as pd

from data_loader import load_from_yahoo, load_from_csv, generate_synthetic
from jump_detector import detect_jumps, jump_summary
from forward_returns import analyze_jump_outcomes, interpret_results
from features import build_features, FEATURE_COLUMNS
from backtest import walk_forward_backtest, summarize_backtest
from visualize import plot_price_with_jumps, plot_forward_return_distributions, plot_backtest_accuracy


def main():
    parser = argparse.ArgumentParser(description="Jump-based stock return analysis and backtest tool")
    parser.add_argument("--ticker", type=str, default="STOCK", help="Ticker symbol (for labeling output)")
    parser.add_argument("--csv", type=str, default=None, help="Path to local CSV file with OHLCV data")
    parser.add_argument("--synthetic", action="store_true", help="Use generated synthetic data instead of real data")
    parser.add_argument("--start", type=str, default="2015-01-01", help="Start date for Yahoo Finance fetch")
    parser.add_argument("--end", type=str, default=None, help="End date for Yahoo Finance fetch")
    parser.add_argument("--z-threshold", type=float, default=2.5, help="Z-score threshold for jump detection")
    parser.add_argument("--vol-window", type=int, default=20, help="Rolling window (days) for volatility estimate")
    parser.add_argument("--horizon", type=int, default=5, help="Forward horizon (days) for the ML backtest target")
    parser.add_argument("--model", type=str, default="logistic", choices=["logistic", "xgboost"],
                         help="Model type for the walk-forward backtest")
    parser.add_argument("--train-window", type=int, default=500, help="Training window size (days) for backtest")
    parser.add_argument("--test-window", type=int, default=60, help="Test window size (days) for backtest")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory (default: output_<TICKER>)")
    args = parser.parse_args()

    outdir = args.outdir or f"output_{args.ticker}"
    os.makedirs(outdir, exist_ok=True)

    # ---- 1. Load data ----
    print(f"[1/5] Loading data for {args.ticker}...")
    if args.synthetic:
        df = generate_synthetic()
        print(f"  Using synthetic data ({len(df)} rows).")
    elif args.csv:
        df = load_from_csv(args.csv)
        print(f"  Loaded {len(df)} rows from {args.csv}")
    else:
        try:
            df = load_from_yahoo(args.ticker, start=args.start, end=args.end)
            print(f"  Fetched {len(df)} rows from Yahoo Finance.")
        except Exception as e:
            print(f"  ERROR fetching from Yahoo Finance: {e}", file=sys.stderr)
            print("  Tip: if you're behind a restrictive network/sandbox, try --csv or --synthetic instead.",
                  file=sys.stderr)
            sys.exit(1)

    if len(df) < args.train_window + args.test_window + 100:
        print(f"  WARNING: only {len(df)} rows of data — results may be unreliable with such a short history. "
              f"Consider a longer date range.")

    # ---- 2. Detect jumps ----
    print("[2/5] Detecting jumps...")
    df = detect_jumps(df, vol_window=args.vol_window, z_threshold=args.z_threshold)
    summary = jump_summary(df)
    with open(os.path.join(outdir, "jump_summary.txt"), "w") as f:
        f.write(f"Jump Detection Summary for {args.ticker}\n")
        f.write(f"(z-threshold={args.z_threshold}, vol_window={args.vol_window} days)\n\n")
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
    for k, v in summary.items():
        print(f"    {k}: {v}")

    # ---- 3. Forward return statistics ----
    print("[3/5] Analyzing forward returns after jumps...")
    fwd_results = analyze_jump_outcomes(df)
    fwd_results.to_csv(os.path.join(outdir, "forward_return_stats.csv"), index=False)
    interpretation = interpret_results(fwd_results)
    with open(os.path.join(outdir, "forward_return_stats.csv").replace(".csv", "_interpretation.txt"), "w") as f:
        f.write(interpretation)
    print("    " + interpretation.replace("\n", "\n    "))

    # ---- 4. Walk-forward ML backtest ----
    print(f"[4/5] Running walk-forward backtest (model={args.model}, horizon={args.horizon}d)...")
    df_feat = build_features(df)
    try:
        bt_results = walk_forward_backtest(
            df_feat, FEATURE_COLUMNS, horizon=args.horizon,
            train_window=args.train_window, test_window=args.test_window,
            model_type=args.model
        )
        bt_results.to_csv(os.path.join(outdir, "backtest_results.csv"), index=False)
        bt_summary = summarize_backtest(bt_results)
        with open(os.path.join(outdir, "backtest_summary.txt"), "w") as f:
            for k, v in bt_summary.items():
                f.write(f"{k}: {v}\n")
        print("    Backtest summary:")
        for k, v in bt_summary.items():
            print(f"      {k}: {v}")
    except Exception as e:
        print(f"    Backtest could not be run: {e}")
        bt_results = pd.DataFrame()

    # ---- 5. Plots ----
    print("[5/5] Generating plots...")
    plot_price_with_jumps(df, os.path.join(outdir, "price_with_jumps.png"), ticker=args.ticker)
    plot_forward_return_distributions(df, os.path.join(outdir, "forward_return_dist.png"), horizon=args.horizon)
    if not bt_results.empty:
        plot_backtest_accuracy(bt_results, os.path.join(outdir, "backtest_accuracy.png"))

    print(f"\nDone. All outputs saved to: {outdir}/")


if __name__ == "__main__":
    main()
