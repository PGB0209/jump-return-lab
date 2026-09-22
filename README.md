# Jump-Based Stock Return Analysis Toolkit

A research/backtesting toolkit that asks a specific, testable question:

> **After a stock jumps (an unusually large price move relative to its own recent volatility), what happens to its returns over the following days — and is that pattern real, or just noise?**

It does **not** claim to predict the future. It's built to rigorously test whether a jump-based signal has any historical edge at all, using proper statistical testing and time-series-safe backtesting — and to tell you honestly when it doesn't.

## How it works

1. **`data_loader.py`** — loads OHLCV data from Yahoo Finance, a local CSV, or generates synthetic data for testing.
2. **`jump_detector.py`** — flags "jump" days using a rolling **volatility-adjusted z-score** (not a fixed % threshold), so a 5% move is judged relative to that stock's own recent typical volatility.
3. **`forward_returns.py`** — for each jump, computes forward returns over several horizons (1, 3, 5, 10, 20 days) and runs a **two-sample t-test** against non-jump days, so you get a p-value, not just a "looks different" eyeball comparison.
4. **`features.py`** — builds a feature set (jump z-score, days since last jump, jump clustering, volume anomaly, momentum, volatility regime, distance from moving averages) — every feature is computable using only information available as of the day's close.
5. **`backtest.py`** — runs a **walk-forward backtest**: trains a classifier (logistic regression or XGBoost) on a rolling window, tests on the next unseen window, and repeats. This avoids the lookahead bias of a random train/test split. Compares the model against a "naive always predict up" baseline and a coin flip.
6. **`visualize.py`** — generates diagnostic plots.
7. **`main.py`** — CLI that runs the full pipeline end to end.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Real ticker, live Yahoo Finance data
python main.py --ticker AAPL --start 2015-01-01

# From a CSV you already have (columns: Date, Open, High, Low, Close, Volume)
python main.py --csv my_data.csv --ticker MYSTOCK

# No network needed — synthetic data, useful to see how the tool behaves
python main.py --synthetic --ticker SYNTH

# Tune the jump sensitivity and backtest model
python main.py --ticker TSLA --z-threshold 2.0 --model xgboost --horizon 10
```

### Key arguments

| Flag | Meaning | Default |
|---|---|---|
| `--z-threshold` | How many rolling std-devs counts as a "jump" | 2.5 |
| `--vol-window` | Days used to estimate local volatility | 20 |
| `--horizon` | Forward days the ML model tries to predict | 5 |
| `--model` | `logistic` or `xgboost` | logistic |
| `--train-window` / `--test-window` | Walk-forward fold sizes (days) | 500 / 60 |

## Output

Running `main.py` creates `output_<TICKER>/` with:
- `jump_summary.txt` — how many jumps found, up vs down, average magnitude
- `forward_return_stats.csv` + `..._interpretation.txt` — the statistical test results, in plain English
- `backtest_results.csv` / `backtest_summary.txt` — per-fold and aggregate walk-forward performance vs baselines
- Three PNG charts: price with jumps marked, forward-return distributions, and walk-forward accuracy over time

## Reading the results honestly

- **If `significant_at_5pct` is False everywhere** (very common), that's a real, useful finding: this particular signal, on this stock, over this period, isn't distinguishable from noise. Don't force a "story" onto a null result.
- **If the model "beats" the naive baseline in a backtest**, check `model_beats_naive_pct_of_folds` — a good sign only if it's consistently better across *most* folds, not just one lucky period. Cherry-picking one favorable date range is the single most common way people fool themselves with backtests.
- **Transaction costs and slippage are NOT modeled here.** A signal with a small edge before costs can easily be worthless (or a net loser) after real-world trading costs, bid-ask spreads, and slippage — especially at short horizons.
- **Past patterns are not guarantees.** Markets adapt; a pattern with real historical significance can still stop working once enough capital tries to exploit it.

## Extending this

Some natural next steps, roughly in order of effort:
- Add sector/market-relative jumps (is the jump specific to this stock, or is the whole market moving?)
- Add earnings-date awareness (jumps around earnings behave very differently from unexplained jumps)
- Add transaction cost modeling to the backtest
- Test across a basket of tickers rather than one at a time, to see if any pattern generalizes
- Add short-selling to the backtest (currently long-only)

## Disclaimer

This is a research and educational tool, not investment advice. It is not a licensed financial product, and its outputs (including the backtest results) should not be relied on as a signal for actual trading decisions without significant further validation, out-of-sample testing, and your own risk assessment.
