"""
data_loader.py
--------------
Loads historical OHLCV data for a ticker, either from Yahoo Finance (via
yfinance) or from a local CSV file. Returns a clean pandas DataFrame indexed
by date with columns: Open, High, Low, Close, Adj Close, Volume.

NOTE ON NETWORK ACCESS:
Fetching live data requires outbound access to Yahoo Finance's API
(query1/query2.finance.yahoo.com). If you're running this in a locked-down
sandbox, that call will fail — run it on your own machine instead, or use
load_from_csv() with a file you've downloaded manually.
"""

import pandas as pd
import numpy as np


def load_from_yahoo(ticker: str, start: str = "2015-01-01", end: str = None) -> pd.DataFrame:
    """
    Fetch daily OHLCV data for `ticker` from Yahoo Finance.

    Parameters
    ----------
    ticker : str, e.g. "AAPL"
    start  : str, "YYYY-MM-DD"
    end    : str, "YYYY-MM-DD" or None (defaults to today)

    Returns
    -------
    pd.DataFrame indexed by Date with columns Open, High, Low, Close, Volume
    """
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol and date range.")

    # yfinance sometimes returns MultiIndex columns for single tickers
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"
    return df


def load_from_csv(path: str, date_col: str = "Date") -> pd.DataFrame:
    """
    Load OHLCV data from a local CSV. Expects at least Date, Open, High, Low,
    Close, Volume columns (case-insensitive, flexible naming for 'Close'/'Adj Close').
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # normalize column names
    colmap = {}
    for c in df.columns:
        lc = c.lower().replace(" ", "")
        if lc == "date":
            colmap[c] = "Date"
        elif lc in ("open",):
            colmap[c] = "Open"
        elif lc in ("high",):
            colmap[c] = "High"
        elif lc in ("low",):
            colmap[c] = "Low"
        elif lc in ("close", "adjclose"):
            colmap[c] = "Close"
        elif lc in ("volume",):
            colmap[c] = "Volume"
    df = df.rename(columns=colmap)

    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    return df[["Open", "High", "Low", "Close", "Volume"]]


def generate_synthetic(n_days: int = 2000, seed: int = 42, jump_prob: float = 0.03,
                        jump_scale: float = 0.06, drift: float = 0.0003,
                        vol: float = 0.012) -> pd.DataFrame:
    """
    Generate a synthetic daily price series with embedded random jumps, for
    testing the analysis pipeline when live data isn't reachable.

    The process: daily log-returns are drawn from a normal distribution
    (drift, vol), with an independent Bernoulli(jump_prob) jump component of
    magnitude ~ N(0, jump_scale) added on jump days. This gives us a dataset
    with a *known* jump-generating process we can sanity-check our jump
    detector and downstream analysis against.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-01", periods=n_days)

    base_returns = rng.normal(drift, vol, n_days)
    jump_mask = rng.random(n_days) < jump_prob
    jump_sizes = rng.normal(0, jump_scale, n_days) * jump_mask

    log_returns = base_returns + jump_sizes
    log_price = np.cumsum(log_returns) + np.log(100)
    close = np.exp(log_price)

    # synthesize OHLV around close with small intraday noise
    intraday_noise = rng.normal(0, vol * 0.3, n_days)
    open_ = close * np.exp(-log_returns + intraday_noise)
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol * 0.4, n_days)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol * 0.4, n_days)))
    volume = rng.integers(1_000_000, 8_000_000, n_days).astype(float)
    volume = volume * (1 + 3 * jump_mask)  # volume spikes on jump days, common in reality

    df = pd.DataFrame({
        "Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume
    }, index=dates)
    df.index.name = "Date"
    df.attrs["true_jump_mask"] = jump_mask  # stash ground truth for validation
    return df


if __name__ == "__main__":
    df = generate_synthetic()
    print(df.head())
    print(f"\n{len(df)} rows generated, {df.attrs['true_jump_mask'].sum()} true jump days embedded.")
