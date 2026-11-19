"""Step 1: from the raw 1-second Bybit book, build features and future returns for each coin.

    python src/features.py            # all coins (run as a Slurm job, see slurm/)
    python src/features.py SOLUSDT    # one coin

Every second t we know the top 5 bid and ask levels (price, size) of the book at t.
All features at t use only the book up to and including t.

Own-book features (prices in basis points, bps):
  spread     (ask - bid) / mid
  imb1       (bid size - ask size) / (bid size + ask size), best level
  imb5       same, summed over the top 5 levels
  micro      size-weighted mid ("microprice") minus mid
  depth      log $ value of the top 5 levels, minus its 1-hour average
  activity   log number of book updates in the last 10 s, minus its 1-hour average
  ofi_k      order-flow imbalance at the best level (Cont, Kukanov & Stoikov 2014), summed over the
             last k = 1, 5, 30 seconds, divided by the 1-hour average best-level size.
             Positive = buying pressure. This is our stand-in for signed trade flow (no trade data).
  ofi_deep   same idea for levels 2-5, last 5 seconds
  ret_k      log mid return over the last k = 1, 5, 30, 60, 300 seconds
  dev_60     mid minus its 60-second average
  rv_k       realized volatility: sqrt of the sum of squared 1-second returns, last k = 60, 600 s

Targets: fwd_h = log mid return from t to t+h, in bps, h = 5, 30, 60.

A second is "bad" if Bybit flags the book as invalid or restarted, or the book is crossed or empty.
Features at t need 600 good seconds before t; the target needs good seconds from t to t+h+1.

Output, in data/features/<coin>/:
  grid.parquet     one row every 5 s: features, targets, and bid/ask for the trading simulation
  seconds.parquet  every second: the few inputs of the sequence model (config.SEQUENCE)
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

from config import BOOKS, DELAY, FEATURES, FIRST_DAY, HORIZONS, LAST_DAY, STEP, COINS

LEVELS = 5
HOUR = 3600
LOOKBACK = 600         # longest look-back (rv_600)


# --------------------------------------------------------------------------- raw book

def load_day(coin, day):
    """One day of the book, 86,400 rows. Prices are stored as integer ticks: the best bid, and every
    other level as an offset from it (see crypto_project/orderbook_data/common.py)."""
    df = pd.read_parquet(BOOKS / coin / f"{day}.parquet")
    scale = 10.0 ** df["px_decimals"].to_numpy("float64")
    best = df["bid_px_0"].to_numpy("float64")
    out = {"ts": df["ts"].to_numpy()}
    for i in range(LEVELS):
        out[f"bid_px_{i}"] = (best - (df[f"bid_px_{i}"].to_numpy("float64") if i else 0)) / scale
        out[f"ask_px_{i}"] = (best + df[f"ask_px_{i}"].to_numpy("float64")) / scale
        out[f"bid_sz_{i}"] = df[f"bid_sz_{i}"].to_numpy("float64")
        out[f"ask_sz_{i}"] = df[f"ask_sz_{i}"].to_numpy("float64")
    out["n_updates"] = df["n_updates"].to_numpy("float64")
    out["good"] = (df["valid"] & ~df["restart"]).to_numpy()
    return pd.DataFrame(out)


def load_book(coin):
    """All days on one continuous 1-second clock. Missing days become bad rows."""
    days = pd.date_range(FIRST_DAY, LAST_DAY).strftime("%Y-%m-%d")
    book = pd.concat([load_day(coin, d) for d in days if (BOOKS / coin / f"{d}.parquet").exists()])
    clock = pd.date_range(FIRST_DAY, pd.Timestamp(LAST_DAY) + pd.Timedelta("1D"), freq="1s",
                          inclusive="left", tz="UTC")
    book = book.set_index("ts").reindex(clock)
    book["good"] = book["good"].fillna(False).astype(bool)
    book["good"] &= (book.bid_px_0 > 0) & (book.ask_px_0 > book.bid_px_0)
    book["good"] &= (book.bid_sz_0 > 0) & (book.ask_sz_0 > 0)
    return book


# --------------------------------------------------------------------------- features

def rolling_sum(x, k):
    return x.rolling(k, min_periods=k).sum()


def minus_hour_mean(x):
    return x - x.rolling(HOUR, min_periods=LOOKBACK).mean()


def order_flow(b, b_prev, qb, qb_prev, a, a_prev, qa, qa_prev):
    """Cont-Kukanov-Stoikov order-flow imbalance between two snapshots of one book level."""
    return (np.where(b >= b_prev, qb, 0) - np.where(b <= b_prev, qb_prev, 0)
            - np.where(a <= a_prev, qa, 0) + np.where(a >= a_prev, qa_prev, 0))


def ofi(book, level):
    b, qb = book[f"bid_px_{level}"], book[f"bid_sz_{level}"]
    a, qa = book[f"ask_px_{level}"], book[f"ask_sz_{level}"]
    e = order_flow(b, b.shift(), qb, qb.shift(), a, a.shift(), qa, qa.shift())
    return pd.Series(e, index=book.index)


def build_features(book):
    """All features and targets, every second. Bad seconds are NaN."""
    good = book.good.astype(float)
    book = book.drop(columns="good").where(book.good)   # bad rows -> NaN, so nothing is computed from them
    bid, ask = book.bid_px_0, book.ask_px_0
    qb, qa = book.bid_sz_0, book.ask_sz_0
    mid = (bid + ask) / 2
    log_mid = np.log(mid) * 1e4           # in bps, so differences are returns in bps
    ret_1 = log_mid.diff()

    bid_size = sum(book[f"bid_sz_{i}"] for i in range(LEVELS))
    ask_size = sum(book[f"ask_sz_{i}"] for i in range(LEVELS))
    dollars = sum(book[f"bid_px_{i}"] * book[f"bid_sz_{i}"] + book[f"ask_px_{i}"] * book[f"ask_sz_{i}"]
                  for i in range(LEVELS))
    best_size = ((qb + qa) / 2).rolling(HOUR, min_periods=LOOKBACK).mean()
    deep_size = (sum((book[f"bid_sz_{i}"] + book[f"ask_sz_{i}"]) / 2 for i in range(1, LEVELS))
                 .rolling(HOUR, min_periods=LOOKBACK).mean())
    ofi_1 = ofi(book, 0) / best_size
    ofi_deep = sum(ofi(book, i) for i in range(1, LEVELS)) / deep_size

    f = pd.DataFrame(index=book.index)
    f["spread"] = (ask - bid) / mid * 1e4
    f["imb1"] = (qb - qa) / (qb + qa)
    f["imb5"] = (bid_size - ask_size) / (bid_size + ask_size)
    f["micro"] = ((ask * qb + bid * qa) / (qb + qa) - mid) / mid * 1e4
    f["depth"] = minus_hour_mean(np.log(dollars))
    f["activity"] = minus_hour_mean(np.log1p(rolling_sum(book.n_updates, 10)))
    f["ofi_1"] = ofi_1
    f["ofi_5"] = rolling_sum(ofi_1, 5)
    f["ofi_30"] = rolling_sum(ofi_1, 30)
    f["ofi_deep"] = rolling_sum(ofi_deep, 5)
    for k in [1, 5, 30, 60, 300]:
        f[f"ret_{k}"] = log_mid - log_mid.shift(k)
    f["dev_60"] = log_mid - log_mid.rolling(60, min_periods=60).mean()
    f["rv_60"] = np.sqrt(rolling_sum(ret_1 ** 2, 60))
    f["rv_600"] = np.sqrt(rolling_sum(ret_1 ** 2, 600))

    # Features need LOOKBACK good seconds before t (rolling sums above are NaN otherwise).
    f.loc[rolling_sum(good, LOOKBACK + 1) < LOOKBACK + 1] = np.nan

    # Targets, and the prices a trade would get: enter at t+DELAY, leave at t+DELAY+h.
    for h in HORIZONS:
        window_good = rolling_sum(good, h + DELAY + 1).shift(-(h + DELAY)) == h + DELAY + 1
        f[f"fwd_{h}"] = (log_mid.shift(-h) - log_mid).where(window_good)
        f[f"exit_bid_{h}"] = bid.shift(-(h + DELAY))
        f[f"exit_ask_{h}"] = ask.shift(-(h + DELAY))
    f["entry_bid"] = bid.shift(-DELAY)
    f["entry_ask"] = ask.shift(-DELAY)
    return f


# --------------------------------------------------------------------------- main

def build_coin(coin):
    book = load_book(coin)
    f = build_features(book)
    out = FEATURES / coin
    out.mkdir(parents=True, exist_ok=True)

    seq_cols = ["ret_1", "spread", "imb1", "imb5", "micro", "ofi_1"]
    seconds = f[seq_cols].astype("float32")
    seconds.index.name = "ts"
    seconds.reset_index().to_parquet(out / "seconds.parquet", index=False)

    on_grid = (f.index.second % STEP == 0) & (f.index >= pd.Timestamp(FIRST_DAY, tz="UTC") + pd.Timedelta("1D"))
    grid = f[on_grid].dropna(subset=["ret_300", "rv_600"])
    grid = grid.astype({c: "float64" if "bid" in c or "ask" in c else "float32" for c in grid.columns})
    grid.index.name = "ts"
    grid.reset_index().to_parquet(out / "grid.parquet", index=False)

    bad = 1 - book.good.mean()
    print(f"{coin}: {len(grid):,} grid rows, {bad:.2%} bad seconds", flush=True)


if __name__ == "__main__":
    coins = sys.argv[1:] or COINS
    workers = min(len(coins), int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    with Pool(workers) as pool:
        pool.map(build_coin, coins, chunksize=1)
