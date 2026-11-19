"""Paths and settings shared by all scripts."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"                   # symlink to /scratch/network/an8303/fin516/data
FEATURES = DATA / "features"           # step 1: one folder of files per coin
DATASET = DATA / "dataset"             # step 2: train / val / test tables
PREDICTIONS = DATA / "predictions"     # step 3: model forecasts
RESULTS = ROOT / "results"

# Raw data: Bybit USDT perpetuals, top 5 book levels, one snapshot per second (read only).
BOOKS = Path("/scratch/network/an8303/crypto_project/orderbook_data/bybit")

# 25 liquid contracts with clean data over the whole window. BTC and ETH are also predictors.
COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "ADAUSDT",
         "AVAXUSDT", "LINKUSDT", "SUIUSDT", "LTCUSDT", "BCHUSDT", "DOTUSDT", "NEARUSDT",
         "UNIUSDT", "AAVEUSDT", "HYPEUSDT", "ENAUSDT", "TAOUSDT", "ARBUSDT", "1000PEPEUSDT",
         "XLMUSDT", "HBARUSDT", "ONDOUSDT", "TRXUSDT"]
LEADERS = ["BTCUSDT", "ETHUSDT"]

# Time window (UTC). The first day only warms up the look-back features.
FIRST_DAY, LAST_DAY = "2026-08-02", "2026-10-11"
SPLITS = {                              # [start, end), chronological, the same for every coin
    "train": ("2026-08-03", "2026-09-14"),   # 6 weeks
    "val": ("2026-09-14", "2026-09-28"),     # 2 weeks: pick model settings and trading thresholds
    "test": ("2026-09-28", "2026-10-12"),    # 2 weeks: touched once, for the final numbers
}

HORIZONS = [5, 30, 60]    # seconds ahead
STEP = 5                  # one observation every 5 seconds (features are still built every second)

# Features (definitions in features.py). 18 from the coin's own book + 8 from BTC and ETH.
OWN = ["spread", "imb1", "imb5", "micro", "depth", "activity",
       "ofi_1", "ofi_5", "ofi_30", "ofi_deep",
       "ret_1", "ret_5", "ret_30", "ret_60", "ret_300", "dev_60", "rv_60", "rv_600"]
CROSS = [f"{c}_{f}" for c in ["btc", "eth"] for f in ["ret_1", "ret_5", "ret_30", "ofi_5"]]
FEATURE_SETS = {"own": OWN, "own+cross": OWN + CROSS}

# What the sequence model sees for each of the past 60 seconds
SEQUENCE = ["ret_1", "spread", "imb1", "imb5", "micro", "ofi_1", "btc_ret_1", "eth_ret_1"]
SEQ_LEN = 60

# Trading costs. Bybit standard taker fee on USDT perpetuals: 0.055% per side.
TAKER_FEE_BPS = 5.5
FEE_GRID_BPS = [0.0, 2.0, 5.5]   # also show cheaper (VIP / rebate) fee levels
DELAY = 1                        # seconds between the forecast and the order filling


def split_of(ts):
    """'train', 'val', 'test' or None for each timestamp."""
    out = pd.Series(None, index=ts.index, dtype=object)
    for name, (start, end) in SPLITS.items():
        out[(ts >= pd.Timestamp(start, tz="UTC")) & (ts < pd.Timestamp(end, tz="UTC"))] = name
    return out
