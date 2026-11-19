"""Step 2: put all coins in one table and split it into train / validation / test.

    python src/dataset.py

1. Add BTC and ETH features (same timestamp) to every coin: the "cross" features.
2. Split by time. The same dates for every coin (config.SPLITS).
   Rows whose target window would reach into the next split are dropped (the last 61 s of a split).
3. Scale, using the training period only, coin by coin:
   - each feature: subtract its train mean, divide by its train sd, then clip to [-5, 5]
   - each target: divide by its train sd ("y_h"). Then every coin counts about the same in a pooled
     model. The raw return in bps stays in "fwd_h" for evaluation and trading.

Output: data/dataset/{train,val,test}.parquet and scales.parquet
"""
import numpy as np
import pandas as pd

from config import CROSS, COINS, DATASET, DELAY, FEATURES, HORIZONS, OWN, SPLITS, split_of

CLIP = 5.0
PRICES = ["entry_bid", "entry_ask"] + [f"exit_{s}_{h}" for h in HORIZONS for s in ["bid", "ask"]]


def load_grid(coin):
    return pd.read_parquet(FEATURES / coin / "grid.parquet")


def cross_features():
    """BTC and ETH features, named btc_ret_1, eth_ofi_5, ..., one row per timestamp."""
    parts = []
    for coin, prefix in [("BTCUSDT", "btc"), ("ETHUSDT", "eth")]:
        g = load_grid(coin).set_index("ts")
        names = [c.split("_", 1)[1] for c in CROSS if c.startswith(prefix)]
        parts.append(g[names].add_prefix(prefix + "_"))
    return pd.concat(parts, axis=1, join="inner")


def add_split(df):
    df["split"] = split_of(df.ts)
    last_label_second = df.ts + pd.Timedelta(seconds=max(HORIZONS) + DELAY)
    df["split_end"] = df.split.map({k: pd.Timestamp(v[1], tz="UTC") for k, v in SPLITS.items()})
    keep = df.split.notna() & (last_label_second < df.split_end)
    return df[keep].drop(columns="split_end")


def scale_coin(df):
    """Train-period scaling for one coin. Returns the scaled table and the scales used."""
    train = df[df.split == "train"]
    features = OWN + CROSS
    mean, sd = train[features].mean(), train[features].std()
    df[features] = ((df[features] - mean) / sd).clip(-CLIP, CLIP).astype("float32")
    scales = {f"mean_{c}": mean[c] for c in features} | {f"sd_{c}": sd[c] for c in features}
    for h in HORIZONS:
        sd_y = train[f"fwd_{h}"].std()
        df[f"y_{h}"] = (df[f"fwd_{h}"] / sd_y).astype("float32")
        scales[f"sd_fwd_{h}"] = sd_y
    return df, scales


def main():
    DATASET.mkdir(parents=True, exist_ok=True)
    cross = cross_features()
    tables, scales = [], {}
    for coin in COINS:
        df = load_grid(coin).join(cross, on="ts", how="inner")
        df.insert(1, "coin", coin)
        df, scales[coin] = scale_coin(add_split(df))
        tables.append(df)
        print(coin, df.split.value_counts().to_dict(), flush=True)

    data = pd.concat(tables, ignore_index=True)
    data["coin"] = data.coin.astype("category")
    for name in SPLITS:
        part = data[data.split == name].drop(columns="split")
        if name == "train":
            part = part.drop(columns=PRICES)     # prices are only needed for the trading test
        part.to_parquet(DATASET / f"{name}.parquet", index=False)
        print(name, f"{len(part):,} rows", flush=True)
    pd.DataFrame(scales).T.rename_axis("coin").to_parquet(DATASET / "scales.parquet")


if __name__ == "__main__":
    main()
