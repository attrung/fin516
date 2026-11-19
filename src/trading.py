"""Step 5: does the predictability survive trading costs?

    python src/trading.py

A simple taker strategy, one coin at a time, $1 per trade:
  - every 5 s, if we hold nothing and the forecast |E[return]| is above a threshold, trade in its direction
  - the order fills DELAY = 1 s later, at the ask (buy) or bid (sell): we pay the spread
  - we close h seconds after that, again crossing the spread (sell at the bid / buy at the ask)
  - each side pays a fee; we show 0, 2 and 5.5 bps (5.5 bps = Bybit's standard taker fee)
  - no new trade in a coin until the last one is closed (no overlapping positions)
Fills are at the quoted best price, i.e. we assume our size fits at the touch.

The threshold (in bps of forecast return) is picked on validation data to maximize total profit,
separately for each model, horizon and fee, among thresholds with at least MIN_TRADES validation
trades. If none of them makes money on validation, the strategy does not trade. The chosen
threshold is then applied unchanged to the test period.

We also show, for every threshold, the average test-period return per trade after paying the
spread but before fees ("gross"). Half of it is the highest fee per side a strategy could pay
and still break even.

Output: results/trading.md, results/trading.csv
"""
import numpy as np
import pandas as pd
from numba import njit

from common import load
from config import DATASET, DELAY, FEE_GRID_BPS, HORIZONS, PREDICTIONS, RESULTS, TAKER_FEE_BPS

MODELS = {"Ridge": "ridge", "LightGBM": "lgbm", "MLP": "mlp", "GRU": "gru"}
THRESHOLDS = [0, 0.5, 1, 2, 3, 4, 6, 8, 11, 15, 20, 30]     # bps
MIN_TRADES = 200


@njit
def simulate(second, coin, forecast, entry_bid, entry_ask, exit_bid, exit_ask, threshold, hold):
    """Return (in bps, before fees) of the trade opened at each row; NaN where no trade opens.
    Rows must be sorted by coin, then time."""
    out = np.full(len(second), np.nan)
    free_at, current = -1, -1
    for i in range(len(second)):
        if coin[i] != current:
            current, free_at = coin[i], -1
        if second[i] < free_at or abs(forecast[i]) <= threshold or not np.isfinite(exit_bid[i]):
            continue
        if forecast[i] > 0:
            out[i] = np.log(exit_bid[i] / entry_ask[i]) * 1e4
        else:
            out[i] = np.log(entry_bid[i] / exit_ask[i]) * 1e4
        free_at = second[i] + hold
    return out


def load_split(split, h):
    cols = ["ts", "coin", "entry_bid", "entry_ask", f"exit_bid_{h}", f"exit_ask_{h}", f"fwd_{h}"]
    df = load(split, cols)
    sd = pd.read_parquet(DATASET / "scales.parquet")[f"sd_fwd_{h}"]
    df["scale"] = df.coin.map(sd).astype(float)          # forecasts back to bps
    df["second"] = (df.ts - df.ts.min()) // pd.Timedelta("1s")
    df["code"] = df.coin.cat.codes
    return df


def run(df, forecast_bps, h, threshold):
    return simulate(df.second.to_numpy(np.int64), df.code.to_numpy(np.int64), forecast_bps,
                    df.entry_bid.to_numpy(), df.entry_ask.to_numpy(),
                    df[f"exit_bid_{h}"].to_numpy(), df[f"exit_ask_{h}"].to_numpy(), threshold, h + DELAY)


def summary(df, gross, fee):
    traded = np.isfinite(gross)
    net = gross[traded] - 2 * fee
    days = df.ts[traded].dt.floor("D")
    daily = pd.Series(net).groupby(days.to_numpy()).sum()
    n_days = df.ts.dt.floor("D").nunique()
    return {
        "trades": int(traded.sum()),
        "trades/coin/day": traded.sum() / df.coin.nunique() / n_days,
        "gross bps/trade": gross[traded].mean() if traded.any() else np.nan,
        "net bps/trade": net.mean() if traded.any() else np.nan,
        "win (%)": (net > 0).mean() * 100 if traded.any() else np.nan,
        "net bps/day": daily.sum() / n_days,
        "t(daily)": daily.mean() / daily.std() * np.sqrt(len(daily)) if len(daily) > 1 else np.nan,
    }


def main():
    rows, curves = [], []
    for h in HORIZONS:
        data = {s: load_split(s, h) for s in ["val", "test"]}
        for label, model in MODELS.items():
            path = {s: PREDICTIONS / s / f"{model}_h{h}.parquet" for s in data}
            if not all(p.exists() for p in path.values()):
                continue
            fc = {s: pd.read_parquet(path[s])["own+cross"].to_numpy() * data[s].scale.to_numpy() for s in data}
            val_runs = {t: run(data["val"], fc["val"], h, t) for t in THRESHOLDS}
            test_runs = {t: run(data["test"], fc["test"], h, t) for t in THRESHOLDS}
            for t, g in test_runs.items():
                curves.append({"horizon (s)": h, "model": label, "threshold (bps)": t,
                               "trades": int(np.isfinite(g).sum()), "gross bps/trade": np.nanmean(g)})
            for fee in FEE_GRID_BPS:
                profit = {t: np.nansum(g - 2 * fee) for t, g in val_runs.items()
                          if np.isfinite(g).sum() >= MIN_TRADES}
                best = max(profit, key=profit.get) if profit else None
                if best is None or profit[best] <= 0:
                    best = np.inf                           # nothing worked on validation: don't trade
                no_trades = np.full(len(data["test"]), np.nan)
                test = summary(data["test"], test_runs.get(best, no_trades), fee)
                rows.append({"horizon (s)": h, "model": label, "fee/side (bps)": fee, "threshold (bps)": best,
                             "val net bps/day": profit.get(best, 0) / data["val"].ts.dt.floor("D").nunique()} | test)
                print(rows[-1], flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "trading.csv", index=False)
    text = ["# Trading simulation (test period)\n",
            "Taker in and out at the quoted bid/ask, 1 s delay, one position per coin at a time, $1 per trade.",
            "Threshold picked on validation. \"net bps/day\" adds up all coins' trades per day.",
            f"Bybit's standard taker fee is {TAKER_FEE_BPS} bps per side.\n"]
    for fee, part in table.groupby("fee/side (bps)"):
        text += [f"\n## Fee {fee:g} bps per side\n", part.drop(columns="fee/side (bps)")
                 .to_markdown(index=False, floatfmt=".2f"), "\n"]
    curves = pd.DataFrame(curves)
    text += ["\n## Gross return per trade by threshold (test, after the spread, before fees)\n",
             "Trades in brackets. Half the gross return = the break-even fee per side.\n"]
    for h, part in curves.groupby("horizon (s)"):
        cell = part.apply(lambda r: f"{r['gross bps/trade']:.2f} ({r.trades})" if r.trades else "-", axis=1)
        text += [f"\n### {h}-second horizon\n",
                 part.assign(cell=cell).pivot(index="threshold (bps)", columns="model", values="cell")
                 .to_markdown(), "\n"]
    curves.to_csv(RESULTS / "trading_by_threshold.csv", index=False)
    (RESULTS / "trading.md").write_text("\n".join(text))


if __name__ == "__main__":
    main()
