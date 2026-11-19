"""Step 2b: describe the data (the "summary statistics" table of the paper).

    python src/describe.py

Per coin, over the whole sample (Aug 3 - Oct 11, 2026), from the 5-second grid:
  price         average mid price ($)
  spread        average and median bid-ask spread, bps
  sd fwd_h      standard deviation of the h-second return, bps
  zero fwd_h    share of rows where the mid did not move over h seconds, %

Output: results/data_summary.md, results/data_summary.csv
"""
import numpy as np
import pandas as pd

from config import COINS, FEATURES, HORIZONS, RESULTS


def summarize(coin):
    cols = ["spread", "entry_bid", "entry_ask"] + [f"fwd_{h}" for h in HORIZONS]
    g = pd.read_parquet(FEATURES / coin / "grid.parquet", columns=["ts"] + cols)
    g = g[g.ts >= pd.Timestamp("2026-08-03", tz="UTC")]
    row = {
        "rows": len(g),
        "price ($)": ((g.entry_bid + g.entry_ask) / 2).mean(),
        "spread mean (bps)": g.spread.mean(),
        "spread median (bps)": g.spread.median(),
    }
    for h in HORIZONS:
        r = g[f"fwd_{h}"].dropna()
        row[f"sd fwd_{h} (bps)"] = r.std()
        row[f"zero fwd_{h} (%)"] = (r == 0).mean() * 100
    return row


def main():
    table = pd.DataFrame({coin.removesuffix("USDT"): summarize(coin) for coin in COINS}).T
    table.loc["median"] = table.median()
    table.to_csv(RESULTS / "data_summary.csv", float_format="%.4f")
    fmt = table.copy()
    fmt["rows"] = fmt["rows"].map("{:,.0f}".format)
    fmt["price ($)"] = fmt["price ($)"].map(lambda p: f"{p:,.4g}")
    text = ["# Data summary (Aug 3 - Oct 11, 2026, one row every 5 s)\n",
            "spread = (ask - bid) / mid. sd fwd_h = sd of the h-second mid return. "
            "zero fwd_h = share of rows where the mid did not change over h seconds.\n",
            fmt.to_markdown(floatfmt=".2f")]
    (RESULTS / "data_summary.md").write_text("\n".join(text) + "\n")


if __name__ == "__main__":
    main()
