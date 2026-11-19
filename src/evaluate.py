"""Step 4: score every model's forecasts.

    python src/evaluate.py

Metrics, on the test period unless noted (validation in brackets where shown):
  R2        out-of-sample R2 against a zero forecast, in %: 1 - SSE(model) / SSE(zero).
            Pooled over all coins, on the scaled target (so each coin counts about equally).
  corr      correlation of forecast and realized return
  hit       direction accuracy: share of rows where forecast and return have the same sign
            (rows where either is exactly 0 are left out)
  coins>0   number of coins (out of 25) with a positive R2
  days>0    share of test days with a positive R2
  95% CI    from a block bootstrap: we resample whole hours (all coins together) 1,000 times,
            which keeps the dependence between nearby seconds and between coins.

Output: results/evaluation.md, results/metrics.csv, results/figs/*.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import COINS, HORIZONS, LEADERS, PREDICTIONS, RESULTS
from common import load

MODELS = {                          # label: (prediction file, feature set)
    "Zero": ("zero", "own+cross"),
    "OLS": ("ols", "own+cross"),
    "Ridge (own)": ("ridge", "own"),
    "Ridge": ("ridge", "own+cross"),
    "LightGBM (own)": ("lgbm", "own"),
    "LightGBM": ("lgbm", "own+cross"),
    "MLP": ("mlp", "own+cross"),
    "GRU": ("gru", "own+cross"),
}
BOOTSTRAP = 1000
FIGS = RESULTS / "figs"


def forecasts(split, h):
    """{model label: forecast array}, for the models whose files exist."""
    out = {}
    for label, (model, feature_set) in MODELS.items():
        path = PREDICTIONS / split / f"{model}_h{h}.parquet"
        if path.exists():
            out[label] = pd.read_parquet(path, columns=[feature_set])[feature_set].to_numpy()
    return out


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum(y ** 2)


def hour_sums(df, y, preds):
    """Per hour: sum of y^2 and each model's sum of squared errors (the bootstrap works on these)."""
    hour = df.ts.dt.floor("h").to_numpy()
    sums = pd.DataFrame({"sst": y ** 2} | {m: (y - p) ** 2 for m, p in preds.items()}).groupby(hour).sum()
    return sums


def bootstrap(sums, rng):
    """R2 of every model in BOOTSTRAP resamples of hours. Returns DataFrame (draw, model)."""
    draws = rng.integers(0, len(sums), size=(BOOTSTRAP, len(sums)))
    total = sums.to_numpy()[draws].sum(axis=1)                    # (draw, column)
    total = pd.DataFrame(total, columns=sums.columns)
    return 1 - total.drop(columns="sst").div(total.sst, axis=0)


def ci(x):
    return f"[{np.percentile(x, 2.5) * 100:.2f}, {np.percentile(x, 97.5) * 100:.2f}]"


def score(df, y, preds, rng):
    """The main table for one split and horizon."""
    days = df.ts.dt.floor("D").to_numpy()
    sums = hour_sums(df, y, preds)
    boot = bootstrap(sums, rng)
    rows = {}
    for m, p in preds.items():
        nonzero = (y != 0) & (p != 0)
        by_coin = pd.Series((y - p) ** 2).groupby(df.coin.to_numpy(), observed=True).sum() / \
            pd.Series(y ** 2).groupby(df.coin.to_numpy(), observed=True).sum()
        by_day = pd.Series((y - p) ** 2).groupby(days).sum() / pd.Series(y ** 2).groupby(days).sum()
        rows[m] = {
            "R2 (%)": r2(y, p) * 100,
            "95% CI": ci(boot[m]),
            "R2 - Ridge": f"{(r2(y, p) - r2(y, preds['Ridge'])) * 100:.2f} {ci(boot[m] - boot['Ridge'])}"
            if "Ridge" in preds else "",
            "corr": np.corrcoef(y, p)[0, 1] if p.std() > 0 else np.nan,
            "hit (%)": (np.sign(y[nonzero]) == np.sign(p[nonzero])).mean() * 100 if nonzero.any() else np.nan,
            "coins>0": int((by_coin < 1).sum()),
            "days>0 (%)": (by_day < 1).mean() * 100,
        }
    return pd.DataFrame(rows).T, boot


def per_coin(df, y, preds):
    out = {}
    for m, p in preds.items():
        g = pd.DataFrame({"coin": df.coin.to_numpy(), "e": (y - p) ** 2, "s": y ** 2}).groupby("coin", observed=True)
        out[m] = (1 - g.e.sum() / g.s.sum()) * 100
    return pd.DataFrame(out).reindex(COINS)


def cross_asset_gain(df, y, preds, rng):
    """Does adding BTC and ETH features help the other 23 coins? R2(own+cross) - R2(own)."""
    small = ~df.coin.isin(LEADERS).to_numpy()
    sub = df[small]
    sums = hour_sums(sub, y[small], {m: p[small] for m, p in preds.items()})
    boot = bootstrap(sums, rng)
    rows = {}
    for model in ["Ridge", "LightGBM"]:
        if model in preds and f"{model} (own)" in preds:
            own, both = r2(y[small], preds[f"{model} (own)"][small]), r2(y[small], preds[model][small])
            rows[model] = {"R2 own (%)": own * 100, "R2 own+cross (%)": both * 100,
                           "gain": (both - own) * 100, "95% CI": ci(boot[model] - boot[f"{model} (own)"])}
    return pd.DataFrame(rows).T


def importance_tables():
    tables = {}
    for h in HORIZONS:
        path = RESULTS / f"lgbm_importance_h{h}.csv"
        if path.exists():
            tables[f"{h}s"] = pd.read_csv(path, index_col=0)["own+cross"] * 100
    return pd.DataFrame(tables).sort_values(f"{HORIZONS[0]}s", ascending=False) if tables else None


def ridge_coefficients():
    path = RESULTS / "linear_coefficients.csv"
    if not path.exists():
        return None
    c = pd.read_csv(path, index_col=0)
    cols = {f"{h}s": f"ridge (own+cross) h={h}" for h in HORIZONS}
    return c[[v for v in cols.values()]].set_axis(list(cols), axis=1).drop("const")


def plot_r2(main):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    models = [m for m in MODELS if m != "Zero" and m in main[HORIZONS[0]].index]
    width = 0.8 / len(HORIZONS)
    for i, h in enumerate(HORIZONS):
        ax.bar(np.arange(len(models)) + i * width, main[h].loc[models, "R2 (%)"], width, label=f"{h} s")
    ax.set_xticks(np.arange(len(models)) + width, models, fontsize=9)
    ax.set_ylabel("Test R2 (%)")
    ax.set_title("Out-of-sample R2 by model and horizon", loc="left")
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.grid(axis="y", color="#e4e3df")
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "r2_by_model.png", dpi=150)
    plt.close(fig)


def plot_coins(coins, h):
    model = "LightGBM" if "LightGBM" in coins else "Ridge"
    s = coins[model].sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(s.index, s, color="#2a78d6")
    ax.set_xlabel("Test R2 (%)")
    ax.set_title(f"{model}, {h}-second horizon: test R2 by coin", loc="left")
    ax.axvline(0, color="#52514e", lw=0.8)
    ax.tick_params(labelsize=8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / f"r2_by_coin_h{h}.png", dpi=150)
    plt.close(fig)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    text = ["# Forecast results\n",
            "R2 is out-of-sample, against a zero forecast, in %. 95% CIs: hourly block bootstrap.\n"]
    main_tables, metrics = {}, []
    for h in HORIZONS:
        cols = ["ts", "coin", f"y_{h}"]
        for split in ["test", "val"]:
            df = load(split, cols)
            ok = df[f"y_{h}"].notna().to_numpy()
            df = df[ok].reset_index(drop=True)
            y = df[f"y_{h}"].to_numpy(np.float64)
            preds = {m: p[ok].astype(np.float64) for m, p in forecasts(split, h).items()}
            table, _ = score(df, y, preds, rng)
            metrics.append(table.assign(horizon=h, split=split))
            if split == "test":
                main_tables[h] = table
                text += [f"\n## {h}-second horizon\n", f"\n### Test ({df.ts.min():%b %d} - {df.ts.max():%b %d})\n",
                         table.to_markdown(floatfmt=".3f"), "\n"]
                coins = per_coin(df, y, preds)
                text += ["\n### Test R2 (%) by coin\n", coins.to_markdown(floatfmt=".2f"), "\n"]
                text += ["\n### Do BTC and ETH features help the other 23 coins? (test R2, %)\n",
                         cross_asset_gain(df, y, preds, rng).to_markdown(floatfmt=".3f"), "\n"]
                plot_coins(coins, h)
            else:
                text += ["\n### Validation\n", table.drop(columns=["95% CI", "R2 - Ridge"]).to_markdown(floatfmt=".3f"), "\n"]

    importance = importance_tables()
    if importance is not None:
        text += ["\n## Which features matter\n", "\n### LightGBM: share of total gain (%), all 26 features\n",
                 importance.to_markdown(floatfmt=".1f"), "\n"]
    coef = ridge_coefficients()
    if coef is not None:
        text += ["\n### Ridge coefficients (target and features in sd units)\n", coef.to_markdown(floatfmt=".4f"), "\n"]

    plot_r2(main_tables)
    pd.concat(metrics).to_csv(RESULTS / "metrics.csv")
    (RESULTS / "evaluation.md").write_text("\n".join(text))


if __name__ == "__main__":
    main()
