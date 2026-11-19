# Crypto short-horizon return prediction (FIN 516)

Can machine learning predict 5, 30 and 60 second returns of 25 crypto perpetual futures, using
only the Bybit order book? And does it survive trading costs?

**The full study (data, features, models, results), written to base the paper on: [REPORT.md](REPORT.md).**

## Where the data comes from

Only **Bybit order-book data**: no trades, no other exchanges.

1. **Bybit's public order-book archive** for USDT perpetual futures. One zip per contract per day
   holds the exchange's order-book stream (snapshots and changes, millisecond timestamps):
   `https://quote-saver.bycsi.com/orderbook/linear/<SYMBOL>/<YYYY-MM-DD>_<SYMBOL>_ob200.data.zip`
   (`ob500` before about 2025-08-21).
2. **Converted to one book per second.** The downloader in
   `/scratch/network/an8303/crypto_project/orderbook_data/` (`feeder.py`, `worker.py`) replays the stream.
   It saves the top 5 bid and ask levels at the end of every second, with flags for invalid books and exchange restarts.
3. **This study** reads those files. It doesn't download anything.

## Where the data is (Adroit)

| What | Where |
|---|---|
| Raw 1-second books (input, read only) | `/scratch/network/an8303/crypto_project/orderbook_data/bybit/<SYMBOL>/<YYYY-MM-DD>.parquet` |
| Everything this study makes | `/scratch/network/an8303/fin516/data` (`data/` in this repo is a link to it), about 12 GB |
| Python environment | `/scratch/network/an8303/fin516/venv` (anaconda3/2025.12 + lightgbm + torch CPU) |

```
data/
  features/<SYMBOL>/grid.parquet      26 features + targets + bid/ask, one row every 5 s   (step 1)
  features/<SYMBOL>/seconds.parquet   per-second inputs for the GRU                          (step 1)
  dataset/{train,val,test}.parquet    all contracts, scaled, split by date                   (step 2)
  dataset/scales.parquet              train-period means and sds used for scaling            (step 2)
  predictions/{val,test}/<model>_h<h>.parquet   each model's forecasts                       (step 3)
```

The data is not on GitHub (too big). Only code and results are.

## How to run it

Everything runs as Slurm jobs on Adroit; the whole study takes about 45 minutes.

```bash
sbatch slurm/data.sbatch                           # steps 1, 2, 2b, 3a: features, dataset, summary, linear models
sbatch --export=ALL,GROUP=A slurm/models.sbatch    # step 3b-c: LightGBM + MLP (~10 min)
sbatch --export=ALL,GROUP=B slurm/models.sbatch    # step 3c: GRU (~25 min)
sbatch slurm/results.sbatch                        # steps 4-5: scores and trading (~10 min)
```

Adroit runs at most 32 of our CPUs at a time, so group B starts after group A.

To make the Python environment:

```bash
module load anaconda3/2025.12
python -m venv --system-site-packages /scratch/network/an8303/fin516/venv
/scratch/network/an8303/fin516/venv/bin/pip install lightgbm torch \
    --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

## The code

| Step | File | What it does |
|---|---|---|
| 1 | `src/features.py` | Raw 1-second books → 26 features and the future returns, per contract |
| 2 | `src/dataset.py` | Adds BTC/ETH features, splits by date, scales with train-period numbers |
| 2b | `src/describe.py` | Summary statistics of the data |
| 3a | `src/linear.py` | Zero, OLS, ridge |
| 3b | `src/trees.py` | LightGBM |
| 3c | `src/nn.py` | MLP and GRU (PyTorch) |
| 4 | `src/evaluate.py` | R2, correlation, direction, bootstrap CIs, per contract, BTC/ETH test, feature importance |
| 5 | `src/trading.py` | Trading simulation with bid/ask, delay and fees |

Helpers: `src/config.py` (contracts, dates, features, costs) and `src/common.py` (loading, R2, saving forecasts).

## Output (`results/`)

| File | What |
|---|---|
| `data_summary.md` / `.csv` | spreads, return volatility, share of zero returns, per contract |
| `evaluation.md`, `metrics.csv` | every forecast table: R2 with CIs, per contract, BTC/ETH test, feature importance |
| `trading.md`, `trading.csv`, `trading_by_threshold.csv` | trading results |
| `linear_coefficients.csv`, `lgbm_importance_h*.csv` | what the models use |
| `figs/r2_by_model.png`, `figs/r2_by_coin_h*.png` | figures |
