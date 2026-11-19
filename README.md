# Crypto short-horizon return prediction (FIN 516)

Can machine learning predict 5, 30 and 60 second returns of 25 crypto perpetual futures, using
only the Bybit order book? And does it survive trading costs?

**Results and everything we did: [REPORT.md](REPORT.md).**

## Where the data is

| What | Where |
|---|---|
| Raw Bybit book (read only) | `/scratch/network/an8303/crypto_project/orderbook_data/bybit/<COIN>/<day>.parquet`: top 5 levels, every second |
| Our data | `/scratch/network/an8303/fin516/data` (`data/` here is a link to it), about 8 GB |
| Python env | `/scratch/network/an8303/fin516/venv` (anaconda3/2025.12 + lightgbm + torch CPU) |

```
data/
  features/<COIN>/grid.parquet      features + targets, one row every 5 s      (step 1)
  features/<COIN>/seconds.parquet   per-second inputs of the sequence model    (step 1)
  dataset/{train,val,test}.parquet  all coins, scaled, split by date           (step 2)
  predictions/{val,test}/*.parquet  each model's forecasts                      (step 3)
```

## How to run it

Everything runs as Slurm jobs. The whole thing takes about 45 minutes.

```bash
sbatch slurm/data.sbatch                           # steps 1-3a: features, dataset, linear models (~5 min)
sbatch --export=ALL,GROUP=A slurm/models.sbatch    # step 3b-c: LightGBM + MLP (~10 min)
sbatch --export=ALL,GROUP=B slurm/models.sbatch    # step 3c: GRU (~25 min)
sbatch slurm/results.sbatch                        # steps 4-5: scores and trading (~10 min)
```

Adroit runs at most 32 of our CPUs at once, so group B starts when group A is done.
To set up the env:
`module load anaconda3/2025.12; python -m venv --system-site-packages <path>; <path>/bin/pip install lightgbm torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple`

## The code

| Step | File | What it does |
|---|---|---|
| 1 | `src/features.py` | Raw book → 26 features and the future returns, per coin |
| 2 | `src/dataset.py` | Adds BTC/ETH features, splits by date, scales with train-period numbers |
| 3a | `src/linear.py` | Zero, OLS, ridge |
| 3b | `src/trees.py` | LightGBM |
| 3c | `src/nn.py` | MLP and GRU (PyTorch) |
| 4 | `src/evaluate.py` | R2, correlation, direction, bootstrap CIs, per coin, BTC/ETH test, feature importance |
| 5 | `src/trading.py` | Trading simulation with bid/ask, delay and fees |

Helpers: `src/config.py` (coins, dates, features, costs) and `src/common.py` (loading, R2, saving forecasts).

## Output

- `results/evaluation.md`: all forecast tables
- `results/trading.md`: trading tables
- `results/metrics.csv`, `trading.csv`, `trading_by_threshold.csv`: the same numbers as CSV
- `results/linear_coefficients.csv`, `lgbm_importance_h*.csv`: what the models use
- `results/figs/`: figures
