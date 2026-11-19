"""Step 3b: gradient-boosted trees (LightGBM).

    python src/trees.py 5        # one horizon per run, so the three horizons run as parallel jobs

One pooled model per horizon and feature set (own, own+cross).
Settings: tree size (number of leaves) is picked on validation from LEAVES; the number of trees
by early stopping on validation. To keep run time down, each model trains on a random
TRAIN_ROWS rows of the training period.
Feature importance (share of total gain) goes to results/lgbm_importance_h<h>.csv.
"""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import CPUS, features, load, r2, save, target
from config import FEATURE_SETS, RESULTS

TRAIN_ROWS = 6_000_000
LEAVES = [15, 63]
PARAMS = {
    "objective": "l2",
    "learning_rate": 0.05,
    "min_data_in_leaf": 5000,      # leaves need many observations: returns are mostly noise
    "feature_fraction": 0.8,
    "bagging_fraction": 0.5,
    "bagging_freq": 1,
    "lambda_l2": 10.0,
    "verbose": -1,
    "num_threads": CPUS,
    "seed": 0,
}


def fit(X, y, X_val, y_val, leaves):
    data = lgb.Dataset(X, y, free_raw_data=False)
    valid = lgb.Dataset(X_val, y_val, reference=data)
    return lgb.train({**PARAMS, "num_leaves": leaves}, data, num_boost_round=2000, valid_sets=[valid],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def main(h):
    h = int(h)
    rows = np.random.default_rng(0).permutation(len(load("train", ["ts"])))[:TRAIN_ROWS]
    train = load("train").iloc[np.sort(rows)]
    val, test = load("val"), load("test")
    y = target(train, h, clip=True)
    y_val = target(val, h)
    ok, ok_val = np.isfinite(y), np.isfinite(y_val)

    out = {"val": {}, "test": {}}
    importance = {}
    for name, cols in FEATURE_SETS.items():
        X, X_val, X_test = features(train, cols), features(val, cols), features(test, cols)
        fits = {leaves: fit(X[ok], y[ok], X_val[ok_val], y_val[ok_val], leaves) for leaves in LEAVES}
        scores = {leaves: r2(y_val, m.predict(X_val, num_threads=CPUS)) for leaves, m in fits.items()}
        best = fits[max(scores, key=scores.get)]
        print(f"h={h} {name}: val R2 by leaves", {k: round(float(v) * 100, 3) for k, v in scores.items()},
              "trees", {k: m.best_iteration for k, m in fits.items()}, flush=True)

        out["val"][name] = best.predict(X_val, num_threads=CPUS)
        out["test"][name] = best.predict(X_test, num_threads=CPUS)
        gain = pd.Series(best.feature_importance("gain"), index=cols)
        importance[name] = gain / gain.sum()

    for split, forecasts in out.items():
        save("lgbm", h, split, forecasts)
    pd.DataFrame(importance).to_csv(RESULTS / f"lgbm_importance_h{h}.csv", float_format="%.4f")


if __name__ == "__main__":
    main(*sys.argv[1:])
