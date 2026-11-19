"""Step 3a: linear models.

    python src/linear.py

  zero   always forecast 0 (the benchmark every R2 is measured against)
  ols    ordinary least squares, all 26 features
  ridge  OLS with an L2 penalty, own features (18) and all features (26).
         The penalty (as a share of the sample size) is picked from LAMBDAS on validation data.

One pooled model for all coins, one per horizon. Forecasts for validation and test go to
data/predictions/, coefficients to results/linear_coefficients.csv.
"""
import numpy as np
import pandas as pd

from common import features, load, r2, save, target
from config import FEATURE_SETS, HORIZONS, RESULTS

LAMBDAS = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
CHUNK = 2_000_000


def moments(X, y):
    """X'X and X'y with an intercept column, summed in chunks (float64) to save memory."""
    ok = np.isfinite(y)
    X, y = X[ok], y[ok]
    k = X.shape[1] + 1
    xx, xy = np.zeros((k, k)), np.zeros(k)
    for i in range(0, len(y), CHUNK):
        Xc = np.c_[np.ones(len(y[i:i + CHUNK])), X[i:i + CHUNK]].astype(np.float64)
        xx += Xc.T @ Xc
        xy += Xc.T @ y[i:i + CHUNK]
    return xx, xy, len(y)


def ridge(xx, xy, n, lam):
    penalty = lam * n * np.eye(len(xy))
    penalty[0, 0] = 0                       # no penalty on the intercept
    return np.linalg.solve(xx + penalty, xy)


def predict(beta, X):
    return beta[0] + X @ beta[1:].astype(np.float32)


def main():
    train, val, test = load("train"), load("val"), load("test")
    coefficients = []
    for h in HORIZONS:
        y_train, y_val = target(train, h, clip=True), target(val, h)
        out = {"val": {}, "test": {}}
        for name, cols in FEATURE_SETS.items():
            xx, xy, n = moments(features(train, cols), y_train)
            X_val, X_test = features(val, cols), features(test, cols)

            scores = {lam: r2(y_val, predict(ridge(xx, xy, n, lam), X_val)) for lam in LAMBDAS}
            best = max(scores, key=scores.get)
            print(f"h={h} {name}: val R2 by lambda", {k: round(float(v) * 100, 3) for k, v in scores.items()},
                  "-> best", best, flush=True)

            fits = {"ridge": ridge(xx, xy, n, best)}
            if name == "own+cross":
                fits["ols"] = ridge(xx, xy, n, 0)
            for model, beta in fits.items():
                out["val"].setdefault(model, {})[name] = predict(beta, X_val)
                out["test"].setdefault(model, {})[name] = predict(beta, X_test)
                coefficients.append(pd.Series(beta, index=["const"] + cols)
                                    .rename(f"{model} ({name}) h={h}"))

        for split, models in out.items():
            for model, forecasts in models.items():
                save(model, h, split, forecasts)
            save("zero", h, split, {name: np.zeros(len(val if split == "val" else test), np.float32)
                                     for name in FEATURE_SETS})

    RESULTS.mkdir(exist_ok=True)
    pd.concat(coefficients, axis=1).to_csv(RESULTS / "linear_coefficients.csv", float_format="%.5f")


if __name__ == "__main__":
    main()
