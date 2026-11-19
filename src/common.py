"""Helpers shared by the model scripts: load a split, build X and y, score, save forecasts."""
import os

import numpy as np
import pandas as pd

from config import DATASET, PREDICTIONS

Y_CLIP = 10.0     # training targets are clipped at +-10 sd, so a few huge spikes don't drive the fit
CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))


def load(split, columns=None):
    return pd.read_parquet(DATASET / f"{split}.parquet", columns=columns)


def features(df, names):
    return df[names].to_numpy(np.float32)


def target(df, h, clip=False):
    """Scaled target y_h (NaN where the future window had bad data)."""
    y = df[f"y_{h}"].to_numpy(np.float32)
    return np.clip(y, -Y_CLIP, Y_CLIP) if clip else y


def r2(y, pred):
    """Out-of-sample R2 against a zero forecast: 1 - SSE(model) / SSE(zero)."""
    ok = np.isfinite(y) & np.isfinite(pred)
    return 1 - np.sum((y[ok] - pred[ok]) ** 2) / np.sum(y[ok] ** 2)


def save(model, h, split, forecasts):
    """forecasts: {feature set name: array with one forecast per row of the split}."""
    folder = PREDICTIONS / split
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(forecasts).to_parquet(folder / f"{model}_h{h}.parquet", index=False)
