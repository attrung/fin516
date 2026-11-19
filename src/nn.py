"""Step 3c: neural networks (PyTorch, on CPU).

    python src/nn.py mlp 5
    python src/nn.py gru 5

mlp  feed-forward network on all 26 features: 26 -> 64 -> 32 -> 1 (ReLU).
     The forecast is the average of SEEDS networks trained from different random starts.
gru  sequence model. A GRU (32 units) reads the last 60 seconds of 8 per-second inputs
     (config.SEQUENCE: own return, spread, imbalances, microprice, OFI, BTC and ETH returns).
     Its final state is joined with the 26 features, then 64 -> 1.

Training: Adam, mean squared error on the scaled target (clipped at +-10), batches of 4096.
Each epoch uses EPOCH_ROWS random training rows. Training stops when the validation loss has not
improved for PATIENCE epochs, and the best epoch is kept.
"""
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn

from common import CPUS, features, load, r2, save, target
from config import COINS, FEATURE_SETS, FEATURES, FIRST_DAY, LEADERS, SEQ_LEN, SEQUENCE, SPLITS

FEATS = FEATURE_SETS["own+cross"]
EPOCH_ROWS = 4_000_000
MAX_EPOCHS = 12
PATIENCE = 2
BATCH = 4096
VAL_ROWS = 1_000_000          # validation rows used for early stopping
SEEDS = {"mlp": [0, 1, 2], "gru": [0]}
START = pd.Timestamp(FIRST_DAY, tz="UTC")


# --------------------------------------------------------------------------- data

def sequence_inputs():
    """Per-second inputs for every coin on one clock: tensor (coin, second, input).
    Scaled coin by coin with training-period mean and sd, clipped to [-5, 5]."""
    leaders = {c: pd.read_parquet(FEATURES / c / "seconds.parquet", columns=["ret_1"]).ret_1.to_numpy()
               for c in LEADERS}
    ts = pd.read_parquet(FEATURES / COINS[0] / "seconds.parquet", columns=["ts"]).ts
    train = ((ts >= pd.Timestamp(SPLITS["train"][0], tz="UTC")) &
             (ts < pd.Timestamp(SPLITS["train"][1], tz="UTC"))).to_numpy()
    arrays = []
    for coin in COINS:
        s = pd.read_parquet(FEATURES / coin / "seconds.parquet")
        s["btc_ret_1"], s["eth_ret_1"] = leaders["BTCUSDT"], leaders["ETHUSDT"]
        a = s[SEQUENCE].to_numpy(np.float32)
        mean, sd = np.nanmean(a[train], axis=0), np.nanstd(a[train], axis=0)
        arrays.append(np.nan_to_num(np.clip((a - mean) / sd, -5, 5)).astype(np.float32))
    return torch.from_numpy(np.stack(arrays))


class Split:
    """One split as tensors: features X, target y, and where each row sits in the sequence tensor."""

    def __init__(self, name, h):
        df = load(name, ["ts", "coin", f"y_{h}"] + FEATS)
        self.X = torch.from_numpy(features(df, FEATS))
        self.y = torch.from_numpy(target(df, h, clip=True))
        self.y_raw = target(df, h)
        self.coin = torch.from_numpy(df.coin.map({c: i for i, c in enumerate(COINS)}).to_numpy(np.int64))
        self.second = torch.from_numpy(((df.ts - START) // pd.Timedelta("1s")).to_numpy(np.int64))
        self.labelled = np.flatnonzero(np.isfinite(self.y_raw))

    def batch(self, rows, seq):
        window = None
        if seq is not None:
            steps = torch.arange(-SEQ_LEN + 1, 1)
            window = seq[self.coin[rows, None], self.second[rows, None] + steps]
        return self.X[rows], window, self.y[rows]


# --------------------------------------------------------------------------- models

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(len(FEATS), 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),
                                 nn.Linear(32, 1))

    def forward(self, x, window):
        return self.net(x).squeeze(-1)


class GRUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(len(SEQUENCE), 32, batch_first=True)
        self.head = nn.Sequential(nn.Linear(32 + len(FEATS), 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, window):
        _, state = self.gru(window)
        return self.head(torch.cat([state[-1], x], dim=1)).squeeze(-1)


# --------------------------------------------------------------------------- training

@torch.no_grad()
def predict(model, split, seq, rows=None):
    model.eval()
    rows = torch.arange(len(split.y)) if rows is None else torch.as_tensor(rows)
    out = [model(*split.batch(r, seq)[:2]) for r in rows.split(65536)]
    return torch.cat(out).numpy()


def train(kind, seed, train_split, val_split, seq):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MLP() if kind == "mlp" else GRUNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    val_rows = np.sort(rng.choice(val_split.labelled, min(VAL_ROWS, len(val_split.labelled)), replace=False))
    y_val = val_split.y[val_rows].numpy()
    best, best_loss, waited = None, np.inf, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        rows = torch.from_numpy(rng.choice(train_split.labelled, EPOCH_ROWS, replace=False))
        for r in rows.split(BATCH):
            x, window, y = train_split.batch(r, seq)
            loss = ((model(x, window) - y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        val_loss = np.mean((predict(model, val_split, seq, val_rows) - y_val) ** 2)
        print(f"{kind} seed {seed} epoch {epoch}: val loss {val_loss:.5f}", flush=True)
        if val_loss < best_loss:
            best, best_loss, waited = {k: v.clone() for k, v in model.state_dict().items()}, val_loss, 0
        else:
            waited += 1
            if waited >= PATIENCE:
                break
    model.load_state_dict(best)
    return model


def main(kind, h):
    h = int(h)
    torch.set_num_threads(CPUS)
    seq = sequence_inputs() if kind == "gru" else None
    splits = {name: Split(name, h) for name in ["train", "val", "test"]}
    forecasts = {"val": [], "test": []}
    for seed in SEEDS[kind]:
        model = train(kind, seed, splits["train"], splits["val"], seq)
        for name in forecasts:
            forecasts[name].append(predict(model, splits[name], seq))
        print(f"{kind} h={h} seed {seed}: val R2 {r2(splits['val'].y_raw, forecasts['val'][-1]) * 100:.3f}%",
              flush=True)
    for name, runs in forecasts.items():
        save(kind, h, name, {"own+cross": np.mean(runs, axis=0)})


if __name__ == "__main__":
    main(*sys.argv[1:])
