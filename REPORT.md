# Predicting short-horizon crypto returns from the Bybit order book

FIN 516 project. This file documents the whole study (data, features, models, tests and results)
in enough detail to write the paper from it. Every number here comes from the files in
`results/`, made by the code in `src/`.

Full tables:
- [results/data_summary.md](results/data_summary.md): the data
- [results/evaluation.md](results/evaluation.md): forecasts
- [results/trading.md](results/trading.md): trading

---

## 1. Question and main findings

**Question.** Using only the limit order book, can machine learning predict the mid-price return
of crypto perpetual futures over the next 5, 30 and 60 seconds better than linear models? And is
the predictability large enough to trade after costs?

**Findings.**

1. **Returns are predictable at very short horizons.** At 5 s the best model explains 3.5% of the
   variance out of sample (test R2). R2 is positive for all 25 coins and all 14 test days, for
   every model. At 30 s the best R2 is 1.0%, and at 60 s it is 0.5%.
2. **Machine learning helps, but only at 5 s.**
   - At 5 s, LightGBM, a feed-forward network (MLP) and a sequence model (GRU) all beat ridge by
     0.3–0.5 R2 points, and the gaps are significant.
   - At 30–60 s only LightGBM stays slightly ahead. The neural nets fall below ridge.
3. **The signal sits at the top of the book.** Best-level imbalance and the microprice make up
   about 60–65% of LightGBM's total gain.
4. **BTC and ETH lead the other coins briefly.** Adding BTC and ETH features raises the other 23
   coins' R2 by about 0.2 points at 5 s (significant), 0.04 at 30 s, and 0 at 60 s.
5. **Coins whose price rarely moves are the most predictable.** Across coins, R2 at 5 s rises with
   the share of 5-second intervals in which the mid does not change (Spearman 0.65).
6. **The predictability does not survive trading costs.**
   - The forecast moves are smaller than the bid-ask spread. Trading as a taker on every forecast
     loses 1.0–1.2 bps per trade even with zero fees.
   - At Bybit's 5.5 bps taker fee, no model finds a profitable trading rule on validation data.

---

## 2. Data

### 2.1 Source

We use **only Bybit order-book data**: no trades, no other exchanges.

**The archive.** Bybit publishes a free, public archive of its order-book stream for USDT
perpetual futures ("linear" contracts), one zip file per contract per day:

    https://quote-saver.bycsi.com/orderbook/linear/<SYMBOL>/<YYYY-MM-DD>_<SYMBOL>_<DEPTH>.data.zip

- **Contents.** Each file holds the exchange's order-book WebSocket messages for one UTC day, each
  with a millisecond timestamp. A message is a full snapshot or a change (delta), and every message
  carries an update id.
- **Depth.** The stream covers the top 500 levels until about 2025-08-21, and the top 200 levels
  after. Our sample (Aug–Oct 2026) uses the 200-level stream. We only keep the top 5 levels.

**From messages to 1-second books.** This was done before this study, by the downloader in
`/scratch/network/an8303/crypto_project/orderbook_data/` (`feeder.py` downloads, `worker.py` converts).
- It replays every message in order to rebuild the full book.
- At the end of each second it records the top 5 bid and ask levels (price and size).
  - The book is carried from one day to the next, so a day's first seconds are correct.
  - In seconds without any message, the book is repeated.
- Each second also gets:
  - `n_updates`: the number of messages in that second
  - `restart`: Bybit restarted its service (a new snapshot with update id 1)
  - `valid`: False before the first snapshot, and after a gap in update ids until the next snapshot
- The result is one parquet file per contract per day, 86,400 rows:
  `/scratch/network/an8303/crypto_project/orderbook_data/bybit/<SYMBOL>/<YYYY-MM-DD>.parquet`.

### 2.2 Sample

| | |
|---|---|
| Contracts | 25 USDT perpetuals: BTC, ETH, SOL, XRP, DOGE, BNB, ADA, AVAX, LINK, SUI, LTC, BCH, DOT, NEAR, UNI, AAVE, HYPE, ENA, TAO, ARB, 1000PEPE, XLM, HBAR, ONDO, TRX |
| Period | 2026-08-03 00:00 to 2026-10-11 23:59 UTC (10 weeks, 70 days). Aug 2 is loaded only to warm up the look-back features |
| Frequency | features built every second; one observation every 5 seconds per contract (17,280 per day) |
| Observations | 1,209,600 per contract, 30.2 million in total |

**How the contracts were chosen.** We took the liquid contracts in the archive that have every day
of the window and no invalid books. We dropped:
- PAXG, a gold token, not a crypto asset in the usual sense
- PONS, which starts on Aug 31
- GRAM and USELESS, which have thin books with up to 30-second silences

**Cleaning.**
- A second is **bad** if any of these hold:
  - the book is marked invalid
  - Bybit restarted that second
  - the book is crossed (ask ≤ bid)
  - the best level has zero size
- An observation is kept only if:
  - the 600 seconds before it are all good (the longest look-back), and
  - every second through its target window is good.
- In this window no second was bad, so nothing was dropped for data quality.

### 2.3 Train, validation and test

The split is chronological, with the same dates for every contract:

| Split | Dates | Weeks | Observations | Used for |
|---|---|---|---|---|
| Train | Aug 3 – Sep 13 | 6 | 18,143,700 | fitting |
| Validation | Sep 14 – Sep 27 | 2 | 6,047,700 | ridge penalty, tree size, early stopping, trading threshold |
| Test | Sep 28 – Oct 11 | 2 | 6,047,700 | final numbers, used once |

The last 61 seconds of each split are dropped, so no target window (up to 60 s + 1 s) reaches into
the next split.

### 2.4 Summary statistics

From [results/data_summary.md](results/data_summary.md):

| | Median over the 25 contracts | Range |
|---|---|---|
| Bid-ask spread (bps) | 1.23 | 0.01 (BTC) to 4.75 (XLM) |
| sd of 5 s return (bps) | 3.23 | 0.75 (TRX) to 5.45 (ENA) |
| sd of 30 s return (bps) | 7.83 | 1.67 (TRX) to 13.53 (ENA) |
| sd of 60 s return (bps) | 11.27 | 2.37 (TRX) to 19.17 (ENA) |
| 5 s intervals with no mid change (%) | 49.9 | 25.9 (ENA) to 96.3 (TRX) |
| 60 s intervals with no mid change (%) | 10.0 | 2.7 (TAO) to 72.1 (TRX) |

Two features of the data matter for the results:
- **Many mids don't move.** In half of all 5-second intervals the mid does not move.
- **Spreads differ a lot.** BTC and ETH trade at a one-tick spread far below 1 bp. Contracts like
  ADA, BCH and XLM have spreads of 4–5 bps, wider than a typical 5-second move.

---

## 3. Variables

### 3.1 Target

$$ y_{i,t}^{(h)} = 10^4 \cdot \big(\ln m_{i,t+h} - \ln m_{i,t}\big), \qquad h \in \{5, 30, 60\} \text{ seconds}, $$

where $m_{i,t} = (a_{i,t} + b_{i,t})/2$ is the mid of contract $i$ at second $t$, and $a$ and $b$ are
the best ask and bid. The target is in basis points.

### 3.2 Features (26)

We kept the list short on purpose: 18 come from the contract's own book, and 8 from BTC and ETH.
Every feature at second $t$ uses only books up to $t$.

Notation:
- $q^b_k, q^a_k$ are the sizes at bid and ask level $k$, with $k = 0$ the best level.
- $b_k, a_k$ are the prices at those levels.

**Own book: state**

| Name | Definition |
|---|---|
| `spread` | $10^4 (a_0 - b_0)/m$ |
| `imb1` | $(q^b_0 - q^a_0)/(q^b_0 + q^a_0)$ |
| `imb5` | $(\sum_{k<5} q^b_k - \sum_{k<5} q^a_k)/(\sum_{k<5} q^b_k + \sum_{k<5} q^a_k)$ |
| `micro` | $10^4 (\mu - m)/m$, with microprice $\mu = (a_0 q^b_0 + b_0 q^a_0)/(q^b_0 + q^a_0)$ |
| `depth` | $\ln \sum_{k<5}(b_k q^b_k + a_k q^a_k)$ (the $ value of the top 5 levels) minus its trailing 1-hour mean |
| `activity` | $\ln(1 + \text{book updates in the last 10 s})$ minus its trailing 1-hour mean |

**Own book: order flow.** Order-flow imbalance (OFI) of Cont, Kukanov and Stoikov (2014), from one
second to the next, at level $k$:

$$ e_{k,t} = q^b_{k,t}\mathbb{1}[b_{k,t} \ge b_{k,t-1}] - q^b_{k,t-1}\mathbb{1}[b_{k,t} \le b_{k,t-1}]
            - q^a_{k,t}\mathbb{1}[a_{k,t} \le a_{k,t-1}] + q^a_{k,t-1}\mathbb{1}[a_{k,t} \ge a_{k,t-1}] $$

Positive values mean net buying pressure. With book data only, OFI is our measure of signed order flow.

| Name | Definition |
|---|---|
| `ofi_1`, `ofi_5`, `ofi_30` | $\sum e_{0}$ over the last 1, 5, 30 s, divided by the trailing 1-hour mean of $(q^b_0 + q^a_0)/2$ |
| `ofi_deep` | $\sum_{k=1}^{4} e_k$ over the last 5 s, divided by the trailing 1-hour mean of $\sum_{k=1}^{4}(q^b_k + q^a_k)/2$ |

**Own book: price history**

| Name | Definition |
|---|---|
| `ret_1`, `ret_5`, `ret_30`, `ret_60`, `ret_300` | $10^4(\ln m_t - \ln m_{t-k})$, $k$ = 1, 5, 30, 60, 300 s |
| `dev_60` | $10^4 \ln m_t$ minus its mean over the last 60 s |
| `rv_60`, `rv_600` | $\sqrt{\sum r_s^2}$ over the last 60 and 600 s, with $r_s$ the 1-second return (bps) |

**Cross-asset (8):** `ret_1`, `ret_5`, `ret_30` and `ofi_5` of BTC and of ETH, at the same second,
named `btc_ret_1`, …, `eth_ofi_5`.

For BTC and ETH themselves, these duplicate their own features. That's why the cross-asset test
in Section 5.3 uses only the other 23 contracts.

### 3.3 Scaling

All scaling uses training-period numbers only, contract by contract:
- **Features:** subtract the train mean, divide by the train sd, clip to [−5, 5].
- **Target:** divide by the train sd of that contract's h-second return. A pooled model then weighs
  every contract about equally, instead of being driven by the most volatile ones.
  - Forecasts are converted back to bps for trading.
  - For training only, the scaled target is clipped to [−10, 10], so a few extreme seconds
    (moves above 50 sd occur) don't drive the fit.
  - Evaluation always uses the unclipped target.

---

## 4. Models

For each horizon we fit one **pooled** model across all 25 contracts, as Gu, Kelly and Xiu (2020) do
for stocks.

| Model | Inputs | Settings |
|---|---|---|
| Zero | none | forecast = 0. The benchmark for R2 |
| OLS | 26 features | least squares with intercept |
| Ridge | 18 own, or all 26 | penalty $\lambda n$ with $\lambda \in \{0, 10^{-4}, 10^{-3}, 10^{-2}, 0.1, 1\}$ picked on validation (chosen: 0 at 5 s, 0.01 at 30 s, 0.1 at 60 s). With 18M rows it is nearly OLS |
| LightGBM | 18 own, or all 26 | see the LightGBM settings below |
| MLP | 26 features | 26 → 64 → 32 → 1, ReLU. The forecast averages 3 networks with different random starts |
| GRU | 26 features + last 60 s | see the GRU design below |

**LightGBM settings:**
- squared loss, learning rate 0.05
- 15 or 63 leaves, picked on validation (63 chosen at every horizon)
- min 5,000 rows per leaf, 80% of features per tree, 50% row bagging, L2 penalty 10
- early stopping on validation after 50 rounds without improvement. It stopped after 48–174 trees.
- trained on 6M random training rows

**GRU design:**
- A GRU with 32 units reads the last 60 seconds of 8 per-second inputs: own `ret_1`, `spread`,
  `imb1`, `imb5`, `micro`, `ofi_1`, and BTC and ETH `ret_1`. Each input is standardized per
  contract on train.
- Its final state is joined with the 26 features, then 64 → 1.
- One random start.

**Neural-net training:**
- Adam, learning rate $10^{-3}$, batch 4,096, mean squared error on the clipped scaled target.
- Each epoch uses 4M random training rows.
- Training stops when the loss on 1M validation rows has not improved for 2 epochs, and the best
  epoch is kept (at most 12 epochs).
- Seeds 0, 1, 2 for the MLP; 0 for the GRU.

"(own)" in the tables means the model uses only the 18 own-book features.

---

## 5. Results

### 5.1 How we measure forecast quality

- **Out-of-sample R2** against a zero forecast (as in Gu, Kelly and Xiu 2020):
  $R^2 = 1 - \sum (y - \hat y)^2 / \sum y^2$, pooled over contracts, on the scaled target.
  Per contract, it is the same as R2 on returns in bps.
- **Correlation** between the forecast and the realized return.
- **Direction accuracy (hit):** the share of observations where forecast and realized return have
  the same sign. Observations with a zero return are left out; they are about half of all 5-second intervals (Section 2.4).
- **Confidence intervals:** a block bootstrap of 1,000 draws. We resample whole clock hours
  (336 hours in the test period) with all contracts together. This keeps the dependence between
  overlapping targets, nearby seconds and contracts. The same draws give CIs for differences between models.
- **Robustness:** the number of contracts with positive R2, and the share of test days with positive R2.

### 5.2 Forecast accuracy (test period, Sep 28 – Oct 11)

| Model | 5 s | 30 s | 60 s |
|---|---|---|---|
| OLS | 3.02 | 0.93 | 0.45 |
| Ridge (own) | 2.84 | 0.89 | 0.46 |
| Ridge | 3.02 [2.78, 3.27] | 0.93 [0.83, 1.04] | 0.46 [0.37, 0.55] |
| LightGBM (own) | 3.26 | 0.98 | 0.53 |
| LightGBM | 3.41 [3.15, 3.69] | **1.01** [0.86, 1.15] | **0.52** [0.36, 0.67] |
| MLP | 3.35 [3.09, 3.61] | 0.91 [0.79, 1.03] | 0.35 [0.13, 0.54] |
| GRU | **3.55** [3.29, 3.81] | 0.76 [0.49, 0.99] | −0.01 [−0.48, 0.37] |

The table shows R2 in %, with 95% bootstrap CIs in brackets. Validation R2s are close and give the same
ranking at 5 s (results/evaluation.md).

![Test R2 by model and horizon](results/figs/r2_by_model.png)

| Model | Correlation 5 / 30 / 60 s | Hit (%) 5 / 30 / 60 s | Contracts with R2 > 0 | Test days with R2 > 0 |
|---|---|---|---|---|
| Ridge | 0.174 / 0.097 / 0.068 | 63.4 / 56.0 / 54.1 | 25 / 25 / 25 | 100% / 100% / 100% |
| LightGBM | 0.185 / 0.100 / 0.072 | 63.5 / 56.1 / 54.3 | 25 / 25 / 25 | 100% / 100% / 100% |
| MLP | 0.183 / 0.096 / 0.061 | 63.5 / 56.0 / 54.2 | 25 / 25 / 19 | 100% / 100% / 86% |
| GRU | 0.188 / 0.090 / 0.048 | 63.6 / 56.0 / 54.1 | 25 / 23 / 7 | 100% / 93% / 79% |

### 5.3 Machine learning vs linear

The table shows the test R2 difference against ridge with all 26 features, in % points, with 95% CIs:

| | 5 s | 30 s | 60 s |
|---|---|---|---|
| LightGBM | **+0.40** [0.36, 0.44] | **+0.07** [0.01, 0.13] | +0.06 [−0.03, 0.13] |
| MLP | **+0.33** [0.30, 0.36] | −0.02 [−0.06, 0.01] | −0.11 [−0.25, 0.01] |
| GRU | **+0.53** [0.47, 0.59] | **−0.17** [−0.36, −0.03] | **−0.47** [−0.87, −0.17] |

**At 5 s every non-linear model wins.** That's about 11–18% more explained variance than ridge. The
GRU, which sees the last minute second by second, is best: recent book dynamics carry information
that the 26 summary features miss.

**At 30–60 s the extra flexibility mostly fits noise:**
- The signal-to-noise ratio is about 3–7 times lower than at 5 s.
- The GRU's validation loss stopped improving after 1–2 epochs at these horizons, and on test it
  falls below ridge.
- Only LightGBM, with its large leaves and early stopping, keeps a small edge, and it is
  significant at 30 s only.

### 5.4 By contract

The table shows test R2 at 5 s, in %. All contracts are in results/evaluation.md.

| | Ridge | LightGBM | GRU |
|---|---|---|---|
| Highest: BNB | 9.40 | 11.07 | 12.11 |
| XLM | 6.66 | 8.62 | 9.21 |
| BCH | 6.02 | 6.85 | 6.89 |
| ADA | 5.57 | 6.48 | 6.71 |
| BTC | 3.55 | 4.09 | 4.05 |
| ETH | 2.24 | 2.93 | 2.94 |
| HBAR | 1.09 | 1.26 | 1.33 |
| Lowest: ENA | 0.97 | 1.29 | 1.34 |

![Test R2 by contract, 5 s](results/figs/r2_by_coin_h5.png)

**What drives the differences between contracts:**
- R2 is highest where the mid moves least often.
- Across the 25 contracts, LightGBM's 5-second R2 has a Spearman correlation of 0.65 with the share of
  5-second intervals with no mid change. It has 0.56 with the spread relative to the 5-second return sd.
- BNB, XLM, BCH and ADA have 70–74% zero 5-second returns and spreads of 1.4–4.8 bps. When the price
  is "sticky" relative to the tick, the queue sizes at the best bid and ask say a lot about which
  side moves next.
- ENA, TAO and ARB move almost every interval (26–31% zeros) and are among the hardest to predict.
- TRX is an exception: it barely moves at all (96% zeros), so there is little to predict.

### 5.5 Do BTC and ETH help the other contracts?

The table compares test R2 (%) for the 23 contracts other than BTC and ETH: own-book features only vs own + BTC/ETH.

| Model | Horizon | Own | Own + cross | Gain [95% CI] |
|---|---|---|---|---|
| Ridge | 5 s | 2.83 | 3.02 | **+0.19** [0.16, 0.22] |
| Ridge | 30 s | 0.91 | 0.95 | **+0.04** [0.01, 0.08] |
| Ridge | 60 s | 0.47 | 0.48 | +0.00 [−0.02, 0.04] |
| LightGBM | 5 s | 3.25 | 3.41 | **+0.16** [0.13, 0.20] |
| LightGBM | 30 s | 0.99 | 1.02 | +0.03 [−0.00, 0.06] |
| LightGBM | 60 s | 0.54 | 0.54 | −0.00 [−0.04, 0.04] |

BTC and ETH lead the rest of the market by a few seconds:
- The ridge coefficients on `btc_ret_1`, `eth_ret_1`, `btc_ofi_5` and `eth_ofi_5` are all positive at 5 s.
- By 60 s the lead is fully priced in.

### 5.6 Which features matter

LightGBM's share of total gain (%), with all 26 features:

| Feature | 5 s | 30 s | 60 s |
|---|---|---|---|
| `imb1` | 44.3 | 38.9 | 29.3 |
| `micro` | 18.6 | 24.9 | 26.3 |
| `rv_600` | 5.4 | 4.0 | 5.2 |
| `rv_60` | 3.6 | 2.1 | 4.7 |
| `ret_1` | 2.2 | 3.6 | 5.1 |
| `imb5` | 2.1 | 2.3 | 2.0 |
| `ret_5` | 1.6 | 2.7 | 2.7 |
| `ret_300` | 1.3 | 2.6 | 4.4 |
| `ofi_1` + `ofi_5` + `ofi_30` + `ofi_deep` | 3.9 | 2.5 | 1.8 |
| BTC and ETH (8 features) | 13.4 | 11.8 | 12.7 |
| `ret_30`, `ret_60`, `dev_60`, `spread`, `depth`, `activity` (6) | 3.7 | 4.7 | 5.7 |

Ridge coefficients (target and features in sd units; all in results/evaluation.md):

| Feature | 5 s | 30 s | 60 s | Reading |
|---|---|---|---|---|
| `imb1` | +0.094 | +0.033 | +0.027 | more size on the bid → price goes up |
| `micro` | +0.035 | +0.052 | +0.035 | same idea, size-weighted |
| `ofi_1` | +0.034 | +0.008 | +0.003 | recent buying pressure continues, briefly |
| `ret_1` | −0.030 | −0.009 | −0.006 | the last second's move partly reverses |
| `dev_60` | −0.047 | −0.006 | −0.005 | so does distance from the 1-minute average |
| `eth_ret_1` | +0.029 | +0.012 | +0.006 | ETH leads |
| `btc_ret_1` | +0.013 | +0.003 | +0.004 | BTC leads |
| `rv_60` | +0.009 | +0.017 | +0.018 | small upward drift when volatility is high |

**Summary.** The queue imbalance at the best quote is by far the most useful variable:
- Deeper levels (`imb5`, `ofi_deep`), depth and update activity add little.
- Order flow adds information only at the shortest horizon.
- Volatility matters more for the tree model than for ridge. It works mainly through interactions:
  the same imbalance means a different expected move when volatility is high.

### 5.7 Trading

**Strategy** (`src/trading.py`), one contract at a time, $1 per trade:
1. Every 5 seconds, if no position is open and |forecast| (in bps) exceeds a threshold $c$, trade in
   the forecast's direction.
2. The order fills **1 second later**, as a taker: at the best ask to buy, at the best bid to sell.
3. Close **h seconds after the fill**, again as a taker. No new trade in that contract until then.
4. Pay a fee on each side: 0, 2 or 5.5 bps. 5.5 bps is Bybit's standard taker fee for USDT perpetuals.
5. Choose $c$ from {0, 0.5, 1, 2, 3, 4, 6, 8, 11, 15, 20, 30} bps on **validation**, to maximize total
   profit, among thresholds with at least 200 validation trades. If no threshold makes money on
   validation, the strategy does not trade.
6. Apply that $c$ unchanged to the test period.

We assume our size fills at the quoted price, ignoring queue position and market impact.

**Results on test** (full tables in results/trading.md):

| Fee per side | 5 s | 30 s | 60 s |
|---|---|---|---|
| 5.5 bps | no model trades | no model trades | no model trades |
| 2 bps | no model trades | GRU trades: −10.8 bps/trade | LightGBM −7.6, MLP −7.4, GRU −6.6 bps/trade |
| 0 bps | no model trades | Ridge −0.1, MLP −0.5, GRU −0.6, LightGBM **+1.5** bps/trade (597 trades, t = 0.7) | all four lose: −0.9 to −2.8 bps/trade |

Return per trade after paying the spread, before fees, if we trade on every forecast (threshold 0), on test:

| Model | 5 s | 30 s | 60 s |
|---|---|---|---|
| Ridge | −1.18 | −1.05 | −1.03 |
| LightGBM | −1.18 | −1.05 | −1.00 |
| MLP | −1.18 | −1.05 | −1.03 |
| GRU | −1.18 | −1.06 | −1.07 |

(At 5 s the numbers are identical because, with threshold 0 and non-overlapping trades, all
models trade at the same times; only the direction differs, and it barely changes the average.)

**Why it fails:**
- **The forecasts are small.** On test, the median absolute forecast is 0.3–0.4 bps at 5 s and
  0.5–0.6 bps at 30–60 s. Even the 99th percentile is only 1.4–3.4 bps.
- **The spread is larger.** It is about 1.2 bps for the median contract, and each round trip crosses it.
- Raising the threshold keeps only the large forecasts, but there are few of them (tens to hundreds over two
  weeks), and their test results are noisy and mostly negative.
- The positive cells in the threshold table (results/trading.md) are scattered, don't rise steadily with
  the threshold, mostly rest on a few dozen trades, and don't line up with validation.

So the predictability is statistically strong but economically too small for a taker. A market
maker, who earns the spread instead of paying it, could still use these forecasts: for example, to
step away from the side about to be hit. We did not test that.

---

## 6. Limitations

- **Short sample.** 10 weeks of data in one market regime; the test period is 2 weeks. The bootstrap
  covers within-sample noise, not regime change.
- **Book only.** No trade prints, so no true signed volume; OFI from book changes stands in for it.
  With one snapshot per second, changes inside a second are invisible. Real systems react in milliseconds.
- **Fills.** The trading test assumes we fill at the quoted price with no queue or impact, which makes it
  optimistic. Since it already loses money, better execution modelling would not change the answer.
- **Neural-net tuning.** The MLP and GRU use simple settings, with no hyperparameter search. The GRU ran
  once. More tuning might help at 30–60 s, but the 5 s ranking is clear and significant.
- **One exchange.** Bybit only. Lead-lag across exchanges (e.g. Binance leading Bybit) is not studied.

---

## 7. Changes from the proposal

| Proposal | This study | Reason |
|---|---|---|
| Binance USD-M, live WebSocket collection, ~4 weeks | Bybit USDT perpetuals, public archive, 10 weeks | Bybit book data only (as decided); the archive gives more history |
| Order book + trades | Order book only | Trade flow replaced by order-flow imbalance (OFI) |
| 20–30 contracts | 25 | |
| 5 s, 30 s, 1 min | same | |
| OLS, ridge, zero; LightGBM; MLP; sequence model on the last minute | same (sequence model = GRU) | |
| Chronological split, purged boundaries, train-only preprocessing, validation-based tuning | same | |
| OOS R2, correlation, direction, per contract and per day, time-block bootstrap | same | |
| BTC/ETH returns for smaller contracts | same, plus BTC/ETH order flow | |
| Trading with bid/ask, fees, delay, slippage | bid/ask, 1 s delay, fees of 0 / 2 / 5.5 bps; no extra slippage beyond the spread | |

---

## 8. Reproducing the study

Code, data paths and run commands: [README.md](README.md). Everything is deterministic given the seeds
(0 for data sampling and LightGBM, 0–2 for the MLP, 0 for the GRU). On Adroit the full run takes about
45 minutes of Slurm time.

## References

- Cont, R., Kukanov, A., and Stoikov, S. (2014). The price impact of order book events. *Journal of
  Financial Econometrics*, 12(1), 47–88.
- Gu, S., Kelly, B., and Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of
  Financial Studies*, 33(5), 2223–2273.
- Kolm, P. N., Turiel, J., and Westray, N. (2023). Deep order flow imbalance: Extracting alpha at
  multiple horizons from the limit order book. *Mathematical Finance*, 33(4), 1044–1081.
- Makarov, I., and Schoar, A. (2020). Trading and arbitrage in cryptocurrency markets. *Journal of
  Financial Economics*, 135(2), 293–319.
- Stoikov, S. (2018). The micro-price: a high-frequency estimator of future prices. *Quantitative
  Finance*, 18(12), 1959–1966.
