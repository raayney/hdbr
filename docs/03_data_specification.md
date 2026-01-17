# 3. Data specification

## 3.1 Overview

This document specifies the data universe, fields, preprocessing and feature engineering used in the project. It is written to match the actual implementation:

- underlyings from Yahoo Finance only,
- synthetic three‑month implied volatility surfaces,
- Monte Carlo labels,
- neural network inputs of dimension 274.

It serves as the blueprint for the data pipeline in `output/datasets` and the scripts in `src/data`, `src/pricing` and `src/features`.

---

## 3.2 Data universe and time horizon

### 3.2.1 Market and instruments

Index

- EURO STOXX 50 index, ticker `^STOXX50E` (or a liquid ETF proxy with similar behaviour).
- Synthetic European vanilla options on the index, three‑month maturity.

Constituents

- Top 20 constituents of the EURO STOXX 50 by weight, as listed in `src/config.py` (`TOP_20_CONSTITUENTS`).
- For each constituent, Yahoo Finance spot prices and synthetic three‑month options, consistent with the index set‑up.

### 3.2.2 Historical period and frequency

Backtest and training window

- Start date: `2018-01-01`
- End date: `2025-12-31`

This spans multiple regimes (low volatility, COVID crash, inflation and rate‑hike periods) and provides sufficient data to train and evaluate the model.

Frequency

- Daily end‑of‑day observations (adjusted close from Yahoo Finance).
- No intraday data are used in the current implementation.

---

## 3.3 Underlying price data (real)

Daily prices for the index and the 20 constituents are downloaded from Yahoo Finance, for example using `yfinance`. For each asset and date the following fields are retrieved:

- open
- high
- low
- close
- adjusted close
- volume

The adjusted close series is used for all return and realised variance calculations, as it incorporates splits and dividends.

After download and basic cleaning, these are stored in Parquet files under:

- `output/datasets/processed/`  
  for example
  - `processed_prices.parquet` (wide table: dates × tickers)
  - `returns.parquet` (daily log returns)
  - `rolling_vol.parquet` (for realised volatility estimates)

Daily log returns for asset $`i`$ on day $`t_k`$ are computed as

$`\log\frac{S^i_{t_k}}{S^i_{t_{k-1}}}.`$

These returns are used to:

- estimate short‑window realised volatility for ATM implied volatilities,
- estimate rolling correlation matrices for the Monte Carlo simulation,
- compute realised dispersion in the backtest.

---

## 3.4 Synthetic implied volatility surfaces and option prices

There is no free historical options database for the EURO STOXX 50 and its constituents, so the project uses synthetic but deliberately structured implied volatility surfaces.

### 3.4.1 Moneyness grid

The project fixes a symmetric grid of log‑moneyness levels, defined in `src/config.py` as

$`\text{MONEYNESS\_GRID} = [-0.20,\,-0.10,\,-0.05,\,0.00,\,0.05,\,0.10,\,0.20].`$

For each asset $`i`$ and date $`t`$, synthetic implied volatilities are generated at the seven log‑moneyness points

$`k \in \{-0.20,\,-0.10,\,-0.05,\,0.00,\,0.05,\,0.10,\,0.20\}.`$

### 3.4.2 ATM volatility from realised volatility

For each asset and date, an annualised short‑window realised volatility is computed from the adjusted close series, for example over a 30‑day window:

$`\widehat{\sigma}_{\text{real},i}^{(t)} = \sqrt{252 \cdot \frac{1}{W} \sum_{k=1}^{W} r_{t-k,i}^2},`$

where $`r_{t-k,i}`$ are daily log returns and $`W`$ is the window size. This realised volatility is scaled to obtain an ATM implied volatility

$`\sigma_{\text{ATM},i}^{(t)} \approx c \cdot \widehat{\sigma}_{\text{real},i}^{(t)},`$

with a constant factor $`c`$ chosen to represent typical equity option levels (for example $`c \approx 1.2`$).

### 3.4.3 Quadratic smile model (implemented)

The script `src/pricing/synthetic_surfaces.py` defines `SyntheticIVSurfaceGenerator`, which uses a quadratic smile model over log‑moneyness:

$`\sigma_i^{(t)}(k) = \sigma_{\text{ATM},i}^{(t)} + \alpha k + \beta k^2,`$

with configuration parameters in `src/config.py`:

- $`\alpha = \text{SKEW\_PARAM\_ALPHA} = -0.15`$
- $`\beta = \text{SMILE\_PARAM\_BETA} = 0.05`$

Implied volatilities are clipped to a reasonable range, for example between 0.1 per cent and 200 per cent, to avoid degenerate values.

### 3.4.4 Option prices via Black–Scholes

Given

- spot price $`S_0^i`$,
- forward price $`F_0^i`$,
- three‑month maturity $`T = 0.25`$,
- risk‑free rate $`r_i`$,
- dividend yield $`q_i`$,
- implied volatility $`\sigma_i^{(t)}(k)`$,

call and put prices on strike $`K = F_0^i e^k`$ are computed under the Black–Scholes formula. These synthetic option prices are stored as part of the option surface outputs, for example in:

- `output/datasets/synthetic_iv.parquet`
- `output/datasets/synthetic_options.parquet`

In the theoretical model‑free view, these synthetic prices are treated as if they were observed market vanillas when defining $`\mathcal{M}_{\text{vanilla}}(x)`$.

---

## 3.5 Derived time series for labels and backtesting

The project computes several derived time series from the processed prices:

1. Daily log returns for all 21 assets (index + 20 stocks), stored in `returns.parquet`.
2. Rolling realised volatility, used to derive ATM implied volatilities.
3. Rolling correlation matrices for the 21‑dimensional return vector, estimated over a 60‑day window.

A correlation matrix $`\widehat{\Sigma}_t`$ is estimated at each date $`t`$ with a lookback of 60 trading days, and adjusted to be positive definite (by lifting small eigenvalues if necessary). This matrix is used as the correlation input to the Monte Carlo simulator, but is not included as part of the neural network feature vector.

In the backtest, realised dispersion for a three‑month window is computed by applying the same realised variance formula used inside the Monte Carlo label generator, applied now to actual Yahoo returns over each 63‑day horizon.

---

## 3.6 Feature engineering and feature vector

### 3.6.1 Feature builder

The class `FeatureBuilder` in `src/features/build_features.py` constructs a fixed‑length feature vector for each date from dictionaries of prices, implied volatilities, rates and yields.

Given

- `tickers`: the ordered list of 21 tickers (index first, then 20 constituents),
- `moneyness_grid`: the seven log‑moneyness levels,
- `feature_dim = 274`,

the method `build_feature_vector` proceeds as follows for each date.

Spots and forwards

For each ticker:

- read the spot price $`S_0^i`$,
- read rate $`r_i`$ and dividend yield $`q_i`$,
- compute the three‑month forward

$`F_0^i = S_0^i \exp\big((r_i - q_i)\,T\big), \quad T = 0.25,`$

and append both spot and forward.

This yields $`21 \times 2 = 42`$ features.

Rates and dividend yields

For each ticker:

- append the three‑month rate $`r_i`$,
- append the dividend yield $`q_i`$.

This yields another $`21 \times 2 = 42`$ features.

Implied volatilities

For each ticker:

- obtain the implied volatility curve on the seven log‑moneyness points,
- for each moneyness $`k`$ in the fixed grid, append $`\sigma_i^{(t)}(k)`$.

This yields $`21 \times 7 = 147`$ features.

Zero padding

At this point there are $`42 + 42 + 147 = 231`$ economically meaningful features. The implementation then pads with zeros until length 274:

- append zeros until the vector length reaches 274,
- truncate to length 274 in case of any overshoot.

Thus the remaining $`274 - 231 = 43`$ entries are pure padding and are not interpreted anywhere else in the code. No correlation‑based or term‑structure features are currently present; those are left as future extensions.

### 3.6.2 Feature scaling

The class `FeatureScaler` in `src/features/build_features.py` standardises the features:

- compute per‑dimension mean and standard deviation on the training set:

  $`\mu_j = \text{mean of feature } j,`$  
  $`\sigma_j = \text{standard deviation of feature } j,`$

- transform any feature matrix $`X`$ via

  $`X_{\text{scaled},j} = \frac{X_j - \mu_j}{\sigma_j + \varepsilon},`$

  with $`\varepsilon = \text{SCALER\_EPSILON} = 10^{-8}`$ to avoid division by zero.

The scaler is fitted on the training period and then applied consistently to training, validation and test sets. The scaler object is saved as `scaler.pkl` in the features directory and later used to un‑scale features in the label generation script when reconstructing raw quantities such as spot and implied volatility.

---

## 3.7 Dataset files and directory structure

The root data directory is defined in `src/config.py` as

- `DATA_DIR = "output/datasets"`.

Within this directory, the key subdirectories and files are:

Processed underlying data

- `output/datasets/processed/processed_prices.parquet`  
  daily adjusted prices for all tickers,
- `output/datasets/processed/returns.parquet`  
  daily log returns,
- `output/datasets/processed/rolling_vol.parquet`  
  rolling realised volatilities (if saved).

Synthetic options and implied volatilities

- `output/datasets/synthetic_iv.parquet`  
  implied volatilities for all 21 assets across the seven moneyness levels,
- `output/datasets/synthetic_options.parquet`  
  Black–Scholes option prices on the same grid.

Features

- `output/datasets/features/X_train.npy`
- `output/datasets/features/X_val.npy`
- `output/datasets/features/X_test.npy`
- `output/datasets/features/dates_train.npy`
- `output/datasets/features/dates_val.npy`
- `output/datasets/features/dates_test.npy`
- `output/datasets/features/scaler.pkl`

Labels

- `output/datasets/labels/y_train.npy`
- `output/datasets/labels/y_val.npy`
- `output/datasets/labels/y_test.npy`

The feature and label arrays are aligned by index and date; the date arrays provide an explicit mapping back to calendar time and underlying raw data.

---

## 3.8 Train, validation and test splits

The temporal splits are defined to avoid look‑ahead bias and to evaluate performance on recent unseen 

- training period: from `BACKTEST_START = "2018-01-01"` up to and including `TRAIN_END = "2023-12-31"`,
- validation period: from `VAL_START = "2024-01-01"` to `VAL_END = "2024-09-30"`,
- test period: from `TEST_START = "2024-10-01"` to `TEST_END = "2025-12-31"`.

These constants are set in `src/config.py` and are used consistently in the pipeline that constructs `X_train`, `X_val`, `X_test` and the corresponding date arrays.

---

## 3.9 Data quality checks

Before features and labels are used for training and backtesting, the following checks are applied:

- no missing values in the processed price series after cleaning,
- log returns and realised volatilities within reasonable bounds,
- correlation matrices positive definite (with minimal regularisation if needed),
- labels finite and in a plausible range (for example not exploding due to extremely large simulated moves),
- successful standardisation (no NaNs after scaling).

The progress report files record aggregate statistics such as mean and standard deviation of labels across train, validation and test sets, and confirm that there are no NaNs or infinite values.

---

## 3.10 Summary

This document describes the data used by the project in a way that matches the current implementation:

- market universe: EURO STOXX 50 index and 20 main constituents between 2018 and 2025,
- underlying  daily adjusted closes and volumes from Yahoo Finance,
- options: synthetic three‑month implied volatility surfaces on a fixed log‑moneyness grid, converted to Black–Scholes prices,
- derived series: rolling realised volatility and correlation matrices from historical returns,
- feature engineering: a 274‑dimensional vector per date, consisting of 231 meaningful features (spots, forwards, rates, yields, IVs) plus 43 zero padding entries,
- scaling: per‑feature standardisation based on the training set,
- storage: a clear layout under `output/datasets/processed`, `output/datasets/features` and `output/datasets/labels`,
- splits: strictly time‑ordered train, validation and test periods.

These specifications ensure that the data pipeline is reproducible and that anyone inspecting the repository can trace exactly how the neural network inputs and labels are constructed from raw Yahoo Finance data and synthetic implied volatility surfaces.
