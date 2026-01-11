# 3. Data Specification

## 3.1 Overview

This section specifies the **exact data sources, fields, formats, and feature engineering** required to train and backtest the dispersion trading system. It serves as the blueprint for the data pipeline in Phase 2.

## 3.2 Data universe and time horizon

### 3.2.1 Market and instruments

**Primary index:** EuroStoxx 50 (SX5E)
- Underlying: SX5E index level.
- Options: European vanilla calls and puts on SX5E.
- Maturities of interest: 3 months (primary focus), with secondary maturities (1M, 6M) for term-structure features.

**Constituents:** Top 20 stocks by current index weight in SX5E
- Underlyings: Individual stock prices (ISIN or Bloomberg ticker).
- Options: European vanilla calls and puts on each stock.
- Same maturities as index.

### 3.2.2 Historical time period

**Backtest window:** 2018-01-01 to 2025-12-31 (approximately 8 years of history)
- Rationale: Captures diverse market regimes (post-crisis recovery, low-vol period, COVID crash, rate hikes, geopolitical events).
- Sufficient history for robust statistical validation.

**Data frequency:** Daily
- Sampling: End-of-day (close) prices.
- Alternative intraday data may be considered later but not required for Phase 1.

## 3.3 Data fields and raw formats

### 3.3.1 Index data (SX5E)

For each trading day, record:

| Field | Type | Description |
|-------|------|-------------|
| `date` | YYYYMMDD | Trading date |
| `sx5e_spot` | float | SX5E index level at close |
| `sx5e_forward_3m` | float | 3-month forward level (or computed from spot + rates) |
| `sx5e_dividend_yield_3m` | float | Annualized dividend yield for 3M horizon |
| `sx5e_rate_3m` | float | 3-month risk-free rate (e.g. EURIBOR 3M) |

**Options on SX5E (3-month maturity):**

For each day and maturity, record vanilla option prices or implied vols on a strike grid:

| Field | Type | Description |
|-------|------|-------------|
| `date` | YYYYMMDD | Trading date |
| `maturity` | float | Years to maturity (e.g., 0.25 for 3M) |
| `strike_log_moneyness` | float | \(\log(K / F)\) where \(K\) = strike, \(F\) = forward |
| `call_bid`, `call_mid`, `call_ask` | float | Call option prices (bid/mid/ask) |
| `put_bid`, `put_mid`, `put_ask` | float | Put option prices (bid/mid/ask) |
| `call_iv`, `put_iv` | float | Implied volatilities (Black–Scholes convention) |

**Strike grid:** moneyness levels
\[
\log(K/F) \in \{-0.30, -0.20, -0.10, 0.00, 0.10, 0.20, 0.30\}.
\]
This gives 7 strikes per day, balancing granularity vs data availability.

### 3.3.2 Constituent data (each of 20 stocks)

Repeat the structure above for each constituent. For constituent \(i\), record:

| Field | Type | Description |
|-------|------|-------------|
| `date` | YYYYMMDD | Trading date |
| `stock_i_spot` | float | Stock price at close |
| `stock_i_forward_3m` | float | 3-month forward |
| `stock_i_dividend_yield_3m` | float | Dividend yield |
| `stock_i_call_mid`, `stock_i_call_iv` | float | Call prices / IVs on 7-strike grid |
| `stock_i_put_mid`, `stock_i_put_iv` | float | Put prices / IVs on 7-strike grid |

**Index weights:**

For each day, record the weight of each constituent in the SX5E:

| Field | Type | Description |
|-------|------|-------------|
| `date` | YYYYMMDD | Trading date |
| `weight_i` | float | Weight of constituent \(i\), \(\sum_i w_i = 1\) |

These weights are used to construct the weighted dispersion payoff and the basket for comparison.

## 3.4 Data sources and access

### 3.4.1 Underlying price data (real)

Historical prices for the SX5E index (or a liquid ETF proxy) and the 20
constituent stocks are obtained from Yahoo Finance using its public API
(e.g. via the `yfinance` Python package).

For each asset and date we retrieve:

- Open, high, low, close, adjusted close
- Volume

Adjusted close prices are used to compute log-returns and realised variance,
as they account for splits and dividends.

### 3.4.2 Option surface data (synthetic)

Historical 3-month option surfaces for SX5E and the 20 constituents are not
directly available from free sources. Instead, this project generates
synthetic but realistic implied volatility surfaces as follows:

1. For each date and asset, estimate a short-term realised volatility from
Yahoo Finance returns (e.g., 30–60 day window).
2. Use a parametric smile model (e.g., quadratic in log-moneyness or SVI-like)
to construct a 3-month implied volatility curve across strikes \(\log(K/F) \in \{-0.3, -0.2, \ldots, 0.3\}\).
3. Convert implied volatilities into option prices using the Black–Scholes
formula, given the current spot, forward, and discount factor.

These synthetic option prices are treated as if they were observed market
vanilla prices when defining \(\mathcal{M}_{\text{vanilla}}\) and running the
MOT/LP solver. The synthetic IV model will be calibrated to reproduce
typical EuroStoxx 50 volatility levels and skew documented in the literature
and from current market snapshots.

### 3.4.3 Institutional data (future extension)

The data pipeline and model are designed so that, if institutional options
data (e.g., OptionMetrics, Bloomberg, or iVolatility) becomes available in
the future, the synthetic surface generator can be replaced with real
historical option quotes without changing the overall architecture.

## 3.5 Data preprocessing and feature engineering

### 3.5.1 Handling missing or illiquid data

**Missing strikes / maturities:**
- If a strike is not traded on a given day, use last-observation-carried-forward (LOCF) or interpolation.
- Alternatively, drop that day from training if coverage is too sparse.

**Bid–ask spreads:**
- Work with mid prices: \((P_{\text{bid}} + P_{\text{ask}}) / 2\).
- Flag observations with bid–ask spread > some threshold (e.g., 5% of mid) as potentially illiquid; consider downweighting in training loss.

**Corporate actions:**
- Adjust for stock splits, dividends, and mergers.
- Maintain continuity of forward prices and implied vol surfaces.

### 3.5.2 Implied volatility surface construction

For each asset on each day:

1. **Raw ** Bid/ask option prices on the 7-strike grid, maturity \(T=0.25\) years.
2. **Mid prices:** Compute mid of bid/ask.
3. **Implied vols:** Use Black–Scholes inversion to convert prices to IVs.
   - For puts: \(P_{\text{put}} = C_{\text{call}} - F + K \cdot \beta(T)\) (put-call parity).
   - Invert Black–Scholes formula: \(C_{\text{BS}}(S, K, T, r, \sigma) = P_{\text{call}}\) to solve for \(\sigma\).
4. **Smoothing (optional):** If raw IVs are noisy, fit a smooth spline or SABR parametrization.

**Output:** For each asset and day, an implied vol **curve** across the 7 strikes.

### 3.5.3 Feature normalization

All features will be **standardized** before input to the neural network:

\[
x_{\text{normalized}} = \frac{x - \mu}{\sigma + \varepsilon},
\]

where \(\mu, \sigma\) are the mean and standard deviation computed on the training set, and \(\varepsilon > 0\) is a small constant to avoid division by zero.

Normalization is crucial for:
- Stable neural network training (faster convergence, avoiding gradient explosion/vanishing).
- Fair weighting of features with different scales (e.g., spot price in EUR vs. implied vol in %).

## 3.6 Feature vector construction

### 3.6.1 Flattened feature vector (baseline approach)

The network input \(x \in \mathbb{R}^{n_{\text{in}}}\) is constructed by flattening market data into a single vector.

**Spot and forward prices:**
- Index spot and forward: 2 values.
- 20 constituent spots and forwards: 40 values.
- Total: 42 values.

**Risk-free rates and yields:**
- 3M EURIBOR (single value for all assets): 1 value.
- SX5E dividend yield: 1 value.
- 20 constituent dividend yields: 20 values.
- Total: 22 values.

**Implied volatilities (strikes × assets):**
- 7 moneyness levels × (1 index + 20 constituents) = 7 × 21 = 147 values.

**Additional features (term structure and skew):**
- Variance term structure (IV at 1M and 6M relative to 3M): 2 × 21 = 42 values.
- Skew measures (25-delta IV-RR): 1 × 21 = 21 values.
- Total: 63 values.

**Grand total:** \(n_{\text{in}} = 42 + 22 + 147 + 63 = 274\).

### 3.6.2 Structured input (future extension)

Alternatively, preserve the structure:

- Input tensor shape: \([21 \text{ assets} \times 7 \text{ strikes} \times N_{\text{features}}]\).
- Feed into convolutional layers or attention mechanisms.
- Advantage: network can exploit spatial structure (smoothness across strikes, similarity across assets).

For Phase 1, we use the **flattened approach** (Section 3.6.1) for simplicity. Structured inputs can be explored in Phase 2 if needed.

## 3.7 Dataset statistics and summary

### 3.7.1 Training / validation / test splits

- **Training set:** 2018-01-01 to 2023-12-31 (roughly 1500 trading days)
  - Used to learn network parameters.
- **Validation set:** 2024-01-01 to 2024-09-30 (roughly 180 trading days)
  - Used for hyperparameter tuning and early stopping.
- **Test set:** 2024-10-01 to 2025-12-31 (roughly 60 trading days)
  - Held-out for final model evaluation; no parameter updates.

Rationale: Time-ordered splits to avoid lookahead bias and test on recent unseen data.

### 3.7.2 Expected feature statistics

**Spot prices (normalized by forward):**
- Mean moneyness \(\log(S/F) \approx 0\) (by definition).
- Std dev: ~1–3% depending on near-term volatility and dividends.

**Implied volatilities (3M, at-the-money):**
- Mean: 15–25% annualized (typical for equity index and stocks).
- Std dev: 5–15% (varies with market regime).
- Range: 5% (low vol) to 60%+ (crisis periods).

**Term structure (IV_1M / IV_3M):**
- Mean ratio: ~1.0–1.2 (term structure typically upward-sloping).
- Std dev: ~0.2.

## 3.8 Label generation (ground truth for training)

Labels \(y_i = (L_i, U_i)\) or \(y_i = P_{\text{fair},i}\) are computed offline via solving the MOT / LP problem (detailed in Step 5). This section defines the placeholder:

**For each training sample (date and market state \(x_i\)):**

1. Use the observed vanilla option prices on that date.
2. Solve the martingale optimal transport problem (Section 2.3.1) to compute:
   - \(L(x_i)\): lower bound on dispersion payoff.
   - \(U(x_i)\): upper bound.
3. Compute fair value: \(P_{\text{fair}}(x_i) = \frac{L(x_i) + U(x_i)}{2}\).
4. Store as label \(y_i\).

**Computational note:** Solving the MOT problem for a single date can take minutes to hours depending on grid resolution. For 1500 training dates, expect significant offline compute. This is covered in detail in Step 5.

## 3.9 Data quality checks and validation

Before using data in training, perform:

1. **Completeness:** Check that all required fields are present for each date and asset.
2. **Arbitrage checks:** Verify put-call parity (within bid-ask spread).
3. **Outlier detection:** Flag extreme option prices or IVs that may indicate data errors.
4. **Continuity:** Check for gaps or jumps in time series inconsistent with market events.
5. **Dividend adjustments:** Ensure forward prices correctly account for corporate actions.

Example thresholds:
- IV > 100% or IV < 1%: investigate.
- Bid > Ask: data error.
- Spread > 10% of mid: potentially illiquid, flag for downweighting.

## 3.10 Summary

This section specified:

- **Universe:** SX5E + 20 constituents, daily data, 2018–2025 (Section 3.2).
- **Raw data fields:** Spots, forwards, rates, yields, option prices/IVs (Section 3.3).
- **Data sources:** OptionMetrics, Bloomberg, Refinitiv (Section 3.4).
- **Preprocessing:** Smoothing, normalization, corporate actions (Section 3.5).
- **Feature engineering:** Flattened vector, 274 dimensions (Section 3.6).
- **Dataset splits:** Train/val/test by time period (Section 3.7).
- **Labels:** MOT-derived bounds (Section 3.8).
- **Quality checks:** Arbitrage, outliers, continuity (Section 3.9).

In **Phase 2**, the data pipeline will be implemented to:
1. Load raw data from chosen source.
2. Preprocess and engineer features.
3. Generate labels via MOT solver.
4. Prepare batches for neural network training.
