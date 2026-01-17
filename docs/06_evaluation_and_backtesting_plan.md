# 6. Evaluation and backtesting plan

## 6.1 Overview

This document describes how the dispersion pricing model is evaluated, both

- offline, by comparing neural network predictions to Monte Carlo labels, and
- online, by embedding the model in a simple trading strategy and testing it on historical data.

The goals are:

- to verify that the neural network has learned a stable and accurate approximation of the Monte Carlo pricing map
  $`x_t \mapsto P_{\text{MC}}(x_t)`$,
- to assess whether the resulting pricing signal can be turned into a sensible dispersion trading strategy once realistic implementation details (transaction costs, position management) are taken into account.

The evaluation is divided into:

- offline model evaluation on the test set,
- backtesting on the full historical period with the trained model frozen,
- robustness checks (regimes, walk‑forward validation, sensitivity to parameters),
- reporting and visualisation.

---

## 6.2 Offline evaluation of the neural network

Offline evaluation compares the model’s predictions $`\hat{y}_t`$ to labels $`y_t = P_{\text{MC}}(x_t)`$ on the validation and test sets, without any trading rules.

### 6.2.1 Data splits

Time‑based splits are used to avoid look‑ahead bias:

- training: 2018‑01‑01 to 2023‑12‑31,
- validation: 2024‑01‑01 to 2024‑09‑30,
- test: 2024‑10‑01 to 2025‑12‑31.

The network is fitted only on the training set, hyperparameters are tuned on the validation set, and all final metrics are reported on the test set.

### 6.2.2 Pointwise error metrics

For each date $`t`$ in a given split, the model outputs

$`\hat{y}_t := V_{\text{NN}}(x_t;\theta^{\star}),`$

where $`\theta^{\star}`$ is the parameter set obtained after training and early stopping. The primary scalar error metrics are:

- mean squared error (MSE):

  $`\text{MSE}
    = \frac{1}{N} \sum_{t} (\hat{y}_t - y_t)^2,`$

- root mean squared error (RMSE):

  $`\text{RMSE} = \sqrt{\text{MSE}},`$

- mean absolute error (MAE):

  $`\text{MAE}
    = \frac{1}{N} \sum_{t} |\hat{y}_t - y_t|,`$

- mean absolute percentage error (MAPE), when $`y_t`$ are not too close to zero:

  $`\text{MAPE}
    = \frac{100}{N} \sum_{t}
        \left|\frac{\hat{y}_t - y_t}{y_t}\right|`$ per cent.

These metrics are computed separately for training, validation and test sets and logged to `evaluationreport.json`.

### 6.2.3 Goodness‑of‑fit and correlation metrics

In addition to pointwise errors, the following statistics are used:

- coefficient of determination $`R^2`$ on the test set:

  $`R^2
    = 1 - \frac{\sum_t (\hat{y}_t - y_t)^2}
               {\sum_t (y_t - \bar{y})^2},`$

  where $`\bar{y}`$ is the mean of the labels on the test set;

- Pearson correlation between predictions and labels:

  $`\rho(\hat{y}, y)
    = \frac{\sum_t (\hat{y}_t - \bar{\hat{y}})(y_t - \bar{y})}
           {\sqrt{\sum_t (\hat{y}_t - \bar{\hat{y}})^2}
            \sqrt{\sum_t (y_t - \bar{y})^2}}.`$

These provide a scale‑independent sense of how well the model captures the structure of the Monte Carlo map.

### 6.2.4 Diagnostics and visual checks

The notebook `notebooks/05eval.ipynb` produces a standard set of plots:

- scatter plot of $`\hat{y}_t`$ versus $`y_t`$ on the test set, with the line $`\hat{y} = y`$ as reference,
- histogram of residuals $`\hat{y}_t - y_t`$,
- residuals versus predictions to check for heteroskedasticity or systematic bias,
- temporal plot of labels and predictions over the test period to see whether errors concentrate in particular regimes (for example crisis periods or low‑volatility regimes).

If substantial patterns or biases are observed, they can guide further feature engineering or architecture tweaks.

### 6.2.5 Regime‑specific analysis

Because dispersion is regime‑dependent, performance is also broken down by simple regime definitions. For example:

- volatility regimes:

  - low volatility: index ATM IV below some quantile,
  - medium volatility,
  - high volatility: above a high quantile;

- correlation regimes:

  - low average correlation across constituents,
  - medium,
  - high.

For each regime, the same error and goodness‑of‑fit metrics are computed. This helps determine whether the network is robust across market conditions or whether it overfits to a specific regime.

---

## 6.3 Signal construction for backtesting

Backtesting requires converting neural network outputs into trading signals. Conceptually, on each date $`t`$ the following quantities are defined:

- $`V_{\text{NN}}(t) := V_{\text{NN}}(x_t;\theta^{\star})`$  
  neural network estimate of the fair value;

- $`V_{\text{impl}}(t)`$  
  implementation price of an actual dispersion trade constructed from the synthetic or modelled option surfaces on date $`t`$.

The core mispricing signal is the difference between these two:

$`s_t := V_{\text{NN}}(t) - V_{\text{impl}}(t).`$

Interpretation:

- if $`s_t > 0`$, the model considers dispersion cheap relative to its implementation cost (fair value above market price);
- if $`s_t < 0`$, the model considers dispersion rich relative to implementation.

In the simplest specification, the trading rule is:

- take a long dispersion position when $`s_t`$ is sufficiently positive,
- take a short dispersion position when $`s_t`$ is sufficiently negative,
- stay flat when $`|s_t|`$ is small.

In practice, the backtest code may use a normalised variant of this mispricing (for example dividing by $`|V_{\text{NN}}(t)|`$) to make the signal more scale‑invariant across regimes. The exact formula and sign convention follow the implementation in the backtest module and are fully documented there.

---

## 6.4 Backtest design

The backtesting engine, implemented under `src/backtest`, evaluates how a trading strategy driven by the signal $`s_t`$ would have performed historically using:

- the trained network for fair values,
- synthetic dispersion implementation prices,
- realised dispersion computed from actual Yahoo returns,
- a simplified but realistic model of transaction costs and position management.

### 6.4.1 Trading horizon and trade timing

Each trade corresponds to a three‑month dispersion position, consistent with the label horizon:

- trade horizon: 63 trading days ($`\approx 0.25`$ years),
- entry dates: typically aligned with the feature dates (daily), subject to entry filters.

A straightforward design is:

- on each eligible date $`t`$:
  - compute $`s_t`$,
  - if $`|s_t|`$ exceeds a threshold $`\theta`$, open a new position:
    - long dispersion if $`s_t > \theta`$,
    - short dispersion if $`s_t < -\theta`$;
  - hold the position for a fixed number of days (for example 63) until maturity, at which point the payoff is realised;

- no overlapping positions on the same notional, or if overlapping positions are allowed, cap portfolio exposure.

Exact choices for holding period and overlap are adjustable in the backtest configuration.

### 6.4.2 Implementation price and payoff

On the trade entry date $`t`$:

- implementation price $`V_{\text{impl}}(t)`$ is derived from the synthetic option surfaces or from a dedicated synthetic market module (for example including a realistic bid–ask spread around the Monte Carlo fair value).

At trade expiry $`T_t`$ (for example $`t + 63`$ trading days):

- realised dispersion payoff $`\Pi_{\text{real}}(t)`$ is computed from actual Yahoo returns using the same realised variance formula as in the label generator:

  $`\widehat{\mathrm{RV}}_T^i
    = 252 \sum_{k=1}^{N}
          \left(
            \log\frac{S^i_{t_k}}{S^i_{t_{k-1}}}
          \right)^2,`$

  and

  $`\Pi_{\text{real}}(t)
    = \widehat{\mathrm{RV}}_T^0
      - \sum_{i=1}^{20} w_i \,\widehat{\mathrm{RV}}_T^i,`$

  where $`w_i`$ are the basket weights.

The trade profit and loss, before costs, for a unit notional trade opened at $`t`$ with sign $`\sigma_t \in \{+1,-1\}`$ is:

$`\text{P\&L}_{\text{gross}}(t)
  = \sigma_t \big(\Pi_{\text{real}}(t) - V_{\text{impl}}(t)\big).`$

Transaction costs and bid–ask effects are then deducted to obtain net P&L.

### 6.4.3 Transaction costs and slippage

Transaction cost modelling is handled by modules such as:

- `src/backtest/transactioncosts.py`,
- `src/backtest/syntheticmarket.py`,

and typically includes:

- a fixed bid–ask spread on the dispersion implementation price,
- proportional costs on notionals to reflect slippage and fees,
- possibly a cost of carry or financing for the position over the holding period.

These costs can be specified in basis points or as a fraction of the notional, and are configurable. The backtest reports both gross and net performance to show the impact of costs.

### 6.4.4 Position sizing and risk management

The simplest position sizing rule is to trade a fixed notional per signal, for example one unit of the dispersion payoff. Extensions may include:

- scaling trade size with signal strength, for example

  $`\text{size}_t = \min\left(S_{\max},
     S_0 + k \frac{|s_t|}{\theta}\right),`$

  with caps $`S_{\max}`$ and base size $`S_0`$;

- capping the number of concurrent open trades or total portfolio exposure;

- imposing hard stop‑loss rules (for example closing trades early if mark‑to‑market P&L breaches a negative threshold).

These extensions can be explored in later phases; the initial backtest uses a simple, transparent sizing rule for interpretability.

---

## 6.5 Performance metrics

Performance is summarised using standard portfolio metrics computed on the sequence of daily P&L from the backtest.

Let $`r_d`$ denote the daily return of the strategy (P&L divided by notional or starting capital) on day $`d`$.

### 6.5.1 Return and Sharpe ratio

The annualised return is

$`R_{\text{ann}}
  = 252 \,\bar{r},`$

where $`\bar{r}`$ is the mean daily return. The annualised volatility is

$`\sigma_{\text{ann}}
  = \sqrt{252} \,\hat{\sigma}_r,`$

where $`\hat{\sigma}_r`$ is the standard deviation of daily returns.

The annualised Sharpe ratio (with zero risk‑free rate for simplicity) is

$`\text{Sharpe}
  = \frac{R_{\text{ann}}}{\sigma_{\text{ann}}}.`$

### 6.5.2 Drawdowns and risk

The equity curve is the cumulative sum of daily P&L. From this, the maximum drawdown is computed as:

- running peak of the equity curve,
- maximum peak‑to‑trough drop over the backtest period.

The Calmar ratio (if desired) is

$`\text{Calmar}
  = \frac{R_{\text{ann}}}{|\text{MaxDrawdown}|}.`$

These metrics quantify the risk of the strategy and its tolerance to adverse regimes.

### 6.5.3 Hit rate and trade‑level statistics

On a per‑trade basis, the backtest collects:

- number of trades,
- proportion of profitable trades (hit rate),
- average P&L per trade,
- distribution of trade P&L (histogram, quantiles),
- average and distribution of holding periods,
- turnover (for example annualised number of trades).

These help diagnose whether performance relies on a few large wins, on many small wins, or on a particular side (long versus short dispersion).

### 6.5.4 Benchmarks

To interpret results, the following benchmarks can be considered:

- a naive strategy that always trades in one direction (for example always long dispersion) with the same trade timing and horizon;
- a simple heuristic strategy driven by historical correlation only;
- a strategy that uses the Monte Carlo labels directly (if available in the backtest window) as a hypothetical oracle.

Comparing against these baselines helps isolate the value added by the neural network relative to simple rules.

---

## 6.6 Robustness and validation

To reduce the risk of overfitting and to understand when the strategy works, several robustness checks are planned.

### 6.6.1 Regime‑based backtest breakdown

Performance metrics are broken down by simple regimes:

- volatility regimes (low, medium, high) based on index realised or implied volatility,
- correlation regimes (low, medium, high) based on rolling average constituent correlation,
- bull versus bear markets (for example based on index trend).

For each regime, the backtest engine reports Sharpe, drawdown, hit rate and other metrics, so that sensitivity to regime is explicit.

### 6.6.2 Threshold sensitivity

The entry threshold $`\theta`$ is a key parameter controlling trade frequency and average mispricing per trade. A grid search is performed over a set of candidate thresholds, for example:

$`\theta \in \{0.5, 1.0, 1.5, 2.0, 3.0, 4.0\},`$

where $`\theta`$ is measured in units of the chosen signal definition (for example standard deviations or percentage mispricing).

For each threshold, the backtest is rerun, and metrics such as Sharpe ratio, number of trades and maximum drawdown are recorded in a table (for example `step7_threshold_grid.csv`). This allows selection of a threshold that balances performance and robustness.

### 6.6.3 Walk‑forward validation

To test stability over time and guard against overfitting to a single training–test split, walk‑forward validation can be performed:

1. Choose an initial training window, for example 2018–2022.
2. Train the network on this window and evaluate on a subsequent window (for example 2023).
3. Slide the training window forward (for example 2019–2023) and test on the next period (2024), retraining the model each time.
4. Aggregate performance statistics across all test windows.

This procedure, implemented in `src/backtest/walkforward.py`, provides a more realistic sense of how the model might behave when periodically retrained in production.

### 6.6.4 Stress tests

Additional stress tests can include:

- artificially increasing transaction costs and checking whether the strategy remains profitable;
- halving the number of Monte Carlo paths used in labels to simulate noisier training targets;
- perturbing features (for example adding noise) at inference time to test the robustness of predictions.

These experiments help determine whether the strategy’s performance is fragile or robust to implementation imperfections.

---

## 6.7 Reporting and visualisation

The final evaluation step is to summarise results in a human‑readable form.

The report generator in `src/backtest/reportgenerator.py` creates:

- tables of offline metrics (MSE, MAE, $`R^2`$, correlation) for train, validation and test sets;
- summary tables of backtest performance by threshold and by regime;
- plots:
  - cumulative P&L (equity curve),
  - drawdown curve,
  - histogram of trade P&L,
  - scatter plots of signal versus subsequent P&L;
- commentary on where the strategy performs best or fails.

A timestamped text report (for example in `output/models/reports/`) collects all key numbers and comments in one document. Optional notebooks (for example `notebooks/06backtestanalysis.ipynb`) provide interactive visual exploration.

---

## 6.8 Summary

This document defines how the dispersion pricing model is evaluated:

- offline, by comparing neural network predictions to Monte Carlo labels using error metrics and correlation measures;
- online, by embedding the trained model in a simple dispersion trading strategy and backtesting it on historical data, with realistic payoff computation from Yahoo returns and a basic transaction cost model;
- via robustness checks, including regime analysis, threshold sensitivity and walk‑forward validation.

These procedures provide a complete view of model quality, from pure function approximation to trading performance, and form the basis for deciding whether to iterate on the architecture, features or trading rules, or to progress to more advanced extensions such as model‑free MOT labels and structured network architectures.