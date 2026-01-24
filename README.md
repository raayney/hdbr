# High-Dimensional Basket Dispersion Trading via Model-Free Deep Learning

[![Sharpe](https://img.shields.io/badge/Walk--Forward%20Sharpe-0.62-blue)](https://github.com/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-orange)](https://pytorch.org/)

This is a personal research prototype: feedforward NNs for data-driven pricing of high-dim basket dispersion (basket var minus index var) on EURO STOXX 50 stocks from market snapshots. [file:6]

Uses 274D features (spots/forwards, synthetic IV surfaces/skew/term, 21×21 corrs, RV) → NN → fair value → signals (|pred - mkt| > 2σ). Walk-forward Sharpe 0.62 (462 trades, 2023-2025, 35bps RT costs). [file:6]

Complete as of Jan 2026: analysis notebooks, backtests; research-focused, not production-ready. [file:6]

## Theoretical Foundation

Data-driven NN pricing without model params, per Neufeld & Sester (2021). [web:0]

**Theorem 11.2 (Continuity of Hedging Prices)**: For continuous parametric payoff families $`\theta \mapsto \Phi_\theta`$ on compact $`K_1`$, super/sub-hedging prices $`(D^{B,\bar{B}}(K,\pi,S_0)(\Phi_\theta)`$, $`D^{B,\bar{B}}(K,\pi,S_0)(\Phi_\theta)`$ continuous w.r.t. market data $`(K, \pi, S_0, \theta)`$ in suitable topology. [web:0]

Implies NNs (universal approximators) can learn bounds for dispersion $`\Phi(S_T) = \sum w_i \sigma_i^2 - \sigma_b^2`$. Trained via MSE:
\[
$$`\mathcal{L}(\theta) = \frac{1}{N} \sum (f_\theta(x_i) - c_i)^2 + \lambda \|\theta\|_2^2
\]`$$
Adam $`lr=10^{-3}`$, early stopping; val R²=0.88, test 0.9712. [web:0][file:3][file:6]

## Key Results

| Metric     | Single-Fold (2024-25) | Walk-Forward (2023-25) | Kelly Impact     |
|------------|-----------------------|-------------------------|------------------|
| Sharpe     | 0.84 [file:6]        | 0.62 ± 0.10 [file:6]  | -                |
| Win Rate   | 61% [file:6]         | 43% [file:6]           | -                |
| Total P&L  | 885 [file:6]         | 977 [file:6]           | +127% [file:6]   |
| # Trades   | 298 [file:6]         | 462 [file:6]           | -                |
| Max DD     | -2.9% [file:6]       | -7.8% [file:6]         | -                |

1,926 obs (2018-25), zero NaNs. Opt threshold 2σ. [file:6]

## Pipeline

* **21 Stocks** $\rightarrow$ Prices/IV synth $\rightarrow$ 274 feats (PCA $r=0.58\text{-}0.72$)
* $\downarrow$
* **BasketPricingNet**: $274 \rightarrow 256 \rightarrow 128 \rightarrow 64 \rightarrow 1$ (ReLU, drop=0.2)
* $\downarrow$
* **Signal** $\rightarrow$ **Kelly** $\rightarrow$ **Backtest** (walk-forward CV)

## Notebooks

- **00_monte_carlo_synthetic_options.ipynb
- **01_framework.ipynb: Model-free pricing framework & data pipeline. [file:4]
- **02_feature_architecture.md**: 274-dimensional feature engineering & signal strength. [file:5]
- **03_neural_network_training_final.ipynb**: NN training & test evaluation (test R²=0.9712). [file:3]
- **04_signal_generation.ipynb**: Signal generation & threshold optimisation. [file:1]
- **05_walk_forward_analysis.ipynb**: Regime-aware robustness & position sizing. [file:2]

## Setup

1. `pip install -r requirements.txt`
2. Jupyter: Run 00→05.
3. `streamlit run dashboard.py` (interactive views). [file:6]

Stack: PyTorch 2.0.1, Pandas/NumPy/sklearn, Streamlit/Plotly. Data: Train/Val/Test splits ~1251/325/350 days. [file:6]

## Limitations

- Synthetic IVs (no live options).
- Historical only; no real-time/live trading.
- Single NN; no ensembles/regimes/SHAP. [file:6]

## References

- Neufeld & Sester, arXiv:2103.11435v3 [web:0]
- PROJECT_OUTLINE.md [file:6]

MIT License. Paris, FR – Jan 24, 2026.
