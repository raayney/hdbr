# 1. Problem and Payoff Specification

## 1.1 High-level objective

This project's goal is to build a model-free trading system powered on a deep-learning-based pricing engine for **high-dimensional basket dispersion trades**.

At a high level, the system will:

- Observe liquid market data for an index and a set of its constituents (spot prices and vanilla options).
- Produce a **model-free fair value (or no-arbitrage band)** for a dispersion-style payoff.
- Compare this to the market-implied price of an implementable dispersion trade.
- Generate a **trading signal** indicating whether dispersion is cheap or expensive and by how much.

This document defines the **exact payoff(s)** and **signal(s)** the system focuses on.

## 1.2 Market setup and notation

Let us consider a financial market with:

- The **EuroStoxx 50 index** \(S^0 = (S_t^0)_{t \ge 0}\).
- The **top 20 constituents by index weight** \(S^1,\dots,S^{20}\), where each \(S^i = (S_t^i)_{t \ge 0}\) denotes the price process of asset \(i\).

Let \(d = 20\) denote the number of constituents considered.

- A fixed **trading horizon** \([0,T]\) for each dispersion trade, where in this project we focus on
  \[
  T = 3 \text{ months} \approx 0.25 \text{ years}.
  \]

At time \(t = 0\), the observable quantities include:

- Current index level \(S_0^0\) and constituent levels \(S_0^1,\dots,S_0^d\).
- Vanilla European call and put prices (or equivalently implied volatilities) on \(S^0,S^1,\dots,S^d\) for a range of strikes and maturities including \(T\).
- Interest rate and dividend information encoded via discount factors or forward prices.

Let \(w_1,\dots,w_{20}\) be the **constituent weights in the SX5E index** satisfying
\[
\sum_{i=1}^d w_i = 1, \quad w_i \ge 0.
\]
Define the corresponding **basket** at time \(t\) by
\[
B_t := \sum_{i=1}^d w_i S_t^i.
\]

## 1.3 Variance dispersion trade (canonical payoff)

A classical variance dispersion trade compares:

- The **realised variance** of the index.
- The **weighted sum of realised variances** of the constituents.

In model-free continuous-time notation, for an asset \(S^i\) over \([0,T]\), the quadratic variation of its log-price is
\[
\langle \log S^i \rangle_T,
\]
which is the object replicated by a variance swap.

For the index \(S^0\) and each constituent \(S^i\), define:

- Index realised variance: \(\mathrm{RV}^0_T := \langle \log S^0 \rangle_T\).
- Constituent realised variances: \(\mathrm{RV}^i_T := \langle \log S^i \rangle_T\), \(i=1,\dots,d\).

A **normalized dispersion payoff** at time \(T\) is then defined as
\[
\Phi_{\text{disp}} := \sum_{i=1}^d w_i \mathrm{RV}^i_T - \mathrm{RV}^0_T.
\]

Positive values of \(\Phi_{\text{disp}}\) correspond to “more idiosyncratic volatility” relative to index volatility, i.e. high dispersion.

In practice, \(\Phi_{\text{disp}}\) is approximated by implementable positions in:

- A **short position** in an index variance swap (or option strip proxy).
- **Long positions** in single-name variance swaps (or option strip proxies) with weights \(w_i\).

This project treats \(\Phi_{\text{disp}}\) as an abstract payoff functional and focuses on **model-free pricing bounds** and **fair values** for this payoff given current market option prices.

### 1.3.1 Discrete-time / implementable approximation

For implementation and backtesting, realised variance can be computed discretely. Given a partition \(0 = t_0 < t_1 < \dots < t_N = T\), define
\[
\widehat{\mathrm{RV}}^i_T := \sum_{k=1}^N \left( \log \frac{S^i_{t_k}}{S^i_{t_{k-1}}} \right)^2.
\]

The corresponding **discrete dispersion payoff** is
\[
\widehat{\Phi}_{\text{disp}} := \sum_{i=1}^d w_i \widehat{\mathrm{RV}}^i_T - \widehat{\mathrm{RV}}^0_T.
\]

In backtests, \(\widehat{\Phi}_{\text{disp}}\) will be used as the realised P\&L driver of the dispersion leg.

For this project, we focus on dispersion trades with a **maturity of \(T = 3 \text{ months}\)**, leveraging the liquid 3M options market on SX5E and its constituents.

### Data and implementation note

In an institutional setting, the option prices used to compute model-free bounds would be
actual market quotes for SX5E and its constituents. In this project, due to the absence of institutional options data access, historical underlying prices are sourced from Yahoo Finance, and 3-month vanilla option surfaces are generated synthetically using a calibrated implied volatility model designed to mimic EuroStoxx 50 market behaviour.

All theoretical results in this document are formulated for true market option prices; in the
implementation, synthetic surfaces play the role of these market prices.

## 1.4 What the pricing engine will produce

Let \(x\) denote the **observable market state at time 0**, including:

- Index and constituent prices.
- Vanilla option prices or implied volatilities on index and constituents at maturity \(T\) (and possibly other maturities).
- Any additional features (term-structure indicators, skew measures, etc., to be specified in the data document).

The deep-learning pricing engine is intended to approximate **model-free no-arbitrage bounds** for the dispersion payoff given \(x\).

Formally, define:

- \(L(x)\): model-free **lower no-arbitrage bound** on the time-0 price of \(\Phi_{\text{disp}}\) consistent with observed vanilla prices.
- \(U(x)\): model-free **upper no-arbitrage bound**.

The engine will output:

1. The no-arbitrage bounds \((L(x), U(x))\).  
2. The derived **fair value** defined as the midpoint:
   \[
   P_{\text{fair}}(x) := \frac{L(x) + U(x)}{2}.
   \]

Both outputs will be used: \([L,U]\) for robustness assessment and constraint checking, and \(P_{\text{fair}}\) as the basis for trading signals.

## 1.5 Trading signal definition

Assume there exists a **market-implementable dispersion trade** whose time-0 premium is \(P_{\text{mkt}}(x)\). The relationship between \(P_{\text{mkt}}(x)\) and the model-free band \([L(x),U(x)]\) yields the trading signal.

Define a **relative mispricing signal**
\[
\Delta(x) := \frac{P_{\text{mkt}}(x) - P_{\text{fair}}(x)}{|P_{\text{fair}}(x)| + \varepsilon},
\]
where \(\varepsilon > 0\) is a small stabilization constant.

Interpretation:

- If \(\Delta(x) \ll 0\): dispersion is **cheap** relative to model-free fair value → candidate to **buy dispersion** (long \(\Phi_{\text{disp}}\)).  
- If \(\Delta(x) \gg 0\): dispersion is **rich** → candidate to **sell dispersion** (short \(\Phi_{\text{disp}}\)).  
- If \(P_{\text{mkt}}(x)\) lies well inside \([L(x),U(x)]\) and \(|\Delta(x)|\) is small: no strong signal.

Thresholds (e.g. \(\Delta(x) < -\theta\) or \(\Delta(x) > \theta\)) and position sizing rules will be specified in the evaluation and backtesting document.

## 1.6 Scope and possible extensions

The core of Phase 1 and initial implementation will focus on the **variance dispersion payoff** \(\Phi_{\text{disp}}\) as defined above.

Natural extensions include:

- Alternative dispersion payoffs based on **basket options**, such as:
  - Calls on the index vs baskets of single-name calls.
  - Worst-of or best-of options.
- Hybrid payoffs combining variance and level moves.

These can be incorporated later by extending the family of payoffs \(\Phi_\theta\) and treating \(\theta\) as an additional input to the pricing network.