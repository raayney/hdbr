# 4. Label Generation and Training Data

## 4.1 Overview

The neural network requires a large set of **labeled examples** to learn the mapping from market states to model-free fair values (or bounds). This section describes how these labels are generated and how the training dataset is constructed.

## 4.2 Synthetic IV surface model

### 4.2.1 Conceptual approach

For each historical date and each asset (index + 20 constituents), we need a plausible 3-month implied volatility surface. We generate these synthetically via a **parametric smile model** rather than reading from live market data.

The synthetic IV surface serves two purposes:

1. **Feature construction:** The NN input \(x_i\) includes IVs (and option prices derived from them) for each strike and maturity.
2. **Label generation:** The same synthetic surfaces are fed into the MOT/LP solver to compute ground-truth bounds \((L(x_i), U(x_i))\).

### 4.2.2 Functional form

For each asset \(i\) on date \(t\), the 3-month implied volatility is modeled as a quadratic function of log-moneyness:

\[
\sigma_i(k,t) = a_i(t) + b_i(t) \cdot k + c_i(t) \cdot k^2,
\]

where \(k = \log(K/F)\) is the log-moneyness (log of strike divided by forward), and \((a_i, b_i, c_i)\) are time-dependent parameters.

**Interpretation:**

- \(a_i(t)\): ATM (at-the-money) level of implied volatility on date \(t\).
- \(b_i(t)\): skew (asymmetry across strikes).
- \(c_i(t)\): smile curvature.

### 4.2.3 Parameter calibration

Parameters are calibrated to match typical EuroStoxx 50 behaviour:

**ATM level \(a_i(t)\):**

- For the index (i=0): Use **historical realised volatility** (e.g., 30-day rolling window) from Yahoo prices, scaled by a factor (e.g., 1.2) to approximate implied vs realised.
  
  \[
  a_0(t) = 1.2 \times \text{RV}^0_{t-30:t}.
  \]

- For each constituent \(i \in \{1,\dots,20\}\): Similarly,
  
  \[
  a_i(t) = 1.2 \times \text{RV}^i_{t-30:t}.
  \]

**Skew \(b_i(t)\):**

- For equity indices and single stocks, typical skew is **negative** (puts more expensive than calls relatively).
- Set \(b_i(t) = -0.05\) (constant across all dates and assets, or vary slightly with regime if desired).

**Smile \(c_i(t)\):**

- Positive curvature, typical for equity options.
- Set \(c_i(t) = 0.01\) (constant, or calibrated to match observed option surfaces if real data is available).

**Strikes grid:**

- Compute IVs on the fixed moneyness grid:
  \[
  k \in \{-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3\}.
  \]

### 4.2.4 Converting IV to option prices

Once the IV surface is constructed, convert to option prices using the **Black–Scholes formula**:

\[
C_{\text{BS}}(S, K, T, r, q, \sigma) = S e^{-q T} \Phi(d_1) - K e^{-rT} \Phi(d_2),
\]

where \(\sigma = \sigma_i(k, t)\) is the strike-dependent IV from above, \(q\) is the dividend yield, and \(d_1, d_2\) are the usual Black–Scholes parameters.

For puts, use put-call parity:

\[
P = C + K e^{-rT} - S e^{-q T}.
\]

### 4.2.5 Implementation in code

Create a function (pseudocode):

```python
def synthetic_iv_surface(spot, forward, t_now, t_mat, 
                          realized_vol, skew=-0.05, smile=0.01):
    """
    Args:
        spot: current spot price
        forward: forward price at maturity t_mat
        t_now: current date (numeric, e.g., datetime to float)
        t_mat: maturity date (numeric)
        realized_vol: historical realized vol (30-day window)
        skew, smile: model parameters
    
    Returns:
        dict with keys 'moneyness', 'iv', 'call_price', 'put_price'
    """
    tau = t_mat - t_now  # years to maturity
    atm_iv = 1.2 * realized_vol
    
    moneyness_grid = [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]
    ivs = []
    call_prices = []
    put_prices = []
    
    for k in moneyness_grid:
        # Quadratic smile
        iv = atm_iv + skew * k + smile * k**2
        ivs.append(iv)
        
        # Compute strike from moneyness
        K = forward * np.exp(k)
        
        # Black-Scholes prices
        c = black_scholes_call(spot, K, tau, r, q, iv)
        p = c - forward * np.exp(-r * tau) + K * np.exp(-r * tau)
        
        call_prices.append(c)
        put_prices.append(p)
    
    return {
        'moneyness': moneyness_grid,
        'iv': ivs,
        'call_price': call_prices,
        'put_price': put_prices
    }
```

## 4.3 Feature construction

For each date \(t\) in the historical sample, construct a feature vector
\(x_t \in \mathbb{R}^{n_{\text{in}}}\) summarising the market state
(index + constituents + option surfaces). This vector is the input to the
neural network.

### 4.3.1 Spot and forward prices

For maturity \(T = t + 0.25\) years (3 months), record:

- SX5E spot price: \(S_0^0(t)\).
- SX5E forward price: \(F_0^0(t,T)\).
- Constituent spot prices: \(S_0^i(t)\), \(i = 1,\dots,20\).
- Constituent forward prices: \(F_0^i(t,T)\), \(i = 1,\dots,20\).

Total: \( (1 + 20) \times 2 = 42 \) scalar features.

### 4.3.2 Risk-free rates and dividend yields

- 3-month risk-free rate \(r(t)\) (single value, applied to all assets).
- Index dividend yield \(q^0(t)\).
- Constituent dividend yields \(q^i(t)\), \(i = 1,\dots,20\), inferred from
  spot–forward relationships if not available directly.

Total: \(1 + (1 + 20) = 22\) features.

### 4.3.3 Implied volatility curves

For each asset \(i = 0,1,\dots,20\) (index + 20 stocks) and each moneyness
level \(k \in \{-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3\}\):

- 3-month implied volatility \(\sigma_i(k,t)\) from the synthetic IV model.

This gives:

- 21 assets × 7 strikes = 147 IV features.

If desired, option prices (calls or puts) at the same grid can be added,
but they are one-to-one with IVs given \(S, F, r, q\), so using IVs alone
is sufficient.

### 4.3.4 Optional term-structure and skew features

To enrich the feature set, the synthetic IV model can be extended to
additional maturities, e.g. 1M and 6M. Then define:

- **Term structure ratios:**
  - \(\sigma_i^{1M}(0) / \sigma_i^{3M}(0)\) and
    \(\sigma_i^{6M}(0) / \sigma_i^{3M}(0)\), for each asset \(i\).
- **Skew indicators:**
  - \(\sigma_i(k_{\text{put}}) - \sigma_i(k_{\text{call}})\) for
    symmetric OTM strikes (e.g., \(k_{\text{put}} = -0.2\),
    \(k_{\text{call}} = 0.2\)).

These additional features are optional and can be introduced later if the
baseline model underperforms.

### 4.3.5 Normalisation

All features are normalised before being fed into the neural network:

\[
x_{\text{norm}} = \frac{x - \mu}{\sigma + \varepsilon},
\]

where \(\mu\) and \(\sigma\) are the mean and standard deviation computed
across the **training set only**, and \(\varepsilon > 0\) is a small number
to avoid division by zero.

The normalisation parameters \((\mu, \sigma)\) are stored (e.g. in a YAML
file under `config/`) and applied consistently to validation and test
features.

---

## 4.4 Label generation

Each date \(t\) yields a label \(y_t\) that the neural network is trained
to approximate. Two approaches are considered: a fully model-free MOT
approach and a simpler model-based Monte Carlo approach.

### 4.4.1 Approach A: model-free MOT / linear programming

This approach is conceptually aligned with the Neufeld–Sester framework.

1. **Discretise the terminal space.**  
   Construct a finite grid of possible terminal values
   \(\{(S_{T,n}^0,\dots,S_{T,n}^{20})\}_{n=1}^{N_{\text{grid}}}\) based on
   current forwards and synthetic IVs.

2. **Impose marginal constraints.**  
   For each asset \(i\), match the marginal distribution implied by the
   synthetic vanilla prices at maturity \(T\), for example via
   Breeden–Litzenberger-type approximations or by fitting a discrete
   distribution whose option prices are close to the synthetic ones.

3. **Impose martingale constraints.**  
   Ensure that for each asset the discounted price process is a martingale,
   typically implemented as linear constraints linking today’s forward
   price and the expected terminal value under the discrete measure.

4. **Optimise dispersion payoff.**  
   Define the dispersion payoff \(\Phi_{\text{disp}}\) on each grid node
   using the continuous-time or discrete-time formula. Then solve:

   \[
   \begin{aligned}
   U(x_t) &=
   \max_{\pi \in \mathbb{R}^{N_{\text{grid}}}}
   \sum_{n=1}^{N_{\text{grid}}} \pi_n \,
   \Phi_{\text{disp}}(S_{T,n}^0,\dots,S_{T,n}^{20}), \\
   L(x_t) &=
   \min_{\pi \in \mathbb{R}^{N_{\text{grid}}}}
   \sum_{n=1}^{N_{\text{grid}}} \pi_n \,
   \Phi_{\text{disp}}(S_{T,n}^0,\dots,S_{T,n}^{20}),
   \end{aligned}
   \]

   subject to the marginal and martingale constraints, and
   \(\pi_n \ge 0, \sum_n \pi_n = 1\).

5. **Define the fair value.**  
   For training the NN, a scalar label can be used:

   \[
   P_{\text{fair}}(x_t) := \frac{L(x_t) + U(x_t)}{2}.
   \]

**Pros:** Theoretically clean, truly model-free bounds.  
**Cons:** Computationally heavy in high dimension; may be limited to a
subset of dates or to a reduced number of constituents.

### 4.4.2 Approach B: model-based Monte Carlo pseudo-labels

This approach uses a simpler multi-asset stochastic model to generate
labels and is more tractable on a laptop.

1. **Specify a multi-asset model.**  
   - Each asset \(S^i\) follows a geometric Brownian motion under the
     risk-neutral measure:
     \[
     \frac{dS_t^i}{S_t^i} = (r - q^i) \, dt + \sigma_i \, dW_t^i,
     \]
     where \(\sigma_i\) is taken from the synthetic ATM IV \(a_i(t)\).
   - Brownian motions \(W^i\) are correlated with correlation matrix
     \(\Sigma\) estimated from historical Yahoo return data.

2. **Simulate paths.**  
   - For each date \(t\), simulate \(N_{\text{sim}}\) paths of the
     21-dimensional process over \([t, T]\), using time discretisation
     aligned with the realised variance computation.

3. **Compute dispersion payoff.**  
   - For each simulated path \(j\), compute the discrete-time realised
     variance for index and constituents and evaluate the dispersion payoff
     \(\widehat{\Phi}_{\text{disp}}^{(j)}\).

4. **Estimate fair value.**  
   - Define the pseudo-label:
     \[
     P_{\text{fair}}(x_t) \approx
     \frac{1}{N_{\text{sim}}} \sum_{j=1}^{N_{\text{sim}}}
     \widehat{\Phi}_{\text{disp}}^{(j)}.
     \]

**Pros:** Much cheaper computationally; feasible for many dates and
constituents.  
**Cons:** Not strictly model-free; labels depend on the GBM and
correlation assumptions.

### 4.4.3 Chosen approach for initial implementation

For the first implementation, the project will use **Approach B
(model-based Monte Carlo)** to generate labels \(P_{\text{fair}}(x_t)\)
for all training dates. This enables end-to-end experimentation on a
MacBook Air.

If time and computational resources permit, **Approach A (MOT/LP)** will
be applied on a smaller subset of dates and/or a reduced number of
constituents to:

- Validate the Monte Carlo labels.
- Demonstrate that the NN approximates true model-free bounds in those
  cases.

---

## 4.5 Dataset composition and splits

### 4.5.1 Date range and sampling

Using Yahoo Finance data, select a historical window such as:

- Start date: 2015‑01‑01  
- End date: 2024‑12‑31

For each business day \(t\) in this range where all 21 assets have valid
prices and the synthetic IV model can be constructed, generate one
feature-label pair \((x_t, y_t)\).

### 4.5.2 Train / validation / test splits

To avoid look-ahead bias, use **time-based splits**:

- **Training set:** 2015‑01‑01 to 2023‑12‑31  
- **Validation set:** 2024‑01‑01 to 2024‑09‑30  
- **Test set:** 2024‑10‑01 to 2024‑12‑31

Exact dates can be adjusted depending on data availability.

### 4.5.3 Quality filters

Before accepting a date \(t\) into the dataset:

1. Check that all 21 assets have valid prices and no extreme outliers.
2. Ensure realised volatility estimates used in the synthetic IV model are
   within reasonable bounds (e.g., between 5% and 100% annualised).
3. Verify that synthetic option prices do not exhibit obvious arbitrage
   (e.g., negative prices, violations of basic monotonicity or
   convexity), within numerical tolerance.

Dates failing these checks can be excluded or down-weighted.

---

## 4.6 Storage format and metadata

### 4.6.1 Feature and label files

Store the final datasets in a binary, columnar format such as Parquet:

- `data/labels/training_set.parquet`
- `data/labels/validation_set.parquet`
- `data/labels/test_set.parquet`

Each file contains:

- `date`
- Feature columns: `x_1, x_2, ..., x_{n_in}`
- Label column(s): `y_fair` (and optionally `y_lower`, `y_upper` if MOT
  bounds are used)

### 4.6.2 Normalisation parameters and configuration

Store normalisation statistics and label-generation settings in a config
file, e.g. `config/data_config.yml`:

```yaml
train_dates:
  start: "2015-01-01"
  end: "2023-12-31"
val_dates:
  start: "2024-01-01"
  end: "2024-09-30"
test_dates:
  start: "2024-10-01"
  end: "2024-12-31"

label_generation_method: "model_based"  # or "mot"
n_simulations: 10000

feature_normalization:
  mean_file: "data/labels/feature_means.npy"
  std_file: "data/labels/feature_stds.npy"
