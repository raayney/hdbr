# 2. Model‑Free Pricing and Neural Network Formulation

## 2.1 Motivation: model‑free no‑arbitrage bounds

Classical derivative pricing (e.g. Black–Scholes) assumes a specific stochastic model for the underlying assets and calibrates model parameters to market data. This introduces **model risk**: if the true market dynamics differ from the assumed model, prices and hedges can be systematically wrong.

A **model‑free approach** instead asks:

> Given observed prices of liquid instruments (e.g. vanilla options), what can we say about the price of a derivative *without* committing to a particular stochastic model?

The model‑free answer is an **interval of arbitrage‑free prices**: an interval $[L(x), U(x)]$ such that any price $P$ for the derivative outside this band would admit a model‑independent arbitrage. These bounds depend only on:

- market‑observable vanilla prices,
- absence of arbitrage,
- and the payoff’s structure,

not on any assumed diffusion, jump, or volatility model. This is precisely the perspective formalized in Neufeld–Sester’s model‑free pricing framework (Theorem II.2, etc.).

For high‑dimensional payoffs like dispersion on an index and 20 constituents, this is particularly attractive because **correlation** and joint tail behaviour are notoriously hard to model correctly; model‑free bounds sidestep misspecified parametric correlation dynamics.

---

## 2.2 Model‑free framework for dispersion pricing

This section summarizes the Neufeld–Sester model‑free setup, specialized to the dispersion payoff from Section 1.3 but still independent of any GBM or Monte Carlo approximation.

### 2.2.1 Observable market data

At time $t = 0$, the observable state is:

- Spot prices:
  $S_0^0, S_0^1,\dots,S_0^d$,  
  where $S^0$ is the index and $S^1,\dots,S^d$ are the constituents.

- Vanilla European options on each asset:
  - For asset $i$, maturity $T$, and a continuum (or dense grid) of strikes $K$, we observe call and put prices
    $C^i(K,T)$ and $P^i(K,T)$.
- Discounting and carry:
  - A discount factor $\beta(T) = e^{-rT}$ or a term structure of rates.
  - Dividend yields (or forwards) to connect spot and forward prices.

**Practical implementation note (this project).**  
In this project, mid‑market option prices $C^i(K,T)$, $P^i(K,T)$ are **not** observed directly. Instead, for each historical date we:

1. Use real underlying prices from Yahoo Finance.
2. Build a **synthetic 3‑month implied volatility surface** for the index and 20 constituents using a quadratic smile model.
3. Convert these IVs into call/put prices via the Black–Scholes formula.

Conceptually, these synthetic prices play the role of “observed vanilla prices” in the model‑free set $\mathcal{M}_{\text{vanilla}}(x)$ below.

### 2.2.2 Risk‑neutral measures consistent with vanillas

Fix a filtered probability space and consider candidate **risk‑neutral measures** $\mathbb{Q}$ such that:

- Discounted asset prices are martingales under $\mathbb{Q}$.
- The discounted price of any replicable payoff equals its initial cost.

Let $\mathcal{M}_{\text{vanilla}}(x)$ be the set of all such measures consistent with the vanilla option surface observed at time 0 (encoded in state $x$):

$\mathcal{M}_{\text{vanilla}}(x)
  := \{\mathbb{Q} : \text{discounted } S \text{ is a martingale under } \mathbb{Q} \text{ and under } \mathbb{Q} \text{ the prices of all observed vanillas match } C^i(K,T), P^i(K,T)\}.$

This set is typically **not a singleton**: many joint distributions and dependence structures are compatible with the same system of vanilla prices. In particular, the **marginal distributions** of each $S_T^i$ may be pinned down by the full strike surface, but **joint dependence and pathwise structure** (correlations, co‑jumps, etc.) remain ambiguous.

Neufeld–Sester work exactly with such sets of model‑independent martingale measures, and prove that arbitrage‑free price bounds for a broad class of payoffs depend continuously on the market input data.

### 2.2.3 Model‑free bounds for a payoff

Let $\Phi_{\text{disp}}$ be the dispersion payoff defined in Section 1.3 (continuous or discrete). For a generic payoff $\Phi$ depending on the multi‑asset path up to time $T$, the **model‑free lower and upper no‑arbitrage bounds** at state $x$ are

$$\begin{aligned}
L(x) &= \inf_{\mathbb{Q} \in \mathcal{M}_{\text{vanilla}}(x)}
        \mathbb{E}_{\mathbb{Q}} \big[ \Phi(S^0,\dots,S^d) \mid \mathcal{F}_0 \big], \\
U(x) &= \sup_{\mathbb{Q} \in \mathcal{M}_{\text{vanilla}}(x)}
        \mathbb{E}_{\mathbb{Q}} \big[ \Phi(S^0,\dots,S^d) \mid \mathcal{F}_0 \big].
\end{aligned}$$

Any price $P$ for $\Phi$ in the interval $\,[L(x),U(x)]\,$ is **arbitrage‑free** in the model‑independent sense; any $P$ outside this interval admits a model‑independent arbitrage strategy (sub‑/super‑replication duality).

A natural **model‑free fair value** is the midpoint

$$P_{\text{fair}}(x) := \frac{L(x) + U(x)}{2}.$$

This is one of the canonical targets Neufeld–Sester approximate by neural networks.

In theory, the dispersion pricing problem in this project is:

- Payoff: $\Phi = \Phi_{\text{disp}}$.
- Market state: $x$ encodes spot, IV surfaces, rates, etc.
- Theoretical target: $x \mapsto P_{\text{fair}}(x)$ coming from the MOT/LP bounds.

---

## 2.3 Practical implementation in this project: Monte Carlo surrogate

Computing $(L(x),U(x))$ exactly by solving a high‑dimensional **martingale optimal transport (MOT)** or linear programming problem for each historical date $x_t$ would be computationally prohibitive in this project, especially with 21 underlyings.

Instead, Phase 2 uses a **model‑based Monte Carlo surrogate** to approximate a fair value for each date, which then serves as the supervised learning target for the neural network.

### 2.3.1 Risk‑neutral GBM approximation

For each historical date $t$, the code constructs a multi‑asset GBM model calibrated to the synthetic IVs and historical correlations:

- Let $S_0 \in \mathbb{R}^{21}$ be the vector of index and constituent prices.
- Let $r_i$ be the rate and $q_i$ the dividend yield for asset $i$.
- Let $\sigma_i^{\text{ATM}}$ be the ATM implied volatility extracted from the synthetic IV curve.

We set **risk‑neutral drifts**

$$\mu_i = r_i - q_i, \quad i = 0,\dots,20.$$

From a 60‑day window of historical returns, we estimate a **$21\times 21$ correlation matrix** $\widehat{\Sigma}_t$ and apply a small eigenvalue regularization to ensure positive definiteness.

Under this approximation, log‑prices are simulated as a correlated Gaussian process with:

- drift vector $\mu$,
- volatility vector $\sigma$,
- correlation matrix $\widehat{\Sigma}_t$,

over $N_{\text{steps}}$ trading days (here $63$, roughly 3 months).

### 2.3.2 Pathwise dispersion payoff and Monte Carlo estimator

For each simulated path $j = 1,\dots,N_{\text{paths}}$, and each asset $i$, the code computes **annualized discrete realised variance** over the simulation window:

$$\widehat{\mathrm{RV}}_{T}^{i,(j)}
  := 252 \sum_{k=1}^{N}
        \left(
          \log\frac{S^{i,(j)}_{t_k}}{S^{i,(j)}_{t_{k-1}}}
        \right)^2.$$

The simulated dispersion payoff on path $j$ is then, by construction:

$$\widehat{\Phi}_{\text{disp}}^{(j)}
  := \widehat{\mathrm{RV}}_{T}^{0,(j)}
     - \sum_{i=1}^d w_i \,\widehat{\mathrm{RV}}_{T}^{i,(j)}.$$

This matches exactly the implementation in `DispersionPayoffCalculator`:

- `realized_variance(paths)` computes the annualized variance via the $252$‑scaled squared log‑returns.
- `dispersion_payoff(index_paths, stock_paths)` returns `var_index - var_basket`.

The **Monte Carlo pseudo‑fair value** used as label for state $x_t$ is the sample mean:

$$P_{\text{MC}}(x_t)
  := \frac{1}{N_{\text{paths}}}
     \sum_{j=1}^{N_{\text{paths}}}
       \widehat{\Phi}_{\text{disp}}^{(j)}.$$

Under the GBM approximation, $P_{\text{MC}}(x_t)$ is an unbiased estimator of the GBM‑model price $\mathbb{E}_{Q^{\text{GBM}}_t}[\widehat{\Phi}_{\text{disp}}]$, and converges at rate $O(N_{\text{paths}}^{-1/2})$ by the law of large numbers and central limit theorem.

Conceptually, this quantity is treated as a **proxy for the model‑free fair value** $P_{\text{fair}}(x_t)$. In later phases, small‑scale MOT experiments (reduced dimension and fewer dates) can be used to compare:

- MOT midpoint $P_{\text{fair}}(x_t)$,
- Monte Carlo surrogate $P_{\text{MC}}(x_t)$,
- neural network output $V_{\text{NN}}(x_t)$.

---

## 2.4 Neural network formulation

Given the Monte Carlo surrogate described above, the neural network learning problem in this project is:

- **Input:** feature vector $x_t \in \mathbb{R}^{274}$ built from spot, forward, rate, yield and IV features as specified in the data document (and summarized in doc 01).
- **Target label:** scalar pseudo‑fair value $P_{\text{MC}}(x_t)$.
- **Output:** scalar prediction $V_{\text{NN}}(x_t;\theta)$ approximating $P_{\text{MC}}(x_t)$.

### 2.4.1 Supervised learning objective

The training dataset is

$$\mathcal{D} = \{(x_t, y_t)\}_{t \in \mathcal{T}}, \quad y_t := P_{\text{MC}}(x_t).$$

The loss function is the mean squared error with L2 regularization:

$$\mathcal{L}(\theta)
  = \frac{1}{|\mathcal{T}_{\text{train}}|}
    \sum_{t \in \mathcal{T}_{\text{train}}}
      \big( V_{\text{NN}}(x_t;\theta) - y_t \big)^2
    + \lambda \,\|\theta\|_2^2,$$

with $\lambda > 0$ a weight‑decay coefficient.

The implementation uses:

- Adam optimizer with learning rate $10^{-3}$.
- Weight decay $10^{-4}$.
- Batch size $64$.
- Early stopping with patience $20$ based on validation loss.
- A scheduler (ReduceLROnPlateau) to reduce learning rate when validation loss plateaus.

These choices are standard deep‑learning practice rather than specific to Neufeld–Sester.

### 2.4.2 Network architecture (pointer to doc 05)

The architecture actually implemented (`BasketPricingNet`) is a fully connected MLP:

- Input dimension: $274$.
- Hidden layers: $256 \to 128 \to 64$.
- Activation: ReLU between layers.
- Dropout: $0.2$ on the first two hidden layers.
- Output: linear layer with $1$ neuron (scalar).

This is a **simplified 3‑hidden‑layer architecture** relative to the original conceptual 4‑layer spec, chosen to balance capacity and training time. Full architectural details and constraints (e.g. possible monotonicity / convexity penalties) are documented in `05_nn_architecture_and_constraints.md`.

---

## 2.5 Relationship to Neufeld–Sester and MOT

Conceptually, this project is a **specialized instance** of the Neufeld–Sester framework:

1. **Same high‑level problem.**  
   Given market state $x$ (spot + vanillas), learn a mapping to model‑free price information for a high‑dimensional payoff $\Phi$ via supervised learning on precomputed labels.

2. **Same theoretical target.**  
   In principle, the target is the pair $(L(x), U(x))$ or midpoint $P_{\text{fair}}(x)$ of model‑independent bounds for $\Phi_{\text{disp}}$, exactly as in their Theorem II.2 and related results.

3. **Same methodology.**  
   Train a single offline neural network that, once trained, can be used online for fast pricing of the payoff for any new market state $x$, instead of repeatedly solving large LP/MOT problems.

4. **Difference in current implementation.**  
   - Neufeld–Sester explicitly construct labels by **solving MOT/LP problems** (or related algorithms) for each sample.
   - In this project’s Phase 2, labels are generated by a **GBM‑based Monte Carlo surrogate** calibrated to synthetic IVs and historical correlations.
   - MOT is reserved for later, smaller‑scale validation experiments (reduced dimension, fewer dates) due to computational constraints.

5. **Continuity and approximation.**  
   Neufeld–Sester prove that, under continuity assumptions on the payoff family and market data, the map from input data to price bounds is continuous and can be approximated arbitrarily well by neural networks. This justifies the overall approach of learning $x \mapsto P_{\text{fair}}(x)$ (or a surrogate thereof) with a feed‑forward network.

In summary, the **mathematical foundation** of the project’s pricing approach is Neufeld–Sester’s model‑free pricing and MOT theory, while the **practical implementation path** in Phase 2 uses Monte Carlo pseudo‑labels and a relatively simple MLP to deliver a working, end‑to‑end dispersion pricing and trading system on free data.

Subsequent documents will:

- Specify the data and feature construction in detail (doc 03).
- Fully describe label generation (doc 04, including the Monte Carlo engine and planned MOT experiments).
- Detail the neural network architecture and training regime (doc 05).
- Define evaluation metrics and backtesting methodology (doc 06).