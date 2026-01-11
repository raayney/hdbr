# 2. Model-Free Pricing and Neural Network Formulation

## 2.1 Motivation: model-free no-arbitrage bounds

In classical derivative pricing (e.g. Black–Scholes), one assumes a specific stochastic model for the underlying assets and calibrates model parameters to market data. This introduces **model risk**: if the true market dynamics differ from the assumed model, prices and hedges can be systematically wrong.

A **model-free approach** instead asks: *given observed market prices of liquid instruments (e.g. vanilla options), what can we say about the price of a derivative without assuming any specific model?*

The answer is **no-arbitrage bounds**: intervals \([L,U]\) such that any price \(P\) outside these bounds would admit an arbitrage (a risk-free profit). These bounds depend only on market-observable prices and financial logic, not on a choice of stochastic model.

### 2.1.1 Why model-free bounds matter for dispersion

A dispersion trade's P&L is driven by **correlation and co-movement** of constituents relative to the index. Classical models often misspecify correlation or assume it is constant. By computing model-free bounds, we avoid embedding such assumptions and directly extract what market vanilla prices imply about dispersion value.

## 2.2 Market setup and martingale measures

### 2.2.1 Observable market data

At time \(t=0\), we observe:

- Spot prices of the index \(S_0^0\) and constituents \(S_0^1,\dots,S_0^d\).
- Vanilla European option prices on each asset:
  - For asset \(i\), calls and puts with strikes \(K\) and maturity \(T\) are traded.
  - Let \(C^i(K,T)\) and \(P^i(K,T)\) denote mid market prices (or we may work with bid/ask).
- The discount factor \(\beta(T) = e^{-rT}\) (or a discount curve if \(r\) is stochastic).
- Dividend yields or forward prices to relate spot and forward.

**Practical implementation note.** In this project, historical option prices
\(C^i(K,T)\) and \(P^i(K,T)\) are not observed directly from the market. Instead,
for each historical date (with real underlying prices from Yahoo Finance), a
synthetic 3-month implied volatility surface is generated for SX5E and its 20
constituents. Option prices are then computed from these synthetic IV surfaces
using the Black–Scholes formula. Conceptually, these synthetic prices play the
role of observed vanilla market prices in the definition of
\(\mathcal{M}_{\text{vanilla}}\).

### 2.2.2 Risk-neutral measures and no-arbitrage

A **risk-neutral measure** \(\mathbb{Q}\) is a probability measure under which:

- All tradable assets, when discounted, are martingales.
- The discounted price of any replicable payoff equals the initial investment to replicate it.

The set of all such measures consistent with observed vanilla option prices is denoted \(\mathcal{M}_{\text{vanilla}}\):
\[
\mathcal{M}_{\text{vanilla}} := \left\{ \mathbb{Q} : \mathbb{E}_\mathbb{Q}[\beta(T) \cdot C(S_T^i, K)] = C^i(K,T) \, \forall i,K,T \right\},
\]
where we implicitly include the put-call parity constraint and smoothness assumptions on the option surface.

In practice, \(\mathcal{M}_{\text{vanilla}}\) is a large set (many measures fit the observed vanilla surface), reflecting the ambiguity in correlation and higher moments not pinned down by the marginal distributions.

### 2.2.3 Model-free bounds for a payoff

Given a payoff functional \(\Phi : \mathbb{R}^{d+1} \to \mathbb{R}\) (e.g. the dispersion payoff \(\Phi_{\text{disp}}\) from Section 1.3), the **model-free upper and lower bounds** are:

\[
\begin{align}
U_{\Phi} &:= \sup_{\mathbb{Q} \in \mathcal{M}_{\text{vanilla}}} \mathbb{E}_\mathbb{Q}[\beta(T) \cdot \Phi(S_T^0,\dots,S_T^d)], \\
L_{\Phi} &:= \inf_{\mathbb{Q} \in \mathcal{M}_{\text{vanilla}}} \mathbb{E}_\mathbb{Q}[\beta(T) \cdot \Phi(S_T^0,\dots,S_T^d)].
\end{align}
\]

Any price \(P\) in the interval \([L_{\Phi}, U_{\Phi}]\) is arbitrage-free; prices outside this interval admit profitable arbitrage strategies.

## 2.3 Computational approach: linear programming / optimal transport

### 2.3.1 Discrete approximation

In practice, we cannot work with the infinite-dimensional space of all measures \(\mathbb{Q} \in \mathcal{M}_{\text{vanilla}}\). Instead, we discretize.

**Approach 1: Grid-based martingale optimal transport (MOT)**

- Discretize the state space of \((S_T^0,\dots,S_T^d)\) into a finite grid of nodes.
- Represent each measure \(\mathbb{Q}\) by a probability vector \((\pi_1,\dots,\pi_N)\) over the grid nodes.
- Write linear constraints ensuring:
  - The marginal distributions of \(S_T^i\) match those implied by vanilla prices via the Breeden–Litzenberger formula.
  - \(\mathbb{Q}\) is a martingale (discounted prices remain martingales under \(\mathbb{Q}\)).
- Formulate the bound computation as a linear program:
  \[
  U_{\Phi} = \max_{\pi} \sum_{n=1}^N \pi_n \cdot \Phi(S_{T,n}^0,\dots,S_{T,n}^d) \quad \text{s.t.} \quad \text{marginal and martingale constraints}.
  \]

This is the **martingale optimal transport (MOT)** problem studied in the referenced Neufeld–Sester paper.

**Approach 2: Direct LP on replicating portfolios**

An alternative (equivalent in the limit) is to formulate the dual problem: find the optimal static hedge (positions in vanillas) that super-replicates the payoff. The dual value gives the upper bound, and the primal optimum gives the bound itself.

### 2.3.2 Computational cost

Solving the LP or MOT problem for a single market state \(x\) can be expensive, especially for high-dimensional grids or many constraints. This is where the **neural network** comes in.

## 2.4 Neural network as a surrogate: the learning task

### 2.4.1 Supervised learning problem

Instead of solving a new LP/MOT problem every time we observe market data \(x\), we train a neural network \(f_\theta\) to **learn the mapping** from market state to bounds:

\[
\begin{align}
f_\theta(x) &\approx \begin{pmatrix} L(x) \\ U(x) \end{pmatrix}, \quad \text{or} \\
f_\theta(x) &\approx P_{\text{fair}}(x) = \frac{L(x) + U(x)}{2}.
\end{align}
\]

**Training dataset:**

Generate a large offline dataset of pairs \(\{(x_i, y_i^L, y_i^U)\}_{i=1}^{N_{\text{train}}}\) where:

- \(x_i\): a **market state** (spot prices, vanilla option prices or implied vols, risk-free rate, etc.).
- \(y_i^L, y_i^U\): the **true lower and upper no-arbitrage bounds**, computed by solving the MOT/LP problem for market state \(x_i\).

**Training objective:**

Minimise a loss function such as mean squared error. If the network has
two separate outputs for the lower and upper bounds, denoted
\(f_\theta^L(x)\) and \(f_\theta^U(x)\), and the corresponding labels are
\(y_i^L\) and \(y_i^U\), then

\[
\mathcal{L}(\theta)
= \frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}}
\left[
\big(f_\theta^L(x_i) - y_i^L\big)^2
+
\big(f_\theta^U(x_i) - y_i^U\big)^2
\right]
+ \lambda \|\theta\|^2,
\]

where \(\lambda > 0\) is a regularization coefficient.

Alternatively, if the network outputs a single scalar corresponding to the
fair value midpoint \(P_{\text{fair}}(x)\), the loss is

\[
\mathcal{L}(\theta)
= \frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}}
\big(f_\theta(x_i) - P_{\text{fair}}(x_i)\big)^2
+ \lambda \|\theta\|^2.
\]

### 2.4.2 Advantages of the NN approach

Once trained, the network can be used **online** for real-time pricing:

- **Speed:** A forward pass through the network is orders of magnitude faster than solving an LP or MOT problem.
- **Scalability:** The same pre-trained network can be applied to any market state \(x\), without retraining, allowing for rapid re-pricing as markets move.
- **Consistency:** A single network learned offline ensures consistent pricing rules across different market regimes.

## 2.5 Input and output specifications

### 2.5.1 Input features \(x\)

The network takes as input a **feature vector \(x \in \mathbb{R}^{n_{\text{in}}}\)** encoding the current market state. At minimum, this includes:

**Asset prices and risk-free **
- Spot prices: \(S_0^0, S_0^1,\dots,S_0^{20}\).
- Forward prices or yields (derived from spot and rates).
- Risk-free discount factor \(\beta(T)\).

**Vanilla options **
For each asset \(i = 0,1,\dots,20\) and maturity \(T\):
- Implied volatilities (or option prices) on a grid of strikes.
  - Example: moneyness levels \(\log(K/F) \in \{-0.3, -0.2, \ldots, 0, \ldots, 0.3\}\).
  - Yields a 2D surface: strikes \(\times\) maturities (though we focus on \(T=3\text{M}\) initially).
- Bid–ask spreads or confidence measures (optional, for filtering illiquid strikes).

**Additional features (to be determined):**
- Term structure of variance (implied vol for different maturities).
- Skew measures (e.g. 25-delta call IV minus put IV).
- Historical realized correlation (recent window).

**Normalization:**
Features will be standardized (zero mean, unit variance) based on training set statistics to improve network training.

### 2.5.2 Output specifications

The network outputs one of the following:

**Option 1: Two outputs (lower and upper bounds)**
\[
f_\theta(x) = (L(x), U(x)) \in \mathbb{R}^2.
\]

**Option 2: Single output (fair value midpoint)**
\[
f_\theta(x) = P_{\text{fair}}(x) \in \mathbb{R}.
\]

**Option 3: Three outputs (bounds and midpoint)**
\[
f_\theta(x) = (L(x), P_{\text{fair}}(x), U(x)) \in \mathbb{R}^3.
\]

For this project, we will use **Option 2 initially** (single fair-value output), with the understanding that bounds can be estimated via ensemble methods or by training separate networks for \(L\) and \(U\) if needed later.

## 2.6 Constraints and regularization

### 2.6.1 No-arbitrage and monotonicity constraints

The learned function \(f_\theta(x)\) should respect financial constraints:

1. **Convexity in spot:** For a given vanilla surface (holding strikes/maturities fixed), the option price should be a convex function of the underlying spot. The learned price function should respect this.

2. **Monotonicity in volatility:** Higher implied volatility should generally increase option prices. The network should learn this monotone relationship.

3. **Smoothness:** Prices should be continuous in market state; sudden jumps are unphysical.

**Implementation:**
- Soft penalties in the loss function rewarding monotone / convex predictions.
- Or, architectural choices (e.g. positive-weight constraints on certain layers) enforcing constraints by design.

### 2.6.2 Regularization

To avoid overfitting on the training set and to improve generalization to new market states:

- **L2 regularization** on network weights: \(\lambda \|\theta\|^2\).
- **Dropout** on hidden layers during training.
- **Early stopping** based on validation set performance.

## 2.7 Training and validation protocol

### 2.7.1 Data generation

**Phase A (MOT/LP label generation):**
1. Sample representative market states \(x_i\) from historical use real SX5E and constituent prices from Yahoo Finance, and for each date construct a synthetic 3-month option surface for all assets using the implied volatility model described in the data document.
2. For each \(x_i\), solve the MOT/LP problem to compute true bounds \((L(x_i), U(x_i))\).
3. Assemble the training dataset \(\{(x_i, y_i)\}_{i=1}^{N_{\text{train}}}\).

**Phase B (Network training):**
1. Split data into training, validation, and test sets (e.g. 70%, 15%, 15%).
2. Train the network on the training set, monitoring loss on the validation set.
3. Tune hyperparameters (learning rate, dropout, regularization) based on validation performance.
4. Evaluate final model on the held-out test set.

### 2.7.2 Evaluation metrics

**Offline (label prediction accuracy):**
- Mean Absolute Error (MAE) between NN output and true LP/MOT bounds.
- Mean Squared Error (MSE).
- Quantile metrics: median and 95th percentile errors, to detect outliers.

**Coverage / no-arbitrage check:**
- Fraction of test samples where the implementation price of the chosen dispersion trade (computed from the same synthetic option surfaces) lies inside the NN-estimated bounds.
With access to real historical option quotes, the same check would be performed against actual market prices instead of synthetic ones.
- Violations: if observed market price is above \(U(x)\) or below \(L(x)\), investigate.

**Online (trading simulation, covered in Step 7):**
- Sharpe ratio, maximum drawdown, hit rate of signals, turnover.

## 2.8 Relationship to the Neufeld–Sester framework

The Neufeld–Sester paper (referenced in the project brief) proposes a similar supervised learning approach for model-free pricing of high-dimensional derivatives using neural networks. Key points of alignment:

1. **Problem formulation:** We use their supervised learning framework where the network learns the mapping from market features to model-free bounds.

2. **MOT foundation:** The labels are generated by solving martingale optimal transport problems, consistent with their methodology.

3. **Application to high-dimensional baskets:** Their framework scales naturally to our case (\(d=20\) constituents), whereas classical PDE / tree-based methods struggle with dimensionality.

4. **Theoretical guarantees:** Neufeld–Sester show that the learned function converges to the true no-arbitrage bound as the network approximation capacity and training set size increase (under regularity conditions).

Our project is a concrete instantiation of their framework applied to the specific domain of dispersion trading on EuroStoxx 50.

## 2.9 Summary

This section formalized:

- **Model-free bounds** as the theoretical target (Section 2.2–2.3).
- **Neural network as a surrogate** (Section 2.4), learning to map market state to fair value / bounds.
- **Input/output specifications** concretely (Section 2.5).
- **Constraints and regularization** to enforce financial realism (Section 2.6).
- **Training protocol** (Section 2.7).
- **Alignment with Neufeld–Sester** framework (Section 2.8).

In **Step 3**, we will specify the data sources and features more precisely. In **Step 5**, we will detail the label generation procedure (MOT problem setup and solution). In **Step 6**, we will design the network architecture.