# 1. Problem and Payoff Specification

## 1.1 High‑level objective

The project builds a trading system for **variance dispersion** on the EURO STOXX 50 (SX5E) that is powered by a **deep‑learning pricing engine** inspired by the model‑free framework of Neufeld–Sester for computing no‑arbitrage bounds via neural networks and martingale optimal transport (MOT) (Neufeld–Sester, Sections II.A–II.B, III).

At a high level, the system:

- Observes market data for an index and its constituents (spot prices and synthetic vanilla options).
- Treats a variance‑based dispersion payoff as an abstract contingent claim $\Phi_{\text{disp}}$ on these assets. [SUPPOSED TO BE PRIOR KNOWLEDGE]
- Conceptually targets the **model‑free no‑arbitrage band** and an associated fair value for $\Phi_{\text{disp}}$, defined exactly as in the Neufeld–Sester framework for general high‑dimensional payoffs (Neufeld–Sester, Theorem II.2).
- In practice, uses **Monte Carlo pseudo‑labels** to approximate a fair value $P_{\text{MC}}(x)$ and trains a neural network surrogate $V_{\text{NN}}(x)$ to learn this mapping from high‑dimensional market state $x$ to price, following their supervised learning approach (Neufeld–Sester, Section II.B–II.C, Algorithm 1).
- Compares $V_{\text{NN}}(x)$ to an implementation price derived from synthetic option surfaces to generate actionable trading signals. [SUPPOSED TO BE PRIOR KNOWLEDGE]

This document defines rigorously:

1. The **market setup and notation**.
2. The **continuous‑time dispersion payoff** and its discrete implementation for backtesting.
3. The **conceptual model‑free pricing problem**, and how it connects to the Neufeld–Sester theory.
4. The **practical surrogate** used in this project.
5. The **trading signal** the rest of the system is built around.

---

## 1.2 Market setup and notation

Consider a financial market on a filtered probability space $(\Omega,\mathcal F,(\mathcal F_t)_{t\ge 0})$, where $(\mathcal F_t)_t$ satisfies the usual conditions and represents the flow of market information. [SUPPOSED TO BE PRIOR KNOWLEDGE]

The traded assets are:

- **Index process**
  $$S^0 = (S_t^0)_{t\ge 0},$$
  representing the EURO STOXX 50 index (or a liquid ETF proxy). [SUPPOSED TO BE PRIOR KNOWLEDGE]

- **Constituent processes**
  $$
  S^1,\dots,S^{20}, \quad S^i = (S_t^i)_{t\ge 0},
  $$
  corresponding to the top 20 SX5E constituents by index weight. [SUPPOSED TO BE PRIOR KNOWLEDGE]

Write $d = 20$ for the number of constituents. [SUPPOSED TO BE PRIOR KNOWLEDGE] Each $S^i$ is modeled as a (discounted) semimartingale under risk‑neutral measures; this is the same general setup used in Neufeld–Sester for multi‑asset derivatives, where the joint path
$$
S = (S^0,\dots,S^d)
$$
is the underlying process for a payoff functional (Neufeld–Sester, Sections I, II.A).

The **trading horizon per dispersion trade** is fixed to
$$
T = 3 \text{ months} \approx 0.25 \text{ years},
$$
a design choice reflecting the liquid 3‑month options market on SX5E and its constituents. [SUPPOSED TO BE PRIOR KNOWLEDGE]

At time $t=0$, the observable state includes:

- Current index and stock prices $S_0^0, S_0^1,\dots,S_0^d$. [SUPPOSED TO BE PRIOR KNOWLEDGE]
- Bid/ask prices or implied volatilities of European calls and puts on each $S^i$ with maturity $T$ across a grid of strikes (here represented by synthetic implied volatility surfaces). [SUPPOSED TO BE PRIOR KNOWLEDGE]
- Risk‑free discount factors and dividend yields (encoded via forwards in the feature set). [SUPPOSED TO BE PRIOR KNOWLEDGE]
- Additional engineered quantities (realised volatilities, historical correlations, etc.), as defined in the data and feature specification. [SUPPOSED TO BE PRIOR KNOWLEDGE]

Let $w_1,\dots,w_d$ denote the **constituent weights** in SX5E, satisfying
$$
\sum_{i=1}^d w_i = 1, \quad w_i \ge 0.
$$
[SUPPOSED TO BE PRIOR KNOWLEDGE]

Define the corresponding **basket level**:
$$
B_t := \sum_{i=1}^d w_i S_t^i.
$$
[SUPPOSED TO BE PRIOR KNOWLEDGE]

The continuous‑time objects $S^i$ and their quadratic variations will be used to justify the dispersion payoff mathematically; their discrete‑time approximations (based on daily data) are used for concrete label generation and backtesting. [SUPPOSED TO BE PRIOR KNOWLEDGE]

---

## 1.3 Variance dispersion payoff

### 1.3.1 Continuous‑time definition

In continuous time, for each asset $S^i$ over $[0,T]$, define the **realised variance** as the quadratic variation of its log‑price:
$$
\mathrm{RV}_T^i := \langle \log S^i \rangle_T.
$$
[SUPPOSED TO BE PRIOR KNOWLEDGE]

This is the standard variance‑swap object: under mild assumptions on $S^i$, a variance swap with payoff $\mathrm{RV}_T^i$ can be replicated by a static position in a strip of options plus dynamic trading in the underlying. This classical model‑free variance replication result comes from the robust options literature (e.g. Carr–Madan, Dupire) rather than Neufeld–Sester. [SUPPOSED TO BE PRIOR KNOWLEDGE]

For the index and constituents, define:

- Index realised variance: $\mathrm{RV}_T^0 := \langle \log S^0 \rangle_T$. [SUPPOSED TO BE PRIOR KNOWLEDGE]
- Constituent realised variances: $\mathrm{RV}_T^i := \langle \log S^i \rangle_T$, $i=1,\dots,d$. [SUPPOSED TO BE PRIOR KNOWLEDGE]

In this implementation, the **dispersion payoff** is defined as
$$
\Phi_{\text{disp}}
  := \mathrm{RV}_T^0 \;-\; \sum_{i=1}^d w_i \,\mathrm{RV}_T^i.
$$
[SUPPOSED TO BE PRIOR KNOWLEDGE]

That is, dispersion is measured as **index variance minus weighted single‑name variance** (this is the convention actually used in the code), rather than the opposite sign. Interpretation:

- If $\Phi_{\text{disp}} > 0$: the **index variance exceeds** the weighted average of single‑name variances; loosely, the index is moving more as a bloc relative to idiosyncratic volatility.
- If $\Phi_{\text{disp}} < 0$: the **weighted single‑name variance dominates** index variance.

These interpretations follow from standard variance–correlation decompositions of basket variance and are not specific to Neufeld–Sester. [SUPPOSED TO BE PRIOR KNOWLEDGE]

Mathematically, $\Phi_{\text{disp}}$ is a functional of the full multi‑asset path $(S_t^0,\dots,S_t^d)_{0\le t\le T}$. It fits into the class of high‑dimensional payoff functions $\varphi(S)$ considered by Neufeld–Sester when proving continuity of model‑free price bounds and universal approximability by neural networks (Neufeld–Sester, Theorem II.2, Remark II.3).

### 1.3.2 Discrete‑time / implementable approximation

For implementation and backtesting, only discrete observations at dates
$$
0 = t_0 < t_1 < \dots < t_N = T
$$
are available (here, trading days over a 3‑month window).

The **discrete realised variance** for asset $i$ is defined in the code as
$$
\widehat{\mathrm{RV}}_T^i
  := 252 \sum_{k=1}^N
        \left(
          \log\frac{S^i_{t_k}}{S^i_{t_{k-1}}}
        \right)^2,
$$
where the factor $252$ annualizes daily returns to variance per year. [SUPPOSED TO BE PRIOR KNOWLEDGE]

This is the usual discrete‑time estimator of quadratic variation used in empirical finance (with annualization) and converges to $\langle \log S^i \rangle_T$ up to a deterministic scaling factor as the mesh of the partition goes to zero, under standard conditions on $S^i$. [SUPPOSED TO BE PRIOR KNOWLEDGE]

Given these, the **discrete dispersion payoff** implemented in the code is
$$
\widehat{\Phi}_{\text{disp}}
  := \widehat{\mathrm{RV}}_T^0
     \;-\; \sum_{i=1}^d w_i \,\widehat{\mathrm{RV}}_T^i.
$$
[SUPPOSED TO BE PRIOR KNOWLEDGE]

This $\widehat{\Phi}_{\text{disp}}$ drives:

- The **Monte Carlo labels**, where it is evaluated on simulated price paths.
- The **backtest P\&L**, where it is computed from realised Yahoo Finance prices over each 3‑month window.

While the theory is naturally formulated in terms of continuous‑time quadratic variation, this discrete definition (with annualization) is the standard plug‑in estimator used in both academic and practitioner work on variance products and is consistent with treating $\widehat{\Phi}_{\text{disp}}$ as a payoff in the Neufeld–Sester sense (they also approximate path‑dependent functionals by discrete‑time grids when computing bounds numerically; see Neufeld–Sester, Section II.C, Examples II.5–II.8).

---

## 1.4 Conceptual model‑free pricing of dispersion

### 1.4.1 Martingale measures consistent with vanilla options

Let $x$ denote the **market state at time 0**, i.e. the collection of:

- Current spots $S_0^0,\dots,S_0^d$.
- Vanilla option quotes (or equivalently implied volatilities) on each asset at maturity $T$ across the moneyness grid.
- Discount factors and dividend yields.

Following Neufeld–Sester (Section II.A), define $\mathcal M_{\text{vanilla}}(x)$ as the set of **risk‑neutral martingale measures** on the multi‑asset path space such that:

- The discounted processes are martingales under $Q$.
- The prices of all liquid vanillas (our synthetic call and put prices) match their observed levels at time 0 under $Q$.
- Bid/ask and transaction cost constraints are respected if modelling frictions are included (Neufeld–Sester, Sections II.A–II.B).

This is exactly the “model‑independent” set of candidate measures considered by Neufeld–Sester when defining model‑free bounds: no parametric dynamics are assumed beyond compatibility with vanilla prices and the martingale property.

### 1.4.2 Model‑free bounds and fair value

Given $\Phi_{\text{disp}}$, define **model‑free no‑arbitrage bounds**:
$$
\begin{aligned}
L(x) &:= \inf_{Q \in \mathcal M_{\text{vanilla}}(x)}
          \mathbb E_Q[\Phi_{\text{disp}} \mid \mathcal F_0], \\
U(x) &:= \sup_{Q \in \mathcal M_{\text{vanilla}}(x)}
          \mathbb E_Q[\Phi_{\text{disp}} \mid \mathcal F_0].
\end{aligned}
$$

This is a direct specialization of the Neufeld–Sester definition of model‑free upper and lower bounds for a general payoff functional $\varphi$ (Neufeld–Sester, Section II.A, Theorem II.2).

A **conceptual fair value** is then
$$
P^*_{\text{fair}}(x) := \frac{L(x) + U(x)}{2}.
$$

In Neufeld–Sester’s language, this corresponds to the midpoint of the super‑ and sub‑hedging prices and is one of the natural scalar targets they consider when training neural networks: they explicitly discuss outputting either $(L(x),U(x))$ or a single scalar fair value $P_{\text{fair}}(x)$ (Neufeld–Sester, Section II.5, “Output specifications”).

This project conceptually adopts the same target: the pricing engine is meant to approximate the map
$$
x \;\mapsto\; P^*_{\text{fair}}(x),
$$
for the specific payoff $\Phi_{\text{disp}}$.

---

## 1.5 Practical surrogate used in this project

### 1.5.1 Monte Carlo pseudo‑fair value

Solving the full multi‑asset MOT / LP problem for dispersion on 21 underlyings at each historical date would be computationally extremely demanding. Instead, Phase 1–2 of this project adopts the following **model‑based approximation**:

1. For each historical date $t$, construct synthetic 3‑month IV surfaces for SX5E and its 20 constituents on a fixed moneyness grid. [SUPPOSED TO BE PRIOR KNOWLEDGE]
2. Extract:
   - Per‑asset volatilities from synthetic ATM IVs. [SUPPOSED TO BE PRIOR KNOWLEDGE]
   - A $21\times 21$ **historical correlation matrix** from 60‑day rolling log‑returns (used internally for simulation, not as NN features). [SUPPOSED TO BE PRIOR KNOWLEDGE]
3. Define a **multi‑asset GBM** under a risk‑neutral measure $Q^{\text{MC}}_x$ whose:
   - Marginal volatilities match synthetic ATM IVs. [SUPPOSED TO BE PRIOR KNOWLEDGE]
   - Correlations match the historical correlation estimate. [SUPPOSED TO BE PRIOR KNOWLEDGE]
4. Simulate $N_{\text{paths}}$ paths $(S^{0,(j)},\dots,S^{d,(j)})$ over $[0,T]$ under this calibrated model. [SUPPOSED TO BE PRIOR KNOWLEDGE]
5. For each path $j$, compute the discrete dispersion payoff $\widehat{\Phi}^{(j)}_{\text{disp}}$ using the annualized definition from Section 1.3.2. [SUPPOSED TO BE PRIOR KNOWLEDGE]
6. Define the **Monte Carlo pseudo‑fair value**:
   $$
   P_{\text{MC}}(x)
     := \frac{1}{N_{\text{paths}}}
        \sum_{j=1}^{N_{\text{paths}}}
          \widehat{\Phi}^{(j)}_{\text{disp}}.
   $$
   [SUPPOSED TO BE PRIOR KNOWLEDGE]

This is the standard Monte Carlo estimator of $\mathbb E_{Q^{\text{MC}}_x}[\widehat{\Phi}_{\text{disp}}]$: it is unbiased and converges at rate $O(N_{\text{paths}}^{-1/2})$ by the law of large numbers and the central limit theorem, facts which come from basic probability theory rather than Neufeld–Sester. [SUPPOSED TO BE PRIOR KNOWLEDGE]

The idea of first computing (possibly expensive) labels for a family of payoffs and then training a neural network to approximate the input–output map is exactly the supervised‑learning paradigm advocated by Neufeld–Sester (Algorithm 1), but here the label engine is a Monte Carlo simulator instead of an LP/MOT solver.

### 1.5.2 Neural network surrogate and feature vector

The neural network $V_{\text{NN}}(x;\theta)$ is trained in Phase 2 to approximate $P_{\text{MC}}(x)$:
$$
V_{\text{NN}}(x;\theta) \approx P_{\text{MC}}(x).
$$
[SUPPOSED TO BE PRIOR KNOWLEDGE]

The **input feature vector** $x_t \in \mathbb{R}^{274}$ used in the implementation is constructed as follows (consistent with the code):

- Spots and forwards:
  - For each of the 21 assets: spot $S_0^i$ and 3‑month forward $F_0^i = S_0^i e^{(r_i - q_i)T}$
  - Total: $21 \times 2 = 42$ features. [SUPPOSED TO BE PRIOR KNOWLEDGE]
- Rates and dividend yields:
  - For each asset: 3‑month rate $r_i$ and dividend yield $q_i$
  - Total: $21 \times 2 = 42$ features. [SUPPOSED TO BE PRIOR KNOWLEDGE]
- IV surfaces:
  - For each asset and each of 7 moneyness levels $k$ in the fixed grid: implied volatility $\sigma_i(k)$ from the quadratic smile model
  - Total: $21 \times 7 = 147$ features. [SUPPOSED TO BE PRIOR KNOWLEDGE]
- Padding:
  - The above yield $42 + 42 + 147 = 231$ economically meaningful features.
  - The remaining $274 - 231 = 43$ coordinates are **zero padding** used purely to match the desired NN input dimension; they are not interpreted or used downstream. [SUPPOSED TO BE PRIOR KNOWLEDGE]

Importantly, the **60‑day rolling correlation matrix** used in Monte Carlo label generation is *not included* in the feature vector; it is computed directly from historical returns when simulating paths and does not appear as an NN input. [SUPPOSED TO BE PRIOR KNOWLEDGE]

The network architecture implemented in code is:

- Input layer: dimension $274$.
- Hidden layers: fully‑connected MLP with widths $256 \to 128 \to 64$.
- Activation: ReLU after each hidden layer.
- Dropout: $0.2$ on the first two hidden layers for regularization.
- Output layer: linear layer with 1 neuron (scalar fair value). [SUPPOSED TO BE PRIOR KNOWLEDGE]

This is a **simplified 3‑layer architecture** relative to the original 4‑layer conceptual spec (512 → 256 → 128 → 64), chosen for computational efficiency on CPU while retaining sufficient capacity for this problem.

The training configuration is:

- Loss: mean squared error (MSE) between $V_{\text{NN}}(x_t)$ and $P_{\text{MC}}(x_t)$.
- Optimizer: Adam with learning rate $10^{-3}$ and weight decay $10^{-4}$.
- Batch size: $64$.
- Early stopping: patience of 20 epochs based on validation loss.
- Learning‑rate scheduler: ReduceLROnPlateau on validation loss. [SUPPOSED TO BE PRIOR KNOWLEDGE]

---

## 1.6 Trading signal definition

*(Signal math and its exact normalization will be refined once all implementation issues are fully reconciled; for now, this section describes the conceptual intent.)*

Let:

- $V_{\text{NN}}(t) := V_{\text{NN}}(x_t)$ be the **neural‑network fair value** of the dispersion payoff for the market state $x_t$ on date $t$. [SUPPOSED TO BE PRIOR KNOWLEDGE]
- $V_{\text{impl}}(t)$ be the **implementation price** of the dispersion trade inferred from synthetic option surfaces on date $t$ (i.e. the cost of putting on the implementable dispersion structure consistent with our payoff). [SUPPOSED TO BE PRIOR KNOWLEDGE]

Conceptually, the core **mispricing signal** is defined as the deviation between NN fair value and implementation price:
$$
\text{Signal}_t := V_{\text{NN}}(t) - V_{\text{impl}}(t).
$$
[SUPPOSED TO BE PRIOR KNOWLEDGE]

Interpretation (conceptual target):

- If $\text{Signal}_t > 0$: the neural net judges dispersion to be **underpriced** (fair value above implementation price) $\Rightarrow$ candidate to **buy dispersion** (take a long exposure to $\widehat{\Phi}_{\text{disp}}$). [SUPPOSED TO BE PRIOR KNOWLEDGE]
- If $\text{Signal}_t < 0$: dispersion appears **overpriced** $\Rightarrow$ candidate to **sell dispersion** (short exposure to $\widehat{\Phi}_{\text{disp}}$). [SUPPOSED TO BE PRIOR KNOWLEDGE]
- Trades are entered only when $|\text{Signal}_t|$ exceeds a threshold $\theta > 0$ to filter out noise; detailed thresholding and position‑sizing rules are specified in the evaluation and backtesting document. [SUPPOSED TO BE PRIOR KNOWLEDGE]

In the actual implementation, a **normalized version** of this signal is used to make mispricings comparable across regimes; the exact normalization and sign convention will be documented after all coherence checks are completed.

---

## 1.7 Scope and possible extensions

The **baseline scope** of this document is:

- Fixed 3‑month maturity. [SUPPOSED TO BE PRIOR KNOWLEDGE]
- SX5E index + 20 largest constituents by weight (21 underlyings total). [SUPPOSED TO BE PRIOR KNOWLEDGE]
- Single payoff $\Phi_{\text{disp}}$ as defined in Section 1.3, with the convention
  $$
  \Phi_{\text{disp}} = \mathrm{RV}_T^0 - \sum_{i=1}^d w_i \mathrm{RV}_T^i.
  $$
  [SUPPOSED TO BE PRIOR KNOWLEDGE]
- Neural network outputs a single scalar fair value approximating Monte Carlo pseudo‑labels. [SUPPOSED TO BE PRIOR KNOWLEDGE]

Natural **extensions**, consistent with the Neufeld–Sester framework for parametric families of payoffs (Neufeld–Sester, Remark II.3), include:

- Treating weights $(w_1,\dots,w_d)$, horizon $T$, or even different functionals (e.g. basket calls, best‑of options) as **payoff parameters** $\theta$, and training a single network on a whole parametric family $\Phi_\theta$. [SUPPOSED TO BE PRIOR KNOWLEDGE]
- Outputting **lower/upper bounds $(L(x),U(x))$** jointly with the midpoint, as in their multi‑output architectures. [SUPPOSED TO BE PRIOR KNOWLEDGE]
- Incorporating **variance constraints** or additional distributional information into the MOT problem, following their extension to constrained MOT (Neufeld–Sester, Sections III–IV). [SUPPOSED TO BE PRIOR KNOWLEDGE]

For Phase 1–2, however, this document fixes $\Phi_{\text{disp}}$ and the 3‑month horizon, and serves as the mathematical foundation for:

- The label generation pipeline (Monte Carlo under historically calibrated correlations). [SUPPOSED TO BE PRIOR KNOWLEDGE]
- The network training task (learning $x \mapsto P_{\text{MC}}(x)$). [SUPPOSED TO BE PRIOR KNOWLEDGE]
- The backtest engine (mapping predicted mispricing into realised dispersion P\&L). [SUPPOSED TO BE PRIOR KNOWLEDGE]
