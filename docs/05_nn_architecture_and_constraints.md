# 5. Neural network architecture and constraints

## 5.1 Objectives and design criteria

The neural network acts as a fast surrogate for the Monte Carlo pricing engine described in document 04. Given a feature vector $`x_t \in \mathbb{R}^{274}`$ derived from Yahoo prices and synthetic implied volatility surfaces, the network outputs a scalar estimate of the three‑month dispersion fair value

$`V_{\text{NN}}(x_t;\theta) \approx P_{\text{MC}}(x_t),`$

where $`P_{\text{MC}}(x_t)`$ is the Monte Carlo label computed under the correlated GBM model.

The main design goals are:

- expressiveness  
  capture non‑linear relationships between the option surfaces, volatility levels and the dispersion payoff;

- stability and regularity  
  avoid pathological behaviour in regions with sparse training data and ensure smooth responses to small input perturbations;

- computational efficiency  
  trainable on a laptop CPU and fast enough for real‑time inference once trained;

- financial sanity  
  respect basic qualitative properties of dispersion (for example dependence on volatility and correlation) at least in a soft, empirical sense.

For the initial implementation, the architecture is a fully connected feed‑forward network (multi‑layer perceptron, MLP) with ReLU activations and dropout regularisation. The sizes and dropout rates in this document are chosen to match the actual code in `src/models/traindispersionnn.py` and `src/models/basketpricingnet.py`.

---

## 5.2 Baseline architecture

### 5.2.1 Input layer

The network takes as input the normalised feature vector

$`x_t^{\text{norm}} \in \mathbb{R}^{274},`$

constructed and scaled as described in documents 03 and 04. The FeatureScaler ensures that each coordinate has approximately zero mean and unit variance on the training set.

Formally, the input to the network is

$`h^{(0)} := x_t^{\text{norm}}.`$

### 5.2.2 Hidden layers

The implemented architecture uses three hidden layers with widths

$`274 \rightarrow 256 \rightarrow 128 \rightarrow 64.`$

For $`\ell = 1,2,3`$, the transformation at layer $`\ell`$ is

$`\begin{aligned}
z^{(\ell)} &= W^{(\ell)} h^{(\ell-1)} + b^{(\ell)}, \\
h^{(\ell)} &= \max\{0, z^{(\ell)}\},
\end{aligned}`$

where $`W^{(\ell)}`$ and $`b^{(\ell)}`$ are the learnable weights and biases, and the maximum is applied elementwise (ReLU activation).

Dropout with rate 0.2 is applied after the first and second hidden layers during training:

- after $`h^{(1)}`$,
- after $`h^{(2)}`$,

to reduce overfitting and encourage robustness.

In PyTorch terms, `BasketPricingNet` is defined as:

- `Linear(274, 256)`, `ReLU`, `Dropout(p=0.2)`,
- `Linear(256, 128)`, `ReLU`, `Dropout(p=0.2)`,
- `Linear(128, 64)`, `ReLU`.

This matches the configuration used in the progress report and training script.

### 5.2.3 Output layer

The network outputs a single scalar corresponding to the Monte Carlo fair value label:

$`V_{\text{NN}}(x_t;\theta) = w^{\top} h^{(3)} + b_{\text{out}} \in \mathbb{R},`$

where $`h^{(3)} \in \mathbb{R}^{64}`$, $`w \in \mathbb{R}^{64}`$ and $`b_{\text{out}} \in \mathbb{R}`$.

There is no non‑linear activation on the output layer; it is a pure linear projection. This is appropriate since the dispersion payoff and its Monte Carlo expectation can take both positive and negative values.

If, in a later extension, model‑free lower and upper bounds $`(L(x), U(x))`$ become available from a martingale optimal transport solver, the output layer can be extended to predict a 2‑ or 3‑dimensional vector

$`(L(x), P_{\text{fair}}(x), U(x))`$

rather than a single scalar.

### 5.2.4 Parameter count and feasibility

For an input dimension of 274 and hidden widths 256, 128 and 64, the total number of parameters is approximately 108,545, as reported in the training logs. This size is easily trainable on a laptop CPU with mini‑batch stochastic gradient descent and is modest compared to typical modern neural networks.

The architecture provides sufficient capacity to model the non‑linear dependence of dispersion on synthetic implied volatilities and historical regimes, while remaining small enough to limit overfitting and keep training times reasonable.

---

## 5.3 Loss function and optimisation

### 5.3.1 Loss

The network is trained on scalar Monte Carlo labels $`P_{\text{MC}}(x_t)`$ (see document 04). The primary loss is mean squared error (MSE) with L2 regularisation (weight decay):

$`\mathcal{L}(\theta)
  = \frac{1}{|\mathcal{T}_{\text{train}}|}
    \sum_{t \in \mathcal{T}_{\text{train}}}
      \big(V_{\text{NN}}(x_t;\theta) - P_{\text{MC}}(x_t)\big)^2
    + \lambda \,\|\theta\|_2^2,`$

where:

- $`\mathcal{T}_{\text{train}}`$ is the set of training dates,
- $`\theta`$ collects all network parameters,
- $`\lambda`$ is the weight‑decay coefficient.

In the current configuration, $`\lambda = 10^{-4}`$, matching the weight decay used in the PyTorch optimiser.

No explicit smoothness or financial constraint penalties are active in the training script at present; any such terms remain optional extensions.

### 5.3.2 Optimiser and training schedule

Training is performed using the Adam optimiser with:

- learning rate $`10^{-3}`$,
- weight decay $`10^{-4}`$ (L2 regularisation),
- batch size between 32 and 64 (default 64 in configuration),
- up to approximately 200 epochs with early stopping.

Concretely, the training script `src/models/traindispersionnn.py`:

1. Loads `X_train`, `X_val`, `y_train`, `y_val` from the feature and label directories.
2. Instantiates `BasketPricingNet(input_dim=274, hidden_dims=[256,128,64], dropout=0.2)`.
3. Uses `torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)`.
4. Minimises MSE loss on mini‑batches of the training set.
5. Evaluates validation loss after each epoch.
6. Applies early stopping with patience 20 epochs: if validation loss does not improve for 20 consecutive epochs, training stops and the best model is kept.

A learning‑rate scheduler of type `ReduceLROnPlateau` is used to reduce the learning rate by a fixed factor when validation loss plateaus, improving convergence in later epochs.

---

## 5.4 Regularisation and constraints

Regularisation in the current implementation is purely via standard machine learning techniques rather than explicit financial constraints.

### 5.4.1 Weight decay and dropout

The two main regularisation mechanisms are:

- weight decay  
  L2 penalty on all parameters, implemented via the optimiser’s `weight_decay` argument ($`\lambda = 10^{-4}`$);

- dropout  
  applied with rate 0.2 on the first two hidden layers. During training, each neuron in these layers is set to zero with probability 0.2, independently of others.

These methods help prevent overfitting and promote generalisation to new market states, especially given the limited number of training dates relative to the number of parameters.

### 5.4.2 Optional smoothness penalties (not currently enabled)

If desired in future experiments, an additional smoothness penalty can be added to encourage the mapping $`x \mapsto V_{\text{NN}}(x)`$ to be locally Lipschitz. For instance, for each mini‑batch $`\{x_b\}_{b=1}^B`$ and small perturbations $`\delta_b`$ (for example Gaussian noise),

$`\mathcal{L}_{\text{smooth}}
  = \frac{1}{B} \sum_{b=1}^B
      \big(V_{\text{NN}}(x_b + \delta_b;\theta)
         - V_{\text{NN}}(x_b;\theta)\big)^2.`$

The total loss would then be

$`\mathcal{L}_{\text{total}}
  = \mathcal{L} + \gamma \,\mathcal{L}_{\text{smooth}},`$

with $`\gamma`$ a tunable coefficient. This is not present in the current training script, but can be added if inspection of residuals suggests that the network is too sensitive to small feature perturbations.

### 5.4.3 Financial consistency (soft checks rather than hard constraints)

The network does not currently encode hard financial constraints such as no‑arbitrage inequalities on option prices, because it outputs a dispersion fair value rather than vanilla option prices.

However, certain qualitative relationships can be monitored empirically:

- dependence on volatility  
  on average, higher single‑stock and index volatilities (all else equal) should influence the dispersion value in predictable ways;

- dependence on correlation  
  more highly correlated regimes tend to reduce dispersion (index variance closer to basket variance), so labels and predictions should reflect this behaviour over time.

These relationships are checked through diagnostic plots and partial dependence analysis in the evaluation notebook, rather than being enforced as strict constraints in the loss function.

---

## 5.5 Implementation structure

### 5.5.1 Model class

The network is implemented in PyTorch in `src/models/basketpricingnet.py` as a reusable class. The class signature is approximately:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasketPricingNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=None, dropout: float = 0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = 

        layers = []
        prev_dim = input_dim
        for i, h in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if i < 2 and dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h

        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, 1)  # scalar fair value

    def forward(self, x):
        h = self.feature_extractor(x)
        out = self.output_layer(h)
        return out.squeeze(-1)