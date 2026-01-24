"""Monte Carlo simulator for stock price paths under Black-Scholes."""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    """Simulate forward stock paths under GBM."""

    def __init__(self, n_paths=10000, n_steps=63, dt=1/252, seed=42):
        """
        Args:
            n_paths: Number of Monte Carlo paths
            n_steps: Number of time steps (63 days ≈ 3 months)
            dt: Time step in years (1/252 for daily)
            seed: Random seed for reproducibility
        """
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.dt = dt
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def simulate_gbm(self, S0, mu, sigma):
        """
        Simulate geometric Brownian motion paths.

        Args:
            S0: Initial spot price
            mu: Drift (risk-neutral: mu = r - q)
            sigma: Volatility (from IV surface)

        Returns:
            paths: Array of shape (n_paths, n_steps+1) with price paths
        """
        # Generate random shocks
        Z = self.rng.standard_normal((self.n_paths, self.n_steps))

        # Initialize paths
        paths = np.zeros((self.n_paths, self.n_steps + 1))
        paths[:, 0] = S0

        # Simulate forward
        for t in range(1, self.n_steps + 1):
            paths[:, t] = paths[:, t-1] * np.exp(
                (mu - 0.5 * sigma**2) * self.dt + sigma * np.sqrt(self.dt) * Z[:, t-1]
            )

        return paths

    def simulate_basket(self, spots, mus, sigmas, correlation_matrix):
        """
        Simulate correlated basket of stocks.

        Args:
            spots: Array of initial spot prices (21,)
            mus: Array of drifts (21,)
            sigmas: Array of volatilities (21,)
            correlation_matrix: Correlation between returns (21, 21)

        Returns:
            all_paths: Array of shape (21, n_paths, n_steps+1)
        """
        n_assets = len(spots)

        # Cholesky decomposition for correlation
        try:
            L = np.linalg.cholesky(correlation_matrix)
        except np.linalg.LinAlgError:
            logger.warning("Correlation matrix not positive definite, using identity")
            L = np.eye(n_assets)

        # Generate independent standard normal shocks
        Z_independent = self.rng.standard_normal((n_assets, self.n_paths, self.n_steps))

        # Apply correlation via Cholesky
        Z_correlated = np.einsum('ij,jkl->ikl', L, Z_independent)

        # Initialize paths for all assets
        all_paths = np.zeros((n_assets, self.n_paths, self.n_steps + 1))
        all_paths[:, :, 0] = spots[:, np.newaxis]

        # Simulate forward for each asset
        for i in range(n_assets):
            for t in range(1, self.n_steps + 1):
                all_paths[i, :, t] = all_paths[i, :, t-1] * np.exp(
                    (mus[i] - 0.5 * sigmas[i]**2) * self.dt +
                    sigmas[i] * np.sqrt(self.dt) * Z_correlated[i, :, t-1]
                )

        return all_paths