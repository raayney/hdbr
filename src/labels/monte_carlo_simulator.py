import numpy as np
import logging

logger = logging.getLogger(__name__)


class MonteCarloSimulator:

    def __init__(self, n_paths=10000, n_steps=63, dt=1/252, seed=42):
        
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.dt = dt
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def simulate_gbm(self, S0, mu, sigma):
        
        Z = self.rng.standard_normal((self.n_paths, self.n_steps))

        paths = np.zeros((self.n_paths, self.n_steps + 1))
        paths[:, 0] = S0

        for t in range(1, self.n_steps + 1):
            paths[:, t] = paths[:, t-1] * np.exp(
                (mu - 0.5 * sigma**2) * self.dt + sigma * np.sqrt(self.dt) * Z[:, t-1]
            )

        return paths

    def simulate_basket(self, spots, mus, sigmas, correlation_matrix):
        """Cholesky decomposition"""
        n_assets = len(spots)

        try:
            L = np.linalg.cholesky(correlation_matrix)
        except np.linalg.LinAlgError:
            logger.warning("Correlation matrix not positive definite, using identity")
            L = np.eye(n_assets)

        Z_independent = self.rng.standard_normal((n_assets, self.n_paths, self.n_steps))

        Z_correlated = np.einsum('ij,jkl->ikl', L, Z_independent)

        all_paths = np.zeros((n_assets, self.n_paths, self.n_steps + 1))
        all_paths[:, :, 0] = spots[:, np.newaxis]

        for i in range(n_assets):
            for t in range(1, self.n_steps + 1):
                all_paths[i, :, t] = all_paths[i, :, t-1] * np.exp(
                    (mus[i] - 0.5 * sigmas[i]**2) * self.dt +
                    sigmas[i] * np.sqrt(self.dt) * Z_correlated[i, :, t-1]
                )

        return all_paths
