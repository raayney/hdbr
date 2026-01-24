"""Compute realized variance and dispersion payoff from simulated paths."""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class DispersionPayoffCalculator:
    """Calculate realized variance and dispersion spread."""

    def __init__(self, weights=None):
        """
        Args:
            weights: Array of basket weights (20,). If None, use equal weights.
        """
        if weights is None:
            # Equal weights for 20 stocks
            self.weights = np.ones(20) / 20
        else:
            self.weights = np.array(weights)
            assert len(self.weights) == 20, "Must have 20 weights"
            assert np.isclose(self.weights.sum(), 1.0, atol=1e-3), "Weights must sum to 1"


    def realized_variance(self, paths):
        """
        Compute realized variance from price paths.

        Realized variance = sum of squared log-returns (annualized).

        Args:
            paths: Array of shape (n_paths, n_steps+1)

        Returns:
            realized_var: Array of shape (n_paths,) with realized variance per path
        """
        # Compute log-returns
        log_returns = np.diff(np.log(paths), axis=1)  # (n_paths, n_steps)

        # Sum of squared returns (already in variance units if daily)
        realized_var = np.sum(log_returns**2, axis=1) * 252  # Annualize

        return realized_var

    def dispersion_payoff(self, index_paths, stock_paths):
        """
        Compute dispersion payoff: Var(Index) - weighted sum of Var(Stocks).

        Args:
            index_paths: Array of shape (n_paths, n_steps+1) for index
            stock_paths: Array of shape (20, n_paths, n_steps+1) for stocks

        Returns:
            payoff: Array of shape (n_paths,) with dispersion payoff per path
        """
        # Realized variance of index
        var_index = self.realized_variance(index_paths)

        # Realized variance of each stock
        var_stocks = np.array([self.realized_variance(stock_paths[i]) for i in range(20)])

        # Weighted sum of stock variances
        var_basket = np.dot(self.weights, var_stocks)

        # Dispersion payoff
        payoff = var_index - var_basket

        return payoff

    def expected_payoff(self, index_paths, stock_paths):
        """
        Compute expected dispersion payoff (mean over all MC paths).

        Args:
            index_paths: Array of shape (n_paths, n_steps+1)
            stock_paths: Array of shape (20, n_paths, n_steps+1)

        Returns:
            expected_value: Scalar, mean payoff
        """
        payoffs = self.dispersion_payoff(index_paths, stock_paths)
        return np.mean(payoffs)