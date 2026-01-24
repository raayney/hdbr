import numpy as np
import logging

logger = logging.getLogger(__name__)


class DispersionPayoffCalculator:

    def __init__(self, weights=None):
        
        if weights is None:
            self.weights = np.ones(20) / 20
        else:
            self.weights = np.array(weights)
            assert len(self.weights) == 20, "Must have 20 weights"
            assert np.isclose(self.weights.sum(), 1.0, atol=1e-3), "Weights must sum to 1"


    def realized_variance(self, paths):
        
        log_returns = np.diff(np.log(paths), axis=1)  

        realized_var = np.sum(log_returns**2, axis=1) * 252  

        return realized_var

    def dispersion_payoff(self, index_paths, stock_paths):
        
        var_index = self.realized_variance(index_paths)

        var_stocks = np.array([self.realized_variance(stock_paths[i]) for i in range(20)])

        var_basket = np.dot(self.weights, var_stocks)

        payoff = var_index - var_basket

        return payoff

    def expected_payoff(self, index_paths, stock_paths):
        
        payoffs = self.dispersion_payoff(index_paths, stock_paths)
        return np.mean(payoffs)
