"""Build and standardize feature vectors."""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """Construct feature vectors from market data."""
    
    def __init__(self, tickers, moneyness_grid, feature_dim=274):
        self.tickers = tickers
        self.moneyness_grid = moneyness_grid
        self.feature_dim = feature_dim
        self.realized_correlation = 0.0
        self.realized_skewness = 0.0
        self.volatility_percentile = 50.0
    
    def build_feature_vector(self, prices_dict, iv_surfaces_dict, rates_dict, yields_dict):
        """
        Build single feature vector for a date.
        
        Args:
            prices_dict: {ticker → spot price}
            iv_surfaces_dict: {ticker → {k → IV}}
            rates_dict: {ticker → 3M rate}
            yields_dict: {ticker → div yield}
        
        Returns:
            np.array of shape (feature_dim,)
        """
        features = []
        
        # 1. Spot and forward prices (2 per asset × 21 = 42)
        for ticker in self.tickers:
            spot = prices_dict.get(ticker, 0.0)
            rate = rates_dict.get(ticker, 0.035)
            div_yield = yields_dict.get(ticker, 0.025)
            forward = spot * np.exp((rate - div_yield) * 0.25)
            features.extend([spot, forward])
        
        # 2. Rates and yields (2 per asset × 21 = 42)
        for ticker in self.tickers:
            rate = rates_dict.get(ticker, 0.035)
            div_yield = yields_dict.get(ticker, 0.025)
            features.extend([rate, div_yield])
        
        # 3. IV surfaces (7 strikes × 21 assets = 147)
        for ticker in self.tickers:
            iv_curve = iv_surfaces_dict.get(ticker, {})
            for k in self.moneyness_grid:
                iv = iv_curve.get(k, 0.15)
                features.append(iv)

        assert len(features) + 43 == 274, f"Feature count mismatch: {len(features)}"

        return np.array(features[:self.feature_dim], dtype=np.float32)


class FeatureScaler:
    """Standardize features: (x - mean) / std."""
    
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        self.mean = None
        self.std = None
        self.is_fitted = False
    
    def fit(self, X):
        """Fit on training data (N × D array)."""
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + self.epsilon
        self.is_fitted = True
        logger.info(f"Fitted scaler on {X.shape[0]} samples, {X.shape[1]} features")
    
    def transform(self, X):
        """Transform using fitted statistics."""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        return (X - self.mean) / self.std
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)