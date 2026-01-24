"""Preprocess prices: compute returns, variance, correlations."""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PricePreprocessor:
    """Preprocess price data."""
    
    def __init__(self, data_dict, trading_days_per_year=252):
        self.data_dict = data_dict
        self.trading_days_per_year = trading_days_per_year
    
    def align_and_merge(self):
        """Merge all assets on common dates."""
        dfs = [df[['Adj Close']].rename(columns={'Adj Close': ticker}) 
               for ticker, df in self.data_dict.items()]
        merged = pd.concat(dfs, axis=1, join='inner').dropna()
        logger.info(f"Aligned to {len(merged)} trading dates")
        return merged
    
    def compute_log_returns(self, prices):
        """Daily log-returns: log(P_t / P_{t-1})."""
        returns = np.log(prices / prices.shift(1)).dropna()
        logger.info(f"Computed returns: {returns.shape}")
        return returns
    
    def compute_rolling_variance(self, returns, window=30):
        """Rolling realized variance: sum(r_t^2) over window."""
        var = (returns ** 2).rolling(window=window).sum()
        logger.info(f"Computed rolling variance (window={window})")
        return var
    
    def compute_rolling_vol(self, rolling_var, annualize=True):
        """Convert variance to volatility."""
        vol = np.sqrt(rolling_var)
        if annualize:
            vol = vol * np.sqrt(self.trading_days_per_year)
        logger.info(f"Computed rolling vol (annualized={annualize})")
        return vol
    
    def compute_correlation_matrix(self, returns, window=60):
        """Rolling correlation between all assets."""
        corr_matrices = {}
        for date in returns.index:
            subset = returns.loc[:date].tail(window)
            if len(subset) >= 5:
                corr_matrices[date] = subset.corr()
        logger.info(f"Computed {len(corr_matrices)} correlation matrices")
        return corr_matrices
    def compute_realized_correlation_scalar(self, returns, window=60):
        """
        Compute rolling mean correlation as a scalar regime indicator.
        For each date, computes the 60-day rolling correlation matrix of all 21 assets,
        then extracts the mean of the upper triangle to get a single scalar correlation measure.
        """
        import numpy as np
        import pandas as pd
    
        logger.info(f"Computing {window}-day rolling correlation matrices...")
        correlation_series = {}
    
        for date_idx in range(window, len(returns)):
            # Get last N days of returns
            lookback_window = returns.iloc[date_idx - window : date_idx]
        
            # Compute correlation matrix (21x21)
            corr_matrix = lookback_window.corr()
        
            # Extract upper triangle (exclude diagonal)
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            upper_triangle = corr_matrix.values[mask]
        
            # Single scalar: mean of all pairwise correlations
            mean_correlation = np.mean(upper_triangle)
            correlation_series[returns.index[date_idx]] = mean_correlation
    
        logger.info(f"Computed correlations for {len(correlation_series)} dates")
        return pd.Series(correlation_series)


    def compute_realized_skewness_scalar(self, returns, window=60):
        """
        Compute rolling mean skewness as a scalar regime indicator.
        For each date, computes the 60-day rolling skewness of all 21 assets,
        then takes the mean to get a single scalar skewness measure.
        """
        from scipy.stats import skew
        import numpy as np
        import pandas as pd
    
        logger.info(f"Computing {window}-day rolling skewness...")
        skewness_series = {}
    
        for date_idx in range(window, len(returns)):
            # Get last N days of returns
            lookback_window = returns.iloc[date_idx - window : date_idx]
        
            # Compute skewness for each asset
            asset_skews = skew(lookback_window.values, axis=0, bias=False)
        
            # Single scalar: mean skewness across 21 assets
            mean_skewness = np.mean(asset_skews)
            kewness_series[returns.index[date_idx]] = mean_skewness
    
        logger.info(f"Computed skewness for {len(skewness_series)} dates")
        return pd.Series(skewness_series)


    def compute_volatility_percentile(self, rolling_vol, window=252):
        """
        Compute rolling volatility percentile rank as a scalar regime indicator.
        For each date, ranks the current 30-day realized vol as a percentile
        against the 252-day (1-year) history.
        """
        import numpy as np
        import pandas as pd
    
        logger.info(f"Computing volatility percentile vs {window}-day history...")
        vol_percentile_series = {}
    
        for date_idx in range(window, len(rolling_vol)):
            # Get last 252 days of vol history
            lookback_vol_history = rolling_vol.iloc[date_idx - window : date_idx]
        
            # Current vol
            current_vol = rolling_vol.iloc[date_idx]
        
            # Percentile: what % of past 252 vols are <= current vol?
            percentile_rank = (lookback_vol_history <= current_vol).sum() / len(lookback_vol_history) * 100.0
            vol_percentile_series[rolling_vol.index[date_idx]] = percentile_rank
    
        logger.info(f"Computed vol percentile for {len(vol_percentile_series)} dates")
        return pd.Series(vol_percentile_series)