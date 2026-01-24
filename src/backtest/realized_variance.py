# src/backtest/realized_variance.py
import numpy as np
import pandas as pd
from typing import Dict

class RealizedVarianceCalculator:
    """
    Compute realized variance from actual returns.
    
    Formula: σ²_realized = (252 / n) × Σ(r_t²)
    where n = number of days, 252 = trading days per year
    
    References:
    - Realized variance definition: Wikipedia, Barndorff-Nielsen & Shephard (2002)
    - Annualization convention: Standard in volatility literature
    """
    
    def __init__(self, holding_period_days=20, trading_days_per_year=252):
        """
        Args:
            holding_period_days: Standard holding period (20 days = 1 month)
            trading_days_per_year: Convention for annualization (252)
        """
        self.holding_period = holding_period_days
        self.trading_days = trading_days_per_year
        
    def compute_realized_variance(self, returns_series):
        """
        Compute annualized realized variance from daily log returns.
        
        Args:
            returns_series: pandas Series or numpy array of daily log returns
            
        Returns:
            float: Annualized realized variance
        """
        # Convert to numpy array if needed
        if isinstance(returns_series, pd.Series):
            returns = returns_series.values
        else:
            returns = np.array(returns_series)
        
        # Remove NaN values
        returns = returns[~np.isnan(returns)]
        
        if len(returns) == 0:
            return np.nan
        
        # Sum of squared returns
        squared_returns_sum = np.sum(returns ** 2)
        
        # Annualize: (252 / n) × Σ(r_t²)
        realized_var = (self.trading_days / len(returns)) * squared_returns_sum
        
        return realized_var
    
    def compute_realized_volatility(self, returns_series):
        """
        Compute annualized realized volatility (sqrt of variance).
        
        Args:
            returns_series: pandas Series or array of daily log returns
            
        Returns:
            float: Annualized realized volatility (in decimal form, e.g., 0.20 = 20%)
        """
        rv = self.compute_realized_variance(returns_series)
        return np.sqrt(rv) if not np.isnan(rv) else np.nan
    
    def compute_dispersion_payoff(self, 
                                   index_returns, 
                                   constituent_returns_dict, 
                                   weights_dict):
        """
        Compute dispersion payoff: σ²_index - Σ(w_i × σ²_i).
        
        This is the core payoff for variance dispersion trading.
        
        Args:
            index_returns: Series of index returns over holding period
            constituent_returns_dict: Dict mapping ticker -> returns Series
            weights_dict: Dict mapping ticker -> weight (should sum to ~1.0)
            
        Returns:
            float: Realized dispersion (can be positive or negative)
        """
        # Compute index realized variance
        index_rv = self.compute_realized_variance(index_returns)
        
        if np.isnan(index_rv):
            return np.nan
        
        # Compute weighted sum of constituent realized variances
        constituent_rv_weighted = 0.0
        total_weight = 0.0
        
        for ticker, weight in weights_dict.items():
            if ticker in constituent_returns_dict:
                constituent_rv = self.compute_realized_variance(
                    constituent_returns_dict[ticker]
                )
                if not np.isnan(constituent_rv):
                    constituent_rv_weighted += weight * constituent_rv
                    total_weight += weight
        
        # Normalize weights if needed
        if total_weight > 0 and abs(total_weight - 1.0) > 0.01:
            constituent_rv_weighted /= total_weight
        
        # Dispersion = index variance - weighted constituent variance
        # Typically negative (index var < constituent var due to diversification)
        dispersion = index_rv - constituent_rv_weighted
        
        return dispersion
    
    def batch_compute_dispersion(self, 
                                  dates, 
                                  returns_loader,
                                  index_ticker='STOXX50E',
                                  constituent_tickers=None,
                                  weights=None):
        """
        Compute realized dispersion for multiple dates.
        
        Args:
            dates: List of entry dates
            returns_loader: ReturnsDataLoader instance
            index_ticker: Index ticker
            constituent_tickers: List of constituent tickers
            weights: Dict of weights (default: equal weight)
            
        Returns:
            Dict: {date: realized_dispersion}
        """
        if weights is None and constituent_tickers is not None:
            weights = {t: 1.0/len(constituent_tickers) for t in constituent_tickers}
        
        results = {}
        
        for date in dates:
            forward_returns = returns_loader.get_forward_returns(date, self.holding_period)
            
            if forward_returns is None or len(forward_returns) < self.holding_period:
                results[date] = np.nan
                continue
            
            # Extract returns
            index_returns = forward_returns[index_ticker]
            constituent_returns_dict = {
                ticker: forward_returns[ticker]
                for ticker in constituent_tickers
                if ticker in forward_returns.columns
            }
            
            # Compute dispersion
            dispersion = self.compute_dispersion_payoff(
                index_returns,
                constituent_returns_dict,
                weights
            )
            
            results[date] = dispersion
        
        return results