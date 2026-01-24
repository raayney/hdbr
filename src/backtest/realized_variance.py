import numpy as np
import pandas as pd
from typing import Dict

class RealizedVarianceCalculator:
    
    def __init__(self, holding_period_days=20, trading_days_per_year=252):
        
        self.holding_period = holding_period_days
        self.trading_days = trading_days_per_year
        
    def compute_realized_variance(self, returns_series):
        
        if isinstance(returns_series, pd.Series):
            returns = returns_series.values
        else:
            returns = np.array(returns_series)
        
        returns = returns[~np.isnan(returns)]
        
        if len(returns) == 0:
            return np.nan
        
        squared_returns_sum = np.sum(returns ** 2)
        
        realized_var = (self.trading_days / len(returns)) * squared_returns_sum
        
        return realized_var
    
    def compute_realized_volatility(self, returns_series):
        
        rv = self.compute_realized_variance(returns_series)
        return np.sqrt(rv) if not np.isnan(rv) else np.nan
    
    def compute_dispersion_payoff(self, 
                                   index_returns, 
                                   constituent_returns_dict, 
                                   weights_dict):
        
        index_rv = self.compute_realized_variance(index_returns)
        
        if np.isnan(index_rv):
            return np.nan
        
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
        
        if total_weight > 0 and abs(total_weight - 1.0) > 0.01:
            constituent_rv_weighted /= total_weight
        
        dispersion = index_rv - constituent_rv_weighted
        
        return dispersion
    
    def batch_compute_dispersion(self, 
                                  dates, 
                                  returns_loader,
                                  index_ticker='STOXX50E',
                                  constituent_tickers=None,
                                  weights=None):
        
        if weights is None and constituent_tickers is not None:
            weights = {t: 1.0/len(constituent_tickers) for t in constituent_tickers}
        
        results = {}
        
        for date in dates:
            forward_returns = returns_loader.get_forward_returns(date, self.holding_period)
            
            if forward_returns is None or len(forward_returns) < self.holding_period:
                results[date] = np.nan
                continue
            
            index_returns = forward_returns[index_ticker]
            constituent_returns_dict = {
                ticker: forward_returns[ticker]
                for ticker in constituent_tickers
                if ticker in forward_returns.columns
            }
            
            dispersion = self.compute_dispersion_payoff(
                index_returns,
                constituent_returns_dict,
                weights
            )
            
            results[date] = dispersion
        
        return results
