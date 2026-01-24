# src/backtest/payoff_engine.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

@dataclass
class Trade:
    """Represents a single dispersion trade with full lifecycle."""
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float          # Market price at entry (what we paid)
    nn_fair_value: float        # NN prediction at entry
    signal: float               # (Market - NN) / |NN|
    realized_dispersion: float  # Actual realized dispersion at settlement
    pnl: float                  # Realized - Entry
    holding_days: int           # Actual holding period
    
    def to_dict(self):
        """Convert to dictionary for DataFrame construction."""
        d = asdict(self)
        # Convert timestamps to strings for JSON serialization
        d['entry_date'] = str(self.entry_date.date())
        d['exit_date'] = str(self.exit_date.date())
        return d
    
    def __repr__(self):
        return (f"Trade(entry={self.entry_date.date()}, "
                f"signal={self.signal:.2f}, "
                f"entry_price={self.entry_price:.2f}, "
                f"realized={self.realized_dispersion:.2f}, "
                f"pnl={self.pnl:.2f})")

class PayoffSettlementEngine:
    """
    Compute P&L for dispersion trades with full position lifecycle.
    
    Trade lifecycle:
    1. Entry: Buy dispersion at market price (NN signal triggers entry)
    2. Holding: Monitor position over holding period
    3. Settlement: Compute realized dispersion, settle P&L
    
    P&L = Realized_dispersion - Entry_price
    """
    
    def __init__(self, 
                 returns_loader,
                 realized_var_calculator,
                 holding_period_days=20):
        """
        Args:
            returns_loader: ReturnsDataLoader instance
            realized_var_calculator: RealizedVarianceCalculator instance
            holding_period_days: Holding period (20 days = 1 month)
        """
        self.returns_loader = returns_loader
        self.rv_calc = realized_var_calculator
        self.holding_period = holding_period_days
        
        print(f"Payoff Settlement Engine initialized:")
        print(f"  - Holding period: {self.holding_period} trading days")
        
    def settle_trade(self, 
                     entry_date, 
                     entry_price, 
                     nn_fair_value,
                     signal,
                     index_ticker='STOXX50E',
                     constituent_tickers=None,
                     weights=None) -> Optional[Trade]:
        """
        Settle a single trade by computing realized dispersion.
        
        Args:
            entry_date: Trade entry date
            entry_price: Market price at entry
            nn_fair_value: NN prediction at entry
            signal: Trading signal value
            index_ticker: Index ticker symbol
            constituent_tickers: List of constituent tickers
            weights: Dict of constituent weights
            
        Returns:
            Trade object with P&L, or None if insufficient data
        """
        # Default equal weights
        if weights is None and constituent_tickers is not None:
            n_assets = len(constituent_tickers)
            weights = {t: 1.0/n_assets for t in constituent_tickers}
        
        # Get forward returns over holding period
        forward_returns = self.returns_loader.get_forward_returns(
            entry_date, 
            self.holding_period
        )
        
        # Check data availability
        if forward_returns is None:
            return None
        
        if len(forward_returns) < self.holding_period:
            return None  # Insufficient forward data
        
        # Extract index and constituent returns
        if index_ticker not in forward_returns.columns:
            return None
        
        index_returns = forward_returns[index_ticker]
        
        constituent_returns_dict = {}
        for ticker in constituent_tickers:
            if ticker in forward_returns.columns:
                constituent_returns_dict[ticker] = forward_returns[ticker]
        
        if len(constituent_returns_dict) < len(constituent_tickers) * 0.8:
            # More than 20% of constituents missing
            return None
        
        # Compute realized dispersion
        realized_dispersion = self.rv_calc.compute_dispersion_payoff(
            index_returns,
            constituent_returns_dict,
            weights
        )
        
        if np.isnan(realized_dispersion):
            return None
        
        # Compute P&L
        # Long dispersion: profit when realized > entry
        pnl = realized_dispersion - entry_price
        
        # Exit date
        exit_date = forward_returns.index[-1]
        
        # Create trade record
        trade = Trade(
            entry_date=pd.to_datetime(entry_date),
            exit_date=exit_date,
            entry_price=entry_price,
            nn_fair_value=nn_fair_value,
            signal=signal,
            realized_dispersion=realized_dispersion,
            pnl=pnl,
            holding_days=len(forward_returns)
        )
        
        return trade
    
    def settle_multiple_trades(self, 
                               trades_config: List[Dict],
                               verbose=True) -> pd.DataFrame:
        """
        Settle multiple trades and return summary DataFrame.
        
        Args:
            trades_config: List of trade configuration dicts, each containing:
                - entry_date
                - entry_price
                - nn_fair_value
                - signal
                - (optional) index_ticker, constituent_tickers, weights
            verbose: Print progress
            
        Returns:
            pd.DataFrame: Trade log with all results
        """
        trades = []
        
        for i, config in enumerate(trades_config):
            if verbose and (i+1) % 10 == 0:
                print(f"  Settling trades: {i+1}/{len(trades_config)}")
            
            trade = self.settle_trade(**config)
            
            if trade is not None:
                trades.append(trade)
        
        # Convert to DataFrame
        if trades:
            trades_df = pd.DataFrame([t.to_dict() for t in trades])
            
            if verbose:
                print(f"\nSettled {len(trades_df)}/{len(trades_config)} trades successfully")
                print(f"  Total P&L: {trades_df['pnl'].sum():.2f}")
                print(f"  Mean P&L: {trades_df['pnl'].mean():.2f}")
                print(f"  Win rate: {(trades_df['pnl'] > 0).mean()*100:.1f}%")
            
            return trades_df
        else:
            return pd.DataFrame()
    
    def compute_summary_statistics(self, trades_df: pd.DataFrame) -> Dict:
        """
        Compute comprehensive summary statistics for trade log.
        
        Args:
            trades_df: Trade log DataFrame
            
        Returns:
            Dict of summary statistics
        """
        if len(trades_df) == 0:
            return {}
        
        pnl = trades_df['pnl'].values
        
        stats = {
            'num_trades': len(trades_df),
            'total_pnl': pnl.sum(),
            'mean_pnl': pnl.mean(),
            'median_pnl': np.median(pnl),
            'std_pnl': pnl.std(),
            'min_pnl': pnl.min(),
            'max_pnl': pnl.max(),
            'win_rate': (pnl > 0).mean(),
            'profit_factor': pnl[pnl > 0].sum() / abs(pnl[pnl < 0].sum()) if (pnl < 0).any() else np.inf,
            'avg_win': pnl[pnl > 0].mean() if (pnl > 0).any() else 0,
            'avg_loss': pnl[pnl < 0].mean() if (pnl < 0).any() else 0,
            'sharpe_ratio': pnl.mean() / pnl.std() if pnl.std() > 0 else 0,
        }
        
        return stats