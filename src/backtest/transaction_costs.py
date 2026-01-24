import numpy as np
from dataclasses import dataclass

@dataclass
class TransactionCosts:
    """Transaction cost configuration."""
    entry_spread_bps: float = 15.0      # Bid-ask at entry (bps)
    exit_spread_bps: float = 15.0       # Bid-ask at exit (bps)
    commission_bps: float = 5.0         # Commission per side (bps)
    slippage_bps: float = 10.0          # Market impact (bps)
    
    def total_round_trip_bps(self):
        """Total cost for round-trip trade (entry + exit)."""
        entry_cost = self.entry_spread_bps / 2 + self.commission_bps + self.slippage_bps / 2
        exit_cost = self.exit_spread_bps / 2 + self.commission_bps + self.slippage_bps / 2
        return entry_cost + exit_cost
    
    def entry_cost_bps(self):
        """Cost at entry."""
        return self.entry_spread_bps / 2 + self.commission_bps + self.slippage_bps / 2
    
    def exit_cost_bps(self):
        """Cost at exit."""
        return self.exit_spread_bps / 2 + self.commission_bps + self.slippage_bps / 2

class TransactionCostCalculator:
    """
    Calculate transaction costs for dispersion trades.
    
    For dispersion trading:
    - Entry cost reduces entry price (makes spread wider)
    - Exit cost reduces P&L at settlement
    """
    
    def __init__(self, costs: TransactionCosts = None):
        """
        Args:
            costs: TransactionCosts config (default: typical market)
        """
        if costs is None:
            costs = TransactionCosts(
                entry_spread_bps=15.0,
                exit_spread_bps=15.0,
                commission_bps=5.0,
                slippage_bps=10.0
            )
        
        self.costs = costs
        
        print(f"Transaction Cost Calculator initialized:")
        print(f"  - Entry spread: {self.costs.entry_spread_bps} bps")
        print(f"  - Exit spread: {self.costs.exit_spread_bps} bps")
        print(f"  - Commission: {self.costs.commission_bps} bps")
        print(f"  - Slippage: {self.costs.slippage_bps} bps")
        print(f"  - Total round-trip: {self.costs.total_round_trip_bps():.1f} bps")
    
    def compute_entry_cost(self, entry_price):
        """
        Compute cost applied at entry.
        
        For dispersion: negative entry price (selling upside)
        Cost makes entry worse (more negative)
        
        Args:
            entry_price: Market price at entry
            
        Returns:
            float: Cost amount (positive value to subtract)
        """
        cost_fraction = self.costs.entry_cost_bps() / 10000
        return abs(entry_price) * cost_fraction
    
    def compute_exit_cost(self, entry_price, exit_price):
        """
        Compute cost applied at exit.
        
        Args:
            entry_price: Original entry price
            exit_price: Price at exit (realized dispersion)
            
        Returns:
            float: Cost amount to reduce P&L
        """
        # Use entry price as reference
        cost_fraction = self.costs.exit_cost_bps() / 10000
        return abs(entry_price) * cost_fraction
    
    def adjust_entry_price(self, entry_price):
        """
        Adjust entry price for transaction costs.
        
        For dispersion (short upside):
        - Market price is negative (e.g., -70)
        - Adding cost makes it more negative (worse entry)
        
        Args:
            entry_price: Market price at entry
            
        Returns:
            float: Adjusted entry price (including costs)
        """
        entry_cost = self.compute_entry_cost(entry_price)
        
        if entry_price < 0:
            # Short position: cost makes it worse (more negative)
            return entry_price - entry_cost
        else:
            # Long position: cost makes it worse (less positive)
            return entry_price - entry_cost
    
    def adjust_pnl(self, entry_price, realized_price, pnl):
        """
        Adjust P&L for transaction costs.
        
        Args:
            entry_price: Original entry price
            realized_price: Realized price at exit
            pnl: Unadjusted P&L
            
        Returns:
            tuple: (adjusted_pnl, entry_cost, exit_cost)
        """
        entry_cost = self.compute_entry_cost(entry_price)
        exit_cost = self.compute_exit_cost(entry_price, realized_price)
        
        adjusted_pnl = pnl - entry_cost - exit_cost
        
        return adjusted_pnl, entry_cost, exit_cost
    
    def get_config(self):
        """Return configuration as dict."""
        return {
            'entry_spread_bps': self.costs.entry_spread_bps,
            'exit_spread_bps': self.costs.exit_spread_bps,
            'commission_bps': self.costs.commission_bps,
            'slippage_bps': self.costs.slippage_bps,
            'total_round_trip_bps': self.costs.total_round_trip_bps()
        }