import numpy as np
from dataclasses import dataclass

@dataclass
class TransactionCosts:
    
    entry_spread_bps: float = 15.0      
    exit_spread_bps: float = 15.0       
    commission_bps: float = 5.0         
    slippage_bps: float = 10.0          
    
    def total_round_trip_bps(self):
        
        entry_cost = self.entry_spread_bps / 2 + self.commission_bps + self.slippage_bps / 2
        exit_cost = self.exit_spread_bps / 2 + self.commission_bps + self.slippage_bps / 2
        return entry_cost + exit_cost
    
    def entry_cost_bps(self):
        
        return self.entry_spread_bps / 2 + self.commission_bps + self.slippage_bps / 2
    
    def exit_cost_bps(self):
        
        return self.exit_spread_bps / 2 + self.commission_bps + self.slippage_bps / 2

class TransactionCostCalculator:
    
    def __init__(self, costs: TransactionCosts = None):
        
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
        
        cost_fraction = self.costs.entry_cost_bps() / 10000
        return abs(entry_price) * cost_fraction
    
    def compute_exit_cost(self, entry_price, exit_price):
        
        cost_fraction = self.costs.exit_cost_bps() / 10000
        return abs(entry_price) * cost_fraction
    
    def adjust_entry_price(self, entry_price):
        
        entry_cost = self.compute_entry_cost(entry_price)
        
        if entry_price < 0:
            
            return entry_price - entry_cost
        else:
            
            return entry_price - entry_cost
    
    def adjust_pnl(self, entry_price, realized_price, pnl):
        
        entry_cost = self.compute_entry_cost(entry_price)
        exit_cost = self.compute_exit_cost(entry_price, realized_price)
        
        adjusted_pnl = pnl - entry_cost - exit_cost
        
        return adjusted_pnl, entry_cost, exit_cost
    
    def get_config(self):
        
        return {
            'entry_spread_bps': self.costs.entry_spread_bps,
            'exit_spread_bps': self.costs.exit_spread_bps,
            'commission_bps': self.costs.commission_bps,
            'slippage_bps': self.costs.slippage_bps,
            'total_round_trip_bps': self.costs.total_round_trip_bps()
        }
