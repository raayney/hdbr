import numpy as np

class SyntheticMarketPriceGenerator:
    
    
    def __init__(self, 
                 implied_realized_spread_pct=0.10,
                 bid_ask_bps=15,
                 seed=42):
        
        self.ir_spread = implied_realized_spread_pct
        self.bid_ask_bps = bid_ask_bps
        self.rng = np.random.RandomState(seed)
        
        print(f"Market Price Generator initialized:")
        print(f"  - Implied-Realized spread: {self.ir_spread*100:.1f}%")
        print(f"  - Bid-ask spread: {self.bid_ask_bps} bps")
        
    def generate_market_price(self, nn_fair_value, market_vol=20.0):
        
        spread_adjustment = abs(nn_fair_value) * self.ir_spread
        
        market_price_base = nn_fair_value - spread_adjustment
        
        bid_ask_noise = self.rng.uniform(
            -self.bid_ask_bps / 10000, 
            self.bid_ask_bps / 10000
        ) * abs(nn_fair_value)
        
        market_price = market_price_base + bid_ask_noise
        
        return market_price
    
    def generate_batch_prices(self, nn_predictions, market_vols=None):
        
        nn_predictions = np.array(nn_predictions)
        
        if market_vols is None:
            market_vols = np.ones_like(nn_predictions) * 20.0
        else:
            market_vols = np.array(market_vols)
        
        market_prices = np.array([
            self.generate_market_price(fv, vol)
            for fv, vol in zip(nn_predictions, market_vols)
        ])
        
        return market_prices
    
    def calibrate_from_historical(self, realized_vars, implied_vars):
        
        realized_vars = np.array(realized_vars)
        implied_vars = np.array(implied_vars)
        
        mask = np.isfinite(realized_vars) & np.isfinite(implied_vars) & (realized_vars > 0)
        realized_vars = realized_vars[mask]
        implied_vars = implied_vars[mask]
        
        #compute spread ratio
        spread_ratios = (implied_vars - realized_vars) / realized_vars
        
        self.ir_spread = np.median(spread_ratios)
        
        print(f"Calibrated IR spread: {self.ir_spread*100:.2f}%")
        print(f"  (Mean: {np.mean(spread_ratios)*100:.2f}%, Std: {np.std(spread_ratios)*100:.2f}%)")
        
        return self.ir_spread
    
    def get_config(self):
        """Return current configuration."""
        return {
            'implied_realized_spread_pct': self.ir_spread,
            'bid_ask_bps': self.bid_ask_bps
        }
