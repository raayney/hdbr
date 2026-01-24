# src/backtest/synthetic_market.py
import numpy as np

class SyntheticMarketPriceGenerator:
    """
    Generate realistic market prices for dispersion trades.
    
    Market microstructure modeling:
    1. Implied-realized spread: IV typically exceeds RV by 5-15%
    2. Bid-ask spread: 10-20 bps for liquid dispersion trades
    3. Index overpricing: Hedging demand inflates index volatility
    
    References:
    - Christensen et al. (1998): Implied vs realized volatility
    - BSIC (2020): Dispersion trading mechanics
    - Stoxx (2024): Dispersion index methodology
    """
    
    def __init__(self, 
                 implied_realized_spread_pct=0.10,
                 bid_ask_bps=15,
                 seed=42):
        """
        Args:
            implied_realized_spread_pct: Mean (IV - RV) / RV ratio (default 10%)
            bid_ask_bps: Bid-ask spread in basis points (default 15 bps)
            seed: Random seed for reproducibility
        """
        self.ir_spread = implied_realized_spread_pct
        self.bid_ask_bps = bid_ask_bps
        self.rng = np.random.RandomState(seed)
        
        print(f"Market Price Generator initialized:")
        print(f"  - Implied-Realized spread: {self.ir_spread*100:.1f}%")
        print(f"  - Bid-ask spread: {self.bid_ask_bps} bps")
        
    def generate_market_price(self, nn_fair_value, market_vol=20.0):
        """
        Generate synthetic market price from NN fair value.
        
        Formula:
            Market_price = NN_fair_value × (1 + IR_spread) + Bid_ask_noise
        
        Args:
            nn_fair_value: Neural network prediction of fair value
            market_vol: Current market volatility level (%) for scaling
            
        Returns:
            float: Synthetic market price (with spread and noise)
        """
        # Apply implied-realized spread
        # Market overprices dispersion → add spread adjustment
        spread_adjustment = abs(nn_fair_value) * self.ir_spread
        
        # Direction: If NN says dispersion is negative (typical), market makes it more negative
        # This means we can "buy cheap" when signal is strong
        market_price_base = nn_fair_value - spread_adjustment
        
        # Add bid-ask noise (transaction cost)
        bid_ask_noise = self.rng.uniform(
            -self.bid_ask_bps / 10000, 
            self.bid_ask_bps / 10000
        ) * abs(nn_fair_value)
        
        market_price = market_price_base + bid_ask_noise
        
        return market_price
    
    def generate_batch_prices(self, nn_predictions, market_vols=None):
        """
        Generate market prices for batch of NN predictions.
        
        Args:
            nn_predictions: Array of NN fair value predictions
            market_vols: Optional array of market volatility levels (for scaling)
            
        Returns:
            np.array: Synthetic market prices
        """
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
        """
        Calibrate implied-realized spread from historical data.
        
        Args:
            realized_vars: Array of historical realized variances
            implied_vars: Array of historical implied variances (from IV surfaces)
            
        Returns:
            float: Calibrated IR spread
        """
        realized_vars = np.array(realized_vars)
        implied_vars = np.array(implied_vars)
        
        # Remove NaN/inf
        mask = np.isfinite(realized_vars) & np.isfinite(implied_vars) & (realized_vars > 0)
        realized_vars = realized_vars[mask]
        implied_vars = implied_vars[mask]
        
        # Compute spread ratio
        spread_ratios = (implied_vars - realized_vars) / realized_vars
        
        # Use median for robustness
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