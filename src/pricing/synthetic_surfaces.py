"""Generate realistic synthetic IV surfaces."""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class SyntheticIVSurfaceGenerator:
    """Generate IV curves using quadratic smile model."""
    
    def __init__(self, moneyness_grid, alpha=-0.15, beta=0.05):
        self.moneyness_grid = moneyness_grid
        self.alpha = alpha
        self.beta = beta
    
    def generate_iv_curve(self, atm_vol):
        """
        Generate IV curve for given ATM vol.
        
        Formula: σ(k) = σ_ATM + α·k + β·k²
        where k = log(strike / forward)
        """
        curve = {}
        for k in self.moneyness_grid:
            iv = atm_vol + self.alpha * k + self.beta * k**2
            # Floor at 0.1%, cap at 200%
            iv = np.clip(iv, 0.001, 2.00)
            curve[k] = iv
        return curve
    
    def add_seasonal_variation(self, atm_vol, date, amplitude=0.02):
        """Add seasonal variation: σ_ATM × (1 + amp × sin(day))."""
        day_of_year = date.timetuple().tm_yday
        seasonal = 1 + amplitude * np.sin(2 * np.pi * day_of_year / 365)
        return atm_vol * seasonal