"""Position sizing for basket dispersion trading backtest.

Implements Kelly Criterion and vol-targeting to scale trade sizes dynamically.
New file - doesn't modify existing backtest.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PositionSizer:
    """Add Kelly Criterion + vol-targeting to backtest trades."""
    
    def __init__(self, kelly_fraction=0.25, vol_target=0.15):
        """
        Args:
            kelly_fraction: Conservative Kelly (0.25 = 25% of Kelly optimal)
            vol_target: Target annual volatility (0.15 = 15%)
        """
        self.kelly_fraction = kelly_fraction
        self.vol_target = vol_target
        logger.info(f"PositionSizer: Kelly={kelly_fraction}, VolTarget={vol_target}")
    
    def kelly_position_size(self, edge: float, win_rate: float) -> float:
        """Kelly Criterion: position = (win_rate - loss_rate) / 1.
        
        Simplified: position = edge (how mispriced) * kelly_fraction
        """
        position = self.kelly_fraction * edge
        return np.clip(position, 0.1, 3.0)  # 10%-300% limits
    
    def vol_target_position_size(self, current_vol: float) -> float:
        """Vol-targeting: scale position inversely to volatility.
        
        If vol spikes to 30%, reduce position by half.
        """
        if current_vol == 0:
            return 1.0
        position = self.vol_target / current_vol
        return np.clip(position, 0.1, 2.0)  # 10%-200% limits
    
    def add_position_sizing(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Add position sizing columns to trades."""
        trades_df = trades_df.copy()
        
        # Compute signal strength (absolute value of mispricing)
        trades_df['signal_strength'] = np.abs(trades_df['signal'])
        
        # Rolling realized volatility (simplified: use std of last 20 trades)
        rolling_vol = trades_df['signal_strength'].rolling(20).std().fillna(0.20)
        trades_df['rolling_vol'] = rolling_vol
        
        # Kelly position sizing
        trades_df['position_kelly'] = trades_df.apply(
            lambda row: self.kelly_position_size(row['signal_strength'], 0.6),
            axis=1
        )
        
        # Vol-target position sizing
        trades_df['position_vol_target'] = trades_df.apply(
            lambda row: self.vol_target_position_size(row['rolling_vol']),
            axis=1
        )
        
        # Scaled P&L (example: use Kelly sizing)
        trades_df['pnl_kelly_scaled'] = trades_df['pnl_after_costs'] * trades_df['position_kelly']
        trades_df['pnl_vol_scaled'] = trades_df['pnl_after_costs'] * trades_df['position_vol_target']
        
        logger.info(f"Position sizing applied to {len(trades_df)} trades")
        logger.info(f"  Mean Kelly position: {trades_df['position_kelly'].mean():.2f}x")
        logger.info(f"  Mean Vol-target position: {trades_df['position_vol_target'].mean():.2f}x")
        logger.info(f"  Kelly-scaled total P&L: ${trades_df['pnl_kelly_scaled'].sum():.2f}")
        logger.info(f"  Vol-scaled total P&L: ${trades_df['pnl_vol_scaled'].sum():.2f}")
        
        return trades_df


def main():
    """Load backtest results, add position sizing, save."""
    trades_path = Path('output/backtest/regime_aware_trades.csv')
    
    if not trades_path.exists():
        logger.error(f"Trades not found: {trades_path}")
        return
    
    # Load trades
    trades_df = pd.read_csv(trades_path)
    
    # Add position sizing
    sizer = PositionSizer(kelly_fraction=0.25, vol_target=0.15)
    trades_sized = sizer.add_position_sizing(trades_df)
    
    # Save
    output_path = Path('output/backtest/regime_aware_trades_sized.csv')
    trades_sized.to_csv(output_path, index=False)
    logger.info(f"✓ Saved sized trades: {output_path}")
    
    # Summary comparison
    logger.info("\n" + "="*70)
    logger.info("POSITION SIZING COMPARISON")
    logger.info("="*70)
    
    comparison = pd.DataFrame({
        'Strategy': ['Fixed (1.0)', 'Kelly Scaled', 'Vol-Target Scaled'],
        'Total P&L': [
            f"${trades_df['pnl_after_costs'].sum():.2f}",
            f"${trades_sized['pnl_kelly_scaled'].sum():.2f}",
            f"${trades_sized['pnl_vol_scaled'].sum():.2f}"
        ],
        'Mean Position': [
            '1.00x',
            f"{trades_sized['position_kelly'].mean():.2f}x",
            f"{trades_sized['position_vol_target'].mean():.2f}x"
        ]
    })
    
    logger.info("\n" + comparison.to_string(index=False))


if __name__ == "__main__":
    main()