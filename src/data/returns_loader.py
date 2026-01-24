# src/data/returns_loader.py
import pandas as pd
import numpy as np
from pathlib import Path

class ReturnsDataLoader:
    """Load and query historical returns data from Phase 1 pipeline."""
    
    def __init__(self, returns_path="output/datasets/processed/returns.parquet"):
        """
        Load returns data generated in Phase 1.
        
        Args:
            returns_path: Path to returns parquet file
        """
        self.returns_df = pd.read_parquet(returns_path)
        self.returns_df.index = pd.to_datetime(self.returns_df.index)
        print(f"Loaded returns  {self.returns_df.shape[0]} dates, {self.returns_df.shape[1]} assets")
        
    def get_forward_returns(self, start_date, holding_days=20):
        """
        Get forward-looking returns over holding period.
        
        Args:
            start_date: Entry date (pd.Timestamp or string)
            holding_days: Number of trading days to hold (default 20 = 1 month)
            
        Returns:
            DataFrame: Daily returns for holding period, or None if insufficient data
        """
        start_date = pd.to_datetime(start_date)
        
        # Find start index
        try:
            start_idx = self.returns_df.index.get_loc(start_date)
        except KeyError:
            # Date not found, try nearest
            nearest_dates = self.returns_df.index[self.returns_df.index >= start_date]
            if len(nearest_dates) == 0:
                return None
            start_idx = self.returns_df.index.get_loc(nearest_dates[0])
        
        # Get forward window
        end_idx = min(start_idx + holding_days, len(self.returns_df))
        
        if end_idx - start_idx < holding_days:
            return None  # Insufficient forward data
        
        return self.returns_df.iloc[start_idx:end_idx]
    
    def get_all_dates(self):
        """Return all available dates."""
        return self.returns_df.index.tolist()
    
    def get_date_range(self):
        """Return (start_date, end_date) tuple."""
        return (self.returns_df.index[0], self.returns_df.index[-1])