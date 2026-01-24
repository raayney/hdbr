import pandas as pd
import numpy as np
from pathlib import Path

class ReturnsDataLoader:
    
    def __init__(self, returns_path="output/datasets/processed/returns.parquet"):
        
        self.returns_df = pd.read_parquet(returns_path)
        self.returns_df.index = pd.to_datetime(self.returns_df.index)
        print(f"Loaded returns  {self.returns_df.shape[0]} dates, {self.returns_df.shape[1]} assets")
        
    def get_forward_returns(self, start_date, holding_days=20):
        
        start_date = pd.to_datetime(start_date)
        
        try:
            start_idx = self.returns_df.index.get_loc(start_date)
        except KeyError:
            
            nearest_dates = self.returns_df.index[self.returns_df.index >= start_date]
            if len(nearest_dates) == 0:
                return None
            start_idx = self.returns_df.index.get_loc(nearest_dates[0])
        
        
        end_idx = min(start_idx + holding_days, len(self.returns_df))
        
        if end_idx - start_idx < holding_days:
            return None  
        
        return self.returns_df.iloc[start_idx:end_idx]
    
    def get_all_dates(self):
        return self.returns_df.index.tolist()
    
    def get_date_range(self):
        return (self.returns_df.index[0], self.returns_df.index[-1])
