"""Download historical OHLCV data from Yahoo Finance."""
import os
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YahooDataDownloader:
    """Download and cache Yahoo Finance data."""
    
    def __init__(self, tickers, start_date, end_date, output_dir):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

        # Robust directory creation
        self.output_dir = Path(output_dir)
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except FileExistsError:
            # If it already exists as a directory, ignore
            if not self.output_dir.is_dir():
                raise

        self.data = {}
    
    def download(self):
        """Download all tickers."""
        logger.info(f"Downloading {len(self.tickers)} tickers...")
        logger.info(f"Date range: {self.start_date} to {self.end_date}")
        
        for i, ticker in enumerate(self.tickers):
            try:
                logger.info(f"[{i+1}/{len(self.tickers)}] {ticker}")
                df = yf.download(
                    ticker,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False,   # keep or remove as you like
                    auto_adjust=False # explicit, safe default
                )

                
                if df.shape[0] > 0:
                    # Standardize columns
                    df.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                    df.index.name = 'Date'
                    self.data[ticker] = df
                    logger.info(f"  ✓ {len(df)} rows")
                else:
                    logger.warning(f"  ⚠ No data returned")
            
            except Exception as e:
                logger.error(f"  ✗ Failed: {e}")
        
        logger.info(f"Downloaded {len(self.data)}/{len(self.tickers)} successfully")
        return self.data
    
    def save(self, data=None):
        """Save to parquet files."""
        if data is None:
            data = self.data
        
        for ticker, df in data.items():
            safe_name = ticker.replace(".", "_").replace("^", "").replace("/", "_")
            path = self.output_dir / f"{safe_name}.parquet"
            df.to_parquet(path)
            logger.info(f"Saved {ticker} → {path}")
    
    def load_all(self):
        """Load all saved parquet files."""
        logger.info(f"Loading from {self.output_dir}...")
        for f in self.output_dir.glob("*.parquet"):
            try:
                ticker_name = f.stem.replace("_", ".").replace("STOXX50E", "^STOXX50E")
                self.data[ticker_name] = pd.read_parquet(f)
            except Exception as e:
                logger.error(f"Failed to load {f}: {e}")
        return self.data
    
    def align_dates(self):
        """Align all datasets to common trading dates."""
        logger.info("Aligning dates...")
        dfs = [df[['Adj Close']].rename(columns={'Adj Close': ticker}) 
               for ticker, df in self.data.items()]
        merged = pd.concat(dfs, axis=1, join='inner').dropna()
        logger.info(f"Aligned to {len(merged)} common trading dates")
        return merged