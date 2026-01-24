import sys
sys.path.insert(0, '/Users/rayney/Documents/basket-dispersion-project')

import yfinance as yf
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.config import TOP_20_CONSTITUENTS, SX5E_TICKER, BACKTEST_START, BACKTEST_END, DATA_DIR

TOP_20_WEIGHT_IN_INDEX = 0.70

def extract_close_price(data, ticker):

    if isinstance(data.columns, pd.MultiIndex):
        if ('Close', ticker) in data.columns:
            return data[('Close', ticker)]
        elif ('Adj Close', ticker) in data.columns:
            return data[('Adj Close', ticker)]
        else:
            price_cols = [col for col in data.columns if 'Close' in col[0]]
            if price_cols:
                return data[price_cols[0]]
                
    elif 'Adj Close' in data.columns:
        return data['Adj Close']
    elif 'Close' in data.columns:
        return data['Close']
    elif isinstance(data, pd.Series):
        return data
    
    return None

def main():
    print("\nDownloading TOP_20 constituents + index...")
    print(f"Index: {SX5E_TICKER}")
    print(f"Constituents ({len(TOP_20_CONSTITUENTS)}): {TOP_20_CONSTITUENTS}\n")
    
    all_prices = {}
    
    for ticker in [SX5E_TICKER] + TOP_20_CONSTITUENTS:
        try:
            data = yf.download(ticker, start=BACKTEST_START, end=BACKTEST_END, progress=False)
            
            if data is None or data.empty:
                print(f"  ✗ {ticker:15} (no data)")
                continue
            
            prices = extract_close_price(data, ticker)
            
            if prices is None:
                print(f"  ✗ {ticker:15} (could not extract prices)")
                continue
            
            if len(prices) > 100 and prices.notna().sum() > 100:
                all_prices[ticker] = prices
                print(f"  ✓ {ticker:15} ({len(prices)} days, {prices.notna().sum()} non-null)")
            else:
                print(f"  ✗ {ticker:15} (insufficient data)")
        
        except Exception as e:
            print(f"  ✗ {ticker:15} ({str(e)[:40]})")
    
    if not all_prices:
        print("\n❌ No tickers downloaded!")
        return
    
    df = pd.concat([s.to_frame(name=t) for t, s in all_prices.items()], axis=1)
    df = df.sort_index()
    print(f"\n✓ Merged: {df.shape[0]} trading days, {df.shape[1]} tickers\n")
    
    monthly = df.resample('ME').last()
    
    if SX5E_TICKER in monthly.columns:
        constituent_cols = [c for c in monthly.columns if c != SX5E_TICKER]
    else:
        constituent_cols = list(monthly.columns)
    
    print(f"Constituents: {len(constituent_cols)}\n")
    
    constituent_prices = monthly[constituent_cols].dropna(how='any')
    print(f"After removing NaN: {constituent_prices.shape[0]} dates\n")
    
    if constituent_prices.empty:
        print("❌ No data after removing NaN!")
        return
    
    constituent_total = constituent_prices.sum(axis=1)
    raw_weights = constituent_prices.div(constituent_total, axis=0)
    scaled_weights = raw_weights * TOP_20_WEIGHT_IN_INDEX
    
    output_dir = Path(DATA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_dir / 'sx5e_prices_daily.csv')
    scaled_weights.to_csv(output_dir / 'sx5e_weights_monthly.csv')
    
    print(f"✓ Saved:")
    print(f"  - {output_dir / 'sx5e_prices_daily.csv'} ({df.shape})")
    print(f"  - {output_dir / 'sx5e_weights_monthly.csv'} ({scaled_weights.shape})\n")
    
    latest = scaled_weights.iloc[-1].sort_values(ascending=False)
    print(f"Latest weights ({scaled_weights.index[-1].date()}):")
    print(latest)
    print(f"\nSum: {latest.sum():.4f}")

if __name__ == '__main__':
    main()
