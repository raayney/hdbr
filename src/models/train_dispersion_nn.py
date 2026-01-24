import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import json
import logging

from src.config import DATA_DIR, N_MC_PATHS, MC_HORIZON_DAYS
from src.models.basket_pricing_net import BasketPricingNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_realized_volatility(returns_df, date, window=60):
    try:
        if date not in returns_df.index:
            return np.nan
        
        end_idx = returns_df.index.get_loc(date)
        start_idx = max(0, end_idx - window)
        hist_returns = returns_df.iloc[start_idx:end_idx]
        
        if len(hist_returns) < 20:
            return np.nan
        
        realized_vol = np.sqrt(252 * np.mean(hist_returns ** 2))
        return realized_vol
    except:
        return np.nan

def categorize_regime(realized_vol, vol_percentiles):
    if realized_vol < vol_percentiles[33]:
        return 'low'
    elif realized_vol < vol_percentiles[67]:
        return 'mid'
    else:
        return 'high'

def train_regime_aware(X_train, X_val, X_test, y_train, y_val, y_test, 
                       dates_train, dates_val, dates_test, returns_df):
    
    
    logger.info("Computing realized volatility for all dates...")
    
    all_dates = np.concatenate([dates_train, dates_val, dates_test])
    all_X = np.vstack([X_train, X_val, X_test])
    all_y = np.concatenate([y_train, y_val, y_test])
    
    vols = []
    for date in all_dates:
        vol = compute_realized_volatility(returns_df, pd.Timestamp(str(date)))
        vols.append(vol)
    
    vols = np.array(vols)
    
    valid_vols = vols[~np.isnan(vols)]
    vol_p33 = np.percentile(valid_vols, 33)
    vol_p67 = np.percentile(valid_vols, 67)
    
    logger.info(f"Volatility percentiles: 33%={vol_p33:.4f}, 67%={vol_p67:.4f}")
    
    regimes = []
    for i, vol in enumerate(vols):
        if np.isnan(vol):
            regime = 'mid'  
        elif vol < vol_p33:
            regime = 'low'
        elif vol < vol_p67:
            regime = 'mid'
        else:
            regime = 'high'
        regimes.append(regime)
    
    regimes = np.array(regimes)
    
    logger.info(f"Regime distribution: low={np.sum(regimes=='low')}, "
                f"mid={np.sum(regimes=='mid')}, high={np.sum(regimes=='high')}")
    
    logger.info("\nTraining single model on all regimes...")
    
    model = BasketPricingNet(input_dim=274, hidden_dims=[256, 128, 64], dropout=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    train_dataset = TensorDataset(
        torch.tensor(all_X[:len(X_train)], dtype=torch.float32),
        torch.tensor(all_y[:len(y_train)], dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(all_X[len(X_train):len(X_train)+len(X_val)], dtype=torch.float32),
        torch.tensor(all_y[len(y_train):len(y_train)+len(y_val)], dtype=torch.float32)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    
    logger.info("Starting training loop...")
    
    for epoch in range(1, 201):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_dataset)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        
        val_loss /= len(val_dataset)
        
        if (epoch % 20 == 0) or (epoch == 1):
            logger.info(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            Path(f"{DATA_DIR}/../models").mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), f"{DATA_DIR}/../models/best_model_regime_aware.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
    
    logger.info("\n" + "="*60)
    logger.info("TEST SET PERFORMANCE BY REGIME")
    logger.info("="*60)
    
    test_indices = len(X_train) + len(X_val)
    test_regimes = regimes[test_indices:]
    test_X = all_X[test_indices:]
    test_y = all_y[test_indices:]
    
    model.eval()
    with torch.no_grad():
        test_preds = model(torch.tensor(test_X, dtype=torch.float32)).numpy()
    
    for regime in ['low', 'mid', 'high']:
        mask = test_regimes == regime
        if np.sum(mask) > 0:
            regime_mse = np.mean((test_preds[mask] - test_y[mask]) ** 2)
            regime_mae = np.mean(np.abs(test_preds[mask] - test_y[mask]))
            regime_r2 = 1 - np.sum((test_preds[mask] - test_y[mask])**2) / np.sum((test_y[mask] - np.mean(test_y[mask]))**2)
            
            logger.info(f"\n{regime.upper()} VOLATILITY ({np.sum(mask)} samples):")
            logger.info(f"  MSE: {regime_mse:.4f}")
            logger.info(f"  MAE: {regime_mae:.4f}")
            logger.info(f"  R²:  {regime_r2:.4f}")
    
    logger.info("\n" + "="*60 + "\n")

if __name__ == "__main__":
    import pandas as pd
    
    logger.info("Loading data...")
    X_train = np.load(f"{DATA_DIR}/features/X_train.npy")
    X_val = np.load(f"{DATA_DIR}/features/X_val.npy")
    X_test = np.load(f"{DATA_DIR}/features/X_test.npy")
    
    y_train = np.load(f"{DATA_DIR}/labels/y_train.npy")
    y_val = np.load(f"{DATA_DIR}/labels/y_val.npy")
    y_test = np.load(f"{DATA_DIR}/labels/y_test.npy")
    
    dates_train = np.load(f"{DATA_DIR}/features/dates_train.npy", allow_pickle=True)
    dates_val = np.load(f"{DATA_DIR}/features/dates_val.npy", allow_pickle=True)
    dates_test = np.load(f"{DATA_DIR}/features/dates_test.npy", allow_pickle=True)
    
    returns_df = pd.read_parquet(f"{DATA_DIR}/processed/returns.parquet")
    if not isinstance(returns_df.index, pd.DatetimeIndex):
        returns_df.index = pd.to_datetime(returns_df.index)
    
    logger.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    logger.info(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    logger.info(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    train_regime_aware(X_train, X_val, X_test, y_train, y_val, y_test,
                       dates_train, dates_val, dates_test, returns_df)
