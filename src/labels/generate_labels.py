"""Generate Monte Carlo labels for all feature vectors."""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import logging

from src.config import DATA_DIR, N_MC_PATHS, MC_HORIZON_DAYS
from src.labels.monte_carlo_simulator import MonteCarloSimulator
from src.labels.dispersion_payoff import DispersionPayoffCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_sx5e_weights(weights_csv_path, date):
    """Load SX5E weights for a specific date from the monthly weights CSV."""
    weights_df = pd.read_csv(weights_csv_path, index_col=0, parse_dates=True)
    
    # Find closest month-end date <= date
    available_dates = weights_df.index
    date_ts = pd.Timestamp(str(date))  # Convert numpy.str_ to str first

def estimate_correlation_matrix(returns_df, date, window=60):
    """Historical correlation estimation"""
    try:
        if date not in returns_df.index:
            return np.eye(returns_df.shape[1])
        
        end_idx = returns_df.index.get_loc(date)
        start_idx = max(0, end_idx - window)
        hist_returns = returns_df.iloc[start_idx:end_idx]
        
        if len(hist_returns) < 20:
            return np.eye(returns_df.shape[1])
        
        corr_matrix = hist_returns.corr().values
        
        eigvals = np.linalg.eigvalsh(corr_matrix)
        if eigvals.min() < 1e-8:
            corr_matrix += np.eye(corr_matrix.shape[0]) * 1e-6
        
        return corr_matrix
    except:
        return np.eye(returns_df.shape[1])

def compute_historical_correlation(spot_prices, window=60):
    """
    Compute realized correlation from historical spot prices (60-day rolling).
    
    Args:
        spot_prices: Array shape (n_dates, 21)
        window: Lookback window in days
    
    Returns:
        21x21 correlation matrix (positive definite)
    """
    if len(spot_prices) < window + 1:
        return np.eye(21)
    
    # Log returns
    log_ret = np.diff(np.log(spot_prices), axis=0)
    
    # Last 60 days
    recent = log_ret[-window:]
    
    # Correlation
    corr = np.corrcoef(recent.T)
    
    # Make positive definite
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 1e-6)
    corr_adj = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    return corr_adj

def generate_labels(feature_dir, output_dir):
    """
    Generate MC labels for all dates.

    Args:
        feature_dir: Path to features (X_train.npy, etc.)
        output_dir: Path to save labels
    """
    logger.info("Loading features and metadata...")

    returns_df = pd.read_parquet(f"{feature_dir.replace('/features', '')}/processed/returns.parquet")
    if not isinstance(returns_df.index, pd.DatetimeIndex):
        returns_df.index = pd.to_datetime(returns_df.index)

    # Load features
    X_train = np.load(f"{feature_dir}/X_train.npy")
    X_val = np.load(f"{feature_dir}/X_val.npy")
    X_test = np.load(f"{feature_dir}/X_test.npy")

    dates_train = np.load(f"{feature_dir}/dates_train.npy", allow_pickle=True)
    dates_val = np.load(f"{feature_dir}/dates_val.npy", allow_pickle=True)
    dates_test = np.load(f"{feature_dir}/dates_test.npy", allow_pickle=True)

    # Load scaler to un-standardize features
    with open(f"{feature_dir}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Combine all datasets
    X_all = np.vstack([X_train, X_val, X_test])
    dates_all = np.concatenate([dates_train, dates_val, dates_test])

    logger.info(f"Generating labels for {len(X_all)} samples...")

    # Initialize simulator and payoff calculator
    simulator = MonteCarloSimulator(n_paths=N_MC_PATHS, n_steps=MC_HORIZON_DAYS)
    weights_csv = f"{DATA_DIR}/sx5e_weights_monthly.csv"

    # Inside the loop, before computing payoff:
    for i, (features, date) in enumerate(zip(X_all, dates_all)):
        # ... existing code ...
    
        # Load date-specific weights
        weights = load_sx5e_weights(weights_csv, date)
        payoff_calc = DispersionPayoffCalculator(weights=weights)
    
    labels = []

    for i, (features, date) in enumerate(zip(X_all, dates_all)):
        if (i + 1) % 200 == 0:
            logger.info(f"  {i+1}/{len(X_all)} samples processed...")

        # Un-standardize features
        # Manually un-standardize: x_raw = x_scaled * std + mean
        features_raw = features * scaler.std + scaler.mean

        # Extract components (21 spots, 21 forwards, 21 rates, 21 yields, 147 IVs)
        spots = features_raw[:21]
        forwards = features_raw[21:42]
        rates = features_raw[42:63]
        yields = features_raw[63:84]
        ivs = features_raw[84:231].reshape(21, 7)  # 21 assets × 7 strikes

        # Compute drifts (risk-neutral: mu = r - q)
        mus = rates - yields

        # Use ATM volatility (middle strike, index 3)
        sigmas = ivs[:, 3]

        # Correlation matrix
        # Correlation matrix - convert numpy.str_ to str first
        date_ts = pd.Timestamp(str(date))
        correlation_matrix = estimate_correlation_matrix(
            returns_df=returns_df,
            date=date_ts,
            window=60
        )

        # Simulate paths
        all_paths = simulator.simulate_basket(spots, mus, sigmas, correlation_matrix)

        # Compute expected payoff
        index_paths = all_paths[0]  # First asset is the index
        stock_paths = all_paths[1:]  # Remaining 20 are stocks

        expected_payoff = payoff_calc.expected_payoff(index_paths, stock_paths)

        labels.append(expected_payoff)

    labels = np.array(labels)

    # Split labels back into train/val/test
    n_train = len(X_train)
    n_val = len(X_val)

    y_train = labels[:n_train]
    y_val = labels[n_train:n_train + n_val]
    y_test = labels[n_train + n_val:]

    # Save labels
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(f"{output_dir}/y_train.npy", y_train)
    np.save(f"{output_dir}/y_val.npy", y_val)
    np.save(f"{output_dir}/y_test.npy", y_test)

    logger.info(f"✓ Labels saved to {output_dir}")
    logger.info(f"  y_train: {y_train.shape}")
    logger.info(f"  y_val:   {y_val.shape}")
    logger.info(f"  y_test:  {y_test.shape}")
    logger.info(f"  Mean label: {labels.mean():.4f}")
    logger.info(f"  Std label:  {labels.std():.4f}")


if __name__ == "__main__":
    generate_labels(
        feature_dir=f"{DATA_DIR}/features",
        output_dir=f"{DATA_DIR}/labels"
    )