import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import logging

from src.config import (
    SX5E_TICKER, TOP_20_CONSTITUENTS,
    BACKTEST_START, BACKTEST_END,
    TRAIN_END, VAL_START, VAL_END,
    MONEYNESS_GRID, REALIZED_VOL_WINDOW,
    SKEW_PARAM_ALPHA, SMILE_PARAM_BETA,
    EURIBOR_3M, DIV_YIELD_INDEX, DIV_YIELD_STOCK,
    FEATURE_DIM, SCALER_EPSILON,
    DATA_DIR
)
from src.data.download_yahoo import YahooDataDownloader
from src.data.preprocess_prices import PricePreprocessor
from src.pricing.synthetic_surfaces import SyntheticIVSurfaceGenerator
from src.features.build_features import FeatureBuilder, FeatureScaler

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPipeline:
    
    
    def __init__(self):
        self.tickers = [SX5E_TICKER] + TOP_20_CONSTITUENTS
        self.raw_prices = {}
        self.processed_prices = None
        self.returns = None
        self.rolling_vol = None
        self.iv_surfaces = {}
        self.features = None
        self.dates = []
        self.scaler = None
    
    def run(self):
        
        print("\n" + "="*70)
        print("STEP 1: DATA PIPELINE")
        print("="*70 + "\n")
        
        try:
            
            print("[1/6] Downloading prices from Yahoo Finance...")
            self._download_prices()
            
            
            print("\n[2/6] Preprocessing & computing statistics...")
            self._preprocess()
            
            
            print("\n[3/6] Generating synthetic IV surfaces...")
            self._generate_iv_surfaces()
            
            
            print("\n[4/6] Building feature vectors...")
            self._build_features()
            
            
            print("\n[5/6] Standardizing and splitting data...")
            self._standardize_and_save()
            
            
            print("\n[6/6] Validating outputs...")
            self._validate()
            
            print("\n" + "="*70)
            print("✓ STEP 1 COMPLETE!")
            print("="*70 + "\n")
        
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
    
    def _download_prices(self):
        
        downloader = YahooDataDownloader(
            tickers=self.tickers,
            start_date=BACKTEST_START,
            end_date=BACKTEST_END,
            output_dir=f"{DATA_DIR}/raw_prices"
        )
        self.raw_prices = downloader.download()
        downloader.save()

    
        successful_tickers = sorted(list(self.raw_prices.keys()))
        print(f"\nSuccessfully downloaded {len(successful_tickers)} tickers:")
        for t in successful_tickers:
            print(f"  - {t}")
        self.tickers = successful_tickers

    
    def _preprocess(self):
        
        preprocessor = PricePreprocessor(self.raw_prices)
        self.processed_prices = preprocessor.align_and_merge()
        self.returns = preprocessor.compute_log_returns(self.processed_prices)
        
        rolling_var = preprocessor.compute_rolling_variance(self.returns, window=REALIZED_VOL_WINDOW)
        self.rolling_vol = preprocessor.compute_rolling_vol(rolling_var, annualize=True)
        
        logger.info("Computing regime features...")
        self.realized_correlation = self.preprocessor.compute_realized_correlation_scalar(
            self.returns, window=60
        )
        self.realized_skewness = self.preprocessor.compute_realized_skewness_scalar(
            self.returns, window=60
        )
        self.volatility_percentile = self.preprocessor.compute_volatility_percentile(
            self.rolling_vol, window=252
        )

        
        regime_dir = Path(DATADIR) / "processed"
        regime_dir.mkdir(parents=True, exist_ok=True)
        self.realized_correlation.to_csv(regime_dir / "realized_correlation.csv")
        self.realized_skewness.to_csv(regime_dir / "realized_skewness.csv")
        self.volatility_percentile.to_csv(regime_dir / "volatility_percentile.csv")

        correlations = preprocessor.compute_correlation_matrix(self.returns, window=60)
        
        
        Path(f"{DATA_DIR}/processed").mkdir(parents=True, exist_ok=True)
        self.processed_prices.to_parquet(f"{DATA_DIR}/processed/prices.parquet")
        self.returns.to_parquet(f"{DATA_DIR}/processed/returns.parquet")
        self.rolling_vol.to_parquet(f"{DATA_DIR}/processed/rolling_vol.parquet")
    
    def _generate_iv_surfaces(self):
        
        generator = SyntheticIVSurfaceGenerator(
            moneyness_grid=MONEYNESS_GRID,
            alpha=SKEW_PARAM_ALPHA,
            beta=SMILE_PARAM_BETA
        )
        
        for i, date in enumerate(self.rolling_vol.index):
            self.iv_surfaces[date] = {}
            for ticker in self.tickers:
                atm_vol = self.rolling_vol.loc[date, ticker]
                if pd.isna(atm_vol):
                    atm_vol = 0.15
                atm_vol_adj = generator.add_seasonal_variation(atm_vol, date)
                iv_curve = generator.generate_iv_curve(atm_vol_adj)
                self.iv_surfaces[date][ticker] = iv_curve
            
            if (i + 1) % 500 == 0:
                logger.info(f"  {i+1} dates processed...")
    
    def _build_features(self):
        
        builder = FeatureBuilder(
            tickers=self.tickers,
            moneyness_grid=MONEYNESS_GRID,
            feature_dim=FEATURE_DIM
        )
        
        features_list = []
        dates_list = []
        
        for i, date in enumerate(self.processed_prices.index):
            
            if date in self.realized_correlation.index:
                builder.realized_correlation = self.realized_correlation[date]
                builder.realized_skewness = self.realized_skewness[date]
                builder.volatility_percentile = self.volatility_percentile[date]
            else:
                
                builder.realized_correlation = 0.0
                builder.realized_skewness = 0.0
                builder.volatility_percentile = 50.0

            
            X = builder.build_feature_vector(pricesdict, ivsurfacesdict, ratesdict, yieldsdict)

            features_list.append(X)
            dates_list.append(date)
            
            if (i + 1) % 500 == 0:
                logger.info(f"  {i+1} dates processed...")
        
        self.features = np.array(features_list, dtype=np.float32)
        self.dates = dates_list
        logger.info(f"Built {len(self.features)} feature vectors")
    
    def _standardize_and_save(self):
        
        dates_pd = pd.to_datetime(self.dates)
        
        
        train_mask = dates_pd <= pd.to_datetime(TRAIN_END)
        val_mask = (dates_pd > pd.to_datetime(TRAIN_END)) & (dates_pd <= pd.to_datetime(VAL_END))
        test_mask = dates_pd > pd.to_datetime(VAL_END)
        
        X_train = self.features[train_mask]
        X_val = self.features[val_mask]
        X_test = self.features[test_mask]
        
        
        self.scaler = FeatureScaler(epsilon=SCALER_EPSILON)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        
        Path(f"{DATA_DIR}/features").mkdir(parents=True, exist_ok=True)
        np.save(f"{DATADIR}/features/X_train_regime.npy", Xtrainscaled)
        np.save(f"{DATADIR}/features/X_val_regime.npy", Xvalscaled)
        np.save(f"{DATADIR}/features/X_test_regime.npy", Xtestscaled)
        
        
        np.save(f"{DATA_DIR}/features/dates_train.npy", 
               np.array([str(d.date()) for d in dates_pd[train_mask]]))
        np.save(f"{DATA_DIR}/features/dates_val.npy", 
               np.array([str(d.date()) for d in dates_pd[val_mask]]))
        np.save(f"{DATA_DIR}/features/dates_test.npy", 
               np.array([str(d.date()) for d in dates_pd[test_mask]]))
        
        
        with open(f"{DATA_DIR}/features/scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, open(f"{DATADIR}/features/scaler_regime.pkl", "wb"))
        
        print("\nDataset shapes:")
        print(f"  Training:   {X_train_scaled.shape}")
        print(f"  Validation: {X_val_scaled.shape}")
        print(f"  Test:       {X_test_scaled.shape}")
    
    def _validate(self):
        
        X_train = np.load(f"{DATA_DIR}/features/X_train.npy")
        X_val = np.load(f"{DATA_DIR}/features/X_val.npy")
        X_test = np.load(f"{DATA_DIR}/features/X_test.npy")

        
        assert X_train.shape[1] == FEATURE_DIM, "Train feature dim mismatch"
        assert X_val.shape[1] == FEATURE_DIM, "Val feature dim mismatch"
        assert X_test.shape[1] == FEATURE_DIM, "Test feature dim mismatch"

        
        max_abs_mean = np.abs(X_train.mean(axis=0)).max()
        max_std_err = np.abs(X_train.std(axis=0) - 1.0).max()
        print(f"Max abs mean after scaling: {max_abs_mean:.4f}")
        print(f"Max std dev error after scaling: {max_std_err:.4f}")

        if max_abs_mean >= 0.05:
            print(f"WARNING: Train mean not perfectly centered (max |mean| = {max_abs_mean:.4f})")
        if max_std_err >= 0.05:
            print(f"WARNING: Train std not perfectly 1 (max |std-1| = {max_std_err:.4f})")

        
        assert not np.isnan(X_train).any(), "NaNs in train"
        assert not np.isnan(X_val).any(), "NaNs in val"
        assert not np.isnan(X_test).any(), "NaNs in test"

        print("✓ All validation checks passed (with soft warnings if any).")


if __name__ == "__main__":
    pipeline = DataPipeline()
    pipeline.run()
