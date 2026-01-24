import pandas as pd
import numpy as np
import torch
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import json
import logging

from src.backtest.enhanced_backtest import EnhancedBacktestEngine
from src.backtest.transaction_costs import TransactionCosts
from src.models.basket_pricing_net import BasketPricingNet
from src.config import DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WalkForwardValidator:
    
    def __init__(self, test_threshold: float = 2.0):
        
        self.test_threshold = test_threshold
        self.transaction_costs = TransactionCosts(
            entry_spread_bps=15.0,
            exit_spread_bps=15.0,
            commission_bps=5.0,
            slippage_bps=10.0
        )
        self.fold_results = []
        
        logger.info(f"WalkForwardValidator initialized (threshold={test_threshold})")
    
    def load_full_dataset(self) -> Dict:
    
        logger.info("Loading full dataset...")
        
        X_train = np.load(f"{DATA_DIR}/features/X_train.npy")
        X_val = np.load(f"{DATA_DIR}/features/X_val.npy")
        X_test = np.load(f"{DATA_DIR}/features/X_test.npy")
        
        y_train = np.load(f"{DATA_DIR}/labels/y_train.npy")
        y_val = np.load(f"{DATA_DIR}/labels/y_val.npy")
        y_test = np.load(f"{DATA_DIR}/labels/y_test.npy")
        
        dates_train = np.load(f"{DATA_DIR}/features/dates_train.npy", allow_pickle=True)
        dates_val = np.load(f"{DATA_DIR}/features/dates_val.npy", allow_pickle=True)
        dates_test = np.load(f"{DATA_DIR}/features/dates_test.npy", allow_pickle=True)
        
        with open(f"{DATA_DIR}/features/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        

        X_full = np.vstack([X_train, X_val, X_test])
        y_full = np.concatenate([y_train, y_val, y_test])
        dates_full = np.concatenate([dates_train, dates_val, dates_test])
        
        logger.info(f"Full dataset: X={X_full.shape}, y={y_full.shape}, dates={len(dates_full)}")
        logger.info(f"Date range: {pd.Timestamp(str(dates_full[0])).date()} to {pd.Timestamp(str(dates_full[-1])).date()}")
        
        return {
            'X_full': X_full,
            'y_full': y_full,
            'dates_full': dates_full,
            'scaler': scaler,
          
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'dates_train': dates_train,
            'dates_val': dates_val,
            'dates_test': dates_test
        }
    
    def get_folds(self, X_full: np.ndarray, dates_full: np.ndarray, n_folds: int = 3) -> List[Dict]:
        n_total = len(X_full)
        
        
        base_train_size = int(n_total * 0.65)  
        test_size = int(n_total * 0.12)  
        
        folds = []
        for fold_num in range(1, n_folds + 1):
            train_end = base_train_size + (fold_num - 1) * test_size
            test_end = min(train_end + test_size, n_total)
            
            if train_end >= n_total - test_size:
                logger.warning(f"Fold {fold_num}: insufficient data, skipping")
                continue
            
            fold = {
                'fold_num': fold_num,
                'train_start': 0,
                'train_end': train_end,
                'test_start': train_end,
                'test_end': test_end,
                'train_dates': (pd.Timestamp(str(dates_full[0])).date(), 
                               pd.Timestamp(str(dates_full[train_end])).date()),
                'test_dates': (pd.Timestamp(str(dates_full[train_end])).date(), 
                              pd.Timestamp(str(dates_full[test_end-1])).date()),
                'n_train': train_end,
                'n_test': test_end - train_end
            }
            folds.append(fold)
            
            logger.info(f"\nFold {fold_num}:")
            logger.info(f"  Train: {fold['train_dates'][0]} to {fold['train_dates'][1]} ({fold['n_train']} days)")
            logger.info(f"  Test:  {fold['test_dates'][0]} to {fold['test_dates'][1]} ({fold['n_test']} days)")
        
        return folds
    
    def train_model_on_fold(self, X_train: np.ndarray, y_train: np.ndarray, fold_num: int) -> BasketPricingNet:
        
        logger.info(f"Training model for fold {fold_num} ({len(X_train)} samples)...")
        
        from torch.utils.data import DataLoader, TensorDataset
        import torch.nn as nn
        
        model = BasketPricingNet(input_dim=274, hidden_dims=[256, 128, 64], dropout=0.2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.MSELoss()
        
        n_train = int(len(X_train) * 0.8)
        train_dataset = TensorDataset(
            torch.tensor(X_train[:n_train], dtype=torch.float32),
            torch.tensor(y_train[:n_train], dtype=torch.float32)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_train[n_train:], dtype=torch.float32),
            torch.tensor(y_train[n_train:], dtype=torch.float32)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        epochs_trained = 0
        
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
            
            if epoch % 50 == 0:
                logger.info(f"  Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    epochs_trained = epoch
                    logger.info(f"  Early stopping at epoch {epoch}")
                    break
            
            epochs_trained = epoch
        
        logger.info(f"Fold {fold_num} training complete ({epochs_trained} epochs)")
        return model
    
    def run_backtest_on_fold(self, model: BasketPricingNet, X_test: np.ndarray, 
                            y_test: np.ndarray, dates_test: np.ndarray, fold_num: int) -> Dict:
        logger.info(f"Running backtest for fold {fold_num} ({len(X_test)} samples)...")
        
        engine = EnhancedBacktestEngine(transaction_costs=self.transaction_costs)
        
        data = {
            'model': model,
            'X_test': X_test,
            'y_test': y_test,
            'dates_test': dates_test,
            'scaler': None
        }
        
        data = engine.generate_predictions(data)
        trades = engine.generate_signals(data, threshold=self.test_threshold)
        pnl_pre, positions, total_pnl_pre, mean_pnl = engine.compute_pnl(trades)
        
        data['trades'] = pd.DataFrame(trades)
        data['pnl_pre'] = pnl_pre
        data['positions'] = positions
        
        data = engine.apply_transaction_costs(data)
        data = engine.compute_metrics(data)
        
        metrics = data['metrics']
        
        logger.info(f"Fold {fold_num} results: Sharpe={metrics['sharpe_ratio']:.4f}, "
                   f"PnL={metrics['pnl_after_costs']:.2f}, Win%={metrics['win_rate']*100:.1f}%")
        
        return {
            'fold_num': fold_num,
            'metrics': metrics,
            'trades': data['trades']
        }
    
    def run_all_folds(self, dataset: Dict, n_folds: int = 3) -> List[Dict]:
        """Execute walk-forward validation across all folds."""
        logger.info(f"\n{'='*70}")
        logger.info("STARTING WALK-FORWARD VALIDATION")
        logger.info(f"{'='*70}\n")
        
        X_full = dataset['X_full']
        y_full = dataset['y_full']
        dates_full = dataset['dates_full']
        
        folds = self.get_folds(X_full, dates_full, n_folds=n_folds)
        
        fold_results = []
        for fold in folds:
            fold_num = fold['fold_num']
            
            
            X_train = X_full[fold['train_start']:fold['train_end']]
            y_train = y_full[fold['train_start']:fold['train_end']]
            
            X_test = X_full[fold['test_start']:fold['test_end']]
            y_test = y_full[fold['test_start']:fold['test_end']]
            dates_test = dates_full[fold['test_start']:fold['test_end']]
            
            
            model = self.train_model_on_fold(X_train, y_train, fold_num)
            
            
            result = self.run_backtest_on_fold(model, X_test, y_test, dates_test, fold_num)
            
            
            result['fold_info'] = fold
            fold_results.append(result)
        
        self.fold_results = fold_results
        return fold_results
    
    def summarize_results(self) -> pd.DataFrame:
        
        logger.info(f"\n{'='*70}")
        logger.info("WALK-FORWARD VALIDATION SUMMARY")
        logger.info(f"{'='*70}\n")
        
        summary = []
        for result in self.fold_results:
            fold_num = result['fold_num']
            metrics = result['metrics']
            fold_info = result['fold_info']
            
            summary.append({
                'Fold': fold_num,
                'Test Period': f"{fold_info['test_dates'][0]} to {fold_info['test_dates'][1]}",
                'N Test': fold_info['n_test'],
                'Sharpe': metrics['sharpe_ratio'],
                'Sortino': metrics['sortino_ratio'],
                'PnL Post-Costs': metrics['pnl_after_costs'],
                'Win Rate': metrics['win_rate'],
                'Max DD': metrics['max_drawdown'],
                'Num Trades': int(metrics['num_trades'])
            })
        
        summary_df = pd.DataFrame(summary)
        
        
        agg_row = {
            'Fold': 'MEAN',
            'Test Period': 'All folds',
            'N Test': summary_df['N Test'].sum(),
            'Sharpe': summary_df['Sharpe'].mean(),
            'Sortino': summary_df['Sortino'].mean(),
            'PnL Post-Costs': summary_df['PnL Post-Costs'].sum(),
            'Win Rate': summary_df['Win Rate'].mean(),
            'Max DD': summary_df['Max DD'].mean(),
            'Num Trades': summary_df['Num Trades'].sum()
        }
        summary_df = pd.concat([summary_df, pd.DataFrame([agg_row])], ignore_index=True)
        
        logger.info(summary_df.to_string(index=False))
        logger.info("")
        
        return summary_df
    
    def save_results(self, summary_df: pd.DataFrame, output_dir: str = 'output/backtest/'):
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        
        summary_csv = output_path / 'walk_forward_results.csv'
        summary_df.to_csv(summary_csv, index=False)
        logger.info(f"Saved summary: {summary_csv}")
        
        
        detailed = {
            'threshold': self.test_threshold,
            'n_folds': len(self.fold_results),
            'folds': []
        }
        
        for result in self.fold_results:
            fold_data = {
                'fold_num': result['fold_num'],
                'test_period': {
                    'start': str(result['fold_info']['test_dates'][0]),
                    'end': str(result['fold_info']['test_dates'][1])
                },
                'metrics': {k: float(v) if isinstance(v, (np.number, float)) else v 
                           for k, v in result['metrics'].items()}
            }
            detailed['folds'].append(fold_data)
        
        detailed_json = output_path / 'walk_forward_detailed.json'
        with open(detailed_json, 'w') as f:
            json.dump(detailed, f, indent=2)
        logger.info(f"Saved detailed results: {detailed_json}")
        
        return {
            'summary': str(summary_csv),
            'detailed': str(detailed_json)
        }
    
    def compare_to_original(self, original_results: Dict) -> pd.DataFrame:
        
        logger.info(f"\n{'='*70}")
        logger.info("COMPARISON: ORIGINAL vs WALK-FORWARD")
        logger.info(f"{'='*70}\n")
        
        
        wf_sharpe = np.mean([r['metrics']['sharpe_ratio'] for r in self.fold_results])
        wf_sharpe_std = np.std([r['metrics']['sharpe_ratio'] for r in self.fold_results])
        wf_win_rate = np.mean([r['metrics']['win_rate'] for r in self.fold_results])
        wf_pnl = np.sum([r['metrics']['pnl_after_costs'] for r in self.fold_results])
        
        comparison = pd.DataFrame({
            'Metric': ['Sharpe Ratio', 'Sharpe Std Dev', 'Win Rate', 'Total PnL', 'Num Folds'],
            'Original (Single Test)': [
                original_results['sharpe_ratio'],
                'N/A',
                original_results['win_rate'],
                original_results['pnl_after_costs'],
                1
            ],
            'Walk-Forward (Multiple Folds)': [
                f"{wf_sharpe:.4f}",
                f"{wf_sharpe_std:.4f}",
                f"{wf_win_rate:.2%}",
                f"{wf_pnl:.2f}",
                len(self.fold_results)
            ],
            'Interpretation': [
                f"{'Stable ✓' if wf_sharpe_std < 0.1 else 'Variable'}",
                f"{wf_sharpe_std:.4f}",
                "Consistent?",
                "Aggregate",
                "Robustness"
            ]
        })
        
        logger.info(comparison.to_string(index=False))
        logger.info("")
        
        return comparison


def main():
    
    import json
    
    
    original_metrics_path = Path('output/backtest/regime_aware_metrics.json')
    if not original_metrics_path.exists():
        logger.error(f"Original metrics not found at {original_metrics_path}")
        logger.error("Run: python -m src.backtest.enhanced_backtest first")
        return
    
    with open(original_metrics_path, 'r') as f:
        original_results = json.load(f)
    
    logger.info(f"Original backtest Sharpe: {original_results['sharpe_ratio']:.4f}")
    
    validator = WalkForwardValidator(test_threshold=2.0)
    dataset = validator.load_full_dataset()
    
    fold_results = validator.run_all_folds(dataset, n_folds=3)
    summary_df = validator.summarize_results()
    
    save_paths = validator.save_results(summary_df)
    logger.info(f"\nResults saved to: {save_paths}")
    
    comparison = validator.compare_to_original(original_results)
    comparison.to_csv('output/backtest/walk_forward_comparison.csv', index=False)
    
    logger.info("\n✓ Walk-forward validation complete")
    logger.info(f"✓ Summary: output/backtest/walk_forward_results.csv")
    logger.info(f"✓ Detailed: output/backtest/walk_forward_detailed.json")
    logger.info(f"✓ Comparison: output/backtest/walk_forward_comparison.csv")


if __name__ == "__main__":
    main()
