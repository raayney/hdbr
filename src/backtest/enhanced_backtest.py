import pandas as pd
import numpy as np
import torch
import pickle
from pathlib import Path
from typing import Dict, List
import json
import logging

from src.backtest.transaction_costs import TransactionCostCalculator, TransactionCosts
from src.backtest.risk_metrics import RiskMetricsCalculator
from src.models.basket_pricing_net import BasketPricingNet
from src.config import DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedBacktestEngine:
    
    def __init__(self, 
                 transaction_costs: TransactionCosts = None,
                 risk_free_rate: float = 0.04):
        if transaction_costs is None:
            transaction_costs = TransactionCosts(
                entry_spread_bps=15.0,
                exit_spread_bps=15.0,
                commission_bps=5.0,
                slippage_bps=10.0
            )
        
        self.tc_calc = TransactionCostCalculator(transaction_costs)
        self.risk_calc = RiskMetricsCalculator(risk_free_rate)
        
        logger.info("Enhanced Backtest Engine initialized")
    
    def load_model_and_data(self, model_path: str, features_dir: str):
        logger.info(f"Loading model from {model_path}...")
        
        model = BasketPricingNet(input_dim=274, hidden_dims=[256, 128, 64], dropout=0.2)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        X_test = np.load(f"{features_dir}/X_test.npy")
        y_test = np.load(f"{features_dir.replace('features', 'labels')}/y_test.npy")
        dates_test = np.load(f"{features_dir}/dates_test.npy", allow_pickle=True)
        
        with open(f"{features_dir}/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        
        logger.info(f"Loaded {len(X_test)} test samples")
        logger.info(f"Date range: {pd.Timestamp(str(dates_test[0])).date()} to {pd.Timestamp(str(dates_test[-1])).date()}")
        
        return {
            'model': model,
            'X_test': X_test,
            'y_test': y_test,
            'dates_test': dates_test,
            'scaler': scaler
        }
    
    def generate_predictions(self, data: Dict):
        logger.info("Generating NN predictions...")

        model = data['model']
        X_test = torch.FloatTensor(data['X_test']).to(next(model.parameters()).device)

        with torch.no_grad():
            predictions = model(X_test).cpu().numpy().flatten()

        logger.info(f"Predictions - mean: {predictions.mean():.4f}, std: {predictions.std():.4f}")
        logger.info(f"Predictions - range: [{predictions.min():.4f}, {predictions.max():.4f}]")

        data['predictions'] = predictions
        return data
    
    def generate_signals(self, data: Dict, threshold: float = 0.0):
        
        logger.info(f"Generating signals (threshold={threshold})...")

        predictions = data['predictions']
        labels = data['y_test']  #true fair value
        dates = data['dates_test']

        signals = labels - predictions

        trades = []
        buys, sells = 0, 0

        for i, (date, signal, pred, label) in enumerate(zip(dates, signals, predictions, labels)):
            action = 'HOLD'
            if signal > threshold:
                action = 'BUY'
                buys += 1
            elif signal < -threshold:
                action = 'SELL'  
                sells += 1

            trades.append({
                'date': date,
                'signal': float(signal),
                'prediction': float(pred),
                'label': float(label),
                'action': action
            })

        logger.info(f"Generated {len(trades)} signals ({buys} buys, {sells} sells)")
        return trades
    
    def compute_pnl(self, trades: List[Dict]):
    
        logger.info("Computing P&L...")

        pnl_pre = []
        positions = []

        for trade in trades:
            action = trade['action']
            label = trade['label']
            prediction = trade['prediction']

            if action == 'BUY':
                position = 1.0
                pnl = label - prediction
            elif action == 'SELL':
                position = -1.0
                pnl = -(label - prediction)
            else:
                position = 0.0
                pnl = 0.0

            pnl_pre.append(pnl)
            positions.append(position)

        total_pnl_pre = np.sum(pnl_pre)
        mean_pnl = np.mean(pnl_pre)

        logger.info(f"Total P&L (pre-costs): {total_pnl_pre:.2f}")
        logger.info(f"Mean P&L per trade: {mean_pnl:.4f}")

        return pnl_pre, positions, total_pnl_pre, mean_pnl
    
    def apply_transaction_costs(self, data: Dict):
        logger.info("Applying transaction costs...")

        trades_df = data['trades'].copy()

        market_noise = np.random.normal(0, 3.0, len(trades_df))
        trades_df['label'] += market_noise
        trades_df['pnl'] = np.where(
            trades_df['action'] == 'BUY', trades_df['label'] - trades_df['prediction'],
            np.where(trades_df['action'] == 'SELL', trades_df['prediction'] - trades_df['label'], 0)
        )

        entry_costs = []
        exit_costs = []
        adjusted_pnls = []

        for idx, row in trades_df.iterrows():
            if row['action'] == 'HOLD':
                cost_amount = 0
            else:
                entry_price = abs(row['prediction'])
                total_cost_bps = (
                    self.tc_calc.costs.entry_spread_bps +
                    self.tc_calc.costs.exit_spread_bps +
                    self.tc_calc.costs.commission_bps +
                    self.tc_calc.costs.slippage_bps
                )
                total_cost_pct = total_cost_bps / 10000
                cost_amount = entry_price * total_cost_pct

            entry_costs.append(cost_amount / 2)
            exit_costs.append(cost_amount / 2)
            adjusted_pnl = row['pnl'] - cost_amount
            adjusted_pnls.append(adjusted_pnl)

        trades_df['entry_cost'] = entry_costs
        trades_df['exit_cost'] = exit_costs  
        trades_df['pnl_after_costs'] = adjusted_pnls

        total_costs = sum(entry_costs) + sum(exit_costs)
        logger.info(f"Transaction costs applied: {total_costs:.2f}")
        logger.info(f"Total P&L (post-costs): {trades_df['pnl_after_costs'].sum():.2f}")

        data['trades'] = trades_df
        return data
    
    def compute_metrics(self, data: Dict):
        logger.info("Computing risk metrics...")
        trades_df = data['trades']
        pnl_series = trades_df['pnl_after_costs'].values
    
        metrics = self.risk_calc.compute_all_metrics(pnl_series, is_trade_pnls=True)
    
        metrics['num_buys'] = len(trades_df[trades_df['action'] == 'BUY'])
        metrics['num_sells'] = len(trades_df[trades_df['action'] == 'SELL'])
        metrics['pnl_before_costs'] = trades_df['pnl'].sum()
        metrics['pnl_after_costs'] = trades_df['pnl_after_costs'].sum()
        metrics['total_costs'] = trades_df['entry_cost'].sum() + trades_df['exit_cost'].sum()
    
        data['metrics'] = metrics
        return data

    
    def print_results(self, data: Dict):
        metrics = data['metrics']
        
        logger.info("\n" + "="*70)
        logger.info("REGIME-AWARE MODEL BACKTEST RESULTS")
        logger.info("="*70)
        
        logger.info("\nTrade Summary:")
        logger.info(f"  Total trades: {int(metrics['num_trades'])}")
        logger.info(f"  Buys: {int(metrics['num_buys'])}, Sells: {int(metrics['num_sells'])}")
        
        logger.info("\nP&L Summary:")
        logger.info(f"  P&L (pre-costs): {metrics['pnl_before_costs']:,.2f}")
        logger.info(f"  P&L (post-costs): {metrics['pnl_after_costs']:,.2f}")
        logger.info(f"  Transaction costs: {metrics['total_costs']:,.2f}")
        logger.info(f"  Mean P&L per trade: {metrics['mean_pnl']:.2f}")
        
        logger.info("\nRisk Metrics:")
        logger.info(f"  Win rate: {metrics['win_rate']*100:.1f}%")
        logger.info(f"  Profit factor: {metrics['profit_factor']:.2f}")
        logger.info(f"  Max drawdown: {metrics['max_drawdown']*100:.2f}%")

        
        logger.info("\nRisk-Adjusted Returns:")
        logger.info(f"  Sharpe ratio: {metrics['sharpe_ratio']:.4f}")
        logger.info(f"  Sortino ratio: {metrics['sortino_ratio']:.4f}")
        logger.info(f"  Calmar ratio: {metrics['calmar_ratio']:.4f}")
        
        logger.info("\n" + "="*70)
    
    def save_results(self, data: Dict, output_dir: str = 'output/backtest/', threshold: float = None):
        """Save backtest results to CSV (only trades are threshold-specific)."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        trades_csv = output_path / f'regime_aware_trades_threshold_{threshold:.1f}.csv'
        data['trades'].to_csv(trades_csv, index=False)
        logger.info(f"Saved trades: {trades_csv}")

        return data['metrics']
    
    def run(self, model_path: str = 'output/models/best_model_regime_aware.pt',
        threshold: float = 0.0,
        output_dir: str = 'output/backtest/'):
        """Run complete backtest pipeline."""
        data = self.load_model_and_data(model_path, f"{DATA_DIR}/features")
        data = self.generate_predictions(data)
        trades = self.generate_signals(data, threshold=threshold) 
        pnl_pre, positions, total_pnl_pre, mean_pnl = self.compute_pnl(trades)  
    
        data = {
            'trades': pd.DataFrame(trades),  
            'pnl_pre': pnl_pre,
            'positions': positions
        }
    
        data = self.apply_transaction_costs(data)
        data = self.compute_metrics(data)
        self.print_results(data)
        metrics = self.save_results(data, output_dir, threshold=threshold)
        return metrics

    def run_grid_search(self, thresholds: List[float],
                        model_path: str = 'output/models/best_model_regime_aware.pt',
                        output_dir: str = 'output/backtest/'):
        logger.info(f"\n{'='*70}")
        logger.info(f"RUNNING GRID SEARCH: {len(thresholds)} thresholds")
        logger.info(f"{'='*70}\n")

        all_results = []

        for threshold in thresholds:
            logger.info(f"\n{'='*50}")
            logger.info(f"THRESHOLD: {threshold}")
            logger.info(f"{'='*50}")

            metrics = self.run(
                model_path=model_path,
                threshold=threshold,
                output_dir=output_dir
            )

            metrics['threshold'] = threshold
            all_results.append(metrics)

        #highest sharpe ratio
        optimal_idx = np.argmax([r['sharpe_ratio'] for r in all_results])
        optimal_result = all_results[optimal_idx]

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        metrics_json = output_path / 'regime_aware_metrics.json'

        # np -> py
        results_serializable = []
        for result in all_results:
            clean_result = {}
            for k, v in result.items():
                if isinstance(v, (np.integer, np.floating)):
                    clean_result[k] = float(v)
                elif np.isinf(v):
                    clean_result[k] = None  #json doesn't support inf
                else:
                    clean_result[k] = v
            results_serializable.append(clean_result)

        output_dict = {
            "grid_search": {
                "thresholds": [float(t) for t in thresholds],
                "results": results_serializable
            },
            "optimal": {
                "threshold": float(optimal_result['threshold']),
                "num_trades": int(optimal_result['num_trades']),
                "win_rate": float(optimal_result['win_rate']),
                "sharpe_ratio": float(optimal_result['sharpe_ratio']),
                "sortino_ratio": float(optimal_result['sortino_ratio']),
                "max_drawdown": float(optimal_result['max_drawdown']),
                "pnl_before_costs": float(optimal_result['pnl_before_costs']),
                "pnl_after_costs": float(optimal_result['pnl_after_costs']),
                "total_costs": float(optimal_result['total_costs']),
                "num_buys": int(optimal_result['num_buys']),
                "num_sells": int(optimal_result['num_sells']),
                "profit_factor": float(optimal_result['profit_factor']) if not np.isinf(optimal_result['profit_factor']) else None
            }
        }

        with open(metrics_json, 'w') as f:
            json.dump(output_dict, f, indent=2)

        logger.info(f"\n{'='*70}")
        logger.info(f"GRID SEARCH COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Optimal threshold: {optimal_result['threshold']}")
        logger.info(f"Optimal Sharpe: {optimal_result['sharpe_ratio']:.4f}")
        logger.info(f"Optimal P&L (post-costs): {optimal_result['pnl_after_costs']:,.2f}")
        logger.info(f"Saved all results to: {metrics_json}\n")

        return output_dict

if __name__ == "__main__":
    engine = EnhancedBacktestEngine()
    
    thresholds = [0.0, 2.0, 4.0, 6.0, 8.0]
    
    results = engine.run_grid_search(
        thresholds=thresholds,
        model_path='output/models/best_model_regime_aware.pt',
        output_dir='output/backtest/'
    )

    summary_data = results['grid_search']['results']
    summary_df = pd.DataFrame([{
        'threshold': r['threshold'],
        'num_trades': r['num_trades'],
        'win_rate': r['win_rate'],
        'sharpe': r['sharpe_ratio'],
        'pnl_post': r['pnl_after_costs']
    } for r in summary_data])

    print("\nSUMMARY TABLE:")
    print(summary_df.round(4))
    summary_df.to_csv('output/backtest/threshold_grid.csv', index=False)