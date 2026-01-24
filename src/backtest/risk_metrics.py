import numpy as np
import pandas as pd

class RiskMetricsCalculator:
    def __init__(self, risk_free_rate=0.04, periods_per_year=252):
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
    
    def compute_win_loss_ratio(self, pnl_series):
        pnl_series = np.array(pnl_series)
        wins = pnl_series[pnl_series > 0]
        losses = pnl_series[pnl_series < 0]
        num_wins = len(wins)
        num_losses = len(losses)
        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        win_rate = num_wins / len(pnl_series) if len(pnl_series) > 0 else 0
        win_loss_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else np.inf
        return {
            'num_wins': num_wins,
            'num_losses': num_losses,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_rate': win_rate,
            'win_loss_ratio': win_loss_ratio
        }
    
    def compute_profit_factor(self, pnl_series):
        pnl_series = np.array(pnl_series)
        total_wins = np.sum(pnl_series[pnl_series > 0])
        total_losses = abs(np.sum(pnl_series[pnl_series < 0]))
        return total_wins / total_losses if total_losses > 0 else np.inf
    
    def compute_max_drawdown(self, pnl_series):
        pnl_series = np.array(pnl_series)
        cumulative = np.cumsum(pnl_series)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / np.maximum(np.abs(running_max), 1e-8)
        return np.min(drawdown)
    
    def compute_sharpe_ratio(self, pnl_series, periods=1):
        pnl_series = np.array(pnl_series)
        mean_ret = np.mean(pnl_series)
        std_ret = np.std(pnl_series)
        return (mean_ret / std_ret * np.sqrt(periods)) if std_ret > 0 else 0
    
    def compute_sortino_ratio(self, pnl_series, periods=1):
        pnl_series = np.array(pnl_series)
        mean_ret = np.mean(pnl_series)
        downside = pnl_series[pnl_series < 0]
        downside_vol = np.std(downside)
        return (mean_ret / downside_vol * np.sqrt(periods)) if downside_vol > 0 else np.inf
    
    def compute_calmar_ratio(self, pnl_series, periods=1):
        annual_ret = np.mean(pnl_series) * periods
        max_dd = self.compute_max_drawdown(pnl_series)
        return annual_ret / abs(max_dd) if max_dd < 0 else np.inf
    
    def compute_all_metrics(self, pnl_series, is_trade_pnls=True):
        pnl_series = np.array(pnl_series)
        
        metrics = {
            'num_trades': len(pnl_series),
            'total_pnl': np.sum(pnl_series),
            'mean_pnl': np.mean(pnl_series),
            'median_pnl': np.median(pnl_series),
            'std_pnl': np.std(pnl_series),
            'min_pnl': np.min(pnl_series),
            'max_pnl': np.max(pnl_series),
        }
        
        wl = self.compute_win_loss_ratio(pnl_series)
        metrics.update(wl)
        metrics['profit_factor'] = self.compute_profit_factor(pnl_series)
        
        periods = 1 if is_trade_pnls else self.periods_per_year
        
        metrics['sharpe_ratio'] = self.compute_sharpe_ratio(pnl_series, periods)
        metrics['sortino_ratio'] = self.compute_sortino_ratio(pnl_series, periods)
        metrics['calmar_ratio'] = self.compute_calmar_ratio(pnl_series, periods)
        metrics['max_drawdown'] = self.compute_max_drawdown(pnl_series)
        
        return metrics
