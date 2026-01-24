import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json

st.set_page_config(
    page_title="Basket Dispersion Trading",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .highlight-green {
        color: #09ab3b;
        font-weight: bold;
    }
    .highlight-red {
        color: #ff2b2b;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD DATA
# ============================================================================

@st.cache_data
def load_backtest_data():
    """Load all backtest results from files."""
    data = {}
    
    # Load metrics
    metrics_path = Path('output/backtest/regime_aware_metrics.json')
    if metrics_path.exists():
        with open(metrics_path) as f:
            data['metrics'] = json.load(f)
    
    # Load trades
    trades_path = Path('output/backtest/regime_aware_trades.csv')
    if trades_path.exists():
        data['trades'] = pd.read_csv(trades_path)
    
    # Load walk-forward results
    wf_path = Path('output/backtest/walk_forward_detailed.json')
    if wf_path.exists():
        with open(wf_path) as f:
            data['walk_forward'] = json.load(f)
    
    # Load threshold grid (if exists from previous grid search)
    grid_path = Path('output/backtest/threshold_grid.csv')
    if grid_path.exists():
        data['threshold_grid'] = pd.read_csv(grid_path)
    
    # Load cleaned metrics
    cleaned_path = Path('output/backtest/regime_aware_metrics_cleaned.json')
    if cleaned_path.exists():
        with open(cleaned_path) as f:
            data['metrics_cleaned'] = json.load(f)
    
    return data


data = load_backtest_data()

# ============================================================================
# SIDEBAR - Navigation & Controls
# ============================================================================

st.sidebar.title("🎛️ Dashboard Controls")

view = st.sidebar.radio(
    "Select View:",
    ["📈 Overview", "🎯 Threshold Optimization", "🔄 Walk-Forward Validation", 
     "💰 Position Sizing", "📊 Trade Analysis"]
)

threshold_display = st.sidebar.slider(
    "Current Threshold",
    min_value=0.0,
    max_value=8.0,
    value=2.0,
    step=0.5,
    help="Threshold for signal generation"
)

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.title("🚀 Basket Dispersion Trading - Backtest Dashboard")
st.markdown("Deep Learning Model-Free Pricing | Martingale Optimal Transport")

# ============================================================================
if view == "📈 Overview":
    st.header("Portfolio Overview")
    
    if 'metrics' in data:
        metrics = data['metrics']
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Sharpe Ratio",
                f"{metrics.get('sharpe_ratio', 0):.4f}",
                help="Risk-adjusted return"
            )
        
        with col2:
            st.metric(
                "Total P&L",
                f"${metrics.get('pnl_after_costs', 0):.2f}",
                help="Post-transaction costs"
            )
        
        with col3:
            st.metric(
                "Win Rate",
                f"{metrics.get('win_rate', 0):.2%}",
                help="% of profitable trades"
            )
        
        with col4:
            st.metric(
                "Max Drawdown",
                f"{metrics.get('max_drawdown', 0):.2%}",
                help="Worst peak-to-trough decline"
            )
        
        # More metrics
        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.4f}")
        with col2:
            st.metric("Calmar Ratio", f"{metrics.get('calmar_ratio', 0):.4f}")
        with col3:
            st.metric("Num Trades", f"{int(metrics.get('num_trades', 0))}")
        with col4:
            st.metric("Mean P&L/Trade", f"${metrics.get('mean_pnl', 0):.4f}")
        
        # Comparison
        if 'metrics_cleaned' in data:
            st.divider()
            st.subheader("Cleaned Results (HOLD-free)")
            
            metrics_clean = data['metrics_cleaned']
            comparison = pd.DataFrame({
                'Metric': ['Sharpe', 'Win Rate', 'P&L', 'Num Trades'],
                'Original': [
                    f"{metrics.get('sharpe_ratio', 0):.4f}",
                    f"{metrics.get('win_rate', 0):.2%}",
                    f"${metrics.get('pnl_after_costs', 0):.2f}",
                    f"{int(metrics.get('num_trades', 0))}"
                ],
                'Cleaned': [
                    f"{metrics_clean.get('sharpe_ratio', 0):.4f}",
                    f"{metrics_clean.get('win_rate', 0):.2%}",
                    f"${metrics_clean.get('pnl_after_costs', 0):.2f}",
                    f"{int(metrics_clean.get('num_trades', 0))}"
                ]
            })
            st.table(comparison)

# ============================================================================
elif view == "🎯 Threshold Optimization":
    st.header("Threshold Grid Search")
    
    if 'threshold_grid' in data:
        grid = data['threshold_grid'].copy()
        
        # Plot: Sharpe vs Threshold
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=grid['threshold'],
            y=grid['sharpe_ratio'],
            mode='lines+markers',
            name='Sharpe Ratio',
            line=dict(color='#09ab3b', width=3),
            marker=dict(size=10)
        ))
        
        # Highlight best threshold
        best_idx = grid['sharpe_ratio'].idxmax()
        best_threshold = grid.loc[best_idx, 'threshold']
        best_sharpe = grid.loc[best_idx, 'sharpe_ratio']
        
        fig.add_vline(
            x=best_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Best: {best_threshold:.1f}",
            annotation_position="top right"
        )
        
        fig.update_layout(
            title="Sharpe Ratio vs Threshold",
            xaxis_title="Threshold",
            yaxis_title="Sharpe Ratio",
            template="plotly_white",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Table
        st.subheader("Grid Results")
        st.dataframe(grid, use_container_width=True)
    else:
        st.info("No threshold grid data available. Run threshold optimization first.")

# ============================================================================
elif view == "🔄 Walk-Forward Validation":
    st.header("Walk-Forward Cross-Validation")
    
    if 'walk_forward' in data:
        wf = data['walk_forward']
        
        st.markdown(f"**Threshold:** {wf.get('threshold', 'N/A')}")
        st.markdown(f"**Number of Folds:** {wf.get('n_folds', 'N/A')}")
        
        # Extract fold results
        folds_list = []
        for fold in wf.get('folds', []):
            metrics = fold.get('metrics', {})
            folds_list.append({
                'Fold': fold['fold_num'],
                'Sharpe': metrics.get('sharpe_ratio', 0),
                'Sortino': metrics.get('sortino_ratio', 0),
                'Win Rate': metrics.get('win_rate', 0),
                'P&L': metrics.get('pnl_after_costs', 0),
                'Max DD': metrics.get('max_drawdown', 0)
            })
        
        folds_df = pd.DataFrame(folds_list)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            mean_sharpe = folds_df['Sharpe'].mean()
            std_sharpe = folds_df['Sharpe'].std()
            st.metric("Mean Sharpe", f"{mean_sharpe:.4f}", f"±{std_sharpe:.4f}")
        
        with col2:
            mean_wr = folds_df['Win Rate'].mean()
            st.metric("Mean Win Rate", f"{mean_wr:.2%}")
        
        with col3:
            total_pnl = folds_df['P&L'].sum()
            st.metric("Total P&L", f"${total_pnl:.2f}")
        
        # Plot fold results
        fig = px.bar(
            folds_df,
            x='Fold',
            y=['Sharpe', 'Win Rate'],
            barmode='group',
            title="Walk-Forward Fold Comparison",
            labels={'value': 'Value', 'variable': 'Metric'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Table
        st.subheader("Detailed Fold Results")
        st.dataframe(folds_df, use_container_width=True)
    else:
        st.info("No walk-forward validation data available.")

# ============================================================================
elif view == "💰 Position Sizing":
    st.header("Position Sizing Analysis")
    
    st.markdown("""
    **Kelly Criterion:** Dynamic position sizing based on edge and volatility
    
    **Vol-Targeting:** Scale positions inversely to realized volatility
    """)
    
    # Kelly parameters
    col1, col2 = st.columns(2)
    with col1:
        kelly_frac = st.slider("Kelly Fraction", 0.1, 1.0, 0.25, 0.05)
    with col2:
        vol_target = st.slider("Vol Target", 0.05, 0.30, 0.15, 0.01)
    
    if 'trades' in data:
        trades = data['trades'].copy()
        
        # Simulate position sizing
        trades['signal_strength'] = np.abs(trades['signal'])
        trades['position_kelly'] = kelly_frac * trades['signal_strength']
        trades['position_kelly'] = trades['position_kelly'].clip(0.1, 3.0)
        
        trades['pnl_kelly'] = trades['pnl_after_costs'] * trades['position_kelly']
        
        # Comparison
        col1, col2, col3 = st.columns(3)
        
        with col1:
            orig_pnl = trades['pnl_after_costs'].sum()
            st.metric("Fixed Size P&L", f"${orig_pnl:.2f}")
        
        with col2:
            kelly_pnl = trades['pnl_kelly'].sum()
            st.metric("Kelly-Scaled P&L", f"${kelly_pnl:.2f}")
        
        with col3:
            improvement = (kelly_pnl - orig_pnl) / orig_pnl * 100 if orig_pnl != 0 else 0
            st.metric("Improvement", f"{improvement:+.1f}%")
        
        # Visualize position sizing
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=trades['position_kelly'].cumsum(),
            mode='lines',
            name='Cumulative Kelly Position',
            line=dict(color='#09ab3b')
        ))
        fig.update_layout(
            title="Position Size Over Time",
            yaxis_title="Cumulative Position",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
elif view == "📊 Trade Analysis":
    st.header("Trade Analysis")
    
    if 'trades' in data:
        trades = data['trades'].copy()
        
        # Trade statistics
        col1, col2, col3, col4 = st.columns(4)
        
        wins = (trades['pnl_after_costs'] > 0).sum()
        losses = (trades['pnl_after_costs'] < 0).sum()
        
        with col1:
            st.metric("Winning Trades", wins)
        with col2:
            st.metric("Losing Trades", losses)
        with col3:
            avg_win = trades[trades['pnl_after_costs'] > 0]['pnl_after_costs'].mean()
            st.metric("Avg Win", f"${avg_win:.2f}")
        with col4:
            avg_loss = trades[trades['pnl_after_costs'] < 0]['pnl_after_costs'].mean()
            st.metric("Avg Loss", f"${avg_loss:.2f}")
        
        # P&L distribution
        fig = px.histogram(
            trades,
            x='pnl_after_costs',
            nbins=50,
            title="P&L Distribution",
            labels={'pnl_after_costs': 'P&L', 'count': 'Frequency'},
            color_discrete_sequence=['#09ab3b']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top trades
        st.subheader("Top 10 Winning Trades")
        top_wins = trades.nlargest(10, 'pnl_after_costs')[['date', 'action', 'signal', 'pnl_after_costs']]
        st.dataframe(top_wins, use_container_width=True)
        
        st.subheader("Top 10 Losing Trades")
        top_losses = trades.nsmallest(10, 'pnl_after_costs')[['date', 'action', 'signal', 'pnl_after_costs']]
        st.dataframe(top_losses, use_container_width=True)

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
<div style="text-align: center; color: #999; font-size: 12px;">
    Hugh-Dimensional Basket Dispersion Trading via Model-Free Deep Learning
    | Powered by Streamlit
</div>
""", unsafe_allow_html=True)