import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Basket Dispersion Trading",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 16px;
    border-radius: 8px;
    margin: 6px 0;
}
.highlight-green {
    color: #09ab3b;
    font-weight: 600;
}
.highlight-red {
    color: #ff2b2b;
    font-weight: 600;
}
</style>
""",
    unsafe_allow_html=True,
)

# data loading

@st.cache_data
def load_backtest_data():
    data = {}
    base = Path("output/backtest")
    
    # Fix paths to match your exact filenames
    data['metrics'] = json.load((base / 'regime_aware_metrics.json').open())
    data['metrics_cleaned'] = json.load((base / 'regime_aware_metrics_cleaned.json').open())
    data['trades'] = pd.read_csv(base / 'regime_aware_trades.csv')
    data['walk_forward_detailed'] = json.load((base / 'walk_forward_detailed.json').open())
    data['walk_forward_results'] = pd.read_csv(base / 'walk_forward_results.csv')
    data['trades_sized'] = pd.read_csv(base / 'regime_aware_trades_sized.csv')
    data['threshold_grid'] = pd.read_csv(base / 'threshold_grid.csv')
    
    return data

data = load_backtest_data()

# sidebar

st.sidebar.title("dashboard")

view = st.sidebar.radio(
    "view",
    [
        "overview",
        "threshold grid",
        "walk-forward",
        "position sizing",
        "trade analysis",
    ],
)

threshold_display = st.sidebar.slider(
    "threshold (display only)",
    min_value=0.0,
    max_value=8.0,
    value=2.0,
    step=0.5,
)

# header

st.title("basket dispersion trading – backtest dashboard")
st.caption("neural pricing of high-dimensional variance dispersion on euro stoxx")

# overview

if view == "overview":
    st.header("portfolio overview")

    if "metrics" not in data:
        st.info("no metrics found under output/backtest.")
    else:
        m = data["metrics"]
        optimal = m["optimal"]
        grid_data = m["grid_search"]["results"]
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("sharpe", f"{optimal.get('sharpe_ratio', 0):.4f}")
        with c2:
            st.metric("total pnl", f"{optimal.get('pnl_after_costs', 0):.2f}")
        with c3:
            st.metric("win rate", f"{optimal.get('win_rate', 0):.2%}")
        with c4:
            dd_raw = optimal.get('max_drawdown', '0')
            dd = (float(str(dd_raw).replace('%', '').strip()))/100
            st.metric("max drawdown", f"{dd:.2%}")

        st.divider()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("sortino", f"{optimal.get('sortino_ratio', 0):.4f}")
        with c2:
            st.metric("profit factor", f"{optimal.get('profit_factor', 0):.2f}")
        with c3:
            st.metric("num trades", f"{optimal.get('num_trades', 0)}")
        with c4:
            st.metric("buys/sells", f"{optimal.get('num_buys', 0)}/{optimal.get('num_sells', 0)}")

        grid = pd.DataFrame(grid_data)
        st.subheader("threshold grid")
        grid_short = grid[["threshold", "sharpe_ratio", "win_rate", "pnl_after_costs"]].round(4)
        st.table(grid_short)

# threshold grid

elif view == "threshold grid":
    st.header("threshold grid search")

    if "threshold_grid" not in data:
        st.info("no threshold_grid.csv found.")
    else:
        grid = data["threshold_grid"].copy()

        if {"threshold", "sharpe_ratio"}.issubset(grid.columns):
            best_idx = grid["sharpe_ratio"].idxmax()
            best_thr = grid.loc[best_idx, "threshold"]
            best_sharpe = grid.loc[best_idx, "sharpe_ratio"]

            c1, c2 = st.columns(2)
            with c1:
                st.metric("best threshold", f"{best_thr:.2f}")
            with c2:
                st.metric("sharpe at best threshold", f"{best_sharpe:.4f}")

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=grid["threshold"],
                    y=grid["sharpe_ratio"],
                    mode="lines+markers",
                    name="sharpe",
                    line=dict(color="#09ab3b", width=2),
                    marker=dict(size=7),
                )
            )
            fig.add_vline(
                x=best_thr,
                line_dash="dash",
                line_color="red",
            )
            fig.update_layout(
                title="sharpe vs threshold",
                xaxis_title="threshold",
                yaxis_title="sharpe",
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("grid data")
        st.dataframe(grid, use_container_width=True)

# walk-forward

elif view == "walk-forward":
    st.header("walk-forward validation")

    if "walk_forward_detailed" not in data:
        st.info("no walkforwarddetailed.json found.")
    else:
        wf = data["walk_forward_detailed"]

        folds = []
        for fold in wf.get("folds", []):
            met = fold.get("metrics", {})
            folds.append(
                {
                    "fold": fold.get("fold_num", None),
                    "sharpe": met.get("sharpe_ratio", 0.0),
                    "sortino": met.get("sortino_ratio", 0.0),
                    "win_rate": met.get("win_rate", 0.0),
                    "pnl": met.get("pnl_after_costs", 0.0),
                    "max_dd": met.get("max_drawdown", 0.0),
                }
            )

        if not folds:
            st.info("no fold-level metrics inside walkforwarddetailed.json.")
        else:
            folds_df = pd.DataFrame(folds)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(
                    "mean sharpe",
                    f"{folds_df['sharpe'].mean():.4f}",
                    f"±{folds_df['sharpe'].std():.4f}",
                )
            with c2:
                st.metric("mean win rate", f"{folds_df['win_rate'].mean():.2%}")
            with c3:
                st.metric("total pnl", f"{folds_df['pnl'].sum():.2f}")

            fig = px.bar(
                folds_df,
                x="fold",
                y="sharpe",
                title="fold sharpe",
            )
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("fold metrics")
            st.dataframe(folds_df, use_container_width=True)

    if "walk_forward_results" in data:
        st.subheader("summary (fixed vs kelly)")
        res = data["walk_forward_results"].copy()
        st.dataframe(res, use_container_width=True)

# position sizing

elif view == "position sizing":
    st.header("position sizing")

    st.markdown(
        """
kelly sizing:  
for a trade with edge \(e\) and variance σ², the full kelly fraction is
f* = µ / σ². to avoid leverage, we cap sizes [0.1,3.0]
"""
    )

    if "trades_sized" in data:
        trades = data["trades_sized"].copy()
    elif "trades" in data:
        trades = data["trades"].copy()
    else:
        trades = None

    if trades is None:
        st.info("no trade data found.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            kelly_frac = st.slider("kelly fraction", 0.1, 0.5, 0.25, 0.05)
        with c2:
            cap = st.slider("max position", 1.0, 5.0, 3.0, 0.5)

        if "signal" in trades.columns:
            trades["signal_strength"] = trades["signal"].abs()
        else:
            trades["signal_strength"] = 1.0

        trades["position_kelly"] = kelly_frac * trades["signal_strength"]
        trades["position_kelly"] = trades["position_kelly"].clip(0.1, cap)

        if "pnl_after_costs" in trades.columns:
            trades["pnl_kelly"] = trades["pnl_after_costs"] * trades["position_kelly"]
            fixed_pnl = trades["pnl_after_costs"].sum()
            kelly_pnl = trades["pnl_kelly"].sum()

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("fixed-size pnl", f"{fixed_pnl:.2f}")
            with c2:
                st.metric("kelly pnl", f"{kelly_pnl:.2f}")
            with c3:
                diff = kelly_pnl - fixed_pnl
                pct = diff / fixed_pnl * 100 if fixed_pnl != 0 else 0.0
                st.metric("improvement", f"{diff:.2f}", f"{pct:+.1f}%")

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    y=trades["pnl_kelly"].cumsum(),
                    mode="lines",
                    name="kelly cumsum",
                    line=dict(color="#09ab3b"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    y=trades["pnl_after_costs"].cumsum(),
                    mode="lines",
                    name="fixed cumsum",
                    line=dict(color="#1f77b4"),
                )
            )
            fig.update_layout(
                title="cumulative pnl (fixed vs kelly)",
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)

# trade analysis

elif view == "trade analysis":
    st.header("trade analysis")

    if "trades" not in data:
        st.info("no regimeawaretrades.csv found.")
    else:
        trades = data["trades"].copy()
        if "date" in trades.columns:
            trades["date"] = pd.to_datetime(trades["date"])

        wins = trades["pnl_after_costs"] > 0
        losses = trades["pnl_after_costs"] < 0

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("winning trades", int(wins.sum()))
        with c2:
            st.metric("losing trades", int(losses.sum()))
        with c3:
            st.metric(
                "avg win",
                f"{trades.loc[wins, 'pnl_after_costs'].mean():.2f}"
                if wins.any()
                else "n/a",
            )
        with c4:
            st.metric(
                "avg loss",
                f"{trades.loc[losses, 'pnl_after_costs'].mean():.2f}"
                if losses.any()
                else "n/a",
            )

        fig = px.histogram(
            trades,
            x="pnl_after_costs",
            nbins=50,
            title="pnl distribution",
            labels={"pnl_after_costs": "pnl"},
            color_discrete_sequence=["#09ab3b"],
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("top 10 winners")
        cols = [c for c in ["date", "action", "signal", "pnl_after_costs"] if c in trades.columns]
        st.dataframe(
            trades.sort_values("pnl_after_costs", ascending=False)[cols].head(10),
            use_container_width=True,
        )

        st.subheader("top 10 losers")
        st.dataframe(
            trades.sort_values("pnl_after_costs", ascending=True)[cols].head(10),
            use_container_width=True,
        )

# footer

st.divider()
st.markdown(
    """
<div style="text-align: center; color: #999; font-size: 11px;">
High-Dimensional Basket Dispersion Trading via Model-Free Deep Learning
</div>
""",
    unsafe_allow_html=True,
)