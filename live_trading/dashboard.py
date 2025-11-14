"""
Live Trading Dashboard
Real-time visualization of bot performance, signals, and trades
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dashboard_data import DashboardDataManager
from datetime import datetime, timedelta
import time

# Page config
st.set_page_config(
    page_title="Bitcoin Live Trading Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .positive {
        color: #00ff00;
    }
    .negative {
        color: #ff0000;
    }
</style>
""", unsafe_allow_html=True)

# Initialize data manager
@st.cache_resource
def get_data_manager():
    return DashboardDataManager()

data_manager = get_data_manager()

# ==================== Header ====================

st.title("🤖 Bitcoin Live Trading Dashboard")
st.markdown("---")

# ==================== Sidebar ====================

st.sidebar.title("⚙️ Dashboard Settings")

# Auto-refresh
auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 1, 60, 5)

# Time range filter
time_range = st.sidebar.selectbox(
    "Time Range",
    ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last Week", "All Time"]
)

# Data export
if st.sidebar.button("📥 Export Data"):
    data_manager.export_data("exports")
    st.sidebar.success("Data exported to exports/")

# Clear data (dangerous!)
if st.sidebar.button("🗑️ Clear All Data", type="secondary"):
    if st.sidebar.checkbox("Are you sure?"):
        data_manager.clear_all_data()
        st.sidebar.warning("All data cleared!")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dashboard Info")
st.sidebar.info(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==================== Main Dashboard ====================

# Get current data
status = data_manager.get_status()
performance = data_manager.get_performance()
trades_df = data_manager.get_trades_df()
signals_df = data_manager.get_signals_df()

# Apply time filter
def filter_by_time(df, time_range):
    if df.empty or 'timestamp' not in df.columns:
        return df

    now = datetime.now()
    if time_range == "Last Hour":
        cutoff = now - timedelta(hours=1)
    elif time_range == "Last 6 Hours":
        cutoff = now - timedelta(hours=6)
    elif time_range == "Last 24 Hours":
        cutoff = now - timedelta(days=1)
    elif time_range == "Last Week":
        cutoff = now - timedelta(days=7)
    else:  # All Time
        return df

    return df[df['timestamp'] >= cutoff]

trades_df = filter_by_time(trades_df, time_range)
signals_df = filter_by_time(signals_df, time_range)

# ==================== Bot Status ====================

st.header("🤖 Bot Status")

col1, col2, col3, col4 = st.columns(4)

with col1:
    status_emoji = "🟢" if status.get('running', False) else "🔴"
    st.metric("Status", f"{status_emoji} {'Running' if status.get('running', False) else 'Stopped'}")

with col2:
    current_price = status.get('current_price', 0)
    st.metric("Current Price", f"${current_price:,.2f}" if current_price else "N/A")

with col3:
    current_regime = status.get('current_regime', 'Unknown')
    regime_color = {
        'Bull Market': '🟢',
        'Sideways': '🟡',
        'Bear Market': '🔴',
        'High Volatility': '🟠'
    }.get(current_regime, '⚪')
    st.metric("Market Regime", f"{regime_color} {current_regime}")

with col4:
    last_signal = status.get('last_signal', 'HOLD')
    signal_emoji = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '⚪'}.get(last_signal, '⚪')
    confidence = status.get('last_confidence', 0)
    st.metric("Last Signal", f"{signal_emoji} {last_signal}", f"{confidence:.0%} confidence")

# ==================== Performance Metrics ====================

st.markdown("---")
st.header("📈 Performance Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_pnl = performance.get('total_pnl', 0)
    pnl_color = "normal" if total_pnl >= 0 else "inverse"
    st.metric("Total PnL", f"${total_pnl:,.2f}", delta=None, delta_color=pnl_color)

with col2:
    win_rate = performance.get('win_rate', 0)
    st.metric("Win Rate", f"{win_rate:.1f}%")

with col3:
    total_trades = performance.get('total_trades', 0)
    winning = performance.get('winning_trades', 0)
    losing = performance.get('losing_trades', 0)
    st.metric("Total Trades", f"{total_trades}", f"✅ {winning} | ❌ {losing}")

with col4:
    sharpe = performance.get('sharpe_ratio', 0)
    st.metric("Sharpe Ratio", f"{sharpe:.2f}")

with col5:
    max_dd = performance.get('max_drawdown', 0)
    st.metric("Max Drawdown", f"${max_dd:,.2f}")

# ==================== Open Position ====================

open_position = status.get('open_position')
if open_position:
    st.markdown("---")
    st.header("📍 Open Position")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Side", open_position.get('side', 'N/A'))

    with col2:
        entry = open_position.get('entry_price', 0)
        st.metric("Entry Price", f"${entry:,.2f}")

    with col3:
        quantity = open_position.get('quantity', 0)
        st.metric("Quantity", f"{quantity:.6f} BTC")

    with col4:
        unrealized_pnl = open_position.get('unrealized_pnl', 0)
        pnl_color = "normal" if unrealized_pnl >= 0 else "inverse"
        st.metric("Unrealized PnL", f"${unrealized_pnl:,.2f}", delta_color=pnl_color)

# ==================== Charts ====================

st.markdown("---")
st.header("📊 Charts")

# Tab for different charts
tab1, tab2, tab3, tab4 = st.tabs(["💰 PnL Chart", "📈 Signals", "🎯 Win Rate", "📊 Trade Distribution"])

with tab1:
    st.subheader("Cumulative PnL Over Time")

    if not trades_df.empty and 'type' in trades_df.columns:
        closed_trades = trades_df[trades_df['type'] == 'CLOSE'].copy()

        if not closed_trades.empty:
            closed_trades = closed_trades.sort_values('timestamp')
            closed_trades['cumulative_pnl'] = closed_trades['pnl'].cumsum()

            fig = go.Figure()

            # PnL line
            fig.add_trace(go.Scatter(
                x=closed_trades['timestamp'],
                y=closed_trades['cumulative_pnl'],
                mode='lines+markers',
                name='Cumulative PnL',
                line=dict(color='#00ff00', width=2),
                fill='tozeroy'
            ))

            # Zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray")

            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="PnL (USDT)",
                hovermode='x unified',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No closed trades yet")
    else:
        st.info("No trade data available")

with tab2:
    st.subheader("Price & Signals")

    if not signals_df.empty:
        # Create price chart with signal markers
        fig = go.Figure()

        # Price line
        fig.add_trace(go.Scatter(
            x=signals_df['timestamp'],
            y=signals_df['price'],
            mode='lines',
            name='Price',
            line=dict(color='blue', width=2)
        ))

        # Buy signals
        buy_signals = signals_df[signals_df['signal'] == 1]
        if not buy_signals.empty:
            fig.add_trace(go.Scatter(
                x=buy_signals['timestamp'],
                y=buy_signals['price'],
                mode='markers',
                name='BUY',
                marker=dict(color='green', size=15, symbol='triangle-up')
            ))

        # Sell signals
        sell_signals = signals_df[signals_df['signal'] == -1]
        if not sell_signals.empty:
            fig.add_trace(go.Scatter(
                x=sell_signals['timestamp'],
                y=sell_signals['price'],
                mode='markers',
                name='SELL',
                marker=dict(color='red', size=15, symbol='triangle-down')
            ))

        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Price (USDT)",
            hovermode='x unified',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Confidence over time
        st.subheader("Signal Confidence Over Time")

        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(
            x=signals_df['timestamp'],
            y=signals_df['confidence'],
            mode='lines',
            name='Confidence',
            line=dict(color='orange', width=2),
            fill='tozeroy'
        ))

        fig2.update_layout(
            xaxis_title="Time",
            yaxis_title="Confidence",
            yaxis_tickformat='.0%',
            hovermode='x unified',
            height=300
        ))

        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No signal data available")

with tab3:
    st.subheader("Win Rate by Regime")

    if not trades_df.empty and 'type' in trades_df.columns:
        closed_trades = trades_df[trades_df['type'] == 'CLOSE'].copy()

        if not closed_trades.empty and 'regime' in closed_trades.columns:
            # Calculate win rate by regime
            regime_stats = closed_trades.groupby('regime').agg({
                'pnl': ['count', lambda x: (x > 0).sum(), 'sum']
            }).round(2)

            regime_stats.columns = ['Total', 'Wins', 'Total PnL']
            regime_stats['Win Rate'] = (regime_stats['Wins'] / regime_stats['Total'] * 100).round(1)

            # Bar chart
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=regime_stats.index,
                y=regime_stats['Win Rate'],
                text=regime_stats['Win Rate'].apply(lambda x: f"{x:.1f}%"),
                textposition='auto',
                marker_color=['#00ff00' if x >= 50 else '#ff0000' for x in regime_stats['Win Rate']]
            ))

            fig.update_layout(
                xaxis_title="Market Regime",
                yaxis_title="Win Rate (%)",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Show table
            st.dataframe(regime_stats, use_container_width=True)
        else:
            st.info("Not enough trade data")
    else:
        st.info("No trade data available")

with tab4:
    st.subheader("PnL Distribution")

    if not trades_df.empty and 'type' in trades_df.columns:
        closed_trades = trades_df[trades_df['type'] == 'CLOSE'].copy()

        if not closed_trades.empty:
            # Histogram of PnL
            fig = go.Figure()

            fig.add_trace(go.Histogram(
                x=closed_trades['pnl'],
                nbinsx=30,
                name='PnL Distribution',
                marker_color='lightblue'
            ))

            fig.add_vline(x=0, line_dash="dash", line_color="red")

            fig.update_layout(
                xaxis_title="PnL (USDT)",
                yaxis_title="Frequency",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No closed trades yet")
    else:
        st.info("No trade data available")

# ==================== Recent Trades ====================

st.markdown("---")
st.header("📋 Recent Trades")

if not trades_df.empty:
    # Show last 20 trades
    recent_trades = trades_df.tail(20).sort_values('timestamp', ascending=False)

    # Format for display
    display_df = recent_trades.copy()

    if 'timestamp' in display_df.columns:
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Color code PnL
    def highlight_pnl(row):
        if 'pnl' in row and pd.notna(row['pnl']):
            if row['pnl'] > 0:
                return ['background-color: #90EE90'] * len(row)
            elif row['pnl'] < 0:
                return ['background-color: #FFB6C1'] * len(row)
        return [''] * len(row)

    st.dataframe(
        display_df.style.apply(highlight_pnl, axis=1),
        use_container_width=True,
        height=400
    )
else:
    st.info("No trades yet")

# ==================== Recent Signals ====================

st.markdown("---")
st.header("📡 Recent Signals")

if not signals_df.empty:
    # Show last 20 signals
    recent_signals = signals_df.tail(20).sort_values('timestamp', ascending=False)

    # Format for display
    display_signals = recent_signals.copy()

    if 'timestamp' in display_signals.columns:
        display_signals['timestamp'] = display_signals['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    if 'confidence' in display_signals.columns:
        display_signals['confidence'] = display_signals['confidence'].apply(lambda x: f"{x:.1%}")

    st.dataframe(display_signals, use_container_width=True, height=400)
else:
    st.info("No signals yet")

# ==================== Auto Refresh ====================

if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
