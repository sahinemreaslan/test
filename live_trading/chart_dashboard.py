"""
Live Trading Chart Dashboard
Real-time candlestick chart with technical indicators
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dashboard_data import DashboardDataManager
from datetime import datetime, timedelta
import time
import os
import sys
import plotly.io as pio

# Fix plotly recursion error
pio.templates.default = "plotly"

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from binance_connector import BinanceConnector
from dotenv import load_dotenv

# Page config
st.set_page_config(
    page_title="Bitcoin Chart Dashboard",
    page_icon="📈",
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
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize
@st.cache_resource
def get_data_manager():
    return DashboardDataManager()

@st.cache_resource
def get_binance_connector():
    load_dotenv()
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')

    if not api_key or not api_secret:
        return None

    testnet = True  # Default to testnet for safety
    return BinanceConnector(api_key, api_secret, testnet=testnet)

data_manager = get_data_manager()
binance = get_binance_connector()

# ==================== Header ====================

st.title("📈 Bitcoin Live Chart Dashboard")
st.markdown("---")

# ==================== Sidebar ====================

st.sidebar.title("⚙️ Chart Settings")

# Auto-refresh
auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 15)

# Timeframe selection
timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["1m", "5m", "15m", "30m", "1h", "4h", "1D"],
    index=2  # Default to 15m
)

# Candle count
candle_count = st.sidebar.slider("Number of Candles", 50, 500, 200)

# Indicators
st.sidebar.markdown("### 📊 Technical Indicators")
show_ma = st.sidebar.checkbox("Moving Averages", value=True)
show_bb = st.sidebar.checkbox("Bollinger Bands", value=False)
show_volume = st.sidebar.checkbox("Volume", value=True)
show_rsi = st.sidebar.checkbox("RSI", value=True)
show_macd = st.sidebar.checkbox("MACD", value=False)

# Trade markers
st.sidebar.markdown("### 🎯 Overlays")
show_trades = st.sidebar.checkbox("Show Trades", value=True)
show_signals = st.sidebar.checkbox("Show Signals", value=True)

st.sidebar.markdown("---")
st.sidebar.info(f"Last Updated: {datetime.now().strftime('%H:%M:%S')}")

# ==================== Main Content ====================

if binance is None:
    st.error("❌ Binance connector not initialized. Please check your .env file.")
    st.stop()

# Fetch data
@st.cache_data(ttl=refresh_interval)
def fetch_chart_data(symbol, interval, limit):
    try:
        df = binance.get_historical_klines(
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Get current price
def get_current_price():
    try:
        return binance.get_current_price("BTCUSDT")
    except:
        return None

# Get market data
df = fetch_chart_data("BTCUSDT", timeframe, candle_count)
current_price = get_current_price()

if df is None or df.empty:
    st.error("❌ Could not fetch chart data")
    st.stop()

# ==================== Price Display ====================

col1, col2, col3, col4 = st.columns(4)

with col1:
    if current_price:
        st.metric("Current Price", f"${current_price:,.2f}")

with col2:
    price_change = df['close'].iloc[-1] - df['close'].iloc[0]
    price_change_pct = (price_change / df['close'].iloc[0]) * 100
    st.metric(
        "Price Change",
        f"${price_change:,.2f}",
        f"{price_change_pct:+.2f}%"
    )

with col3:
    high_24h = df['high'].tail(96).max()  # Assuming 15m candles
    st.metric("24h High", f"${high_24h:,.2f}")

with col4:
    low_24h = df['low'].tail(96).min()
    st.metric("24h Low", f"${low_24h:,.2f}")

st.markdown("---")

# ==================== Calculate Indicators ====================

def calculate_indicators(df):
    """Calculate technical indicators"""
    df = df.copy()

    # Moving Averages
    df['ma7'] = df['close'].rolling(window=7).mean()
    df['ma25'] = df['close'].rolling(window=25).mean()
    df['ma99'] = df['close'].rolling(window=99).mean()

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    return df

df = calculate_indicators(df)

# ==================== Get Trade and Signal Data ====================

trades_df = data_manager.get_trades_df()
signals_df = data_manager.get_signals_df()

# Filter trades and signals to match chart timeframe
if not df.empty and 'timestamp' in df.columns:
    chart_start = df['timestamp'].min()
    chart_end = df['timestamp'].max()

    if not trades_df.empty and 'timestamp' in trades_df.columns:
        trades_df = trades_df[
            (trades_df['timestamp'] >= chart_start) &
            (trades_df['timestamp'] <= chart_end)
        ]

    if not signals_df.empty and 'timestamp' in signals_df.columns:
        signals_df = signals_df[
            (signals_df['timestamp'] >= chart_start) &
            (signals_df['timestamp'] <= chart_end)
        ]

# ==================== Create Chart ====================

# Determine number of subplots
subplot_count = 1
subplot_titles = ['Price']
row_heights = [0.7]

if show_volume:
    subplot_count += 1
    subplot_titles.append('Volume')
    row_heights.append(0.15)

if show_rsi:
    subplot_count += 1
    subplot_titles.append('RSI')
    row_heights.append(0.15)

if show_macd:
    subplot_count += 1
    subplot_titles.append('MACD')
    row_heights.append(0.15)

# Normalize heights
row_heights = [h/sum(row_heights) for h in row_heights]

# Create subplots
fig = make_subplots(
    rows=subplot_count,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    subplot_titles=subplot_titles,
    row_heights=row_heights
)

# ==================== Candlestick Chart ====================

fig.add_trace(
    go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='BTCUSDT',
        increasing_line_color='#00ff00',
        decreasing_line_color='#ff0000'
    ),
    row=1, col=1
)

# ==================== Moving Averages ====================

if show_ma:
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['ma7'],
            mode='lines',
            name='MA7',
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['ma25'],
            mode='lines',
            name='MA25',
            line=dict(color='orange', width=1)
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['ma99'],
            mode='lines',
            name='MA99',
            line=dict(color='purple', width=1.5)
        ),
        row=1, col=1
    )

# ==================== Bollinger Bands ====================

if show_bb:
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['bb_upper'],
            mode='lines',
            name='BB Upper',
            line=dict(color='gray', width=1, dash='dash')
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['bb_middle'],
            mode='lines',
            name='BB Middle',
            line=dict(color='gray', width=1)
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['bb_lower'],
            mode='lines',
            name='BB Lower',
            line=dict(color='gray', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(128,128,128,0.1)'
        ),
        row=1, col=1
    )

# ==================== Trade Markers ====================

if show_trades and not trades_df.empty:
    # Entry points
    entries = trades_df[trades_df['type'] == 'OPEN']
    if not entries.empty:
        fig.add_trace(
            go.Scatter(
                x=entries['timestamp'],
                y=entries['entry_price'],
                mode='markers',
                name='Entry',
                marker=dict(
                    symbol='triangle-up',
                    size=15,
                    color='lime',
                    line=dict(color='darkgreen', width=2)
                ),
                text=[f"Entry: ${p:.2f}<br>Qty: {q:.6f}"
                      for p, q in zip(entries['entry_price'], entries['quantity'])],
                hoverinfo='text'
            ),
            row=1, col=1
        )

    # Exit points
    exits = trades_df[trades_df['type'] == 'CLOSE']
    if not exits.empty:
        fig.add_trace(
            go.Scatter(
                x=exits['timestamp'],
                y=exits['exit_price'],
                mode='markers',
                name='Exit',
                marker=dict(
                    symbol='triangle-down',
                    size=15,
                    color='red',
                    line=dict(color='darkred', width=2)
                ),
                text=[f"Exit: ${p:.2f}<br>PnL: ${pnl:.2f}"
                      for p, pnl in zip(exits['exit_price'], exits['pnl'])],
                hoverinfo='text'
            ),
            row=1, col=1
        )

# ==================== Signal Markers ====================

if show_signals and not signals_df.empty:
    # Buy signals
    buy_signals = signals_df[signals_df['signal'] == 1]
    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_signals['timestamp'],
                y=buy_signals['price'],
                mode='markers',
                name='BUY Signal',
                marker=dict(
                    symbol='circle',
                    size=10,
                    color='cyan',
                    line=dict(color='blue', width=1)
                ),
                text=[f"BUY<br>Confidence: {c:.0%}<br>Regime: {r}"
                      for c, r in zip(buy_signals['confidence'], buy_signals['regime'])],
                hoverinfo='text'
            ),
            row=1, col=1
        )

    # Sell signals
    sell_signals = signals_df[signals_df['signal'] == -1]
    if not sell_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_signals['timestamp'],
                y=sell_signals['price'],
                mode='markers',
                name='SELL Signal',
                marker=dict(
                    symbol='circle',
                    size=10,
                    color='orange',
                    line=dict(color='red', width=1)
                ),
                text=[f"SELL<br>Confidence: {c:.0%}<br>Regime: {r}"
                      for c, r in zip(sell_signals['confidence'], sell_signals['regime'])],
                hoverinfo='text'
            ),
            row=1, col=1
        )

# ==================== Volume ====================

current_row = 2

if show_volume:
    colors = ['red' if df['close'].iloc[i] < df['open'].iloc[i] else 'green'
              for i in range(len(df))]

    fig.add_trace(
        go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='Volume',
            marker_color=colors,
            showlegend=False
        ),
        row=current_row, col=1
    )
    current_row += 1

# ==================== RSI ====================

if show_rsi:
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['rsi'],
            mode='lines',
            name='RSI',
            line=dict(color='purple', width=2)
        ),
        row=current_row, col=1
    )

    # Overbought/Oversold lines
    fig.add_hline(
        y=70,
        line_dash="dash",
        line_color="red",
        row=current_row, col=1
    )
    fig.add_hline(
        y=30,
        line_dash="dash",
        line_color="green",
        row=current_row, col=1
    )

    current_row += 1

# ==================== MACD ====================

if show_macd:
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['macd'],
            mode='lines',
            name='MACD',
            line=dict(color='blue', width=1)
        ),
        row=current_row, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['macd_signal'],
            mode='lines',
            name='Signal',
            line=dict(color='red', width=1)
        ),
        row=current_row, col=1
    )

    colors = ['red' if val < 0 else 'green' for val in df['macd_hist']]
    fig.add_trace(
        go.Bar(
            x=df['timestamp'],
            y=df['macd_hist'],
            name='Histogram',
            marker_color=colors
        ),
        row=current_row, col=1
    )

# ==================== Layout ====================

fig.update_layout(
    title=f'BTCUSDT {timeframe.upper()} Chart',
    xaxis_rangeslider_visible=False,
    height=800,
    hovermode='x unified',
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

# Update y-axis labels
fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
if show_volume:
    fig.update_yaxes(title_text="Volume", row=2, col=1)
if show_rsi:
    row_num = 3 if show_volume else 2
    fig.update_yaxes(title_text="RSI", row=row_num, col=1)
if show_macd:
    row_num = subplot_count
    fig.update_yaxes(title_text="MACD", row=row_num, col=1)

# Display chart
st.plotly_chart(fig, use_container_width=True)

# ==================== Statistics ====================

st.markdown("---")
st.header("📊 Chart Statistics")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    volatility = df['close'].pct_change().std() * 100
    st.metric("Volatility", f"{volatility:.2f}%")

with col2:
    avg_volume = df['volume'].mean()
    st.metric("Avg Volume", f"{avg_volume:,.0f}")

with col3:
    price_range = df['high'].max() - df['low'].min()
    st.metric("Price Range", f"${price_range:,.2f}")

with col4:
    if show_rsi:
        current_rsi = df['rsi'].iloc[-1]
        st.metric("Current RSI", f"{current_rsi:.1f}")

with col5:
    trend = "🟢 Bullish" if df['close'].iloc[-1] > df['ma25'].iloc[-1] else "🔴 Bearish"
    st.metric("Trend", trend)

# ==================== Latest Candles Table ====================

st.markdown("---")
st.header("📋 Latest Candles")

latest_candles = df.tail(10).copy()
latest_candles['timestamp'] = latest_candles['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
latest_candles = latest_candles[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
latest_candles = latest_candles.iloc[::-1]  # Reverse to show newest first

# Format numbers
for col in ['open', 'high', 'low', 'close']:
    latest_candles[col] = latest_candles[col].apply(lambda x: f"${x:,.2f}")
latest_candles['volume'] = latest_candles['volume'].apply(lambda x: f"{x:,.2f}")

st.dataframe(latest_candles, use_container_width=True, hide_index=True)

# ==================== Auto Refresh ====================

if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
