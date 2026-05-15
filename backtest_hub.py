import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Algo Factory | Backtest Hub V3",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
body, [data-testid="stAppViewContainer"] { background-color: #0d1117; color: #e6edf3; }
[data-testid="stSidebar"] { background-color: #010409; border-right: 1px solid #21262d; }
[data-testid="stSidebar"] * { color: #e6edf3; }
h1, h2, h3 { color: #e6edf3; }
.metric-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 10px; padding: 16px 12px; text-align: center;
    margin-bottom: 8px;
}
.metric-value { font-size: 26px; font-weight: 800; margin-bottom: 4px; }
.metric-label { font-size: 11px; color: #8b949e; letter-spacing: 0.5px; text-transform: uppercase; }
.green  { color: #3fb950; }
.red    { color: #f85149; }
.blue   { color: #58a6ff; }
.gold   { color: #d29922; }
.purple { color: #bc8cff; }
div[data-testid="stHorizontalBlock"] { gap: 8px; }
div.stButton > button {
    background: linear-gradient(135deg, #238636, #1a7f37);
    color: white; font-weight: 700; font-size: 15px;
    border: 1px solid #2ea043; border-radius: 8px;
    padding: 12px; width: 100%; letter-spacing: 0.5px;
}
div.stButton > button:hover { background: linear-gradient(135deg, #2ea043, #238636); }
.warning-box {
    background: #272115; border: 1px solid #d29922;
    border-radius: 8px; padding: 12px; margin: 8px 0;
    color: #d29922; font-size: 13px;
}
.error-box {
    background: #1f1215; border: 1px solid #f85149;
    border-radius: 8px; padding: 12px; margin: 8px 0;
    color: #f85149; font-size: 13px;
}
.info-box {
    background: #0d1926; border: 1px solid #58a6ff;
    border-radius: 8px; padding: 12px; margin: 8px 0;
    color: #58a6ff; font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  ASSET LIBRARY
# ─────────────────────────────────────────────
INDICES = {
    "⚡ NIFTY 50 (Spot)"   : "^NSEI",
    "⚡ SENSEX (Spot)"     : "^BSESN",
    "⚡ BANK NIFTY"        : "^NSEBANK",
    "⚡ NIFTY IT"          : "^CNXIT",
    "⚡ NIFTY MIDCAP 50"   : "^NSEMDCP50",
}

NIFTY50 = {
    "Reliance Industries"  : "RELIANCE.NS",
    "TCS"                  : "TCS.NS",
    "HDFC Bank"            : "HDFCBANK.NS",
    "Infosys"              : "INFY.NS",
    "ICICI Bank"           : "ICICIBANK.NS",
    "Hindustan Unilever"   : "HINDUNILVR.NS",
    "ITC"                  : "ITC.NS",
    "SBI"                  : "SBIN.NS",
    "Bharti Airtel"        : "BHARTIARTL.NS",
    "Bajaj Finance"        : "BAJFINANCE.NS",
    "Wipro"                : "WIPRO.NS",
    "HCL Technologies"     : "HCLTECH.NS",
    "Asian Paints"         : "ASIANPAINT.NS",
    "Kotak Mahindra Bank"  : "KOTAKBANK.NS",
    "L&T"                  : "LT.NS",
    "Axis Bank"            : "AXISBANK.NS",
    "Maruti Suzuki"        : "MARUTI.NS",
    "Sun Pharma"           : "SUNPHARMA.NS",
    "NTPC"                 : "NTPC.NS",
    "Power Grid"           : "POWERGRID.NS",
    "UltraTech Cement"     : "ULTRACEMCO.NS",
    "Titan Company"        : "TITAN.NS",
    "Tech Mahindra"        : "TECHM.NS",
    "Bajaj Auto"           : "BAJAJ-AUTO.NS",
    "Nestle India"         : "NESTLEIND.NS",
    "Grasim Industries"    : "GRASIM.NS",
    "JSW Steel"            : "JSWSTEEL.NS",
    "Tata Steel"           : "TATASTEEL.NS",
    "Tata Motors"          : "TATAMOTORS.NS",
    "ONGC"                 : "ONGC.NS",
    "Coal India"           : "COALINDIA.NS",
    "Adani Ports"          : "ADANIPORTS.NS",
    "Adani Enterprises"    : "ADANIENT.NS",
    "Cipla"                : "CIPLA.NS",
    "Divis Labs"           : "DIVISLAB.NS",
    "Eicher Motors"        : "EICHERMOT.NS",
    "HDFC Life"            : "HDFCLIFE.NS",
    "Hero MotoCorp"        : "HEROMOTOCO.NS",
    "Hindalco"             : "HINDALCO.NS",
    "IndusInd Bank"        : "INDUSINDBK.NS",
    "M&M"                  : "M&M.NS",
    "Dr. Reddys"           : "DRREDDY.NS",
    "SBI Life"             : "SBILIFE.NS",
    "Shriram Finance"      : "SHRIRAMFIN.NS",
    "Tata Consumer"        : "TATACONSUM.NS",
    "Vedanta"              : "VEDL.NS",
    "Britannia"            : "BRITANNIA.NS",
    "BPCL"                 : "BPCL.NS",
    "Apollo Hospitals"     : "APOLLOHOSP.NS",
    "Bajaj Finserv"        : "BAJAJFINSV.NS",
}

ALL_ASSETS = {**INDICES, **NIFTY50}

TIMEFRAME_MAP = {
    "1 Minute  (last 7 days)"   : ("1m",  7),
    "5 Minutes (last 60 days)"  : ("5m",  60),
    "15 Minutes(last 60 days)"  : ("15m", 60),
    "1 Hour    (last 730 days)" : ("1h",  730),
    "1 Day     (multi-year)"    : ("1d",  3650),
    "1 Week    (multi-year)"    : ("1wk", 3650),
}

# ─────────────────────────────────────────────
#  INDICATOR FUNCTIONS
# ─────────────────────────────────────────────
def calc_sma(series, length):
    return series.rolling(window=length).mean()

def calc_ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def calc_dema(series, length):
    ema1 = calc_ema(series, length)
    ema2 = calc_ema(ema1, length)
    return (2 * ema1) - ema2

def calc_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

# ─────────────────────────────────────────────
#  PROFIT ENGINE (V3.1 — Weekly/Monthly Aware)
# ─────────────────────────────────────────────

# Indices that expire WEEKLY (Nifty, BankNifty, FinNifty, etc.)
WEEKLY_EXPIRY_TICKERS = {"^NSEI", "^NSEBANK", "^CNXIT", "^NSEMDCP50", "^BSESN"}

def apply_profit_model(strategy_returns, mode, interval, ticker):
    """
    Convert raw strategy percentage returns into model-specific returns.
    Automatically detects Index (Weekly Expiry) vs Stock (Monthly Expiry).
    mode: 'Cash (Equity)', 'Intraday (10x Margin)', 'Options (Synthetic 40x)'
    """
    if mode == "Cash (Equity)":
        return strategy_returns

    elif mode == "Intraday (10x Margin)":
        leveraged = strategy_returns * 10
        # Margin Call: Cap max loss at -100% per candle (broker squares off at zero)
        leveraged = leveraged.clip(lower=-1.0)
        return leveraged

    elif mode == "Options (Synthetic 40x)":
        # Determine candles per trading day based on timeframe
        candles_per_day = {
            "1m": 375, "2m": 188, "5m": 75,
            "15m": 25, "30m": 13, "60m": 6,
            "1h": 6,  "1d": 1,   "1wk": 0.2
        }
        cpd = candles_per_day.get(interval, 1)

        # --- Weekly vs Monthly Detection ---
        is_weekly = ticker in WEEKLY_EXPIRY_TICKERS
        # Weekly (Index): Theta decays 3x faster, leverage is slightly higher
        # Monthly (Stock): Standard theta decay
        base_leverage   = 45.0 if is_weekly else 40.0
        daily_theta     = 0.022 if is_weekly else 0.0075  # 2.2% per day for weekly, 0.75% for monthly
        theta_per_candle = daily_theta / max(cpd, 1)

        # --- Gamma Simulation (Money multiplies faster as trade goes deeper ITM) ---
        # When return is strongly positive, leverage increases by up to 1.5x (Gamma effect)
        cumulative_return = strategy_returns.cumsum()
        gamma_boost = 1.0 + (cumulative_return.clip(lower=0) * 0.5).clip(upper=0.5)

        leveraged = strategy_returns * base_leverage * gamma_boost

        # --- Apply Theta Decay only when IN a trade ---
        in_trade_mask = strategy_returns != 0
        leveraged[in_trade_mask] -= theta_per_candle

        # --- Options Liability Cap: Cannot lose more than premium paid (Hero or Zero) ---
        leveraged = leveraged.clip(lower=-1.0)

        return leveraged

    return strategy_returns

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.markdown("## ⚡ Algo Factory")
st.sidebar.markdown("### 📌 Asset Selection")

asset_name = st.sidebar.selectbox("Select Asset / Index", list(ALL_ASSETS.keys()), index=0)
ticker = ALL_ASSETS[asset_name]

st.sidebar.markdown("---")
st.sidebar.markdown("### 🕐 Timeframe")
tf_label = st.sidebar.select_slider(
    "Timeframe",
    options=list(TIMEFRAME_MAP.keys()),
    value="1 Day     (multi-year)"
)
interval, max_days = TIMEFRAME_MAP[tf_label]

st.sidebar.markdown("---")
st.sidebar.markdown("### 📅 Date Range")
today = datetime.today().date()
default_start = today - timedelta(days=min(365, max_days))
start_date = st.sidebar.date_input("Start Date", default_start)
end_date   = st.sidebar.date_input("End Date",   today)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Chart Style")
chart_type = st.sidebar.radio("Chart Type", ["Candlestick", "Line", "Area"], index=0)

st.sidebar.markdown("### 👁️ Visibility")
show_labels = st.sidebar.checkbox("Show Price Labels on Chart", value=False)
marker_size = st.sidebar.slider("Marker Size", 5, 15, 8)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Strategy Engine")

# Indicator Checklist with Illustrations
use_sma  = st.sidebar.checkbox("Double SMA (Trend)",      value=True)
if use_sma:
    st.sidebar.caption("📘 SMA = Simple Moving Average. Two lines (Fast & Slow) follow the price. When the Fast line crosses ABOVE the Slow line → BUY signal. When it crosses BELOW → SELL signal. Shorter lengths = more sensitive, more trades. Longer = fewer but stronger signals.")

use_dema = st.sidebar.checkbox("Double DEMA (Zero-Lag)",  value=False)
if use_dema:
    st.sidebar.caption("📗 DEMA = Double Exponential Moving Average. Reacts FASTER than SMA with less lag. It uses a mathematical formula (2×EMA - EMA of EMA) to predict price direction sooner. Best for catching quick reversals that SMA would miss.")

use_rsi  = st.sidebar.checkbox("RSI Filter (Momentum)",   value=False)
if use_rsi:
    st.sidebar.caption("📙 RSI = Relative Strength Index (0–100 scale). Measures how 'overbought' or 'oversold' a stock is. Above 70 = Overbought (may fall). Below 30 = Oversold (may rise). Used as a FILTER to confirm or reject SMA/DEMA signals.")

# Strategy Direction
st.sidebar.markdown("#### 🎯 Trade Direction")
direction = st.sidebar.radio(
    "Trade Direction",
    ["Long Only (Call/Buy)", "Short Only (Put/Sell)", "Both Directions"],
    label_visibility="collapsed"
)
st.sidebar.caption("🔹 Long Only = Profit when price goes UP. 🔻 Short Only = Profit when price goes DOWN. 🔄 Both = Trade in both directions automatically.")

# Exit Voting System
st.sidebar.markdown("#### 🚪 Exit Sensitivity")

active_indicator_count = sum([use_sma, use_dema, use_rsi])

if active_indicator_count >= 2:
    exit_mode = st.sidebar.selectbox(
        "Exit trade when:",
        [
            "Any 1 indicator turns against (Aggressive)",
            "At least 2 indicators turn against (Majority)",
            "ALL indicators turn against (Conservative)",
        ],
        index=0
    )
    st.sidebar.caption("⚡ Aggressive = Exit fast at the first warning. 🛡️ Conservative = Stay in trade until ALL indicators agree to exit. Aggressive protects capital. Conservative maximizes winning trades.")
    if "Any 1" in exit_mode:
        req_exit_votes = 1
    elif "At least 2" in exit_mode:
        req_exit_votes = 2
    else:
        req_exit_votes = active_indicator_count
else:
    req_exit_votes = 1
    st.sidebar.info("ℹ️ Enable 2+ indicators to unlock Exit Voting control.")

# --- SMA INDEPENDENT SLIDERS ---
if use_sma:
    st.sidebar.markdown("#### 📊 SMA Lengths")
    fast_len_sma = st.sidebar.slider("SMA Fast Length", 3, 100, 14, key="sma_fast")
    slow_len_sma = st.sidebar.slider("SMA Slow Length", 5, 500, 21, key="sma_slow")
    st.sidebar.caption(f"📐 Fast ({fast_len_sma}) reacts quickly to price changes. Slow ({slow_len_sma}) smooths out noise. The BIGGER the gap between them, the FEWER but STRONGER the signals.")
else:
    fast_len_sma, slow_len_sma = 14, 21

# --- DEMA INDEPENDENT SLIDERS ---
if use_dema:
    st.sidebar.markdown("#### 🌊 DEMA Lengths")
    fast_len_dema = st.sidebar.slider("DEMA Fast Length", 3, 100, 9, key="dema_fast")
    slow_len_dema = st.sidebar.slider("DEMA Slow Length", 5, 500, 21, key="dema_slow")
    st.sidebar.caption(f"📐 DEMA Fast ({fast_len_dema}) catches reversals early. DEMA Slow ({slow_len_dema}) confirms the trend. DEMA is faster than SMA — use shorter lengths here.")
else:
    fast_len_dema, slow_len_dema = 9, 21

# --- RSI CUSTOM LOGIC (Above/Below Toggle) ---
rsi_period   = 14
rsi_buy_logic  = "Above"
rsi_buy_val    = 50
rsi_sell_logic = "Below"
rsi_sell_val   = 50

if use_rsi:
    st.sidebar.markdown("#### 📡 RSI Criteria")
    rsi_period = st.sidebar.slider("RSI Period", 5, 50, 14)
    st.sidebar.caption(f"🔎 RSI Period ({rsi_period}) = How many candles to look back. Lower = more sensitive. Higher = smoother.")

    st.sidebar.markdown("**Entry Condition (When to BUY):**")
    c1, c2 = st.sidebar.columns(2)
    rsi_buy_logic = c1.selectbox("Long if RSI:", ["Above", "Below"], index=0, key="rsi_bl")
    rsi_buy_val   = c2.number_input("Level", value=50, min_value=1, max_value=99, key="rsi_bv")
    if rsi_buy_logic == "Above":
        st.sidebar.caption(f"✅ You are saying: 'Only BUY when RSI is ABOVE {rsi_buy_val}' — This is TREND FOLLOWING. You buy when momentum is already strong.")
    else:
        st.sidebar.caption(f"✅ You are saying: 'Only BUY when RSI is BELOW {rsi_buy_val}' — This is MEAN REVERSION. You buy when the stock is oversold (dip buying).")

    st.sidebar.markdown("**Exit Condition (When to SELL/SHORT):**")
    c3, c4 = st.sidebar.columns(2)
    rsi_sell_logic = c3.selectbox("Short if RSI:", ["Below", "Above"], index=0, key="rsi_sl")
    rsi_sell_val   = c4.number_input("Level", value=50, min_value=1, max_value=99, key="rsi_sv")
    if rsi_sell_logic == "Below":
        st.sidebar.caption(f"🛑 You are saying: 'EXIT/SHORT when RSI drops BELOW {rsi_sell_val}' — Momentum is dying, time to get out.")
    else:
        st.sidebar.caption(f"🛑 You are saying: 'EXIT/SHORT when RSI goes ABOVE {rsi_sell_val}' — Stock is overbought, expecting a pullback.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 💰 Capital & Profit Model")
profit_mode     = st.sidebar.selectbox("Profit Model", ["Cash (Equity)", "Intraday (10x Margin)", "Options (Synthetic 40x)"])
initial_capital = st.sidebar.slider("Initial Capital (₹)", 5000, 2000000, 100000, step=5000)

run = st.sidebar.button("🚀  EXECUTE BACKTEST", use_container_width=True)

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("# ⚡ Algo Factory — Backtest Hub V3")
st.markdown(
    f"**Asset:** `{asset_name}` &nbsp;·&nbsp; "
    f"**Ticker:** `{ticker}` &nbsp;·&nbsp; "
    f"**Timeframe:** `{interval}` &nbsp;·&nbsp; "
    f"**Mode:** `{profit_mode}`"
)
st.markdown("---")

if not run:
    st.markdown(
        '<div class="info-box">👈  Configure your strategy in the sidebar and click <strong>EXECUTE BACKTEST</strong> to begin.</div>',
        unsafe_allow_html=True
    )
    st.stop()

# ─────────────────────────────────────────────
#  GUARD: At least one indicator must be selected
# ─────────────────────────────────────────────
if not use_sma and not use_dema and not use_rsi:
    st.markdown(
        '<div class="error-box">⛔ No indicators selected. Please enable at least one indicator (Double SMA, Double DEMA, or RSI) in the sidebar.</div>',
        unsafe_allow_html=True
    )
    st.stop()

# ─────────────────────────────────────────────
#  GUARD: Fast must be less than Slow
# ─────────────────────────────────────────────
if use_sma and fast_len_sma >= slow_len_sma:
    st.markdown(
        '<div class="warning-box">⚠️ SMA Logic Warning: SMA Fast Length must be smaller than Slow Length. Please adjust SMA sliders.</div>',
        unsafe_allow_html=True
    )
    st.stop()

if use_dema and fast_len_dema >= slow_len_dema:
    st.markdown(
        '<div class="warning-box">⚠️ DEMA Logic Warning: DEMA Fast Length must be smaller than Slow Length. Please adjust DEMA sliders.</div>',
        unsafe_allow_html=True
    )
    st.stop()

# ─────────────────────────────────────────────
#  GUARD: Date range vs timeframe limits
# ─────────────────────────────────────────────
days_requested = (end_date - start_date).days
if days_requested > max_days:
    st.markdown(
        f'<div class="warning-box">⚠️ Data Availability Warning: The <strong>{tf_label.strip()}</strong> timeframe only supports up to <strong>{max_days} days</strong> of history. '
        f'You requested <strong>{days_requested} days</strong>. Please move your Start Date forward or select a larger timeframe like <strong>1 Day</strong>.</div>',
        unsafe_allow_html=True
    )
    st.stop()

# ─────────────────────────────────────────────
#  DATA FETCH
# ─────────────────────────────────────────────
with st.spinner(f"⏳ Fetching {interval} data for {asset_name}..."):
    try:
        raw = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False
        )
    except Exception as e:
        st.markdown(
            f'<div class="error-box">⛔ Data Fetch Error: {str(e)}<br>Please check your internet connection or try a different asset.</div>',
            unsafe_allow_html=True
        )
        st.stop()

if raw is None or raw.empty:
    st.markdown(
        '<div class="error-box">⛔ No data returned from Yahoo Finance. This may be due to an invalid ticker, a holiday period, or a network issue. Try widening your date range.</div>',
        unsafe_allow_html=True
    )
    st.stop()

# Flatten multi-index columns (yfinance sometimes returns these)
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)

data = raw.copy()
data.dropna(subset=['Close'], inplace=True)

# ─────────────────────────────────────────────
#  IST TIMEZONE CONVERSION
# ─────────────────────────────────────────────
try:
    if data.index.tz is not None:
        data.index = data.index.tz_convert('Asia/Kolkata')
    else:
        data.index = data.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
except Exception:
    pass  # Daily/Weekly data has no timezone, safe to skip

# ─────────────────────────────────────────────
#  CALCULATE INDICATORS
# ─────────────────────────────────────────────
close = data['Close'].squeeze()

if use_sma:
    data['Fast_SMA'] = calc_sma(close, fast_len_sma)
    data['Slow_SMA'] = calc_sma(close, slow_len_sma)

if use_dema:
    data['Fast_DEMA'] = calc_dema(close, fast_len_dema)
    data['Slow_DEMA'] = calc_dema(close, slow_len_dema)

if use_rsi:
    data['RSI'] = calc_rsi(close, rsi_period)

# ─────────────────────────────────────────────
#  GENERATE SIGNALS (Voting Engine V3.1)
# ─────────────────────────────────────────────

# --- ENTRY VOTES: Count how many indicators agree to BUY ---
# Entry uses strict AND logic: ALL active indicators must agree

buy_votes  = pd.Series(0, index=data.index)
exit_votes = pd.Series(0, index=data.index)

if use_sma:
    buy_votes  += np.where(data['Fast_SMA'] > data['Slow_SMA'], 1, 0)
    exit_votes += np.where(data['Fast_SMA'] < data['Slow_SMA'], 1, 0)

if use_dema:
    buy_votes  += np.where(data['Fast_DEMA'] > data['Slow_DEMA'], 1, 0)
    exit_votes += np.where(data['Fast_DEMA'] < data['Slow_DEMA'], 1, 0)

if use_rsi:
    if rsi_buy_logic == "Above":
        buy_votes  += np.where(data['RSI'] > rsi_buy_val,  1, 0)
        exit_votes += np.where(data['RSI'] < rsi_sell_val, 1, 0)
    else:
        buy_votes  += np.where(data['RSI'] < rsi_buy_val,  1, 0)
        exit_votes += np.where(data['RSI'] > rsi_sell_val, 1, 0)

# --- ENTRY CONDITION: All active indicators must agree (safe entry) ---
entry_condition = (buy_votes == active_indicator_count)

# --- EXIT CONDITION: User-controlled vote threshold ---
exit_condition = (exit_votes >= req_exit_votes)

# --- FINAL SIGNAL ASSEMBLY ---
raw_signal = pd.Series(0.0, index=data.index)

if direction == "Long Only (Call/Buy)":
    raw_signal[entry_condition] =  1.0   # Enter Long
    raw_signal[exit_condition]  =  0.0   # Exit to Cash

elif direction == "Short Only (Put/Sell)":
    raw_signal[exit_condition]  = -1.0   # Enter Short
    raw_signal[entry_condition] =  0.0   # Exit to Cash

else:  # Both Directions
    raw_signal[entry_condition] =  1.0   # Enter Long
    raw_signal[exit_condition]  = -1.0   # Enter Short (reverse)

data['Signal']          = raw_signal
data['Position_Change'] = data['Signal'].diff()

# ─────────────────────────────────────────────
#  CALCULATE RETURNS
# ─────────────────────────────────────────────
data['Market_Ret'] = close.pct_change()

# Shift signal by 1 candle to prevent look-ahead bias
# (You act on yesterday's signal, earning today's return)
data['Strategy_Ret_Raw'] = data['Market_Ret'] * data['Signal'].shift(1)

# Apply chosen profit model
data['Strategy_Ret'] = apply_profit_model(data['Strategy_Ret_Raw'], profit_mode, interval, ticker)

# Cumulative compounding
data['Equity_Curve']    = (1 + data['Strategy_Ret'].fillna(0)).cumprod()
data['Market_Curve']    = (1 + data['Market_Ret'].fillna(0)).cumprod()
data['Portfolio_Value'] = initial_capital * data['Equity_Curve']
data['BH_Value']        = initial_capital * data['Market_Curve']

# Drawdown
rolling_max      = data['Portfolio_Value'].cummax()
data['Drawdown'] = (data['Portfolio_Value'] / rolling_max - 1) * 100

# ─────────────────────────────────────────────
#  EXTRACT TRADE EVENTS
# ─────────────────────────────────────────────
buy_entries  = data[data['Position_Change'] > 0]   # 0→1 or -1→1
sell_entries = data[data['Position_Change'] < 0]   # 1→0, 0→-1, or 1→-1

# ─────────────────────────────────────────────
#  GUARD: No trades found
# ─────────────────────────────────────────────
total_signals = len(buy_entries) + len(sell_entries)
if total_signals == 0:
    st.markdown(
        '<div class="info-box">ℹ️ Strategy Insight: No trade signals were generated with the current settings. '
        'Your filters may be too strict. Try:<br>'
        '• Loosening the RSI levels (e.g., Buy above 45, Sell below 55)<br>'
        '• Reducing the Fast or Slow MA lengths<br>'
        '• Selecting a longer date range</div>',
        unsafe_allow_html=True
    )
    st.stop()

# ─────────────────────────────────────────────
#  PERFORMANCE METRICS
# ─────────────────────────────────────────────
strat_return  = (data['Equity_Curve'].iloc[-1]  - 1) * 100
market_return = (data['Market_Curve'].iloc[-1]  - 1) * 100
alpha         = strat_return - market_return
final_val     = data['Portfolio_Value'].iloc[-1]
max_dd        = data['Drawdown'].min()
num_longs     = len(buy_entries)
num_shorts    = len(sell_entries)

st.markdown("## 📊 Performance Dashboard")
cols = st.columns(6)

def card(col, label, value, css_class):
    col.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value {css_class}">{value}</div>'
        f'<div class="metric-label">{label}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

card(cols[0], "Strategy Return",   f"{strat_return:+.2f}%",   "green" if strat_return >= 0 else "red")
card(cols[1], "Market Return",     f"{market_return:+.2f}%",  "green" if market_return >= 0 else "red")
card(cols[2], "Alpha Generated",   f"{alpha:+.2f}%",          "green" if alpha >= 0 else "red")
card(cols[3], "Max Drawdown",      f"{max_dd:.2f}%",          "red")
card(cols[4], "Long Entries",  f"{num_longs}",  "blue")
card(cols[5], "Short Entries", f"{num_shorts}", "purple")

# Win Rate Calculation
if num_longs > 0:
    trade_rets = []
    long_idx = list(buy_entries.index)
    exit_idx = list(sell_entries.index)
    for i, entry_t in enumerate(long_idx):
        exits_after = [e for e in exit_idx if e > entry_t]
        if exits_after:
            exit_t = exits_after[0]
            trade_ret = (data.loc[exit_t, 'Close'] - data.loc[entry_t, 'Close']) / data.loc[entry_t, 'Close'] * 100
            trade_rets.append(trade_ret)
    if trade_rets:
        wins = sum(1 for r in trade_rets if r > 0)
        win_rate = (wins / len(trade_rets)) * 100
        avg_ret = np.mean(trade_rets)
        st.markdown("<br>", unsafe_allow_html=True)
        w1, w2 = st.columns(2)
        card(w1, "Win Rate",       f"{win_rate:.1f}%",  "green" if win_rate >= 50 else "red")
        card(w2, "Avg Trade Return", f"{avg_ret:+.2f}%", "green" if avg_ret >= 0 else "red")

st.markdown("<br>", unsafe_allow_html=True)
c1, c2 = st.columns(2)
card(c1, "Initial Capital",       f"₹ {initial_capital:,.0f}", "gold")
card(c2, "Final Portfolio Value", f"₹ {final_val:,.2f}",       "green" if final_val >= initial_capital else "red")

# ─────────────────────────────────────────────
#  MASTER CHART
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🕯️ Terminal Chart")

indicator_tags = []

if use_sma:
    indicator_tags.append(f"SMA {fast_len_sma}/{slow_len_sma}")
if use_dema:
    indicator_tags.append(f"DEMA {fast_len_dema}/{slow_len_dema}")
if use_rsi:
    indicator_tags.append(f"RSI({rsi_period}) | {rsi_buy_logic} {rsi_buy_val} / {rsi_sell_logic} {rsi_sell_val}")

indicator_label = "  ·  ".join(indicator_tags) if indicator_tags else "No Indicators"

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.025,
    row_heights=[0.58, 0.16, 0.26],
    subplot_titles=(
        f"⚡ {asset_name}  |  {chart_type}  |  {indicator_label}  |  {interval.upper()}",
        "📊 Volume",
        f"💰 Equity Curve vs Buy & Hold  [{profit_mode}]"
    )
)

# ── Price Panel
if chart_type == "Candlestick":
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'], high=data['High'],
        low=data['Low'],   close=close,
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350',
        increasing_fillcolor='#26a69a',
        decreasing_fillcolor='#ef5350',
        name='Price', showlegend=False
    ), row=1, col=1)

elif chart_type == "Line":
    fig.add_trace(go.Scatter(
        x=data.index, y=close,
        line=dict(color='#58a6ff', width=1.5),
        name='Close Price'
    ), row=1, col=1)

else:  # Area

    # First add an invisible baseline at the minimum price (NOT zero)
    price_floor = close.min() * 0.999

    fig.add_trace(go.Scatter(
        x=data.index, y=[price_floor] * len(data.index),
        line=dict(color='rgba(0,0,0,0)', width=0),
        showlegend=False, name='_floor', hoverinfo='skip'
    ), row=1, col=1)

    # Then fill from the price line DOWN to the floor (not to zero)
    fig.add_trace(go.Scatter(
        x=data.index, y=close,
        fill='tonexty',
        fillcolor='rgba(88,166,255,0.12)',
        line=dict(color='#58a6ff', width=2),
        name='Close Price'
    ), row=1, col=1)

# ── Indicator Lines
if use_sma:
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Fast_SMA'],
        line=dict(color='#2979ff', width=2),
        name=f'Fast SMA ({fast_len_sma})'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Slow_SMA'],
        line=dict(color='#ff6d00', width=2),
        name=f'Slow SMA ({slow_len_sma})'
    ), row=1, col=1)

if use_dema:
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Fast_DEMA'],
        line=dict(color='#00e5ff', width=1.5, dash='dot'),
        name=f'Fast DEMA ({fast_len_dema})'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Slow_DEMA'],
        line=dict(color='#ff4081', width=1.5, dash='dot'),
        name=f'Slow DEMA ({slow_len_dema})'
    ), row=1, col=1)

# ── Buy & Sell Markers (Cleaned & Optimized)

if not buy_entries.empty:
    fig.add_trace(go.Scatter(
        x=buy_entries.index,
        y=data.loc[buy_entries.index, 'Low'] * 0.998,
        mode='markers+text' if show_labels else 'markers',
        marker=dict(
            symbol='triangle-up',
            size=marker_size,
            color='#3fb950',
            opacity=0.7,
            line=dict(color='#ffffff', width=0.5)
        ),
        text=[f"B ₹{p:.1f}" for p in data.loc[buy_entries.index, 'Close']],
        textposition='bottom center',
        textfont=dict(color='#3fb950', size=8),
        name='Long Entry',
        hoverinfo='text',
        hovertext=[f"BUY Entry: ₹{p:.2f}" for p in data.loc[buy_entries.index, 'Close']]
    ), row=1, col=1)

if not sell_entries.empty:
    fig.add_trace(go.Scatter(
        x=sell_entries.index,
        y=data.loc[sell_entries.index, 'High'] * 1.002,
        mode='markers+text' if show_labels else 'markers',
        marker=dict(
            symbol='triangle-down',
            size=marker_size,
            color='#f85149',
            opacity=0.7,
            line=dict(color='#ffffff', width=0.5)
        ),
        text=[f"S ₹{p:.1f}" for p in data.loc[sell_entries.index, 'Close']],
        textposition='top center',
        textfont=dict(color='#f85149', size=8),
        name='Short Entry / Exit',
        hoverinfo='text',
        hovertext=[f"SELL Entry/Exit: ₹{p:.2f}" for p in data.loc[sell_entries.index, 'Close']]
    ), row=1, col=1)

# ── Volume Panel
if 'Volume' in data.columns:
    vol_colors = [
        '#26a69a' if c >= o else '#ef5350'
        for c, o in zip(data['Close'], data['Open'])
    ]
    fig.add_trace(go.Bar(
        x=data.index, y=data['Volume'],
        marker_color=vol_colors,
        name='Volume', showlegend=False, opacity=0.8
    ), row=2, col=1)

    # Volume rolling average line
    vol_ma = data['Volume'].rolling(20).mean()
    fig.add_trace(go.Scatter(
        x=data.index, y=vol_ma,
        line=dict(color='#d29922', width=1.2),
        name='Vol MA(20)', showlegend=False
    ), row=2, col=1)

# ── Equity Curve Panel
# Profit zone (strategy above BH) filled green, loss zone filled red
fig.add_trace(go.Scatter(
    x=data.index, y=data['BH_Value'],
    line=dict(color='rgba(0,0,0,0)', width=0),
    showlegend=False, name='_bh_base'
), row=3, col=1)

fig.add_trace(go.Scatter(
    x=data.index, y=data['Portfolio_Value'],
    fill='tonexty',
    fillcolor='rgba(63,185,80,0.15)',
    line=dict(color='#3fb950', width=2),
    name='Strategy Portfolio'
), row=3, col=1)

fig.add_trace(go.Scatter(
    x=data.index, y=data['BH_Value'],
    line=dict(color='#8b949e', width=1.5, dash='dot'),
    name='Buy & Hold'
), row=3, col=1)

# ── Rangebreaks: Remove non-trading gaps (IST-precise)

if interval in ['1m', '2m', '5m', '15m', '30m', '60m', '1h']:
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),                   # Remove full weekends
            dict(bounds=[15.5, 9.25], pattern="hour"),     # Remove 3:30 PM to 9:15 AM IST
        ],
        row=1, col=1
    )
elif interval in ['1d', '1wk']:
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),                   # Only remove weekends for daily/weekly
        ],
        row=1, col=1
    )

# ── Layout (Professional Terminal)
fig.update_layout(
    height=880,
    template='plotly_dark',
    paper_bgcolor='#0d1117',
    plot_bgcolor='#0d1117',
    xaxis_rangeslider_visible=False,
    hovermode='closest',
    hoverdistance=30,
    spikedistance=-1,
    legend=dict(
        orientation='h', yanchor='bottom', y=1.02,
        xanchor='right', x=1,
        bgcolor='rgba(22,27,34,0.85)',
        bordercolor='#30363d', borderwidth=1,
        font=dict(size=10, color='#e6edf3')
    ),
    margin=dict(l=10, r=10, t=55, b=10),
    font=dict(color='#8b949e', family='Inter, Roboto, sans-serif'),
    title_font=dict(color='#e6edf3', size=13)
)

# Force Y-axis to auto-scale to visible data range (not to zero)
fig.update_xaxes(
    gridcolor='#21262d', showgrid=True, zeroline=False,
    showspikes=True, spikethickness=1, spikecolor='#58a6ff',
    spikemode='across', spikesnap='cursor',
    tickfont=dict(color='#8b949e', size=10)
)
fig.update_yaxes(
    gridcolor='#161b22', showgrid=True, zeroline=False,
    autorange=True, fixedrange=False,
    showspikes=True, spikethickness=1, spikecolor='#58a6ff',
    spikemode='across', spikesnap='cursor',
    tickfont=dict(color='#8b949e', size=10)
)

st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
#  DRAWDOWN CHART
# ─────────────────────────────────────────────
st.markdown("## 📉 Drawdown Chart")
dd_fig = go.Figure()
dd_fig.add_trace(go.Scatter(
    x=data.index, y=data['Drawdown'],
    fill='tozeroy',
    fillcolor='rgba(248,81,73,0.2)',
    line=dict(color='#f85149', width=1.5),
    name='Drawdown %'
))
dd_fig.update_layout(
    height=220, template='plotly_dark',
    paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
    margin=dict(l=10, r=10, t=20, b=10),
    yaxis_ticksuffix='%', hovermode='x unified',
    font=dict(color='#8b949e')
)
dd_fig.update_xaxes(gridcolor='#21262d')
dd_fig.update_yaxes(gridcolor='#21262d')
st.plotly_chart(dd_fig, use_container_width=True)

# ─────────────────────────────────────────────
#  RSI CHART (only if RSI is active)
# ─────────────────────────────────────────────
if use_rsi and 'RSI' in data.columns:
    st.markdown("## 📡 RSI Indicator")
    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(
        x=data.index, y=data['RSI'],
        line=dict(color='#bc8cff', width=1.8),
        name=f'RSI ({rsi_period})'
    ))
    rsi_fig.add_hline(
        y=rsi_buy_val, line_dash='dot', line_color='#3fb950',
        annotation_text=f'Long Entry: {rsi_buy_logic} {rsi_buy_val}',
        annotation_position='right'
    )
    rsi_fig.add_hline(
        y=rsi_sell_val, line_dash='dot', line_color='#f85149',
        annotation_text=f'Short Entry: {rsi_sell_logic} {rsi_sell_val}',
        annotation_position='right'
    )
    rsi_fig.add_hrect(y0=rsi_buy_val, y1=100, fillcolor='rgba(63,185,80,0.06)', line_width=0)
    rsi_fig.add_hrect(y0=0, y1=rsi_sell_val, fillcolor='rgba(248,81,73,0.06)', line_width=0)
    rsi_fig.update_layout(
        height=220, template='plotly_dark',
        paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
        margin=dict(l=10, r=10, t=20, b=10),
        hovermode='x unified', font=dict(color='#8b949e'),
        yaxis=dict(range=[0, 100])
    )
    rsi_fig.update_xaxes(gridcolor='#21262d')
    rsi_fig.update_yaxes(gridcolor='#21262d')
    st.plotly_chart(rsi_fig, use_container_width=True)

# ─────────────────────────────────────────────
#  TRADE EXECUTION LEDGER
# ─────────────────────────────────────────────
st.markdown("## 📋 Trade Execution Ledger")

trade_rows = data[data['Position_Change'] != 0].copy()

if not trade_rows.empty:
    def signal_label(val):
        if val > 0:
            return '🟢 LONG (Call/Buy)'
        elif val < 0:
            return '🔴 SHORT (Put/Sell)'
        return '⬜ EXIT'

    trade_rows['Signal']              = trade_rows['Position_Change'].apply(signal_label)
    trade_rows['Exec Price (₹)']      = close.loc[trade_rows.index].round(2)
    trade_rows['Portfolio Value (₹)'] = trade_rows['Portfolio_Value'].round(2)
    trade_rows['Drawdown (%)']        = trade_rows['Drawdown'].round(2)

    log = trade_rows[['Signal', 'Exec Price (₹)', 'Portfolio Value (₹)', 'Drawdown (%)']].copy()

    # Format timestamps in IST
    try:
        log.index = log.index.strftime('%d %b %Y  %I:%M %p')
    except Exception:
        log.index = log.index.strftime('%d %b %Y')

    log.index.name = 'Execution Time (IST)'

    st.dataframe(
        log,
        use_container_width=True,
        column_config={
            "Signal"              : st.column_config.TextColumn("Signal"),
            "Exec Price (₹)"      : st.column_config.NumberColumn("Exec Price",      format="₹ %.2f"),
            "Portfolio Value (₹)" : st.column_config.NumberColumn("Portfolio Value", format="₹ %.2f"),
            "Drawdown (%)"        : st.column_config.NumberColumn("Drawdown",        format="%.2f %%"),
        }
    )
else:
    st.markdown(
        '<div class="info-box">ℹ️ No trade events to display in the selected range.</div>',
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#8b949e; font-size:12px;'>"
    "⚡ Algo Factory Backtest Hub V3 &nbsp;·&nbsp; "
    "Data: Yahoo Finance &nbsp;·&nbsp; "
    "Timezone: IST (Asia/Kolkata) &nbsp;·&nbsp; "
    "For Research Only — Not Financial Advice"
    "</center>",
    unsafe_allow_html=True
)
