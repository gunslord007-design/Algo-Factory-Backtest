"""
app.py — Main Streamlit Interface for Algo Factory V4
======================================================
Visual front-end connecting to the engine modules.
Features an institutional tabbed layout and Midnight styling.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import date, timedelta

# Engine imports
from engine.data_loader import (
    fetch_stock_data, get_stocks_by_sector,
    validate_ma_config, SECTOR_MAP, TIMEFRAME_OPTIONS
)
from engine.indicators import get_indicator, calculate_rsi
from engine.strategy import run_backtest
from engine.analytics import compute_full_analytics
from engine.optimizer import run_optimization
from engine.export import generate_csv, generate_pdf_report

# ── PAGE CONFIGURATION ──
st.set_page_config(
    page_title="Algo Factory Terminal V4",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── PREMIUM "MIDNIGHT" CSS ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0B0E14;
    }
    
    /* Fix global text color for contrast */
    h1, h2, h3, p, span, div {
        color: #E6EDF3 !important;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: rgba(30, 35, 41, 0.6);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        border: 1px solid #2B3139;
        text-align: center;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #3772FF;
    }
    .metric-title {
        color: #8B949E !important;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    .metric-value {
        color: #FFFFFF !important;
        font-size: 28px;
        font-weight: 700;
    }
    .metric-positive { color: #2EA043 !important; text-shadow: 0 0 10px rgba(46,160,67,0.3); }
    .metric-negative { color: #F85149 !important; text-shadow: 0 0 10px rgba(248,81,73,0.3); }
    
    /* Subdued Streamlit Buttons */
    .stButton>button {
        background-color: #238636;
        color: white !important;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #2EA043;
        box-shadow: 0 0 15px rgba(46,160,67,0.4);
    }
</style>
""", unsafe_allow_html=True)


# ── CACHED DATA FETCHING ──
@st.cache_data(show_spinner=False, ttl=3600)
def load_data(ticker, start, end, interval):
    return fetch_stock_data(ticker, start, end, interval)


# ── SIDEBAR: SETUP ──
st.sidebar.title("⚡ Algo Factory V4")
st.sidebar.markdown("---")

st.sidebar.header("1. Data Feed")
selected_sector = st.sidebar.selectbox("Sector Filter", list(SECTOR_MAP.keys()))
available_stocks = get_stocks_by_sector(selected_sector)
stock_names = list(available_stocks.keys())
selected_stock_name = st.sidebar.selectbox("Select Asset", stock_names)
ticker = available_stocks[selected_stock_name]

col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start", date.today() - timedelta(days=365))
end_date = col2.date_input("End", date.today())

tf_label = st.sidebar.selectbox("Timeframe", list(TIMEFRAME_OPTIONS.keys()), index=6)
interval = TIMEFRAME_OPTIONS[tf_label]["interval"]

st.sidebar.markdown("---")
st.sidebar.header("2. Strategy Setup")
strategy_mode = st.sidebar.selectbox("Strategy Mode", ["MA Crossover", "RSI Only", "MA + RSI Combined"], help="Choose how the engine triggers trades. 'MA Crossover' uses only moving averages. 'RSI Only' ignores moving averages. 'MA + RSI' requires BOTH indicators to agree before taking a trade.")
direction = st.sidebar.radio("Direction", ["Long Only", "Short Only", "Both"], index=0, horizontal=True)

if "MA" in strategy_mode:
    ma_type = st.sidebar.selectbox("Indicator Type", ["SMA", "EMA", "DEMA", "WMA", "HMA", "VWAP"])
    col3, col4 = st.sidebar.columns(2)
    
    # Surgical fix for Streamlit Numpad Bug: Using text_input and piping through an int converter
    fast_str = col3.text_input("Fast Length", value="12")
    try:
        fast_len = max(1, min(500, int(fast_str)))
    except ValueError:
        fast_len = 12
        
    slow_str = col4.text_input("Slow Length", value="65")
    try:
        slow_len = max(2, min(500, int(slow_str)))
    except ValueError:
        slow_len = 65
else:
    ma_type = "SMA"
    fast_len = 9
    slow_len = 21

if "RSI" in strategy_mode:
    st.sidebar.markdown("---")
    st.sidebar.header("3. RSI Settings")
    rsi_len = st.sidebar.number_input("RSI Length", min_value=2, max_value=100, value=14, help="Standard is 14. Lower numbers make RSI highly sensitive and erratic. Higher numbers make it slower and smoother.")

    rsi_buy_rule = st.sidebar.selectbox("RSI Buy Logic", [
        "Crosses Below Oversold", 
        "Crosses Above Oversold", 
        "Crosses Above Midline (50)"
    ], help="Mean Reversion: Buys exactly when the stock crashes below Oversold (falling knife). Momentum Bounce: Waits for crash, but buys when it bounces ABOVE Oversold (safer). Trend Following: Buys when RSI crosses above 50 (bullish trend).")

    rsi_sell_rule = st.sidebar.selectbox("RSI Sell Logic", [
        "Crosses Above Overbought", 
        "Crosses Below Overbought", 
        "Crosses Below Midline (50)"
    ], help="Mean Reversion: Sells exactly when crossing above Overbought. Momentum Drop: Waits to cross back below Overbought. Trend Reversal: Sells when RSI drops below 50.")

    col5, col6 = st.sidebar.columns(2)
    rsi_lower = col5.slider("Oversold Threshold", 0, 50, 30, help="Standard is 30. Any number below this is considered 'cheap' or oversold.")
    rsi_upper = col6.slider("Overbought Threshold", 50, 100, 70, help="Standard is 70. Any number above this is considered 'expensive' or overbought.")
else:
    rsi_len = 14
    rsi_buy_rule = "Crosses Below Oversold"
    rsi_sell_rule = "Crosses Above Overbought"
    rsi_lower = 30
    rsi_upper = 70

st.sidebar.markdown("---")
st.sidebar.header("4. Risk Management (TSL)")
tsl_enabled = st.sidebar.checkbox("Enable Trailing Stop-Loss", value=False, help="Overrides indicator signals if the asset price drops (or rises) by the specified percentage from its peak/trough.")
tsl_pct = st.sidebar.number_input("TSL Percentage (%)", min_value=0.1, max_value=50.0, value=4.0, step=0.1, help="The trailing percentage drop/rise required to trigger the stop loss.")

st.sidebar.markdown("---")
st.sidebar.header("5. Volume Filter")
vol_filter_enabled = st.sidebar.checkbox("Enable Relative Volume (RVOL) Filter", value=False, help="Only takes trades if current volume is strictly higher than the moving average volume.")
col7, col8 = st.sidebar.columns(2)
vol_lookback = col7.number_input("RVOL Lookback", min_value=5, max_value=100, value=20, help="The period used to calculate the average volume.")
vol_threshold = col8.number_input("Min RVOL", min_value=0.5, max_value=5.0, value=1.5, step=0.1, help="e.g. 1.5 means volume must be 150% of the average volume to take a trade.")

st.sidebar.markdown("---")
st.sidebar.header("6. Capital & Costs")
capital = st.sidebar.number_input("Initial Capital (Rs)", min_value=1000, value=100000, step=10000)
brokerage = st.sidebar.slider("Brokerage per Trade (Rs)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)

st.sidebar.markdown("---")
st.sidebar.header("7. Execution Settings")
execution_mode = st.sidebar.radio("Execution Timing", ["Same Bar Close", "Next Bar Open"], index=0, horizontal=True, help="'Same Bar Close': Execute at the close of the signal candle. 'Next Bar Open': Execute at the open of the NEXT candle (more realistic for live trading).")


# ── MAIN TERMINAL ──
with st.expander("🔥 UPCOMING UPDATE: ALGO MODEL HINDI SUMMARY", expanded=True):
    st.markdown("""
    ### 🚨 **एक नया मॉडल आ रहा है! (New Model is Coming!)**
    हम एक बिलकुल नया, एडवांस्ड AI-संचालित मॉडल (Alpha Discovery Engine) लेकर आ रहे हैं जो पुराने इंडिकेटर्स से कहीं आगे की सोचता है। यह मॉडल न्यूज़ और मैक्रो-डेटा को एनालाइज़ करके आपको मार्केट की चाल पहले से ही भांपने में मदद करेगा।
    
    ---

    > **नोट (Note):** यह हमारे नए मॉडल का बहुत ही सरल और व्यावहारिक सारांश है, जिसमें किसी भी जटिल फॉर्मूले का उपयोग नहीं किया गया है।

    ### 🛠️ हमारा नया मॉडल कैसे काम करता है?
    हमारा **Alpha Discovery Engine** दो मुख्य हिस्सों से मिलकर बना है: 
    1. **Numerical Machine Learning Model** (जो पूरी तरह तैयार है)
    2. **NLP News-Driven Engine** (आगामी अपडेट)

    #### 1. Numerical Machine Learning Model (संख्यात्मक AI मॉडल)
    आमतौर पर ट्रेडर्स Moving Averages या RSI देखकर ट्रेड लेते हैं, जो कि "लैगिंग" (पीछे चलने वाले) होते हैं। हमारा AI मॉडल (Random Forest) 50 अलग-अलग इंडिकेटर्स (जैसे India VIX, Crude Oil, US 10Y Yield और Technicals) को एक साथ देखता है और उनके बीच के गहरे पैटर्न्स को समझता है। यह मॉडल न सिर्फ ट्रेंड बताता है बल्कि यह भी समझता है कि कौन सा इंडिकेटर सबसे ज्यादा काम कर रहा है। इसने निफ्टी मिडकैप्स में 70% तक की एक्यूरेसी साबित की है।

    #### 2. NLP News-Driven Engine (न्यूज़ आधारित AI) और Options Logic
    हर न्यूज़ का मार्केट पर अलग असर होता है। इसलिए हमारा आगामी मॉडल न्यूज़ को 7 पिलर्स के आधार पर एनालाइज़ करेगा (जैसे: क्या न्यूज़ की उम्मीद पहले से थी? न्यूज़ किस सोर्स से आई है? न्यूज़ कितनी तेज़ी से फैल रही है?)। 
    इसके साथ ही, मॉडल **Options (FnO) Data** जैसे Max Pain और PCR का इस्तेमाल करेगा। अगर वोलैटिलिटी (IV) बहुत ज़्यादा है, तो मॉडल ऑप्शन खरीदने की बजाय 'Credit Spreads' (ऑप्शन सेलिंग) का इस्तेमाल करेगा ताकि टाइम डिके (Theta) का फायदा मिल सके।

    #### 🔋 आगे का रास्ता (Continuous Learning)
    सोमवार से यह मॉडल **AngelOne WebSocket** के ज़रिए रोज़ाना लाइव डेटा (Tick/Minute data) लेगा। यह हर रोज़ नए डेटा से सीखेगा (Online Incremental Learning) और अपनी प्रेडिक्शन को खुद-ब-खुद और बेहतर बनाएगा।

    **हमसे जुड़े रहें! 🚀 दिन-ब-दिन (day-by-day) हम आपको नई जानकारी और अपडेट्स देते रहेंगे। अगले 5 दिनों में हम इस नए मॉडल के बारे में आगे की पूरी कवरेज और नतीजे आपके साथ साझा करेंगे, इसलिए हमारी प्रोग्रेस पर नज़र बनाए रखें!**
    """)

st.markdown(f"<h2>{selected_stock_name} ({ticker})</h2>", unsafe_allow_html=True)

with st.spinner("Downloading high-speed market data..."):
    fetch_result = load_data(ticker, start_date, end_date, interval)

if not fetch_result["ok"]:
    st.error(f"**Data Error:** {fetch_result['error']}")
    st.info(f"**Hint:** {fetch_result['hint']}")
    st.stop()

df = fetch_result["data"]
st.caption(f"<span style='color:#8B949E;'>{fetch_result['info']}</span>", unsafe_allow_html=True)

val_result = validate_ma_config(len(df), fast_len, slow_len)
if not val_result["ok"]:
    st.error(f"**Configuration Error:** {val_result['error']}")
    st.info(f"**Hint:** {val_result['hint']}")
    st.stop()

try:
    if "MA" in strategy_mode:
        fast_ma = get_indicator(df, ma_type, fast_len)
        slow_ma = get_indicator(df, ma_type, slow_len)
    else:
        fast_ma = pd.Series(0.0, index=df.index)
        slow_ma = pd.Series(0.0, index=df.index)

    if "RSI" in strategy_mode:
        rsi_series = calculate_rsi(df['Close'], rsi_len)
    else:
        rsi_series = pd.Series(50.0, index=df.index)
except Exception as e:
    st.error(f"Error calculating indicators: {e}")
    st.stop()

bt_result = run_backtest(
    df, fast_ma, slow_ma, direction, capital, brokerage, 
    optimize=False, rsi_series=rsi_series, strategy_mode=strategy_mode,
    rsi_buy_rule=rsi_buy_rule, rsi_sell_rule=rsi_sell_rule,
    rsi_upper=rsi_upper, rsi_lower=rsi_lower,
    tsl_enabled=tsl_enabled, tsl_pct=tsl_pct,
    execution_mode=execution_mode,
    vol_filter_enabled=vol_filter_enabled,
    vol_lookback=vol_lookback,
    vol_threshold=vol_threshold,
    max_lookback=max(fast_len, slow_len)
)

if not bt_result["ok"]:
    st.error(f"**Backtest Error:** {bt_result['error']}")
    if bt_result['hint']: st.info(bt_result['hint'])
    st.stop()

result_df = bt_result["data"]
trades = bt_result["trades"]
analytics = compute_full_analytics(result_df, trades, capital, interval)

# ── TABBED ARCHITECTURE ──
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Terminal", "📈 Deep Analytics", "📋 Trade Book", "⚡ Optimizer", "🔮 Future Test"])

def metric_card(title, value, color_class=""):
    return f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value {color_class}">{value}</div>
    </div>
    """

# ==========================================
# TAB 1: TERMINAL (Dashboard & Charts)
# ==========================================
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    ret = analytics['returns']
    rsk = analytics['risk']
    rts = analytics['ratios']
    trd = analytics['trades']

    c1, c2, c3, c4, c5 = st.columns(5)
    ret_color = "metric-positive" if ret['total_return_pct'] >= 0 else "metric-negative"
    c1.markdown(metric_card("Total Return", f"{ret['total_return_pct']:.2f}%", ret_color), unsafe_allow_html=True)
    c2.markdown(metric_card("CAGR", f"{ret['cagr_pct']:.2f}%"), unsafe_allow_html=True)
    c3.markdown(metric_card("Sharpe Ratio", f"{rts['sharpe_ratio']:.2f}"), unsafe_allow_html=True)
    c4.markdown(metric_card("Max Drawdown", f"{rsk['max_drawdown_pct']:.2f}%", "metric-negative"), unsafe_allow_html=True)
    c5.markdown(metric_card("Win Rate", f"{trd['win_rate']:.1f}%"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    chart_type = st.radio(
        "Chart Display Style",
        ["Candlestick", "Line", "Area", "OHLC"],
        index=0,
        horizontal=True,
        label_visibility="collapsed"
    )

    # ── GROWW-STYLE CHART OVERHAUL ──
    x_ax = df.index
    
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.2]
    )

    # 1. Main Price Trace
    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=x_ax, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price", 
            increasing_line_color='#00B852', increasing_fillcolor='#00B852',
            decreasing_line_color='#FF5000', decreasing_fillcolor='#FF5000', hoverinfo='x+y'
        ), row=1, col=1)
    elif chart_type == "Line":
        fig.add_trace(go.Scatter(x=x_ax, y=df['Close'], mode='lines', name='Price', line=dict(color='#E6EDF3', width=2), hoverinfo='x+y'), row=1, col=1)
    elif chart_type == "Area":
        fig.add_trace(go.Scatter(x=x_ax, y=df['Close'], mode='lines', name='Price', line=dict(color='#00B852', width=2), fill='tozeroy', fillcolor='rgba(0, 184, 82, 0.1)', hoverinfo='x+y'), row=1, col=1)
    elif chart_type == "OHLC":
        fig.add_trace(go.Ohlc(
            x=x_ax, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price",
            increasing_line_color='#00B852', decreasing_line_color='#FF5000', hoverinfo='x+y'
        ), row=1, col=1)

    # Moving Averages
    fig.add_trace(go.Scatter(x=x_ax, y=fast_ma, name=f"Fast {ma_type}", line=dict(color='#3772FF', width=1.5), hoverinfo='y'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_ax, y=slow_ma, name=f"Slow {ma_type}", line=dict(color='#F0B90B', width=1.5), hoverinfo='y'), row=1, col=1)

    # 2. Trade Markers (Precise Arrows)
    for t in trades:
        t_date = t['entry_date']
        e_date = t['exit_date']
        
        if t['direction'] == 'LONG':
            fig.add_trace(go.Scatter(x=[t_date], y=[t['entry_price']], mode='markers', marker=dict(symbol='triangle-up', size=14, color='#00B852', line=dict(color='white', width=1)), name='Buy Entry', hoverinfo='skip'), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=[t_date], y=[t['entry_price']], mode='markers', marker=dict(symbol='triangle-down', size=14, color='#FF5000', line=dict(color='white', width=1)), name='Short Entry', hoverinfo='skip'), row=1, col=1)
            
        exit_reason = t.get('exit_reason', 'Signal')
        if exit_reason == 'TSL':
            fig.add_trace(go.Scatter(x=[e_date], y=[t['exit_price']], mode='markers', marker=dict(symbol='x', size=12, color='#F85149', line=dict(color='#FFFFFF', width=1)), name='TSL Exit', hovertemplate='TSL Exit: %{y}<extra></extra>'), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=[e_date], y=[t['exit_price']], mode='markers', marker=dict(symbol='x', size=10, color='#FFFFFF'), name='Exit', hoverinfo='skip'), row=1, col=1)

    # 3. Volume (Blended Colors)
    if vol_filter_enabled:
        from engine.indicators import calculate_rvol
        rvol = calculate_rvol(df['Volume'], vol_lookback)
        vol_colors = []
        for (i, row), r_val in zip(df.iterrows(), rvol):
            if r_val >= vol_threshold:
                vol_colors.append('#00B852' if row['Open'] <= row['Close'] else '#FF5000')  # Vibrant (Passed)
            else:
                vol_colors.append('#4A4A4A')  # Greyed out (Failed filter)
    else:
        vol_colors = ['#FF5000' if row['Open'] > row['Close'] else '#00B852' for index, row in df.iterrows()]
    fig.add_trace(go.Bar(x=x_ax, y=df['Volume'], marker_color=vol_colors, name="Volume", hoverinfo='y', opacity=0.8), row=2, col=1)

    # 4. RSI Oscillator
    fig.add_trace(go.Scatter(x=x_ax, y=rsi_series, name=f"RSI ({rsi_len})", line=dict(color='#A371F7', width=1.5), hoverinfo='y'), row=3, col=1)
    fig.add_hline(y=rsi_upper, line=dict(color='#F85149', width=1, dash='dash'), row=3, col=1)
    fig.add_hline(y=rsi_lower, line=dict(color='#2EA043', width=1, dash='dash'), row=3, col=1)
    # Add light fill between boundaries (Optional clean styling)
    fig.add_hrect(y0=rsi_lower, y1=rsi_upper, fillcolor="rgba(163,113,247,0.1)", layer="below", line_width=0, row=3, col=1)

    # 5. Equity Curve
    fig.add_trace(go.Scatter(x=x_ax, y=result_df['Portfolio_Value'], name="Strategy Equity", line=dict(color='#3772FF', width=2), fill='tozeroy', fillcolor='rgba(55, 114, 255, 0.1)', hoverinfo='y'), row=4, col=1)
    fig.add_trace(go.Scatter(x=x_ax, y=result_df['BH_Value'], name="Buy & Hold", line=dict(color='#8B949E', width=1.5, dash='dot'), hoverinfo='skip'), row=4, col=1)

    # Layout & Groww Navigation Mechanics
    fig.update_layout(
        height=850,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor='#0B0E14',
        plot_bgcolor='#0B0E14',
        font=dict(color='#8B949E'),
        xaxis_rangeslider_visible=False,
        dragmode='pan',       # Enables dragging left/right naturally
        hovermode='x',        # Cleaner hover box
        showlegend=False      # Removing legend to keep it clean (names are in tooltips)
    )
    
    # Crosshairs & Axis styling
    fig.update_xaxes(
        rangebreaks=[dict(bounds=["sat", "mon"])],  # Hides weekends natively while preserving datetime zooming
        gridcolor='rgba(43,49,57,0.3)', tickfont=dict(color='#8B949E'),
        showspikes=True, spikemode='across', spikethickness=1, spikedash='dot', spikecolor='#E6EDF3'
    )
    fig.update_yaxes(
        fixedrange=False,     # Unlocks Y axis for manual scaling
        gridcolor='rgba(43,49,57,0.3)', zerolinecolor='rgba(43,49,57,0.3)', tickfont=dict(color='#8B949E'),
        showspikes=True, spikemode='across', spikethickness=1, spikedash='dot', spikecolor='#E6EDF3'
    )

    # Injecting the dynamic zoom configs
    chart_config = {
        'scrollZoom': True, 
        'displayModeBar': False,
        'modeBarButtonsToRemove': ['zoomIn2d', 'zoomOut2d', 'autoScale2d']
    }
    st.plotly_chart(fig, use_container_width=True, config=chart_config)

# ==========================================
# TAB 2: DEEP ANALYTICS
# ==========================================
with tab2:
    st.markdown("<br><h3>Hedge-Fund Tearsheet</h3><hr style='border-color:#2B3139;'>", unsafe_allow_html=True)
    
    colA, colB, colC = st.columns(3)
    
    with colA:
        st.markdown("#### Return Metrics")
        st.write(f"**Total Return:** {ret.get('total_return_pct', 0):.2f}%")
        st.write(f"**Benchmark Return:** {ret.get('benchmark_return_pct', 0):.2f}%")
        st.write(f"**Alpha:** {ret.get('alpha_pct', 0):.2f}%")
        st.write(f"**CAGR:** {ret.get('cagr_pct', 0):.2f}%")
        st.write(f"**Final Value:** Rs. {ret.get('final_portfolio_value', 0):,.2f}")
        
    with colB:
        st.markdown("#### Risk & Ratios")
        st.write(f"**Max Drawdown:** {rsk.get('max_drawdown_pct', 0):.2f}%")
        st.write(f"**DD Duration:** {rsk.get('max_drawdown_duration_bars', 0)} bars")
        st.write(f"**Annual Volatility:** {rsk.get('annualized_volatility_pct', 0):.2f}%")
        st.write(f"**Sortino Ratio:** {rts.get('sortino_ratio', 0):.3f}")
        st.write(f"**Calmar Ratio:** {rts.get('calmar_ratio', 0):.3f}")
        
    with colC:
        st.markdown("#### Trade Quality")
        st.write(f"**Total Trades:** {trd.get('total_trades', 0)}")
        st.write(f"**Win / Loss:** {trd.get('winning_trades', 0)} / {trd.get('losing_trades', 0)}")
        st.write(f"**Profit Factor:** {trd.get('profit_factor', 0):.3f}")
        st.write(f"**Average Win:** Rs. {trd.get('avg_win', 0):.2f}")
        st.write(f"**Average Loss:** Rs. {trd.get('avg_loss', 0):.2f}")

# ==========================================
# TAB 3: TRADE BOOK
# ==========================================
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    if trades:
        col_exp1, col_exp2 = st.columns([1, 1])
        ma_config_str = f"{ma_type} {fast_len}/{slow_len} ({direction})"
        
        with col_exp1:
            csv_data = generate_csv(trades)
            st.download_button("💾 Download CSV Trade Log", data=csv_data, file_name=f"{ticker}_trades.csv", mime="text/csv", use_container_width=True)
        
        with col_exp2:
            with st.spinner("Generating PDF..."):
                pdf_bytes = generate_pdf_report(analytics, trades, selected_stock_name, ma_config_str)
                # FIX: Explicitly cast to bytes to prevent StreamlitAPIException
                pdf_bytes_clean = bytes(pdf_bytes) 
            st.download_button("📄 Download PDF Tearsheet", data=pdf_bytes_clean, file_name=f"{ticker}_Report.pdf", mime="application/pdf", use_container_width=True)

        tdf = pd.DataFrame(trades)
        display_cols = ['direction', 'entry_date', 'entry_price', 'exit_date', 'exit_price', 'exit_reason', 'brokerage', 'net_pnl', 'return_pct', 'holding_bars']
        exist_cols = [c for c in display_cols if c in tdf.columns]
        format_dict = {'entry_price': 'Rs {:.2f}', 'exit_price': 'Rs {:.2f}', 'brokerage': 'Rs {:.2f}', 'net_pnl': 'Rs {:.2f}', 'return_pct': '{:.2f}%'}
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(tdf[exist_cols].style.format(format_dict), use_container_width=True, height=600)
    else:
        st.info("No trades executed with current parameters.")

# ==========================================
# TAB 4: OPTIMIZER
# ==========================================
with tab4:
    st.markdown("<br><h3>Strategy Optimizer</h3><hr style='border-color:#2B3139;'>", unsafe_allow_html=True)
    
    col_op1, col_op2 = st.columns([1, 3])
    with col_op1:
        st.markdown("Find the optimal parameters for maximum return.")
        # Dynamic opt_target based on strategy_mode
        opt_options = []
        if "MA" in strategy_mode:
            opt_options.append("Optimize MAs")
        if "RSI" in strategy_mode:
            opt_options.append("Optimize RSI")
        if tsl_enabled:
            opt_options.append("Optimize Strategy + TSL (3D)")
            opt_options.append("Optimize TSL Only (1D)")
            
        opt_target = st.radio("Optimization Target", opt_options, help="MA scan keeps RSI fixed. RSI scan keeps MAs fixed. 3D scan optimizes Strategy + TSL simultaneously.")
        opt_mode = st.radio("Scan Mode", ["Fast Scan", "Deep Scan"])
        
        # Clear session state if it doesn't match the current strategy_mode
        if 'opt_strategy_mode' not in st.session_state or st.session_state['opt_strategy_mode'] != strategy_mode:
            if 'opt_result' in st.session_state:
                del st.session_state['opt_result']
            st.session_state['opt_strategy_mode'] = strategy_mode
        if st.button("Run Grid Search", use_container_width=True):
            mode_str = "fast" if "Fast" in opt_mode else "deep"
            with st.spinner("Running Grid Search..."):
                prog_bar = st.progress(0)
                def update_prog(curr, tot):
                    prog_bar.progress(curr / tot)
                    
                opt_res = run_optimization(
                    df, ma_type, mode_str, capital, brokerage, direction=direction, progress_callback=update_prog,
                    opt_target=opt_target, strategy_mode=strategy_mode, rsi_series=rsi_series,
                    fast_ma=fast_ma, slow_ma=slow_ma, rsi_buy_rule=rsi_buy_rule, rsi_sell_rule=rsi_sell_rule,
                    rsi_lower=rsi_lower, rsi_upper=rsi_upper,
                    tsl_enabled=tsl_enabled, tsl_pct=tsl_pct,
                    execution_mode=execution_mode,
                    vol_filter_enabled=vol_filter_enabled,
                    vol_lookback=vol_lookback,
                    vol_threshold=vol_threshold
                )
                
                if opt_res["ok"]:
                    st.session_state['opt_result'] = opt_res
                    st.success("Optimization Complete!")
                else:
                    st.error(opt_res["error"])

    with col_op2:
        if 'opt_result' in st.session_state:
            res = st.session_state['opt_result']
            b = res['best']
            
            if res.get('y_range') and len(res['y_range']) > 0 and 'fast' not in b:
                # New Split Optimizer logic
                base_opt = res.get('base_opt', 'Optimize MAs')
                
                if base_opt == "Optimize TSL Only (1D)":
                    st.markdown(f"🏆 **Best Combination found:** TSL = `{b['z_val']}%` ➔ **Return: {b['return']}%** (Sharpe: {b['sharpe']})")
                    fig_hm = px.line(
                        x=res['z_range'],
                        y=res['return_matrix'],
                        labels={"x": "TSL Percentage (%)", "y": "Total Return (%)"}
                    )
                    fig_hm.update_traces(
                        hovertemplate="TSL: %{x}%<br>Return: %{y:.2f}%<extra></extra>",
                        line=dict(color="#00B852", width=3),
                        fill='tozeroy',
                        fillcolor='rgba(0, 184, 82, 0.1)'
                    )
                else:
                    y_label = "Fast Length" if "MAs" in base_opt else "RSI Length"
                    x_label = "Slow Length" if "MAs" in base_opt else "Oversold Threshold"
                    
                    is_3d_result = ('z_range' in res and len(res['z_range']) > 1)
                    
                    if is_3d_result:
                        st.markdown(f"🏆 **Best Combination found:** {y_label} = `{b['y_val']}`, {x_label} = `{b['x_val']}`, TSL = `{b['z_val']}%` ➔ **Return: {b['return']}%** (Sharpe: {b['sharpe']})")
                        st.caption(f"Displaying 2D Heatmap slice locked at optimal TSL = {b['z_val']}%")
                        
                        z_idx = res['z_range'].index(b['z_val'])
                        z = res['return_matrix'][:, :, z_idx]
                    else:
                        st.markdown(f"🏆 **Best Combination found:** {y_label} = `{b['y_val']}`, {x_label} = `{b['x_val']}` ➔ **Return: {b['return']}%** (Sharpe: {b['sharpe']})")
                        z = res['return_matrix']
                    
                    # Heatmap
                    fig_hm = px.imshow(
                        z,
                        labels=dict(x=x_label, y=y_label, color="Return %"),
                        x=res['x_range'],
                        y=res['y_range'],
                        color_continuous_scale='RdYlGn',
                        aspect="auto"
                    )
                    
                    fig_hm.update_traces(
                        hovertemplate=f"{y_label}: %{{y}} | {x_label}: %{{x}}<br><b>Return: %{{z:.2f}}%</b><extra></extra>",
                        hoverongaps=False
                    )
            else:
                st.markdown(f"🏆 **Legacy Combination found** ➔ **Return: {b['return']}%**")
                st.warning("Please re-run the Grid Search to update the heatmap.")
                fig_hm = go.Figure()
            
            fig_hm.update_layout(
                height=500, 
                margin=dict(l=0, r=0, t=10, b=0), 
                paper_bgcolor='#0B0E14', 
                plot_bgcolor='#0B0E14',
                font=dict(color='#E6EDF3')
            )
            fig_hm.update_xaxes(gridcolor='rgba(43,49,57,0.5)', zerolinecolor='rgba(0,0,0,0)')
            fig_hm.update_yaxes(gridcolor='rgba(43,49,57,0.5)', zerolinecolor='rgba(0,0,0,0)')
            
            st.plotly_chart(fig_hm, use_container_width=True)
        else:
            st.info("Run the optimizer to view the heatmap matrix.")

# ─────────────────────────────────────────────────────────────
#  TAB 5: FUTURE TESTING (WALK-FORWARD OPTIMIZATION)
# ─────────────────────────────────────────────────────────────
with tab5:
    st.markdown("### 🔮 Institutional Future Testing (WFO)")
    st.markdown("Prove your strategy survives the future by simulating a rolling walk-forward optimization.")
    
    with st.expander("📖 What is Walk-Forward Optimization? (Simple Explanation)"):
        st.markdown("""
        **The Problem with normal Backtesting (Curve Fitting)**
        If you let the computer look at all 5 years of data at once, it acts like a student who memorized the answer key. It will perfectly match the past, but usually fails in the real world tomorrow because it didn't learn true market logic.
        
        **The Solution: The Blind Test (Walk-Forward)**
        Walk-Forward Optimization forces the algorithm to prove its worth by taking the test **blind**, just like real trading.
        
        *   **Step 1 (The Training):** We hide the year 2024. We tell the engine to find the best parameters using only 2020-2023. Let's say it finds the 15 & 40 MA.
        *   **Step 2 (The Blind Test):** We lock the strategy at 15 & 40 MA, and force it to trade on 2024 (data it has never seen).
        
        **Rolling vs Expanding Windows**
        We do this multiple times to simulate a trader re-optimizing their system every year:
        *   **Expanding (Anchored):** The algorithm always remembers everything from the very beginning of your data. It builds long-term memory.
        *   **Rolling (Unanchored):** The algorithm forgets the distant past and only trains on the most recent chunk of data before the blind test. It adapts to new regimes.
        
        Finally, we literally glue all the blind trading years together into one massive master sequence. **The Equity Curve you see below is a pure simulation of you trading completely blind into the future.**
        """)
    
    col_wfo1, col_wfo2, col_wfo3 = st.columns(3)
    with col_wfo1:
        wfo_train_size = st.number_input("Training Size (Bars)", min_value=10, max_value=5000, value=500, help="Exact number of bars to use for historical training.")
    with col_wfo2:
        wfo_test_size = st.number_input("Blind Test Size (Bars)", min_value=5, max_value=2000, value=100, help="Exact number of bars to step forward and trade blindly.")
    with col_wfo3:
        wfo_window_mode = st.selectbox("Window Mode", ["Expanding (Anchored)", "Rolling (Unanchored)"], help="Expanding trains from the beginning. Rolling trains on a fixed trailing chunk.")

    col_wfo4, col_wfo5, col_wfo6 = st.columns(3)
    with col_wfo4:
        wfo_opt_target = st.selectbox("WFO Target", ["Optimize MAs", "Optimize TSL Only (1D)", "No Optimization (Use Sidebar MAs)"], help="What the engine should optimize in the past.")
    with col_wfo5:
        wfo_opt_mode = st.selectbox("Grid Resolution", ["Fast Scan (5 increments)", "Deep Scan (1 increments)"], help="Deep Scan takes much longer but finds mathematically perfect numbers.")
    with col_wfo6:
        st.markdown("<br>", unsafe_allow_html=True)
        run_wfo = st.button("🚀 Run Walk-Forward", use_container_width=True)

    st.markdown("---")
    
    # ── VISUAL WFO TIMELINE (GANTT CHART) ──
    if len(df) > 0:
        st.markdown("### 🗺️ Visual WFO Timeline")
        st.markdown("This map shows exactly how the engine will split your data. **Blue** is the Training (In-Sample) period where the engine learns. **Orange** is the Blind Test (Out-Of-Sample) period where it trades blindly.")
        
        try:
            import plotly.express as px
            
            # Calculate boundaries based on explicit train/test sizes
            timeline_total_bars = len(df)
            if timeline_total_bars < wfo_train_size + wfo_test_size:
                st.warning("Not enough data to run WFO. Increase your data range or decrease the Train/Test sizes.")
                st.stop()
                
            wfo_windows = max(1, (timeline_total_bars - wfo_train_size) // wfo_test_size)
            remainder = (timeline_total_bars - wfo_train_size) % wfo_test_size
            
            timeline_data = []
            
            for i in range(wfo_windows):
                t_end_idx = wfo_train_size + remainder + (i * wfo_test_size)
                t_test_start = t_end_idx
                t_test_end = t_test_start + wfo_test_size
                
                if wfo_window_mode == "Expanding (Anchored)":
                    t_start_idx = 0
                else:
                    t_start_idx = max(0, t_end_idx - wfo_train_size)
                    
                t_start_date = df.index[t_start_idx]
                t_end_date = df.index[t_end_idx - 1] if t_end_idx > 0 else df.index[0]
                t_test_start_date = df.index[t_test_start]
                t_test_end_date = df.index[t_test_end - 1] if t_test_end > 0 else df.index[0]
                
                timeline_data.append({
                    "Window": f"Window {i+1}",
                    "Phase": "Training (In-Sample)",
                    "Start": t_start_date,
                    "End": t_end_date
                })
                timeline_data.append({
                    "Window": f"Window {i+1}",
                    "Phase": "Blind Test (Out-of-Sample)",
                    "Start": t_test_start_date,
                    "End": t_test_end_date
                })
                
            tl_df = pd.DataFrame(timeline_data)
            fig_tl = px.timeline(
                tl_df, 
                x_start="Start", 
                x_end="End", 
                y="Window", 
                color="Phase", 
                color_discrete_map={"Training (In-Sample)": "#1f77b4", "Blind Test (Out-of-Sample)": "#ff7f0e"}
            )
            fig_tl.update_yaxes(autorange="reversed")
            fig_tl.update_layout(height=250 + (wfo_windows * 30), margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_tl, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render timeline: {e}")

    st.markdown("---")
    
    wfo_results_container = st.container()
    
    with wfo_results_container:
        if run_wfo:
            with st.spinner("Executing Institutional Walk-Forward Optimization... (This may take a moment)"):
                total_bars = len(df)
                if total_bars < wfo_train_size + wfo_test_size:
                    st.error("Not enough data to run WFO. Increase your data range or decrease the Train/Test sizes.")
                    st.stop()
                    
                wfo_windows = max(1, (total_bars - wfo_train_size) // wfo_test_size)
                remainder = (total_bars - wfo_train_size) % wfo_test_size
                
                master_oos_trades = []
                master_oos_equity = []
                wfo_history = []
                
                progress_bar = st.progress(0)
                
                for i in range(wfo_windows):
                    train_end = wfo_train_size + remainder + (i * wfo_test_size)
                    test_start = train_end
                    test_end = test_start + wfo_test_size
                    
                    if wfo_window_mode == "Expanding (Anchored)":
                        train_start = 0
                    else:
                        train_start = max(0, train_end - wfo_train_size)
                    
                    df_in = df.iloc[train_start:train_end]
                    df_out = df.iloc[test_start:test_end]
                    
                    # 1. Blind Optimization on df_in
                    actual_mode = "deep" if "Deep" in wfo_opt_mode else "fast"
                    
                    if wfo_opt_target == "Optimize MAs":
                        max_lookback = 200
                    else:
                        max_lookback = max(fast_len, slow_len)
                        
                    if len(df_in) < max_lookback + 10:
                        st.error(f"**Data Starvation Error in Window {i+1}:** Training data ({len(df_in)} bars) is too small to calculate a {max_lookback}-period moving average. Please use a larger dataset (change Start Date) or a smaller timeframe (like 15m).")
                        st.stop()
                    
                    if wfo_opt_target == "No Optimization (Use Sidebar MAs)":
                        in_bt = run_backtest(
                            df_in, fast_ma.iloc[train_start:train_end], slow_ma.iloc[train_start:train_end],
                            direction, capital, brokerage, optimize=True,
                            rsi_series=rsi_series.iloc[train_start:train_end] if rsi_series is not None else None,
                            strategy_mode=strategy_mode, rsi_buy_rule=rsi_buy_rule, rsi_sell_rule=rsi_sell_rule,
                            rsi_upper=rsi_upper, rsi_lower=rsi_lower,
                            tsl_enabled=tsl_enabled, tsl_pct=tsl_pct, execution_mode=execution_mode,
                            vol_filter_enabled=vol_filter_enabled, vol_lookback=vol_lookback,
                            vol_threshold=vol_threshold, max_lookback=max_lookback
                        )
                        final_val = in_bt["data"]["Portfolio_Value"].iloc[-1] if in_bt["ok"] and in_bt["data"] is not None else capital
                        ret = ((final_val - capital) / capital) * 100
                        opt_res = {
                            "ok": True,
                            "best": {
                                "y_val": fast_len,
                                "x_val": slow_len,
                                "z_val": tsl_pct if tsl_enabled else None,
                                "return": ret
                            }
                        }
                    else:
                        opt_res = run_optimization(
                            df_in, ma_type=ma_type, mode=actual_mode, initial_capital=capital,
                            brokerage_per_trade=brokerage, direction=direction,
                            opt_target=wfo_opt_target, strategy_mode=strategy_mode,
                            fast_ma=fast_ma.iloc[train_start:train_end], slow_ma=slow_ma.iloc[train_start:train_end],
                            rsi_series=rsi_series.iloc[train_start:train_end], rsi_lower=rsi_lower, rsi_upper=rsi_upper,
                            tsl_enabled=tsl_enabled, tsl_pct=tsl_pct, execution_mode=execution_mode,
                            vol_filter_enabled=vol_filter_enabled, vol_lookback=vol_lookback, vol_threshold=vol_threshold,
                            max_lookback=max_lookback
                        )
                    
                    if not opt_res["ok"]:
                        st.error(f"Optimization failed in Window {i+1}: {opt_res['error']}")
                        st.stop()
                        
                    b = opt_res['best']
                    best_fast = b.get('y_val', fast_len) if wfo_opt_target == "Optimize MAs" else fast_len
                    best_slow = b.get('x_val', slow_len) if wfo_opt_target == "Optimize MAs" else slow_len
                    best_tsl = b.get('z_val', tsl_pct)
                    
                    is_days = (df_in.index[-1] - df_in.index[0]).days
                    is_days = max(1, is_days)
                    is_cagr = (((1 + b['return']/100.0) ** (365.0 / is_days)) - 1) * 100.0
                    
                    # 2. Blind Execution on df_out
                    from engine.indicators import get_indicator
                    if "MA" in strategy_mode and wfo_opt_target == "Optimize MAs":
                        oos_fast_ma = get_indicator(df, ma_type, best_fast)
                        oos_slow_ma = get_indicator(df, ma_type, best_slow)
                    else:
                        oos_fast_ma = fast_ma
                        oos_slow_ma = slow_ma
                        
                    out_bt = run_backtest(
                        df_out, oos_fast_ma.loc[df_out.index], oos_slow_ma.loc[df_out.index],
                        direction, capital, brokerage, optimize=False,
                        rsi_series=rsi_series.loc[df_out.index], strategy_mode=strategy_mode,
                        rsi_buy_rule=rsi_buy_rule, rsi_sell_rule=rsi_sell_rule,
                        rsi_upper=rsi_upper, rsi_lower=rsi_lower,
                        tsl_enabled=tsl_enabled if best_tsl is None else True, 
                        tsl_pct=best_tsl if best_tsl is not None else tsl_pct,
                        execution_mode=execution_mode,
                        vol_filter_enabled=vol_filter_enabled, vol_lookback=vol_lookback, vol_threshold=vol_threshold
                    )
                    
                    oos_cagr_window = 0.0
                    if out_bt["ok"]:
                        master_oos_trades.extend(out_bt["trades"])
                        master_oos_equity.append(out_bt["data"]['Strategy_Return'])
                        
                        window_ret = out_bt["data"]["Strategy_Return"]
                        if len(window_ret) > 0:
                            oos_capital_window = capital
                            for ret in window_ret:
                                oos_capital_window *= (1 + ret)
                            
                            oos_window_days = max(1, (window_ret.index[-1] - window_ret.index[0]).days)
                            oos_cagr_window = (((oos_capital_window / capital) ** (365.0 / oos_window_days)) - 1) * 100.0

                    is_cagr_val = is_cagr if isinstance(is_cagr, (int, float)) else 0.0
                    oos_cagr_val = oos_cagr_window
                    
                    efficiency = 0.0
                    if is_cagr_val > 0:
                        efficiency = (oos_cagr_val / is_cagr_val) * 100.0

                    wfo_history.append({
                        'Window': f"#{i+1}",
                        'Training Dates': f"{df_in.index[0].strftime('%Y-%m-%d')} to {df_in.index[-1].strftime('%Y-%m-%d')}",
                        'Blind Test Dates': f"{df_out.index[0].strftime('%Y-%m-%d')} to {df_out.index[-1].strftime('%Y-%m-%d')}",
                        'Best Fast': "N/A" if wfo_opt_target == "Optimize TSL Only (1D)" else best_fast,
                        'Best Slow': "N/A" if wfo_opt_target == "Optimize TSL Only (1D)" else best_slow,
                        'Best TSL': "N/A" if wfo_opt_target == "Optimize MAs" else (f"{best_tsl}%" if best_tsl else "OFF"),
                        'Training Profit (Yearly)': f"{round(is_cagr_val, 2)}%",
                        'Blind Test Profit (Yearly)': f"{round(oos_cagr_val, 2)}%",
                        'OOS Efficiency': f"{round(efficiency, 2)}%",
                        '_is_cagr_raw': is_cagr_val,
                        '_oos_cagr_raw': oos_cagr_val,
                        '_efficiency_raw': efficiency
                    })
                    
                    progress_bar.progress((i + 1) / wfo_windows)
                    
                st.session_state['wfo_history'] = wfo_history
                st.session_state['wfo_trades'] = master_oos_trades
                st.session_state['wfo_returns'] = pd.concat(master_oos_equity) if master_oos_equity else pd.Series()
                st.session_state['wfo_avg_is_cagr'] = sum([h['_is_cagr_raw'] for h in wfo_history]) / wfo_windows if wfo_windows > 0 else 0
                st.success("Walk-Forward Optimization Complete!")
                
        # ── RENDER DASHBOARDS (STEP 4) ──
        if 'wfo_history' in st.session_state and len(st.session_state['wfo_history']) > 0:
            hist = st.session_state['wfo_history']
            master_trades = st.session_state['wfo_trades']
            master_returns = st.session_state['wfo_returns']
            avg_is_cagr = st.session_state.get('wfo_avg_is_cagr', 0.0)
            
            # Reconstruct Equity Curve
            oos_capital = capital
            oos_equity_curve = [oos_capital]
            for ret in master_returns:
                oos_capital = oos_capital * (1 + ret)
                oos_equity_curve.append(oos_capital)
                
            if len(master_returns) > 0:
                oos_days = (master_returns.index[-1] - master_returns.index[0]).days
                oos_days = max(1, oos_days)
                oos_cagr = (((oos_equity_curve[-1] / capital) ** (365.0 / oos_days)) - 1) * 100.0
            else:
                oos_cagr = 0.0
            
            # ── MASTER OOS METRICS ──
            if len(master_returns) > 0:
                import numpy as np
                mean_ret = master_returns.mean()
                std_ret = master_returns.std()
                oos_sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0.0
                
                cum_returns = (1 + master_returns).cumprod()
                peak = cum_returns.cummax()
                drawdown = (cum_returns - peak) / peak
                oos_max_dd = drawdown.min() * 100
                
                if len(master_trades) > 0:
                    win_rate = len([t for t in master_trades if t['net_pnl'] > 0]) / len(master_trades) * 100
                else:
                    win_rate = 0.0
            else:
                oos_sharpe = 0.0
                oos_max_dd = 0.0
                win_rate = 0.0
                
            wfe = (oos_cagr / avg_is_cagr) * 100 if avg_is_cagr > 0 else 0.0
            
            # ── ROBUSTNESS GRADE LOGIC ──
            if wfe >= 70 and oos_cagr > 0 and oos_max_dd >= -25:
                grade = "A"
                grade_color = "#00B852"
                grade_desc = "Highly robust. The strategy survives blind testing exceptionally well and is a strong candidate for live trading."
            elif wfe >= 40 and oos_cagr > 0:
                grade = "B"
                grade_color = "#00aaff"
                grade_desc = "Good robustness. It holds up in blind testing, though expect some performance degradation compared to backtests."
            elif wfe >= 10 and oos_cagr > 0:
                grade = "C"
                grade_color = "#f4a261"
                grade_desc = "Marginal. The strategy barely survived the blind test. High risk of breaking down in live markets."
            else:
                grade = "F"
                grade_color = "#ff4b4b"
                grade_desc = "Overfitted / Curve-fit. Do NOT trade this live. The strategy completely fails when exposed to unseen data."

            from engine.analytics import run_monte_carlo
            mc_worst_dd = run_monte_carlo(master_trades, capital, iterations=1000)
            
            st.markdown("### 1. Parameter Stability & WFO History")
            display_hist = pd.DataFrame(hist).drop(columns=['_is_cagr_raw', '_oos_cagr_raw', '_efficiency_raw'], errors='ignore')
            st.dataframe(display_hist, use_container_width=True)
            
            st.markdown("### 2. Institutional Robustness Metrics")
            
            # Render Grade Box
            st.markdown(f"""
            <div style="background-color: rgba(0,0,0,0.2); border-left: 5px solid {grade_color}; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                <h3 style="margin: 0; color: {grade_color};">Robustness Grade: {grade}</h3>
                <p style="margin: 5px 0 0 0; color: #a1a1aa;">{grade_desc}</p>
            </div>
            """, unsafe_allow_html=True)

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                wfe_color = "metric-positive" if wfe > 50 else "metric-negative"
                st.markdown(metric_card("Walk-Forward Efficiency", f"{wfe:.1f}%", wfe_color), unsafe_allow_html=True)
            with col_m2:
                st.markdown(metric_card("Avg Training Profit", f"{avg_is_cagr:.1f}%", "metric-positive" if avg_is_cagr>0 else "metric-negative"), unsafe_allow_html=True)
            with col_m3:
                st.markdown(metric_card("Blind Test Profit", f"{oos_cagr:.1f}%", "metric-positive" if oos_cagr>0 else "metric-negative"), unsafe_allow_html=True)
            with col_m4:
                st.markdown(metric_card("Blind Win Rate", f"{win_rate:.1f}%", "metric-neutral"), unsafe_allow_html=True)
                
            col_m5, col_m6, col_m7, col_m8 = st.columns(4)
            with col_m5:
                st.markdown(metric_card("Blind Sharpe", f"{oos_sharpe:.2f}", "metric-positive" if oos_sharpe>1 else "metric-negative"), unsafe_allow_html=True)
            with col_m6:
                st.markdown(metric_card("Blind Max DD", f"{oos_max_dd:.1f}%", "metric-negative"), unsafe_allow_html=True)
            with col_m7:
                st.markdown(metric_card("Monte Carlo Worst DD", f"{mc_worst_dd:.1f}%", "metric-negative"), unsafe_allow_html=True)
            with col_m8:
                st.empty()
                
            st.markdown("### 3. Out-Of-Sample Master Equity Curve")
            fig_oos = px.line(x=master_returns.index, y=oos_equity_curve[1:], labels={"x": "Date", "y": "Portfolio Value"})
            fig_oos.update_traces(line=dict(color='#00B852', width=2), fill='tozeroy', fillcolor='rgba(0, 184, 82, 0.1)')
            fig_oos.update_layout(
                height=400, margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor='#0B0E14', plot_bgcolor='#0B0E14', font=dict(color='#E6EDF3')
            )
            st.plotly_chart(fig_oos, use_container_width=True)
            
        else:
            if not run_wfo:
                st.caption("Configure parameters and run the Future Test to generate institutional robustness analytics.")
