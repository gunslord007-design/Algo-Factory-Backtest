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
ma_type = st.sidebar.selectbox("Indicator Type", ["SMA", "EMA", "DEMA", "WMA", "HMA", "VWAP"])
direction = st.sidebar.radio("Direction", ["Long Only", "Short Only", "Both"], index=0, horizontal=True)

col3, col4 = st.sidebar.columns(2)
fast_len = col3.number_input("Fast Length", min_value=1, max_value=500, value=9)
slow_len = col4.number_input("Slow Length", min_value=2, max_value=500, value=21)

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

st.sidebar.markdown("---")
st.sidebar.header("4. Capital & Costs")
capital = st.sidebar.number_input("Initial Capital (Rs)", min_value=1000, value=100000, step=10000)
brokerage = st.sidebar.slider("Brokerage per Trade (Rs)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)


# ── MAIN TERMINAL ──
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
    fast_ma = get_indicator(df, ma_type, fast_len)
    slow_ma = get_indicator(df, ma_type, slow_len)
    rsi_series = calculate_rsi(df['Close'], rsi_len)
except Exception as e:
    st.error(f"Error calculating indicators: {e}")
    st.stop()

bt_result = run_backtest(
    df, fast_ma, slow_ma, direction, capital, brokerage, 
    optimize=False, rsi_series=rsi_series, strategy_mode=strategy_mode,
    rsi_buy_rule=rsi_buy_rule, rsi_sell_rule=rsi_sell_rule,
    rsi_upper=rsi_upper, rsi_lower=rsi_lower
)

if not bt_result["ok"]:
    st.error(f"**Backtest Error:** {bt_result['error']}")
    if bt_result['hint']: st.info(bt_result['hint'])
    st.stop()

result_df = bt_result["data"]
trades = bt_result["trades"]
analytics = compute_full_analytics(result_df, trades, capital, interval)

# ── TABBED ARCHITECTURE ──
tab1, tab2, tab3, tab4 = st.tabs(["📊 Terminal", "📈 Deep Analytics", "📋 Trade Book", "⚡ Optimizer"])

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

    # ── GROWW-STYLE CHART OVERHAUL ──
    # Convert dates to strings for categorical X-axis (destroys all gaps: weekends, holidays, overnight)
    x_str = df.index.strftime('%Y-%m-%d %H:%M') if interval != "1d" else df.index.strftime('%Y-%m-%d')
    
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.2]
    )

    # 1. Groww Candlesticks (Solid Green/Red, No Borders)
    fig.add_trace(go.Candlestick(
        x=x_str, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="Price", 
        increasing_line_color='#00B852', increasing_fillcolor='#00B852',
        decreasing_line_color='#FF5000', decreasing_fillcolor='#FF5000',
        hoverinfo='x+y'
    ), row=1, col=1)

    # Moving Averages
    fig.add_trace(go.Scatter(x=x_str, y=fast_ma, name=f"Fast {ma_type}", line=dict(color='#3772FF', width=1.5), hoverinfo='y'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_str, y=slow_ma, name=f"Slow {ma_type}", line=dict(color='#F0B90B', width=1.5), hoverinfo='y'), row=1, col=1)

    # 2. Trade Markers (Precise Arrows)
    for t in trades:
        t_date = t['entry_date'].strftime('%Y-%m-%d %H:%M') if interval != "1d" else t['entry_date'].strftime('%Y-%m-%d')
        e_date = t['exit_date'].strftime('%Y-%m-%d %H:%M') if interval != "1d" else t['exit_date'].strftime('%Y-%m-%d')
        
        if t['direction'] == 'LONG':
            fig.add_trace(go.Scatter(x=[t_date], y=[t['entry_price']], mode='markers', marker=dict(symbol='triangle-up', size=14, color='#00B852', line=dict(color='white', width=1)), name='Buy Entry', hoverinfo='skip'), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=[t_date], y=[t['entry_price']], mode='markers', marker=dict(symbol='triangle-down', size=14, color='#FF5000', line=dict(color='white', width=1)), name='Short Entry', hoverinfo='skip'), row=1, col=1)
            
        fig.add_trace(go.Scatter(x=[e_date], y=[t['exit_price']], mode='markers', marker=dict(symbol='x', size=10, color='#FFFFFF'), name='Exit', hoverinfo='skip'), row=1, col=1)

    # 3. Volume (Blended Colors)
    vol_colors = ['#FF5000' if row['Open'] > row['Close'] else '#00B852' for index, row in df.iterrows()]
    fig.add_trace(go.Bar(x=x_str, y=df['Volume'], marker_color=vol_colors, name="Volume", hoverinfo='y', opacity=0.8), row=2, col=1)

    # 4. RSI Oscillator
    fig.add_trace(go.Scatter(x=x_str, y=rsi_series, name=f"RSI ({rsi_len})", line=dict(color='#A371F7', width=1.5), hoverinfo='y'), row=3, col=1)
    fig.add_hline(y=rsi_upper, line=dict(color='#F85149', width=1, dash='dash'), row=3, col=1)
    fig.add_hline(y=rsi_lower, line=dict(color='#2EA043', width=1, dash='dash'), row=3, col=1)
    # Add light fill between boundaries (Optional clean styling)
    fig.add_hrect(y0=rsi_lower, y1=rsi_upper, fillcolor="rgba(163,113,247,0.1)", layer="below", line_width=0, row=3, col=1)

    # 5. Equity Curve
    fig.add_trace(go.Scatter(x=x_str, y=result_df['Portfolio_Value'], name="Strategy Equity", line=dict(color='#3772FF', width=2), fill='tozeroy', fillcolor='rgba(55, 114, 255, 0.1)', hoverinfo='y'), row=4, col=1)
    fig.add_trace(go.Scatter(x=x_str, y=result_df['BH_Value'], name="Buy & Hold", line=dict(color='#8B949E', width=1.5, dash='dot'), hoverinfo='skip'), row=4, col=1)

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
        type='category', nticks=10,  # Forces gapless chart, limits text overlap
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
        display_cols = ['direction', 'entry_date', 'entry_price', 'exit_date', 'exit_price', 'brokerage', 'net_pnl', 'return_pct', 'holding_bars']
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
        opt_target = st.radio("Optimization Target", ["Optimize MAs", "Optimize RSI"], help="MA scan keeps RSI fixed. RSI scan keeps MAs fixed.")
        opt_mode = st.radio("Scan Mode", ["Fast Scan", "Deep Scan"])
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
                    rsi_lower=rsi_lower, rsi_upper=rsi_upper
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
                y_label = "Fast Length" if "MAs" in opt_target else "RSI Length"
                x_label = "Slow Length" if "MAs" in opt_target else "Oversold Threshold"
                st.markdown(f"🏆 **Best Combination found:** {y_label} = `{b['y_val']}`, {x_label} = `{b['x_val']}` ➔ **Return: {b['return']}%** (Sharpe: {b['sharpe']})")
                
                # Heatmap
                z = res['return_matrix']
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
