"""
optimizer.py — Heatmap Grid Search Engine
==========================================
Automatically finds the best Fast/Slow MA combination for a stock.
Runs backtests across a grid and returns a 2D results matrix.

Two scan modes:
  - Fast Scan: Increments of 5 (e.g., 5, 10, 15...) — finishes in seconds
  - Deep Scan: Every single integer (e.g., 5, 6, 7...) — thorough, takes longer
"""

import pandas as pd
import numpy as np
from engine.indicators import get_indicator
from engine.strategy import run_backtest
from engine.analytics import total_return_pct, sharpe_ratio, compute_trade_stats


def generate_grid(mode: str = "fast") -> tuple:
    """
    Generates the Fast and Slow MA length ranges for the grid search.

    Returns: (fast_range, slow_range)
    """
    if mode == "deep":
        fast_range = list(range(3, 51, 1))       # 3 to 50, step 1
        slow_range = list(range(10, 201, 1))      # 10 to 200, step 1
    else:  # fast
        fast_range = list(range(5, 55, 5))        # 5, 10, 15, ..., 50
        slow_range = list(range(10, 210, 10))      # 10, 20, 30, ..., 200

    return fast_range, slow_range


def run_optimization(
    data: pd.DataFrame,
    ma_type: str = "SMA",
    mode: str = "fast",
    initial_capital: float = 100000,
    brokerage_per_trade: float = 0.0,
    direction: str = "Both",
    progress_callback=None,
    **kwargs
) -> dict:
    """
    Runs the full heatmap grid search.

    Args:
        data: OHLCV DataFrame
        ma_type: MA type to use (SMA, EMA, DEMA, WMA, HMA, VWAP)
        mode: "fast" or "deep"
        initial_capital: Starting capital
        brokerage_per_trade: Brokerage per trade
        direction: Strategy direction ("Long Only", "Short Only", "Both")
        progress_callback: Optional function(current, total) for progress updates

    Returns:
        {
            "ok": True/False,
            "return_matrix": 2D numpy array of returns,
            "sharpe_matrix": 2D numpy array of Sharpe ratios,
            "trades_matrix": 2D numpy array of trade counts,
            "y_range": list of Y-axis values,
            "x_range": list of X-axis values,
            "best": {"y_val": int, "x_val": int, "return": float, "sharpe": float},
            "total_combinations": int,
            "error": str or None
        }
    """
    if data is None or data.empty:
        return {"ok": False, "error": "No data provided for optimization."}

    opt_target = kwargs.get("opt_target", "Optimize MAs")
    strategy_mode = kwargs.get("strategy_mode", "MA Crossover")

    if opt_target == "Optimize RSI" and strategy_mode == "MA Crossover":
        return {"ok": False, "error": "Cannot optimize RSI when Strategy Mode is set to 'MA Crossover'. Change Strategy Mode in sidebar first."}

    # ── GENERATE GRIDS ──
    if opt_target == "Optimize MAs":
        y_range, x_range = generate_grid(mode)
        # Pre-filter based on data length
        y_range = [f for f in y_range if f < len(data)]
        x_range = [s for s in x_range if s < len(data)]
        if not y_range or not x_range:
            return {"ok": False, "error": "Not enough data for MA Optimization."}
    else:
        # RSI Grid: Y = RSI Length, X = Oversold Threshold (Overbought = 100 - Oversold)
        if mode == "deep":
            y_range = list(range(2, 31, 1))    # Length: 2 to 30
            x_range = list(range(10, 46, 1))   # Oversold: 10 to 45
        else:
            y_range = list(range(2, 31, 2))    # Length: 2, 4, 6...
            x_range = list(range(10, 46, 5))   # Oversold: 10, 15, 20...
            
    n_y = len(y_range)
    n_x = len(x_range)
    return_matrix = np.full((n_y, n_x), np.nan)
    sharpe_matrix = np.full((n_y, n_x), np.nan)
    trades_matrix = np.full((n_y, n_x), 0.0)

    # ── CACHING ──
    ma_cache = {}
    rsi_cache = {}
    
    if opt_target == "Optimize MAs":
        all_lengths = sorted(set(y_range + x_range))
        for length in all_lengths:
            try:
                ma_cache[length] = get_indicator(data, ma_type, length)
            except Exception:
                continue
    else:
        from engine.indicators import calculate_rsi
        for length in y_range:
            rsi_cache[length] = calculate_rsi(data['Close'], length)
        # Fixed MAs for RSI run
        fast_ma_fixed = kwargs.get('fast_ma')
        slow_ma_fixed = kwargs.get('slow_ma')

    # Count valid combos
    total_combos = 0
    for y in y_range:
        for x in x_range:
            if opt_target == "Optimize MAs" and y >= x:
                continue
            total_combos += 1

    completed = 0
    best_return = -999999
    best_combo = {"y_val": 0, "x_val": 0, "return": 0.0, "sharpe": 0.0}

    for i, y_val in enumerate(y_range):
        for j, x_val in enumerate(x_range):
            if opt_target == "Optimize MAs":
                if y_val >= x_val or y_val not in ma_cache or x_val not in ma_cache:
                    continue
                fast_ma = ma_cache[y_val]
                slow_ma = ma_cache[x_val]
                rsi_series = kwargs.get('rsi_series')
                rsi_lower = kwargs.get('rsi_lower', 30)
                rsi_upper = kwargs.get('rsi_upper', 70)
            else:
                fast_ma = fast_ma_fixed
                slow_ma = slow_ma_fixed
                rsi_series = rsi_cache[y_val]
                rsi_lower = x_val
                rsi_upper = 100 - x_val

            bt = run_backtest(
                data, fast_ma, slow_ma,
                direction=direction,
                initial_capital=initial_capital,
                brokerage_per_trade=brokerage_per_trade,
                optimize=True,
                rsi_series=rsi_series,
                strategy_mode=strategy_mode,
                rsi_buy_rule=kwargs.get("rsi_buy_rule"),
                rsi_sell_rule=kwargs.get("rsi_sell_rule"),
                rsi_lower=rsi_lower,
                rsi_upper=rsi_upper
            )

            if bt["ok"] and bt["data"] is not None:
                final_val = float(bt["data"]["Portfolio_Value"].iloc[-1])
                ret = total_return_pct(initial_capital, final_val)
                sr = sharpe_ratio(bt["data"]["Strategy_Return"])
                n_trades = bt["trades"] if isinstance(bt["trades"], int) else len(bt["trades"])

                return_matrix[i, j] = ret
                sharpe_matrix[i, j] = sr
                trades_matrix[i, j] = n_trades

                if ret > best_return:
                    best_return = ret
                    best_combo = {
                        "y_val": y_val,
                        "x_val": x_val,
                        "return": round(ret, 2),
                        "sharpe": round(sr, 3),
                        "trades": n_trades
                    }

            completed += 1
            if progress_callback and total_combos > 0:
                progress_callback(completed, total_combos)

    return {
        "ok": True,
        "return_matrix": return_matrix,
        "sharpe_matrix": sharpe_matrix,
        "trades_matrix": trades_matrix,
        "y_range": y_range,
        "x_range": x_range,
        "best": best_combo,
        "total_combinations": total_combos,
        "error": None
    }
