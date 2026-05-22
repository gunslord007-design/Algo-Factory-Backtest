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
    progress_callback=None
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
            "fast_range": list of fast MA lengths,
            "slow_range": list of slow MA lengths,
            "best": {"fast": int, "slow": int, "return": float, "sharpe": float},
            "total_combinations": int,
            "error": str or None
        }
    """
    if data is None or data.empty:
        return {"ok": False, "error": "No data provided for optimization."}

    fast_range, slow_range = generate_grid(mode)

    # Filter out invalid combos where fast >= slow
    # Pre-filter ranges based on data length
    data_len = len(data)
    fast_range = [f for f in fast_range if f < data_len]
    slow_range = [s for s in slow_range if s < data_len]

    if len(fast_range) == 0 or len(slow_range) == 0:
        return {
            "ok": False,
            "error": f"Not enough data ({data_len} candles) to run the optimizer. Need at least 55 candles for Fast Scan or 201 for Deep Scan."
        }

    # Initialize result matrices
    n_fast = len(fast_range)
    n_slow = len(slow_range)
    return_matrix = np.full((n_fast, n_slow), np.nan)
    sharpe_matrix = np.full((n_fast, n_slow), np.nan)
    trades_matrix = np.full((n_fast, n_slow), 0.0)

    # Pre-compute ALL indicator values for every length (cache them)
    # This is the KEY optimization — calculate each MA length ONCE
    all_lengths = sorted(set(fast_range + slow_range))
    ma_cache = {}
    for length in all_lengths:
        try:
            ma_cache[length] = get_indicator(data, ma_type, length)
        except Exception:
            continue  # Skip lengths that fail

    total_combos = 0
    for f in fast_range:
        for s in slow_range:
            if f < s:
                total_combos += 1

    completed = 0
    best_return = -999999
    best_combo = {"fast": 0, "slow": 0, "return": 0.0, "sharpe": 0.0}

    for i, fast_len in enumerate(fast_range):
        for j, slow_len in enumerate(slow_range):
            # Skip invalid combos
            if fast_len >= slow_len:
                continue

            # Skip if either MA was not cached
            if fast_len not in ma_cache or slow_len not in ma_cache:
                continue

            fast_ma = ma_cache[fast_len]
            slow_ma = ma_cache[slow_len]

            # Run backtest using the user-selected direction (not hardcoded)
            bt = run_backtest(
                data, fast_ma, slow_ma,
                direction=direction,
                initial_capital=initial_capital,
                brokerage_per_trade=brokerage_per_trade,
                optimize=True
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
                        "fast": fast_len,
                        "slow": slow_len,
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
        "fast_range": fast_range,
        "slow_range": slow_range,
        "best": best_combo,
        "total_combinations": total_combos,
        "error": None
    }
