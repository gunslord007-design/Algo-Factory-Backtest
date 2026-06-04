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
    if data is None or data.empty:
        return {"ok": False, "error": "No data provided for optimization."}

    opt_target = kwargs.get("opt_target", "Optimize MAs")
    strategy_mode = kwargs.get("strategy_mode", "MA Crossover")

    # ── RESOLVE BASE OPTIMIZATION ──
    is_3d = (opt_target == "Optimize Strategy + TSL (3D)")
    base_opt = "Optimize MAs" if "MA" in strategy_mode else "Optimize RSI"
    if not is_3d:
        base_opt = opt_target

    if base_opt == "Optimize RSI" and strategy_mode == "MA Crossover":
        return {"ok": False, "error": "Cannot optimize RSI when Strategy Mode is set to 'MA Crossover'."}

    # ── GENERATE GRIDS ──
    if base_opt == "Optimize MAs":
        y_range, x_range = generate_grid(mode)
        y_range = [f for f in y_range if f < len(data)]
        x_range = [s for s in x_range if s < len(data)]
        if not y_range or not x_range:
            return {"ok": False, "error": "Not enough data for MA Optimization."}
    elif base_opt == "Optimize RSI":
        # RSI Grid: Y = RSI Length, X = Oversold Threshold
        if mode == "deep":
            y_range = list(range(2, 31, 1))
            x_range = list(range(10, 46, 1))
        else:
            y_range = list(range(2, 31, 2))
            x_range = list(range(10, 46, 5))
    elif base_opt == "Optimize TSL Only (1D)":
        y_range = [0]
        x_range = [0]
            
    if is_3d or base_opt == "Optimize TSL Only (1D)":
        if mode == "deep":
            z_range = [round(x * 0.5, 1) for x in range(2, 41)] # 1.0 to 20.0, step 0.5
        else:
            z_range = list(range(1, 21, 1)) # 1.0 to 20.0, step 1.0
    else:
        z_range = [kwargs.get("tsl_pct", None)]

    n_y, n_x, n_z = len(y_range), len(x_range), len(z_range)
    return_matrix = np.full((n_y, n_x, n_z), np.nan)
    sharpe_matrix = np.full((n_y, n_x, n_z), np.nan)
    trades_matrix = np.full((n_y, n_x, n_z), 0.0)

    # ── CACHING ──
    ma_cache = {}
    rsi_cache = {}
    
    if base_opt == "Optimize MAs":
        all_lengths = sorted(set(y_range + x_range))
        for length in all_lengths:
            try:
                ma_cache[length] = get_indicator(data, ma_type, length)
            except Exception:
                continue
    elif base_opt == "Optimize RSI":
        from engine.indicators import calculate_rsi
        for length in y_range:
            rsi_cache[length] = calculate_rsi(data['Close'], length)
        fast_ma_fixed = kwargs.get('fast_ma')
        slow_ma_fixed = kwargs.get('slow_ma')
    elif base_opt == "Optimize TSL Only (1D)":
        fast_ma_fixed = kwargs.get('fast_ma')
        slow_ma_fixed = kwargs.get('slow_ma')

    # Count valid combos
    total_combos = 0
    for y in y_range:
        for x in x_range:
            if base_opt == "Optimize MAs" and y >= x:
                continue
            for z in z_range:
                total_combos += 1

    completed = 0
    best_return = -999999
    best_combo = {"y_val": 0, "x_val": 0, "z_val": None, "return": 0.0, "sharpe": 0.0}

    for i, y_val in enumerate(y_range):
        for j, x_val in enumerate(x_range):
            if base_opt == "Optimize MAs":
                if y_val >= x_val or y_val not in ma_cache or x_val not in ma_cache:
                    continue
                fast_ma = ma_cache[y_val]
                slow_ma = ma_cache[x_val]
                rsi_series = kwargs.get('rsi_series')
                rsi_lower = kwargs.get('rsi_lower', 30)
                rsi_upper = kwargs.get('rsi_upper', 70)
            elif base_opt == "Optimize RSI":
                fast_ma = fast_ma_fixed
                slow_ma = slow_ma_fixed
                rsi_series = rsi_cache[y_val]
                rsi_lower = x_val
                rsi_upper = 100 - x_val
            elif base_opt == "Optimize TSL Only (1D)":
                fast_ma = fast_ma_fixed
                slow_ma = slow_ma_fixed
                rsi_series = kwargs.get('rsi_series')
                rsi_lower = kwargs.get('rsi_lower', 30)
                rsi_upper = kwargs.get('rsi_upper', 70)

            # Precompute base positions for the current MA/RSI combo to avoid recalculating 
            # them redundantly for every TSL step.
            base_bt = run_backtest(
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
                rsi_upper=rsi_upper,
                tsl_enabled=False,
                execution_mode=kwargs.get("execution_mode", "Same Bar Close")
            )
            base_positions = base_bt.get("positions", None)

            # If the strategy generated 0 trades on its own, it won't make trades with TSL.
            # Skip the Z-loop entirely to save massive time.
            if base_positions is None or (base_positions == 0).all():
                completed += len(z_range)
                if progress_callback and total_combos > 0:
                    progress_callback(completed, total_combos)
                continue

            for k, z_val in enumerate(z_range):
                tsl_enabled_run = True if (is_3d or base_opt == "Optimize TSL Only (1D)") else kwargs.get("tsl_enabled", False)
                tsl_pct_run = z_val

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
                    rsi_upper=rsi_upper,
                    tsl_enabled=tsl_enabled_run,
                    tsl_pct=tsl_pct_run,
                    execution_mode=kwargs.get("execution_mode", "Same Bar Close"),
                    precomputed_positions=base_positions
                )

                if bt["ok"] and bt["data"] is not None:
                    # In optimize mode, data is a dict containing Series, not a DataFrame
                    final_val = float(bt["data"]["Portfolio_Value"].iloc[-1])
                    ret = total_return_pct(initial_capital, final_val)
                    sr = sharpe_ratio(bt["data"]["Strategy_Return"])
                    n_trades = bt["trades"] if isinstance(bt["trades"], int) else len(bt["trades"])

                    return_matrix[i, j, k] = ret
                    sharpe_matrix[i, j, k] = sr
                    trades_matrix[i, j, k] = n_trades

                    if ret > best_return:
                        best_return = ret
                        best_combo = {
                            "y_val": y_val,
                            "x_val": x_val,
                            "z_val": z_val,
                            "return": round(ret, 2),
                            "sharpe": round(sr, 3),
                            "trades": n_trades
                        }

                completed += 1
                if progress_callback and total_combos > 0:
                    progress_callback(completed, total_combos)

    # Flatten out Z dimension if not 3D to maintain 2D compatibility with legacy visualization
    if not is_3d and base_opt != "Optimize TSL Only (1D)":
        return_matrix = return_matrix[:, :, 0]
        sharpe_matrix = sharpe_matrix[:, :, 0]
        trades_matrix = trades_matrix[:, :, 0]
    elif base_opt == "Optimize TSL Only (1D)":
        return_matrix = return_matrix[0, 0, :]
        sharpe_matrix = sharpe_matrix[0, 0, :]
        trades_matrix = trades_matrix[0, 0, :]

    return {
        "ok": True,
        "base_opt": base_opt,
        "return_matrix": return_matrix,
        "sharpe_matrix": sharpe_matrix,
        "trades_matrix": trades_matrix,
        "y_range": y_range,
        "x_range": x_range,
        "z_range": z_range,
        "best": best_combo,
        "total_combinations": total_combos,
        "error": None
    }

