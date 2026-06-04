"""
strategy.py — The Strategy Brain
=================================
Generates crossover signals, manages positions, deducts brokerage,
builds equity curves, and produces a complete trade log.

CRITICAL RULE: All signals use .shift(1) to prevent look-ahead bias.
We compare PREVIOUS candle's MA values to detect crossovers,
then execute on the CURRENT candle's Close price.
"""

import pandas as pd
import numpy as np


def detect_crossovers(fast_ma: pd.Series, slow_ma: pd.Series) -> pd.DataFrame:
    """
    Detects Golden Cross (bullish) and Death Cross (bearish) events.

    Logic:
      - Golden Cross: Fast MA was BELOW Slow MA on previous bar,
                      AND Fast MA is ABOVE Slow MA on current bar.
      - Death Cross:  Fast MA was ABOVE Slow MA on previous bar,
                      AND Fast MA is BELOW Slow MA on current bar.

    Returns DataFrame with columns: ['golden_cross', 'death_cross']
    Both are boolean Series.
    """
    # Current bar relationship
    fast_above_slow = fast_ma > slow_ma

    # Previous bar relationship (shifted by 1 to avoid look-ahead)
    prev_fast_above_slow = fast_above_slow.shift(1)

    # Crossover detection
    golden_cross = (fast_above_slow == True) & (prev_fast_above_slow == False)
    death_cross  = (fast_above_slow == False) & (prev_fast_above_slow == True)

    # First row can never be a crossover (no previous bar to compare)
    golden_cross.iloc[0] = False
    death_cross.iloc[0] = False

    return pd.DataFrame({
        'golden_cross': golden_cross,
        'death_cross': death_cross
    }, index=fast_ma.index)


def generate_positions(
    crossovers: pd.DataFrame, 
    direction: str = "Both",
    strategy_mode: str = "MA Crossover"
) -> pd.Series:
    """
    Converts signals into a position series using highly optimized vectorized logic.

    Position values:
       1 = Long (bought, holding)
       0 = Flat (no position)
      -1 = Short (sold short, holding)
    """
    gc = crossovers['golden_cross']
    dc = crossovers['death_cross']
    
    rsi_buy = crossovers.get('rsi_buy', pd.Series(False, index=crossovers.index))
    rsi_sell = crossovers.get('rsi_sell', pd.Series(False, index=crossovers.index))

    # Determine master buy/sell triggers based on strategy mode
    if strategy_mode == "MA Crossover":
        buy_trigger = gc
        sell_trigger = dc
    elif strategy_mode == "RSI Only":
        buy_trigger = rsi_buy
        sell_trigger = rsi_sell
    else:  # "MA + RSI"
        buy_trigger = gc & rsi_buy
        sell_trigger = dc | rsi_sell  # Sell if either safety condition hits

    if direction == "Long Only":
        signals = np.where(buy_trigger, 1, np.where(sell_trigger, 0, np.nan))
    elif direction == "Short Only":
        signals = np.where(sell_trigger, -1, np.where(buy_trigger, 0, np.nan))
    else:  # Both directions
        signals = np.where(buy_trigger, 1, np.where(sell_trigger, -1, np.nan))

    return pd.Series(signals, index=crossovers.index).ffill().fillna(0).astype(int)


def build_trade_log(data: pd.DataFrame, positions: pd.Series, brokerage_per_trade: float = 0.0, initial_capital: float = 100000.0, tsl_pct: float = None, execution_mode: str = "Same Bar Close") -> tuple:
    """
    Builds a detailed trade-by-trade log from position changes.

    When tsl_pct is provided, applies a Trailing Stop-Loss that overrides
    indicator signals. Uses a 2-step check per bar:
      Step 1: Did the Open gap through the previous bar's stop level?
      Step 2: Update peak/trough, then check if intraday Low/High hit the new stop.
    This correctly handles gap-downs, gap-ups, and intrabar reversals.

    Returns: (trades_list, true_positions_series, tsl_exit_bars_dict)
      - trades_list: list of trade dicts
      - true_positions: pd.Series with corrected positions after TSL overrides
      - tsl_exit_bars: dict mapping bar_index -> exit_price for equity curve sync
    """
    trades = []
    close = data['Close']
    high = data['High']
    low = data['Low']
    open_prices = data['Open']
    current_capital = initial_capital

    tsl_enabled = tsl_pct is not None and tsl_pct > 0
    true_positions = positions.copy()
    tsl_exit_bars = {}  # {bar_index: exit_price} for equity curve synchronization

    # Detect position changes from raw indicator signals
    pos_change = positions.diff().fillna(0)

    current_trade = None
    peak = None          # Highest price since LONG entry (for TSL tracking)
    trough = None        # Lowest price since SHORT entry (for TSL tracking)
    tsl_cooldown = False
    cooldown_direction = 0  # Raw indicator position when TSL fired

    for i in range(len(positions)):
        change = pos_change.iloc[i]
        pos = positions.iloc[i]
        dt = data.index[i]
        price = open_prices.iloc[i] if execution_mode == "Next Bar Open" else close.iloc[i]

        # ── TSL COOLDOWN: Block re-entry until raw indicator signal resets ──
        if tsl_cooldown:
            true_positions.iloc[i] = 0
            # Wait for indicator to change away from the direction we were stopped out of
            if pos != cooldown_direction:
                tsl_cooldown = False
                # If this bar also has a valid new entry signal, let it fall through
                if pos != 0 and change != 0:
                    true_positions.iloc[i] = pos  # Record the new entry position correctly
                    pass  # Will be handled by normal entry logic below
                else:
                    continue
            else:
                continue

        # ── TSL CHECK: Every bar while a trade is open ──
        if tsl_enabled and current_trade is not None and i > current_trade['_entry_idx']:
            tsl_hit = False
            tsl_exit_price = None

            if current_trade['direction'] == 'LONG':
                old_tsl = peak * (1 - tsl_pct / 100)
                
                # STEP 1: Check if Open gapped down past the OLD stop
                if open_prices.iloc[i] <= old_tsl:
                    tsl_exit_price = open_prices.iloc[i]
                    tsl_hit = True
                # STEP 2: Check if intraday low hit the OLD stop
                elif low.iloc[i] <= old_tsl:
                    tsl_exit_price = old_tsl
                    tsl_hit = True
                else:
                    # STEP 3: Trade survived! Update peak for TOMORROW'S check
                    peak = max(peak, high.iloc[i])

            else:  # SHORT
                old_tsl = trough * (1 + tsl_pct / 100)
                
                # STEP 1: Check if Open gapped up past the OLD stop
                if open_prices.iloc[i] >= old_tsl:
                    tsl_exit_price = open_prices.iloc[i]
                    tsl_hit = True
                # STEP 2: Check if intraday high hit the OLD stop
                elif high.iloc[i] >= old_tsl:
                    tsl_exit_price = old_tsl
                    tsl_hit = True
                else:
                    # STEP 3: Trade survived! Update trough for TOMORROW'S check
                    trough = min(trough, low.iloc[i])

            if tsl_hit:
                # Close the trade at the TSL exit price
                entry_price = current_trade['entry_price']
                shares = current_trade['_shares']

                if current_trade['direction'] == 'LONG':
                    gross_pnl = (tsl_exit_price - entry_price) * shares
                else:
                    gross_pnl = (entry_price - tsl_exit_price) * shares

                total_brokerage = brokerage_per_trade * 2
                net_pnl = gross_pnl - total_brokerage
                position_value = entry_price * shares
                return_pct = (net_pnl / position_value) * 100 if position_value != 0 else 0.0

                current_capital += net_pnl

                current_trade['exit_date'] = dt
                current_trade['exit_price'] = round(tsl_exit_price, 2)
                current_trade['shares'] = round(shares, 4)
                current_trade['gross_pnl'] = round(gross_pnl, 2)
                current_trade['brokerage'] = round(total_brokerage, 2)
                current_trade['net_pnl'] = round(net_pnl, 2)
                current_trade['holding_bars'] = i - current_trade['_entry_idx']
                current_trade['return_pct'] = round(return_pct, 2)
                current_trade['exit_reason'] = 'TSL'

                del current_trade['_entry_idx']
                del current_trade['_shares']
                trades.append(current_trade)
                current_trade = None
                peak = None
                trough = None

                # Record for equity curve synchronization
                tsl_exit_bars[i] = tsl_exit_price

                # Enter cooldown: stay flat until indicator signal resets
                true_positions.iloc[i] = 0
                tsl_cooldown = True
                cooldown_direction = pos
                continue

        # ── SKIP unchanged bars (original fast path, preserved exactly) ──
        if change == 0:
            continue

        # ── CLOSE existing trade if we had one ──
        if current_trade is not None:
            entry_price = current_trade['entry_price']
            exit_price = price
            shares = current_trade['_shares']

            if current_trade['direction'] == 'LONG':
                gross_pnl = (exit_price - entry_price) * shares
            else:  # SHORT
                gross_pnl = (entry_price - exit_price) * shares

            total_brokerage = brokerage_per_trade * 2  # Entry + Exit
            net_pnl = gross_pnl - total_brokerage
            position_value = entry_price * shares
            return_pct = (net_pnl / position_value) * 100 if position_value != 0 else 0.0

            current_capital += net_pnl  # Update running cash for next trade sizing

            current_trade['exit_date'] = dt
            current_trade['exit_price'] = exit_price
            current_trade['shares'] = round(shares, 4)
            current_trade['gross_pnl'] = round(gross_pnl, 2)
            current_trade['brokerage'] = round(total_brokerage, 2)
            current_trade['net_pnl'] = round(net_pnl, 2)
            current_trade['holding_bars'] = i - current_trade['_entry_idx']
            current_trade['return_pct'] = round(return_pct, 2)
            current_trade['exit_reason'] = 'Signal'

            # Remove internal tracking keys
            del current_trade['_entry_idx']
            del current_trade['_shares']
            trades.append(current_trade)
            current_trade = None

        # ── OPEN new trade if entering a position ──
        if pos != 0:
            shares = current_capital / price if price > 0 else 0.0
            current_trade = {
                'entry_date': dt,
                'entry_price': price,
                'direction': 'LONG' if pos == 1 else 'SHORT',
                '_entry_idx': i,     # Internal: for holding period calculation
                '_shares': shares    # Internal: for P&L calculation
            }
            # Initialize TSL peak/trough tracking from entry price
            if tsl_enabled:
                if pos == 1:
                    peak = price
                else:
                    trough = price

    # ── Handle open trade at end of data (force close at last price) ──
    if current_trade is not None:
        last_idx = len(data) - 1
        last_price = close.iloc[last_idx]
        last_date = data.index[last_idx]

        entry_price = current_trade['entry_price']
        shares = current_trade['_shares']

        if current_trade['direction'] == 'LONG':
            gross_pnl = (last_price - entry_price) * shares
        else:
            gross_pnl = (entry_price - last_price) * shares

        total_brokerage = brokerage_per_trade * 2
        net_pnl = gross_pnl - total_brokerage
        position_value = entry_price * shares
        return_pct = (net_pnl / position_value) * 100 if position_value != 0 else 0.0

        current_trade['exit_date'] = last_date
        current_trade['exit_price'] = last_price
        current_trade['shares'] = round(shares, 4)
        current_trade['gross_pnl'] = round(gross_pnl, 2)
        current_trade['brokerage'] = round(total_brokerage, 2)
        current_trade['net_pnl'] = round(net_pnl, 2)
        current_trade['holding_bars'] = last_idx - current_trade['_entry_idx']
        current_trade['return_pct'] = round(return_pct, 2)
        current_trade['exit_reason'] = 'End of Data'
        current_trade['still_open'] = True  # Flag: this trade was not naturally closed

        del current_trade['_entry_idx']
        del current_trade['_shares']
        trades.append(current_trade)

    return trades, true_positions, tsl_exit_bars


def build_equity_curve(
    data: pd.DataFrame,
    positions: pd.Series,
    initial_capital: float,
    brokerage_per_trade: float = 0.0,
    true_positions: pd.Series = None,
    tsl_exit_bars: dict = None,
    execution_mode: str = "Same Bar Close"
) -> pd.DataFrame:
    """
    Builds a complete equity curve with portfolio value, benchmark, and drawdown.

    Logic:
      - When position=1 (LONG): portfolio gains when price goes up
      - When position=-1 (SHORT): portfolio gains when price goes down
      - When position=0 (FLAT): portfolio stays flat
      - Brokerage is deducted at every position change

    Args:
      true_positions: If provided (from TSL-corrected trade log), overrides
                      raw indicator positions for return/brokerage calculations.
      tsl_exit_bars:  If provided, dict of {bar_index: exit_price}. On these bars,
                      the strategy return is computed from prev_close to exit_price
                      instead of prev_close to bar_close, keeping equity curve
                      perfectly synchronized with the trade log.

    Returns the original DataFrame with added columns:
      - Strategy_Return (per-bar % return from strategy)
      - Equity_Curve (cumulative multiplier, starts at 1.0)
      - Portfolio_Value (in rupees)
      - BH_Value (Buy & Hold benchmark in rupees)
      - Drawdown (percentage from peak)
    """
    close = data['Close'].copy()

    # Use TSL-corrected positions if available, otherwise raw indicator positions
    effective_positions = true_positions if true_positions is not None else positions

    # ── Per-bar percentage returns of the underlying asset ──
    asset_returns = close.pct_change().fillna(0)

    # ── Strategy returns: only earn when in a position ──
    if execution_mode == "Next Bar Open":
        # Positions already pre-shifted +1 in run_backtest, no extra shift needed
        active_position = effective_positions.fillna(0)
    else:
        # Standard: use PREVIOUS bar's position (anti-look-ahead-bias)
        active_position = effective_positions.shift(1).fillna(0)
    strategy_returns = asset_returns * active_position

    # ── Next Bar Open: fix returns on position-change bars using actual open prices ──
    if execution_mode == "Next Bar Open":
        open_prices = data['Open'].copy()
        prev_close = close.shift(1)
        prev_pos = effective_positions.shift(1).fillna(0)
        pos_changes = effective_positions.diff().fillna(0)
        change_mask = pos_changes != 0

        # Exit portion: prev_close → open, earned at OLD position
        exit_ret = ((open_prices - prev_close) / prev_close).fillna(0) * prev_pos
        # Entry portion: open → close, earned at NEW position
        entry_ret = ((close - open_prices) / open_prices).fillna(0) * effective_positions

        strategy_returns[change_mask] = (exit_ret + entry_ret)[change_mask]

    # ── Sync TSL exit bars: use partial return to exact exit price ──
    if tsl_exit_bars:
        for bar_idx, exit_price in tsl_exit_bars.items():
            if bar_idx > 0:
                prev_close = close.iloc[bar_idx - 1]
                if prev_close != 0:
                    partial_return = (exit_price - prev_close) / prev_close
                    strategy_returns.iloc[bar_idx] = partial_return * active_position.iloc[bar_idx]

    # ── Deduct brokerage at position changes ──
    # A reversal (Long→Short or Short→Long) is a magnitude-2 change — charge double brokerage.
    # Brokerage expressed as fraction of initial_capital (flat fee, NOT per-share of stock price).
    pos_changes = effective_positions.diff().fillna(0)
    change_magnitude = abs(pos_changes)
    num_events = np.where(change_magnitude >= 2, 2, np.where(change_magnitude == 1, 1, 0))
    brokerage_impact = (brokerage_per_trade * num_events) / initial_capital

    strategy_returns = strategy_returns - brokerage_impact

    # ── Equity curve (cumulative product of 1 + returns) ──
    equity_curve = (1 + strategy_returns).cumprod()
    portfolio_value = equity_curve * initial_capital

    # ── Buy & Hold benchmark ──
    bh_returns = asset_returns.copy()
    bh_curve = (1 + bh_returns).cumprod()
    bh_value = bh_curve * initial_capital

    # ── Drawdown calculation ──
    rolling_peak = portfolio_value.cummax()
    drawdown = ((portfolio_value - rolling_peak) / rolling_peak) * 100

    # ── Attach to DataFrame ──
    result = data.copy()
    result['Strategy_Return'] = strategy_returns
    result['Equity_Curve'] = equity_curve
    result['Portfolio_Value'] = portfolio_value
    result['BH_Value'] = bh_value
    result['Drawdown'] = drawdown
    result['Position'] = effective_positions.values

    return result


def run_backtest(
    data: pd.DataFrame,
    fast_ma: pd.Series,
    slow_ma: pd.Series,
    direction: str = "Both",
    initial_capital: float = 100000.0,
    brokerage_per_trade: float = 0.0,
    optimize: bool = False,
    rsi_series: pd.Series = None,
    strategy_mode: str = "MA Crossover",
    rsi_buy_rule: str = "Crosses Below Oversold",
    rsi_sell_rule: str = "Crosses Above Overbought",
    rsi_upper: float = 70.0,
    rsi_lower: float = 30.0,
    tsl_enabled: bool = False,
    tsl_pct: float = None,
    execution_mode: str = "Same Bar Close"
) -> dict:
    """
    Master function that orchestrates the entire backtest pipeline.

    Returns:
        {
            "ok": True/False,
            "data": DataFrame with equity curves (or None),
            "trades": list of trade dicts (or []),
            "positions": pd.Series of position values,
            "error": str or None,
            "hint": str or None
        }
    """
    # ── Validate inputs ──
    if data is None or data.empty:
        return {
            "ok": False, "data": None, "trades": [], "positions": None,
            "error": "No price data provided to the strategy engine.",
            "hint": "Fetch data first using the data loader."
        }

    if len(fast_ma) != len(data) or len(slow_ma) != len(data):
        return {
            "ok": False, "data": None, "trades": [], "positions": None,
            "error": f"MA length mismatch. Data has {len(data)} rows but Fast MA has {len(fast_ma)} and Slow MA has {len(slow_ma)}.",
            "hint": "This is an internal error. The indicator engine produced misaligned output."
        }

    if initial_capital <= 0:
        return {
            "ok": False, "data": None, "trades": [], "positions": None,
            "error": "Initial capital must be greater than zero.",
            "hint": "Set your starting capital to at least 1000."
        }

    if brokerage_per_trade < 0:
        return {
            "ok": False, "data": None, "trades": [], "positions": None,
            "error": "Brokerage cannot be negative.",
            "hint": "Set brokerage to 0 or a positive value."
        }

    # ── Normalize direction string ──
    direction_clean = direction
    if "long" in direction.lower():
        direction_clean = "Long Only"
    elif "short" in direction.lower():
        direction_clean = "Short Only"
    else:
        direction_clean = "Both"

    # ── Step 1: Detect MA crossovers ──
    crossovers = detect_crossovers(fast_ma, slow_ma)

    # ── Step 1b: Detect RSI Signals ──
    rsi_buy = pd.Series(False, index=data.index)
    rsi_sell = pd.Series(False, index=data.index)

    if rsi_series is not None and strategy_mode in ["RSI Only", "MA + RSI"]:
        prev_rsi = rsi_series.shift(1)
        curr_rsi = rsi_series
        
        # BUY LOGIC (Vectorized)
        if rsi_buy_rule == "Crosses Below Oversold":
            rsi_buy = (curr_rsi < rsi_lower) & (prev_rsi >= rsi_lower)
        elif rsi_buy_rule == "Crosses Above Oversold":
            rsi_buy = (curr_rsi > rsi_lower) & (prev_rsi <= rsi_lower)
        elif rsi_buy_rule == "Crosses Above Midline (50)":
            rsi_buy = (curr_rsi > 50.0) & (prev_rsi <= 50.0)
            
        # SELL LOGIC (Vectorized)
        if rsi_sell_rule == "Crosses Above Overbought":
            rsi_sell = (curr_rsi > rsi_upper) & (prev_rsi <= rsi_upper)
        elif rsi_sell_rule == "Crosses Below Overbought":
            rsi_sell = (curr_rsi < rsi_upper) & (prev_rsi >= rsi_upper)
        elif rsi_sell_rule == "Crosses Below Midline (50)":
            rsi_sell = (curr_rsi < 50.0) & (prev_rsi >= 50.0)

    # Append to crossovers DataFrame for clean passing
    crossovers['rsi_buy'] = rsi_buy
    crossovers['rsi_sell'] = rsi_sell

    # ── Step 2: Generate positions ──
    positions = generate_positions(crossovers, direction_clean, strategy_mode)

    # ── Next Bar Open: delay all position changes by 1 bar ──
    if execution_mode == "Next Bar Open":
        positions = positions.shift(1).fillna(0).astype(int)

    # ── Step 3: Build trade log ──
    true_positions = None
    tsl_exit_bars = None
    # Disable the fast-path if TSL is active so the engine evaluates every stop-loss
    if optimize and not tsl_enabled:
        # Vectorized trade count one-liner (avoids heavy loop trade logs)
        trades = int(((positions != 0) & (positions.diff().fillna(0) != 0)).sum())
    else:
        tsl_pct_val = tsl_pct if tsl_enabled else None
        trades, true_positions, tsl_exit_bars = build_trade_log(data, positions, brokerage_per_trade, initial_capital, tsl_pct=tsl_pct_val, execution_mode=execution_mode)

    # ── Step 4: Build equity curve ──
    result_data = build_equity_curve(data, positions, initial_capital, brokerage_per_trade, true_positions=true_positions, tsl_exit_bars=tsl_exit_bars, execution_mode=execution_mode)

    return {
        "ok": True,
        "data": result_data,
        "trades": trades,
        "positions": positions,
        "error": None,
        "hint": None
    }
