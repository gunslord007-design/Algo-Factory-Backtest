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
    strategy_mode: str = "MA Crossover",
    volume_filter: pd.Series = None,
    max_lookback: int = 0
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

    # Apply Volume Filter if provided (gates ONLY entries, not exits)
    if volume_filter is not None:
        entry_long = buy_trigger & volume_filter
        exit_long = sell_trigger  # Exit without volume restriction
        entry_short = sell_trigger & volume_filter
        exit_short = buy_trigger  # Exit without volume restriction
    else:
        entry_long = buy_trigger
        exit_long = sell_trigger
        entry_short = sell_trigger
        exit_short = buy_trigger

    if direction == "Long Only":
        signals = np.where(entry_long, 1, np.where(exit_long, 0, np.nan))
    elif direction == "Short Only":
        signals = np.where(entry_short, -1, np.where(exit_short, 0, np.nan))
    else:  # Both directions
        # If entry signal occurs with high volume -> Enter 1 or -1
        # If exit signal occurs but volume is too low to enter the opposite direction -> Go Flat (0)
        signals = np.where(
            entry_long, 1, 
            np.where(
                entry_short, -1, 
                np.where(exit_long | exit_short, 0, np.nan)
            )
        )

    signals_series = pd.Series(signals, index=crossovers.index).ffill().fillna(0).astype(int)
    
    # ── ENFORCE SYNCHRONIZED STARTING LINE ──
    # Destroys the "Fake Profit" warm-up bias by blocking all trades before max_lookback
    if max_lookback > 0 and len(signals_series) > max_lookback:
        signals_series.iloc[:max_lookback] = 0

    return signals_series


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
    current_capital = initial_capital
    tsl_enabled = tsl_pct is not None and tsl_pct > 0
    tsl_exit_bars = {}  # {bar_index: exit_price} for equity curve synchronization

    # ── EXTRACT ARRAYS FOR C-LEVEL SPEED ──
    # Doing this avoids millions of Pandas .iloc lookups during deep optimization scans
    open_arr = data['Open'].values
    high_arr = data['High'].values
    low_arr = data['Low'].values
    close_arr = data['Close'].values
    dates_arr = data.index.values
    pos_arr = positions.values
    pos_change_arr = positions.diff().fillna(0).values
    true_pos_arr = np.array(positions.values, copy=True)

    current_trade = None
    peak = None          # Highest price since LONG entry (for TSL tracking)
    trough = None        # Lowest price since SHORT entry (for TSL tracking)
    tsl_cooldown = False
    cooldown_direction = 0  # Raw indicator position when TSL fired

    for i in range(len(pos_arr)):
        change = pos_change_arr[i]
        pos = pos_arr[i]
        dt = pd.Timestamp(dates_arr[i])
        price = open_arr[i] if execution_mode == "Next Bar Open" else close_arr[i]

        # ── TSL COOLDOWN: Block re-entry until raw indicator signal resets ──
        if tsl_cooldown:
            true_pos_arr[i] = 0
            # Wait for indicator to change away from the direction we were stopped out of
            if pos != cooldown_direction:
                tsl_cooldown = False
                # If this bar also has a valid new entry signal, let it fall through
                if pos != 0 and change != 0:
                    true_pos_arr[i] = pos  # Record the new entry position correctly
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
                if open_arr[i] <= old_tsl:
                    tsl_exit_price = open_arr[i]
                    tsl_hit = True
                # STEP 2: Check if intraday low hit the OLD stop
                elif low_arr[i] <= old_tsl:
                    tsl_exit_price = old_tsl
                    tsl_hit = True
                else:
                    # STEP 3: Trade survived! Update peak for TOMORROW'S check
                    peak = max(peak, high_arr[i])

            else:  # SHORT
                old_tsl = trough * (1 + tsl_pct / 100)
                
                # STEP 1: Check if Open gapped up past the OLD stop
                if open_arr[i] >= old_tsl:
                    tsl_exit_price = open_arr[i]
                    tsl_hit = True
                # STEP 2: Check if intraday high hit the OLD stop
                elif high_arr[i] >= old_tsl:
                    tsl_exit_price = old_tsl
                    tsl_hit = True
                else:
                    # STEP 3: Trade survived! Update trough for TOMORROW'S check
                    trough = min(trough, low_arr[i])

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
                tsl_exit_bars[i] = (tsl_exit_price, False)

                # Enter cooldown: stay flat until indicator signal resets
                true_pos_arr[i] = 0
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
                
                # ── SAME-DAY TSL CHECK FOR "NEXT BAR OPEN" ──
                if execution_mode == "Next Bar Open":
                    tsl_hit = False
                    tsl_exit_price = None
                    if pos == 1:
                        old_tsl = peak * (1 - tsl_pct / 100)
                        # Intraday drop below Open's calculated stop
                        if low_arr[i] <= old_tsl:
                            tsl_exit_price = old_tsl
                            tsl_hit = True
                        else:
                            peak = max(peak, high_arr[i])
                    else:
                        old_tsl = trough * (1 + tsl_pct / 100)
                        # Intraday rise above Open's calculated stop
                        if high_arr[i] >= old_tsl:
                            tsl_exit_price = old_tsl
                            tsl_hit = True
                        else:
                            trough = min(trough, low_arr[i])
                            
                    if tsl_hit:
                        # Close the trade instantly on the exact same bar
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
                        current_trade['holding_bars'] = 0
                        current_trade['return_pct'] = round(return_pct, 2)
                        current_trade['exit_reason'] = 'TSL'

                        del current_trade['_entry_idx']
                        del current_trade['_shares']
                        trades.append(current_trade)
                        
                        current_trade = None
                        peak = None
                        trough = None
                        tsl_exit_bars[i] = (tsl_exit_price, True)  # True = Same-Day Exit
                        true_pos_arr[i] = 0
                        tsl_cooldown = True
                        cooldown_direction = pos

    # ── Handle open trade at end of data (force close at last price) ──
    if current_trade is not None:
        last_idx = len(pos_arr) - 1
        last_price = close_arr[last_idx]
        last_date = pd.Timestamp(dates_arr[last_idx])

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

    true_positions = pd.Series(true_pos_arr, index=data.index)
    return trades, true_positions, tsl_exit_bars


def build_equity_curve(
    data: pd.DataFrame,
    positions: pd.Series,
    initial_capital: float = 100000.0,
    brokerage_per_trade: float = 0.0,
    true_positions: pd.Series = None,
    tsl_exit_bars: dict = None,
    execution_mode: str = "Same Bar Close",
    optimize: bool = False
):
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
        for bar_idx, exit_data in tsl_exit_bars.items():
            if isinstance(exit_data, tuple):
                exit_price, is_sameday = exit_data
            else:
                exit_price = exit_data
                is_sameday = False
                
            if is_sameday and execution_mode == "Next Bar Open":
                # Special Case: Entered and exited on the EXACT same day.
                # The return is strictly intraday from the Open price.
                open_px = data['Open'].iloc[bar_idx]
                if open_px != 0:
                    partial_return = (exit_price - open_px) / open_px
                    original_pos = positions.iloc[bar_idx]
                    strategy_returns.iloc[bar_idx] = partial_return * original_pos
            else:
                if bar_idx > 0:
                    prev_close = close.iloc[bar_idx - 1]
                    if prev_close != 0:
                        partial_return = (exit_price - prev_close) / prev_close
                        # Use PREVIOUS bar's position. In NBO mode active_position[bar_idx] = 0
                        # because TSL zeroed it, which would erase the exit-day PnL entirely.
                        old_pos = effective_positions.iloc[bar_idx - 1]
                        strategy_returns.iloc[bar_idx] = partial_return * old_pos

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

    if optimize:
        return {"Strategy_Return": strategy_returns, "Portfolio_Value": portfolio_value}

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
    execution_mode: str = "Same Bar Close",
    precomputed_positions: pd.Series = None,
    vol_filter_enabled: bool = False,
    vol_lookback: int = 20,
    vol_threshold: float = 1.5,
    max_lookback: int = 0
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

    if precomputed_positions is not None:
        positions = precomputed_positions
    else:
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

        # ── Step 1c: Calculate Volume Filter ──
        volume_filter = None
        if vol_filter_enabled:
            from engine.indicators import calculate_rvol
            rvol = calculate_rvol(data['Volume'], vol_lookback)
            volume_filter = rvol >= vol_threshold

        # ── Step 2: Generate positions ──
        positions = generate_positions(crossovers, direction_clean, strategy_mode, volume_filter=volume_filter, max_lookback=max_lookback)

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
    result_data = build_equity_curve(
        data, positions, initial_capital, brokerage_per_trade, 
        true_positions=true_positions, tsl_exit_bars=tsl_exit_bars, 
        execution_mode=execution_mode, optimize=optimize
    )

    return {
        "ok": True,
        "data": result_data,
        "trades": trades,
        "positions": positions,
        "error": None,
        "hint": None
    }
