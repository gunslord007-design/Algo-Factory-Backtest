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


def generate_positions(crossovers: pd.DataFrame, direction: str = "Both") -> pd.Series:
    """
    Converts crossover signals into a position series using highly optimized vectorized logic.

    Position values:
       1 = Long (bought, holding)
       0 = Flat (no position)
      -1 = Short (sold short, holding)

    Direction modes:
      "Long Only"  — Only take long positions (golden cross = buy, death cross = exit)
      "Short Only" — Only take short positions (death cross = sell, golden cross = exit)
      "Both"       — Take both long and short positions

    Returns: pd.Series of position values (1, 0, or -1)
    """
    gc = crossovers['golden_cross']
    dc = crossovers['death_cross']

    if direction == "Long Only":
        signals = np.where(gc, 1, np.where(dc, 0, np.nan))
    elif direction == "Short Only":
        signals = np.where(dc, -1, np.where(gc, 0, np.nan))
    else:  # Both directions
        signals = np.where(gc, 1, np.where(dc, -1, np.nan))

    return pd.Series(signals, index=crossovers.index).ffill().fillna(0).astype(int)


def build_trade_log(data: pd.DataFrame, positions: pd.Series, brokerage_per_trade: float = 0.0, initial_capital: float = 100000.0) -> list:
    """
    Builds a detailed trade-by-trade log from position changes.

    Each trade dict contains:
      - entry_date, exit_date
      - entry_price, exit_price
      - direction ("LONG" or "SHORT")
      - gross_pnl (before brokerage)
      - brokerage (total for entry + exit)
      - net_pnl (after brokerage)
      - holding_bars (number of candles held)
      - return_pct (percentage return)

    Returns: list of trade dicts
    """
    trades = []
    close = data['Close']
    current_capital = initial_capital  # Track running cash for share-based position sizing

    # Detect position changes
    pos_change = positions.diff().fillna(0)

    current_trade = None

    for i in range(len(positions)):
        change = pos_change.iloc[i]
        pos = positions.iloc[i]
        dt = data.index[i]
        price = close.iloc[i]

        if change == 0:
            continue  # No change

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
            return_pct = (gross_pnl / position_value) * 100 if position_value != 0 else 0.0

            current_capital += net_pnl  # Update running cash for next trade sizing

            current_trade['exit_date'] = dt
            current_trade['exit_price'] = exit_price
            current_trade['shares'] = round(shares, 4)
            current_trade['gross_pnl'] = round(gross_pnl, 2)
            current_trade['brokerage'] = round(total_brokerage, 2)
            current_trade['net_pnl'] = round(net_pnl, 2)
            current_trade['holding_bars'] = i - current_trade['_entry_idx']
            current_trade['return_pct'] = round(return_pct, 2)

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
        return_pct = (gross_pnl / position_value) * 100 if position_value != 0 else 0.0

        current_trade['exit_date'] = last_date
        current_trade['exit_price'] = last_price
        current_trade['shares'] = round(shares, 4)
        current_trade['gross_pnl'] = round(gross_pnl, 2)
        current_trade['brokerage'] = round(total_brokerage, 2)
        current_trade['net_pnl'] = round(net_pnl, 2)
        current_trade['holding_bars'] = last_idx - current_trade['_entry_idx']
        current_trade['return_pct'] = round(return_pct, 2)
        current_trade['still_open'] = True  # Flag: this trade was not naturally closed

        del current_trade['_entry_idx']
        del current_trade['_shares']
        trades.append(current_trade)

    return trades


def build_equity_curve(
    data: pd.DataFrame,
    positions: pd.Series,
    initial_capital: float,
    brokerage_per_trade: float = 0.0
) -> pd.DataFrame:
    """
    Builds a complete equity curve with portfolio value, benchmark, and drawdown.

    Logic:
      - When position=1 (LONG): portfolio gains when price goes up
      - When position=-1 (SHORT): portfolio gains when price goes down
      - When position=0 (FLAT): portfolio stays flat
      - Brokerage is deducted at every position change

    Returns the original DataFrame with added columns:
      - Strategy_Return (per-bar % return from strategy)
      - Equity_Curve (cumulative multiplier, starts at 1.0)
      - Portfolio_Value (in rupees)
      - BH_Value (Buy & Hold benchmark in rupees)
      - Drawdown (percentage from peak)
    """
    close = data['Close'].copy()

    # ── Per-bar percentage returns of the underlying asset ──
    asset_returns = close.pct_change().fillna(0)

    # ── Strategy returns: only earn when in a position ──
    # Use PREVIOUS bar's position to determine if we earn THIS bar's return
    # This is the anti-look-ahead-bias mechanism
    active_position = positions.shift(1).fillna(0)
    strategy_returns = asset_returns * active_position

    # ── Deduct brokerage at position changes ──
    # A reversal (Long→Short or Short→Long) is a magnitude-2 change — charge double brokerage.
    # Brokerage expressed as fraction of initial_capital (flat fee, NOT per-share of stock price).
    pos_changes = positions.diff().fillna(0)
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
    result['Position'] = positions.values

    return result


def run_backtest(
    data: pd.DataFrame,
    fast_ma: pd.Series,
    slow_ma: pd.Series,
    direction: str = "Both",
    initial_capital: float = 100000.0,
    brokerage_per_trade: float = 0.0,
    optimize: bool = False
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

    # ── Step 1: Detect crossovers ──
    crossovers = detect_crossovers(fast_ma, slow_ma)

    # ── Step 2: Generate positions ──
    positions = generate_positions(crossovers, direction_clean)

    # ── Step 3: Build trade log ──
    if optimize:
        # Vectorized trade count one-liner (avoids heavy loop trade logs)
        trades = int(((positions != 0) & (positions.diff().fillna(0) != 0)).sum())
    else:
        trades = build_trade_log(data, positions, brokerage_per_trade, initial_capital)

    # ── Step 4: Build equity curve ──
    result_data = build_equity_curve(data, positions, initial_capital, brokerage_per_trade)

    return {
        "ok": True,
        "data": result_data,
        "trades": trades,
        "positions": positions,
        "error": None,
        "hint": None
    }
