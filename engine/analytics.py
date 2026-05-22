"""
analytics.py — Institutional-Grade Performance Metrics
======================================================
Calculates hedge-fund-level analytics from backtest results.

All functions handle edge cases gracefully:
  - Zero trades → metrics return 0.0 or "N/A"
  - Zero volatility → ratios return 0.0 (not infinity)
  - All wins or all losses → handled without division errors
  - Very short time periods → CAGR uses actual days, not assumed 252

Risk-Free Rate: 6.5% annualized (India 10-Year Govt Bond Yield)
"""

import pandas as pd
import numpy as np

RISK_FREE_RATE = 0.065  # 6.5% annualized (India benchmark)
TRADING_DAYS_PER_YEAR = 252


# ─────────────────────────────────────────────────────────────
#  RETURN METRICS
# ─────────────────────────────────────────────────────────────

def total_return_pct(initial_capital: float, final_value: float) -> float:
    """Simple total return as a percentage."""
    if initial_capital <= 0:
        return 0.0
    return ((final_value / initial_capital) - 1) * 100


def cagr(initial_capital: float, final_value: float, total_days: int) -> float:
    """
    Compound Annual Growth Rate.
    Formula: (Final/Initial)^(365/Days) - 1
    """
    if initial_capital <= 0 or total_days <= 0 or final_value <= 0:
        return 0.0

    years = total_days / 365.0
    if years < 0.01:  # Less than ~4 days — CAGR is meaningless
        return 0.0

    return ((final_value / initial_capital) ** (1.0 / years) - 1) * 100


def alpha_vs_benchmark(strategy_return: float, benchmark_return: float) -> float:
    """Alpha = Strategy Return - Benchmark Return (in %)."""
    return strategy_return - benchmark_return


# ─────────────────────────────────────────────────────────────
#  RISK METRICS
# ─────────────────────────────────────────────────────────────

def max_drawdown(drawdown_series: pd.Series) -> float:
    """Returns the worst (most negative) drawdown percentage."""
    if drawdown_series is None or drawdown_series.empty:
        return 0.0
    return float(drawdown_series.min())


def max_drawdown_duration(portfolio_values: pd.Series) -> int:
    """
    Returns the longest number of bars spent in drawdown (peak to recovery).
    If the portfolio never recovers, returns the duration until the end of data.
    """
    if portfolio_values is None or len(portfolio_values) < 2:
        return 0

    peak = portfolio_values.iloc[0]
    in_drawdown = False
    current_duration = 0
    max_duration = 0

    for val in portfolio_values:
        if val >= peak:
            peak = val
            if in_drawdown:
                max_duration = max(max_duration, current_duration)
                current_duration = 0
                in_drawdown = False
        else:
            in_drawdown = True
            current_duration += 1

    # If still in drawdown at end of data
    if in_drawdown:
        max_duration = max(max_duration, current_duration)

    return max_duration


def annualized_volatility(strategy_returns: pd.Series) -> float:
    """
    Annualized standard deviation of returns.
    Formula: StdDev(daily returns) * sqrt(252)
    """
    if strategy_returns is None or len(strategy_returns) < 2:
        return 0.0

    daily_std = strategy_returns.std()
    if np.isnan(daily_std) or daily_std < 1e-10:
        return 0.0

    return float(daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)) * 100


# ─────────────────────────────────────────────────────────────
#  RISK-ADJUSTED RATIOS
# ─────────────────────────────────────────────────────────────

def sharpe_ratio(strategy_returns: pd.Series) -> float:
    """
    Sharpe Ratio = (Annualized Return - Risk Free Rate) / Annualized Volatility
    Above 1.0 = Good, Above 2.0 = Excellent, Above 3.0 = Exceptional
    """
    if strategy_returns is None or len(strategy_returns) < 2:
        return 0.0

    mean_daily = strategy_returns.mean()
    std_daily = strategy_returns.std()

    if std_daily == 0 or np.isnan(std_daily):
        return 0.0

    # Annualize
    annual_return = mean_daily * TRADING_DAYS_PER_YEAR
    annual_std = std_daily * np.sqrt(TRADING_DAYS_PER_YEAR)
    daily_rf = RISK_FREE_RATE / TRADING_DAYS_PER_YEAR

    sharpe = (mean_daily - daily_rf) / std_daily * np.sqrt(TRADING_DAYS_PER_YEAR)
    return round(float(sharpe), 3)


def sortino_ratio(strategy_returns: pd.Series) -> float:
    """
    Sortino Ratio = (Annualized Return - Risk Free Rate) / Downside Deviation
    Better than Sharpe because it only penalizes DOWNSIDE volatility, not upside.
    """
    if strategy_returns is None or len(strategy_returns) < 2:
        return 0.0

    mean_daily = strategy_returns.mean()
    daily_rf = RISK_FREE_RATE / TRADING_DAYS_PER_YEAR

    # Downside deviation: std of ONLY negative returns
    negative_returns = strategy_returns[strategy_returns < 0]

    if len(negative_returns) < 2:
        # No negative returns — strategy never lost money — Sortino is infinite
        # Cap at 99.0 to avoid display issues
        if mean_daily > daily_rf:
            return 99.0
        return 0.0

    downside_std = negative_returns.std()
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0

    sortino = (mean_daily - daily_rf) / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR)
    return round(float(sortino), 3)


def calmar_ratio(cagr_pct: float, max_dd_pct: float) -> float:
    """
    Calmar Ratio = CAGR / |Max Drawdown|
    Measures return per unit of drawdown risk.
    Above 1.0 = Good, Above 3.0 = Excellent.
    """
    if max_dd_pct == 0 or abs(max_dd_pct) < 0.001:
        if cagr_pct > 0:
            return 99.0  # No drawdown but positive return — cap
        return 0.0

    return round(cagr_pct / abs(max_dd_pct), 3)


# ─────────────────────────────────────────────────────────────
#  TRADE STATISTICS
# ─────────────────────────────────────────────────────────────

def compute_trade_stats(trades: list) -> dict:
    """
    Computes comprehensive trade-level statistics.

    Input: list of trade dicts from strategy.build_trade_log()
    Returns: dict with all trade-level metrics
    """
    total = len(trades)

    if total == 0:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'profit_factor': 0.0,
            'avg_return_pct': 0.0,
            'avg_holding_bars': 0,
            'total_brokerage': 0.0,
        }

    pnls = [t['net_pnl'] for t in trades]
    returns = [t['return_pct'] for t in trades]
    holdings = [t['holding_bars'] for t in trades]
    brokerages = [t['brokerage'] for t in trades]

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    win_count = len(wins)
    loss_count = len(losses)
    win_rate = (win_count / total) * 100 if total > 0 else 0.0

    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0

    largest_win = max(pnls) if pnls else 0.0
    largest_loss = min(pnls) if pnls else 0.0

    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

    return {
        'total_trades': total,
        'winning_trades': win_count,
        'losing_trades': loss_count,
        'win_rate': round(win_rate, 2),
        'avg_win': round(float(avg_win), 2),
        'avg_loss': round(float(avg_loss), 2),
        'largest_win': round(float(largest_win), 2),
        'largest_loss': round(float(largest_loss), 2),
        'profit_factor': round(float(profit_factor), 3),
        'avg_return_pct': round(float(np.mean(returns)), 2),
        'avg_holding_bars': int(np.mean(holdings)),
        'total_brokerage': round(float(sum(brokerages)), 2),
    }


# ─────────────────────────────────────────────────────────────
#  MASTER ANALYTICS FUNCTION
# ─────────────────────────────────────────────────────────────

def compute_full_analytics(
    backtest_data: pd.DataFrame,
    trades: list,
    initial_capital: float,
    interval: str = "1d"
) -> dict:
    """
    Master function that computes ALL analytics in one call.

    Input:
      - backtest_data: DataFrame from strategy.build_equity_curve()
      - trades: list from strategy.build_trade_log()
      - initial_capital: float
      - interval: str (e.g., "1d", "1h", "15m") — used to estimate trading days

    Returns: dict with every metric organized by category
    """
    if backtest_data is None or backtest_data.empty:
        return {"error": "No backtest data provided for analytics."}

    final_value = float(backtest_data['Portfolio_Value'].iloc[-1])
    bh_final = float(backtest_data['BH_Value'].iloc[-1])
    strat_returns = backtest_data['Strategy_Return']
    drawdowns = backtest_data['Drawdown']
    portfolio_vals = backtest_data['Portfolio_Value']

    # Calculate total calendar days
    total_days = (backtest_data.index[-1] - backtest_data.index[0]).days
    if total_days < 1:
        total_days = 1

    # ── Return Metrics ──
    strat_total_ret = total_return_pct(initial_capital, final_value)
    bh_total_ret = total_return_pct(initial_capital, bh_final)
    strat_cagr = cagr(initial_capital, final_value, total_days)
    alpha = alpha_vs_benchmark(strat_total_ret, bh_total_ret)

    # ── Risk Metrics ──
    mdd = max_drawdown(drawdowns)
    mdd_dur = max_drawdown_duration(portfolio_vals)
    vol = annualized_volatility(strat_returns)

    # ── Risk-Adjusted Ratios ──
    sr = sharpe_ratio(strat_returns)
    so = sortino_ratio(strat_returns)
    cr = calmar_ratio(strat_cagr, mdd)

    # ── Trade Stats ──
    ts = compute_trade_stats(trades)

    return {
        'returns': {
            'total_return_pct': round(strat_total_ret, 2),
            'benchmark_return_pct': round(bh_total_ret, 2),
            'alpha_pct': round(alpha, 2),
            'cagr_pct': round(strat_cagr, 2),
            'final_portfolio_value': round(final_value, 2),
        },
        'risk': {
            'max_drawdown_pct': round(mdd, 2),
            'max_drawdown_duration_bars': mdd_dur,
            'annualized_volatility_pct': round(vol, 2),
        },
        'ratios': {
            'sharpe_ratio': sr,
            'sortino_ratio': so,
            'calmar_ratio': cr,
        },
        'trades': ts,
        'meta': {
            'total_calendar_days': total_days,
            'total_candles': len(backtest_data),
            'initial_capital': initial_capital,
        }
    }
