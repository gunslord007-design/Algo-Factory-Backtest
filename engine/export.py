"""
export.py — CSV & PDF Report Generator
=======================================
Generates downloadable trade logs (CSV) and professional
tearsheet PDF reports with full analytics.

PDF uses fpdf2 for clean, lightweight generation.
Falls back to a text-based report if fpdf2 is not installed.
"""

import pandas as pd
import numpy as np
import io
from datetime import datetime


def generate_csv(trades: list) -> str:
    """
    Converts the trade log list into a CSV string.

    Returns: CSV string ready for download
    """
    if not trades:
        return "No trades to export."

    df = pd.DataFrame(trades)

    # Reorder columns for readability
    desired_order = [
        'direction', 'entry_date', 'entry_price', 'exit_date', 'exit_price',
        'gross_pnl', 'brokerage', 'net_pnl', 'return_pct', 'holding_bars'
    ]
    cols = [c for c in desired_order if c in df.columns]
    remaining = [c for c in df.columns if c not in cols]
    df = df[cols + remaining]

    return df.to_csv(index=False)


def generate_pdf_report(analytics: dict, trades: list, stock_name: str, ma_config: str) -> bytes:
    """
    Generates a professional PDF tearsheet report.

    Tries fpdf2 first. If not available, generates a clean text-based PDF
    using only built-in Python.

    Args:
        analytics: dict from analytics.compute_full_analytics()
        trades: list from strategy.build_trade_log()
        stock_name: display name of the stock
        ma_config: string describing the MA configuration (e.g., "SMA 9/21")

    Returns: bytes of the PDF file
    """
    try:
        return _generate_pdf_fpdf(analytics, trades, stock_name, ma_config)
    except ImportError:
        return _generate_text_report(analytics, trades, stock_name, ma_config)


def _generate_pdf_fpdf(analytics: dict, trades: list, stock_name: str, ma_config: str) -> bytes:
    """PDF generation using fpdf2 library."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── PAGE 1: SUMMARY ──
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 22)
    pdf.cell(0, 15, "Algo Factory - Backtest Report", new_x="LMARGIN", new_y="NEXT", align="C")

    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 8, f"Stock: {stock_name}  |  Strategy: {ma_config}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)

    # ── RETURN METRICS ──
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Return Metrics", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)

    ret = analytics.get('returns', {})
    _add_metric_row(pdf, "Total Return", f"{ret.get('total_return_pct', 0):.2f}%")
    _add_metric_row(pdf, "Benchmark Return (Buy & Hold)", f"{ret.get('benchmark_return_pct', 0):.2f}%")
    _add_metric_row(pdf, "Alpha (vs Benchmark)", f"{ret.get('alpha_pct', 0):.2f}%")
    _add_metric_row(pdf, "CAGR", f"{ret.get('cagr_pct', 0):.2f}%")
    _add_metric_row(pdf, "Final Portfolio Value", f"Rs. {ret.get('final_portfolio_value', 0):,.2f}")
    pdf.ln(5)

    # ── RISK METRICS ──
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Risk Metrics", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)

    risk = analytics.get('risk', {})
    _add_metric_row(pdf, "Max Drawdown", f"{risk.get('max_drawdown_pct', 0):.2f}%")
    _add_metric_row(pdf, "Max Drawdown Duration", f"{risk.get('max_drawdown_duration_bars', 0)} bars")
    _add_metric_row(pdf, "Annualized Volatility", f"{risk.get('annualized_volatility_pct', 0):.2f}%")
    pdf.ln(5)

    # ── RISK-ADJUSTED RATIOS ──
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Risk-Adjusted Ratios", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)

    ratios = analytics.get('ratios', {})
    _add_metric_row(pdf, "Sharpe Ratio", f"{ratios.get('sharpe_ratio', 0):.3f}")
    _add_metric_row(pdf, "Sortino Ratio", f"{ratios.get('sortino_ratio', 0):.3f}")
    _add_metric_row(pdf, "Calmar Ratio", f"{ratios.get('calmar_ratio', 0):.3f}")
    pdf.ln(5)

    # ── TRADE STATISTICS ──
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Trade Statistics", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)

    ts = analytics.get('trades', {})
    _add_metric_row(pdf, "Total Trades", str(ts.get('total_trades', 0)))
    _add_metric_row(pdf, "Winning Trades", str(ts.get('winning_trades', 0)))
    _add_metric_row(pdf, "Losing Trades", str(ts.get('losing_trades', 0)))
    _add_metric_row(pdf, "Win Rate", f"{ts.get('win_rate', 0):.1f}%")
    _add_metric_row(pdf, "Average Win", f"Rs. {ts.get('avg_win', 0):.2f}")
    _add_metric_row(pdf, "Average Loss", f"Rs. {ts.get('avg_loss', 0):.2f}")
    _add_metric_row(pdf, "Largest Win", f"Rs. {ts.get('largest_win', 0):.2f}")
    _add_metric_row(pdf, "Largest Loss", f"Rs. {ts.get('largest_loss', 0):.2f}")
    _add_metric_row(pdf, "Profit Factor", f"{ts.get('profit_factor', 0):.3f}")
    _add_metric_row(pdf, "Avg Holding Period", f"{ts.get('avg_holding_bars', 0)} bars")
    _add_metric_row(pdf, "Total Brokerage Paid", f"Rs. {ts.get('total_brokerage', 0):.2f}")

    # ── PAGE 2: TRADE LOG ──
    if trades:
        pdf.add_page("L")  # Landscape for trade table
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Detailed Trade Log", new_x="LMARGIN", new_y="NEXT")

        # Table header
        pdf.set_font("Helvetica", "B", 8)
        col_widths = [15, 25, 35, 25, 35, 25, 25, 20, 25, 20, 20]
        headers = ["#", "Dir", "Entry Date", "Entry Price", "Exit Date", "Exit Price",
                    "Gross PnL", "Brokerage", "Net PnL", "Return %", "Bars"]

        for w, h in zip(col_widths, headers):
            pdf.cell(w, 7, h, border=1, align="C")
        pdf.ln()

        # Table rows
        pdf.set_font("Helvetica", "", 7)
        for idx, t in enumerate(trades[:100], 1):  # Limit to 100 trades in PDF
            entry_dt = str(t.get('entry_date', ''))[:19]
            exit_dt = str(t.get('exit_date', ''))[:19]

            row = [
                str(idx),
                t.get('direction', ''),
                entry_dt,
                f"{t.get('entry_price', 0):.2f}",
                exit_dt,
                f"{t.get('exit_price', 0):.2f}",
                f"{t.get('gross_pnl', 0):.2f}",
                f"{t.get('brokerage', 0):.2f}",
                f"{t.get('net_pnl', 0):.2f}",
                f"{t.get('return_pct', 0):.2f}%",
                str(t.get('holding_bars', 0))
            ]

            for w, val in zip(col_widths, row):
                pdf.cell(w, 6, val, border=1, align="C")
            pdf.ln()

        if len(trades) > 100:
            pdf.ln(3)
            pdf.set_font("Helvetica", "I", 9)
            pdf.cell(0, 8, f"(Showing 100 of {len(trades)} trades. Full log available in CSV export.)",
                     new_x="LMARGIN", new_y="NEXT")

    # ── FOOTER ──
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(0, 10, "Report generated by Algo Factory Backtest Engine V4", new_x="LMARGIN", new_y="NEXT", align="C")

    return pdf.output()


def _add_metric_row(pdf, label: str, value: str):
    """Helper to add a label-value row in the PDF."""
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(90, 7, f"  {label}:", border=0)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, value, new_x="LMARGIN", new_y="NEXT", border=0)


def _generate_text_report(analytics: dict, trades: list, stock_name: str, ma_config: str) -> bytes:
    """
    Fallback: Generates a plain-text report as bytes if fpdf2 is not installed.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("  ALGO FACTORY — BACKTEST REPORT")
    lines.append("=" * 60)
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Stock: {stock_name}")
    lines.append(f"  Strategy: {ma_config}")
    lines.append("")

    ret = analytics.get('returns', {})
    lines.append("  RETURN METRICS")
    lines.append(f"    Total Return:       {ret.get('total_return_pct', 0):.2f}%")
    lines.append(f"    Benchmark Return:   {ret.get('benchmark_return_pct', 0):.2f}%")
    lines.append(f"    Alpha:              {ret.get('alpha_pct', 0):.2f}%")
    lines.append(f"    CAGR:               {ret.get('cagr_pct', 0):.2f}%")
    lines.append(f"    Final Value:        Rs. {ret.get('final_portfolio_value', 0):,.2f}")
    lines.append("")

    risk = analytics.get('risk', {})
    lines.append("  RISK METRICS")
    lines.append(f"    Max Drawdown:       {risk.get('max_drawdown_pct', 0):.2f}%")
    lines.append(f"    DD Duration:        {risk.get('max_drawdown_duration_bars', 0)} bars")
    lines.append(f"    Volatility:         {risk.get('annualized_volatility_pct', 0):.2f}%")
    lines.append("")

    ratios = analytics.get('ratios', {})
    lines.append("  RISK-ADJUSTED RATIOS")
    lines.append(f"    Sharpe Ratio:       {ratios.get('sharpe_ratio', 0):.3f}")
    lines.append(f"    Sortino Ratio:      {ratios.get('sortino_ratio', 0):.3f}")
    lines.append(f"    Calmar Ratio:       {ratios.get('calmar_ratio', 0):.3f}")
    lines.append("")

    ts = analytics.get('trades', {})
    lines.append("  TRADE STATISTICS")
    lines.append(f"    Total Trades:       {ts.get('total_trades', 0)}")
    lines.append(f"    Win Rate:           {ts.get('win_rate', 0):.1f}%")
    lines.append(f"    Profit Factor:      {ts.get('profit_factor', 0):.3f}")
    lines.append(f"    Total Brokerage:    Rs. {ts.get('total_brokerage', 0):.2f}")
    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines).encode('utf-8')
