"""
data_loader.py — Stock Universe & Data Fetching Engine
=====================================================
Handles the Nifty 500 stock dictionary, yfinance downloads,
data validation, and all error reporting.

Every failure produces a clear, human-readable error dict:
  {"ok": False, "error": "What went wrong", "hint": "How to fix it"}

Success returns:
  {"ok": True, "data": pd.DataFrame, "info": "summary string"}
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────
#  NIFTY 500 STOCK UNIVERSE (Organized by Sector)
#  Format: "Display Name": "Yahoo Finance Ticker"
#  NSE stocks use .NS suffix, BSE stocks use .BO suffix
# ─────────────────────────────────────────────────────────────

STOCK_UNIVERSE = {

    # ── INDICES (Benchmarks) ──
    "Nifty 50 (Index)": "^NSEI",
    "Sensex 30 (Index)": "^BSESN",
    "Nifty Bank (Index)": "^NSEBANK",
    "Nifty IT (Index)": "^CNXIT",
    "Nifty Midcap 50 (Index)": "^NSEMDCP50",

    # ── BANKING & FINANCE ──
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Axis Bank": "AXISBANK.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "Bank of Baroda": "BANKBARODA.NS",
    "Punjab National Bank": "PNB.NS",
    "Federal Bank": "FEDERALBNK.NS",
    "IDFC First Bank": "IDFCFIRSTB.NS",
    "Canara Bank": "CANBK.NS",
    "AU Small Finance Bank": "AUBANK.NS",
    "Bandhan Bank": "BANDHANBNK.NS",
    "Indian Bank": "INDIANB.NS",
    "Bank of India": "BANKINDIA.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "Shriram Finance": "SHRIRAMFIN.NS",
    "Muthoot Finance": "MUTHOOTFIN.NS",
    "Manappuram Finance": "MANAPPURAM.NS",
    "Cholamandalam Inv.": "CHOLAFIN.NS",
    "LIC Housing Finance": "LICHSGFIN.NS",
    "PNB Housing Finance": "PNBHOUSING.NS",
    "HDFC AMC": "HDFCAMC.NS",
    "SBI Life Insurance": "SBILIFE.NS",
    "HDFC Life Insurance": "HDFCLIFE.NS",
    "ICICI Lombard": "ICICIGI.NS",
    "ICICI Prudential": "ICICIPRULI.NS",
    "General Insurance Corp": "GICRE.NS",
    "New India Assurance": "NIACL.NS",

    # ── IT & TECHNOLOGY ──
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HCL Technologies": "HCLTECH.NS",
    "Wipro": "WIPRO.NS",
    "Tech Mahindra": "TECHM.NS",
    "LTIMindtree": "LTIM.NS",
    "Persistent Systems": "PERSISTENT.NS",
    "Mphasis": "MPHASIS.NS",
    "Coforge": "COFORGE.NS",
    "L&T Technology Services": "LTTS.NS",
    "Tata Elxsi": "TATAELXSI.NS",
    "Happiest Minds": "HAPPSTMNDS.NS",
    "Zensar Technologies": "ZENSARTECH.NS",
    "KPIT Technologies": "KPITTECH.NS",
    "Cyient": "CYIENT.NS",

    # ── AUTOMOBILE ──
    "Tata Motors": "TATAMOTORS.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "TVS Motor": "TVSMOTOR.NS",
    "Ashok Leyland": "ASHOKLEY.NS",
    "MRF": "MRF.NS",
    "Apollo Tyres": "APOLLOTYRE.NS",
    "Balkrishna Industries": "BALKRISIND.NS",
    "Bharat Forge": "BHARATFORG.NS",
    "Motherson Sumi": "MOTHERSON.NS",
    "Samvardhana Motherson": "MOTHERSON.NS",
    "Bosch": "BOSCHLTD.NS",
    "Exide Industries": "EXIDEIND.NS",
    "Amara Raja Energy": "AMARAJABAT.NS",
    "Tube Investments": "TIINDIA.NS",

    # ── PHARMA & HEALTHCARE ──
    "Sun Pharma": "SUNPHARMA.NS",
    "Dr. Reddy's": "DRREDDY.NS",
    "Cipla": "CIPLA.NS",
    "Divi's Labs": "DIVISLAB.NS",
    "Lupin": "LUPIN.NS",
    "Aurobindo Pharma": "AUROPHARMA.NS",
    "Torrent Pharma": "TORNTPHARM.NS",
    "Zydus Lifesciences": "ZYDUSLIFE.NS",
    "Biocon": "BIOCON.NS",
    "Alkem Labs": "ALKEM.NS",
    "Glenmark Pharma": "GLENMARK.NS",
    "Ipca Labs": "IPCALAB.NS",
    "Laurus Labs": "LAURUSLABS.NS",
    "Natco Pharma": "NATCOPHARM.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS",
    "Max Healthcare": "MAXHEALTH.NS",
    "Fortis Healthcare": "FORTIS.NS",
    "Dr. Lal PathLabs": "LALPATHLAB.NS",
    "Metropolis Healthcare": "METROPOLIS.NS",

    # ── FMCG & CONSUMER ──
    "Hindustan Unilever": "HINDUNILVR.NS",
    "ITC": "ITC.NS",
    "Nestle India": "NESTLEIND.NS",
    "Britannia": "BRITANNIA.NS",
    "Dabur India": "DABUR.NS",
    "Marico": "MARICO.NS",
    "Godrej Consumer": "GODREJCP.NS",
    "Colgate-Palmolive": "COLPAL.NS",
    "Tata Consumer": "TATACONSUM.NS",
    "Varun Beverages": "VBL.NS",
    "United Spirits": "UNITDSPR.NS",
    "Emami": "EMAMILTD.NS",
    "Pidilite": "PIDILITIND.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Berger Paints": "BERGEPAINT.NS",

    # ── ENERGY & OIL/GAS ──
    "Reliance Industries": "RELIANCE.NS",
    "ONGC": "ONGC.NS",
    "Indian Oil Corp": "IOC.NS",
    "BPCL": "BPCL.NS",
    "HPCL": "HPCL.NS",
    "GAIL": "GAIL.NS",
    "Petronet LNG": "PETRONET.NS",
    "Oil India": "OIL.NS",
    "NTPC": "NTPC.NS",
    "Power Grid Corp": "POWERGRID.NS",
    "Tata Power": "TATAPOWER.NS",
    "Adani Green Energy": "ADANIGREEN.NS",
    "Adani Power": "ADANIPOWER.NS",
    "JSW Energy": "JSWENERGY.NS",
    "NHPC": "NHPC.NS",
    "SJVN": "SJVN.NS",
    "Torrent Power": "TORNTPOWER.NS",
    "CESC": "CESC.NS",

    # ── METALS & MINING ──
    "Tata Steel": "TATASTEEL.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Hindalco": "HINDALCO.NS",
    "Vedanta": "VEDL.NS",
    "Coal India": "COALINDIA.NS",
    "NMDC": "NMDC.NS",
    "Jindal Steel & Power": "JINDALSTEL.NS",
    "SAIL": "SAIL.NS",
    "National Aluminium": "NATIONALUM.NS",
    "Hindustan Zinc": "HINDZINC.NS",
    "Hindustan Copper": "HINDCOPPER.NS",
    "APL Apollo Tubes": "APLAPOLLO.NS",
    "Ratnamani Metals": "RATNAMANI.NS",

    # ── INFRA & CONSTRUCTION ──
    "Larsen & Toubro": "LT.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Ambuja Cements": "AMBUJACEM.NS",
    "ACC": "ACC.NS",
    "Shree Cement": "SHREECEM.NS",
    "Dalmia Bharat": "DALBHARAT.NS",
    "DLF": "DLF.NS",
    "Godrej Properties": "GODREJPROP.NS",
    "Oberoi Realty": "OBEROIRLTY.NS",
    "Phoenix Mills": "PHOENIXLTD.NS",
    "Prestige Estates": "PRESTIGE.NS",
    "Brigade Enterprises": "BRIGADE.NS",
    "IRB Infra": "IRB.NS",
    "KNR Construction": "KNRCON.NS",
    "NCC": "NCC.NS",

    # ── TELECOM & MEDIA ──
    "Bharti Airtel": "BHARTIARTL.NS",
    "Jio Financial Services": "JIOFIN.NS",
    "Vodafone Idea": "IDEA.NS",
    "Indus Towers": "INDUSTOWER.NS",
    "Tata Communications": "TATACOMM.NS",
    "Zee Entertainment": "ZEEL.NS",
    "PVR INOX": "PVRINOX.NS",

    # ── CHEMICALS & FERTILIZERS ──
    "SRF": "SRF.NS",
    "PI Industries": "PIIND.NS",
    "UPL": "UPL.NS",
    "Aarti Industries": "AARTIIND.NS",
    "Deepak Nitrite": "DEEPAKNTR.NS",
    "Navin Fluorine": "NAVINFLUOR.NS",
    "Clean Science": "CLEAN.NS",
    "Gujarat Fluorochemicals": "FLUOROCHEM.NS",
    "Chambal Fertilisers": "CHAMBLFERT.NS",
    "Coromandel International": "COROMANDEL.NS",

    # ── DEFENCE & RAILWAYS ──
    "HAL (Hindustan Aero)": "HAL.NS",
    "BEL (Bharat Electronics)": "BEL.NS",
    "Bharat Dynamics": "BDL.NS",
    "Cochin Shipyard": "COCHINSHIP.NS",
    "Mazagon Dock": "MAZDOCK.NS",
    "IRCTC": "IRCTC.NS",
    "RVNL": "RVNL.NS",
    "IRFC": "IRFC.NS",
    "Titagarh Rail": "TITAGARH.NS",

    # ── CONGLOMERATE & OTHERS ──
    "Adani Wilmar": "AWL.NS",
    "Siemens India": "SIEMENS.NS",
    "ABB India": "ABB.NS",
    "Honeywell Automation": "HONAUT.NS",
    "Havells India": "HAVELLS.NS",
    "Polycab India": "POLYCAB.NS",
    "Crompton Greaves CE": "CROMPTON.NS",
    "Dixon Technologies": "DIXON.NS",
    "Kaynes Technology": "KAYNES.NS",
    "Voltas": "VOLTAS.NS",
    "Blue Star": "BLUESTARCO.NS",
    "Whirlpool India": "WHIRLPOOL.NS",
    "Titan Company": "TITAN.NS",
    "Kalyan Jewellers": "KALYANKJIL.NS",
    "Trent (Westside/Zudio)": "TRENT.NS",
    "Avenue Supermarts (DMart)": "DMART.NS",
    "Zomato": "ZOMATO.NS",
    "Paytm (One97 Comm)": "PAYTM.NS",
    "Nykaa (FSN E-Commerce)": "NYKAA.NS",
    "Info Edge (Naukri)": "NAUKRI.NS",
    "Policy Bazaar (PB Fintech)": "POLICYBZR.NS",
    "Delhivery": "DELHIVERY.NS",
    "Page Industries": "PAGEIND.NS",
    "InterGlobe Aviation (IndiGo)": "INDIGO.NS",
    "Indian Hotels (Taj)": "INDHOTEL.NS",
    "Lemon Tree Hotels": "LEMONTREE.NS",
    "CDSL": "CDSL.NS",
    "BSE Ltd": "BSE.NS",
    "MCX India": "MCX.NS",
    "CAMS": "CAMS.NS",
}

# ── SECTOR GROUPINGS (for sidebar filtering) ──
SECTOR_MAP = {
    "All Stocks": None,
    "Indices (Benchmarks)": ["Nifty 50 (Index)", "Sensex 30 (Index)", "Nifty Bank (Index)", "Nifty IT (Index)", "Nifty Midcap 50 (Index)"],
    "Banking & Finance": [k for k in STOCK_UNIVERSE if STOCK_UNIVERSE[k].replace(".NS","") in [
        "HDFCBANK","ICICIBANK","SBIN","KOTAKBANK","AXISBANK","INDUSINDBK","BANKBARODA","PNB",
        "FEDERALBNK","IDFCFIRSTB","CANBK","AUBANK","BANDHANBNK","INDIANB","BANKINDIA",
        "BAJFINANCE","BAJAJFINSV","SHRIRAMFIN","MUTHOOTFIN","MANAPPURAM","CHOLAFIN",
        "LICHSGFIN","PNBHOUSING","HDFCAMC","SBILIFE","HDFCLIFE","ICICIGI","ICICIPRULI","GICRE","NIACL"
    ]],
    "IT & Technology": [k for k in STOCK_UNIVERSE if STOCK_UNIVERSE[k].replace(".NS","") in [
        "TCS","INFY","HCLTECH","WIPRO","TECHM","LTIM","PERSISTENT","MPHASIS","COFORGE",
        "LTTS","TATAELXSI","HAPPSTMNDS","ZENSARTECH","KPITTECH","CYIENT"
    ]],
    "Automobile": [k for k in STOCK_UNIVERSE if STOCK_UNIVERSE[k].replace(".NS","") in [
        "TATAMOTORS","MARUTI","M&M","BAJAJ-AUTO","HEROMOTOCO","EICHERMOT","TVSMOTOR",
        "ASHOKLEY","MRF","APOLLOTYRE","BALKRISIND","BHARATFORG","MOTHERSON","BOSCHLTD",
        "EXIDEIND","AMARAJABAT","TIINDIA"
    ]],
    "Pharma & Healthcare": [k for k in STOCK_UNIVERSE if STOCK_UNIVERSE[k].replace(".NS","") in [
        "SUNPHARMA","DRREDDY","CIPLA","DIVISLAB","LUPIN","AUROPHARMA","TORNTPHARM","ZYDUSLIFE",
        "BIOCON","ALKEM","GLENMARK","IPCALAB","LAURUSLABS","NATCOPHARM","APOLLOHOSP",
        "MAXHEALTH","FORTIS","LALPATHLAB","METROPOLIS"
    ]],
    "FMCG & Consumer": [k for k in STOCK_UNIVERSE if STOCK_UNIVERSE[k].replace(".NS","") in [
        "HINDUNILVR","ITC","NESTLEIND","BRITANNIA","DABUR","MARICO","GODREJCP","COLPAL",
        "TATACONSUM","VBL","UNITDSPR","EMAMILTD","PIDILITIND","ASIANPAINT","BERGEPAINT"
    ]],
    "Energy & Power": [k for k in STOCK_UNIVERSE if STOCK_UNIVERSE[k].replace(".NS","") in [
        "RELIANCE","ONGC","IOC","BPCL","HPCL","GAIL","PETRONET","OIL","NTPC","POWERGRID",
        "TATAPOWER","ADANIGREEN","ADANIPOWER","JSWENERGY","NHPC","SJVN","TORNTPOWER","CESC"
    ]],
    "Metals & Mining": [k for k in STOCK_UNIVERSE if STOCK_UNIVERSE[k].replace(".NS","") in [
        "TATASTEEL","JSWSTEEL","HINDALCO","VEDL","COALINDIA","NMDC","JINDALSTEL","SAIL",
        "NATIONALUM","HINDZINC","HINDCOPPER","APLAPOLLO","RATNAMANI"
    ]],
    "Infra & Realty": [k for k in STOCK_UNIVERSE if STOCK_UNIVERSE[k].replace(".NS","") in [
        "LT","ADANIENT","ADANIPORTS","ULTRACEMCO","AMBUJACEM","ACC","SHREECEM","DALBHARAT",
        "DLF","GODREJPROP","OBEROIRLTY","PHOENIXLTD","PRESTIGE","BRIGADE","IRB","KNRCON","NCC"
    ]],
    "Defence & Railways": [k for k in STOCK_UNIVERSE if STOCK_UNIVERSE[k].replace(".NS","") in [
        "HAL","BEL","BDL","COCHINSHIP","MAZDOCK","IRCTC","RVNL","IRFC","TITAGARH"
    ]],
}

# ── TIMEFRAME CONFIGURATIONS ──
TIMEFRAME_OPTIONS = {
    "1 Minute":   {"interval": "1m",  "max_days": 7,     "label": "1m"},
    "2 Minutes":  {"interval": "2m",  "max_days": 59,    "label": "2m"},
    "5 Minutes":  {"interval": "5m",  "max_days": 59,    "label": "5m"},
    "15 Minutes": {"interval": "15m", "max_days": 59,    "label": "15m"},
    "30 Minutes": {"interval": "30m", "max_days": 59,    "label": "30m"},
    "1 Hour":     {"interval": "1h",  "max_days": 729,   "label": "1h"},
    "1 Day":      {"interval": "1d",  "max_days": 10000, "label": "1d"},
    "1 Week":     {"interval": "1wk", "max_days": 10000, "label": "1wk"},
    "1 Month":    {"interval": "1mo", "max_days": 10000, "label": "1mo"},
}


# ─────────────────────────────────────────────────────────────
#  CORE DATA FETCHING FUNCTION
# ─────────────────────────────────────────────────────────────

def fetch_stock_data(ticker: str, start_date, end_date, interval: str = "1d") -> dict:
    """
    Downloads OHLCV data from Yahoo Finance with full error handling.

    Returns:
        dict with keys:
          "ok"    : bool   — True if data was fetched successfully
          "data"  : pd.DataFrame or None
          "error" : str or None — Human-readable error message
          "hint"  : str or None — Actionable fix suggestion
          "info"  : str or None — Summary of what was fetched
    """

    # ── STEP 1: Validate inputs ──
    if not ticker or not isinstance(ticker, str) or ticker.strip() == "":
        return {
            "ok": False, "data": None,
            "error": "No ticker symbol provided.",
            "hint": "Please select a stock from the dropdown or type a valid NSE/BSE ticker.",
            "info": None
        }

    if start_date >= end_date:
        return {
            "ok": False, "data": None,
            "error": f"Start date ({start_date}) is on or after End date ({end_date}).",
            "hint": "Make sure your Start Date is before your End Date.",
            "info": None
        }

    # ── STEP 2: Check timeframe limits ──
    days_requested = (end_date - start_date).days
    tf_config = None
    for tf_name, tf_data in TIMEFRAME_OPTIONS.items():
        if tf_data["interval"] == interval:
            tf_config = tf_data
            break

    if tf_config and days_requested > tf_config["max_days"]:
        return {
            "ok": False, "data": None,
            "error": f"Yahoo Finance limits '{interval}' data to {tf_config['max_days']} days. You requested {days_requested} days.",
            "hint": f"Either reduce your date range to {tf_config['max_days']} days, or switch to a longer timeframe (e.g., 1 Day or 1 Hour).",
            "info": None
        }

    # ── STEP 3: Attempt download ──
    try:
        data = yf.download(
            ticker,
            start=str(start_date),
            end=str(end_date),
            interval=interval,
            auto_adjust=True,
            progress=False
        )
    except Exception as e:
        error_str = str(e).lower()
        if "no timezone" in error_str or "connection" in error_str or "urlopen" in error_str:
            return {
                "ok": False, "data": None,
                "error": f"Network error while fetching '{ticker}': {e}",
                "hint": "Check your internet connection. Yahoo Finance may also be temporarily down. Try again in 30 seconds.",
                "info": None
            }
        else:
            return {
                "ok": False, "data": None,
                "error": f"Unexpected error fetching '{ticker}': {e}",
                "hint": "This ticker may not exist on Yahoo Finance. Try adding '.NS' (NSE) or '.BO' (BSE) suffix.",
                "info": None
            }

    # ── STEP 4: Validate downloaded data ──
    if data is None or data.empty:
        return {
            "ok": False, "data": None,
            "error": f"No data returned for ticker '{ticker}' in the selected date range.",
            "hint": "Possible causes:\n• The ticker symbol is incorrect (try RELIANCE.NS instead of RELIANCE)\n• The stock was not listed during this date range\n• Yahoo Finance has no data for this combination of ticker + timeframe",
            "info": None
        }

    # ── STEP 5: Flatten MultiIndex columns (yfinance sometimes returns these) ──
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # ── STEP 6: Check required columns exist ──
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [c for c in required_cols if c not in data.columns]
    if missing_cols:
        return {
            "ok": False, "data": None,
            "error": f"Downloaded data is missing required columns: {missing_cols}",
            "hint": "This is unusual. Try a different ticker or timeframe. Yahoo Finance may have incomplete data for this stock.",
            "info": None
        }

    # ── STEP 7: Drop rows where Close is NaN (non-trading days) ──
    initial_rows = len(data)
    data = data.dropna(subset=['Close'])
    dropped_rows = initial_rows - len(data)

    if data.empty:
        return {
            "ok": False, "data": None,
            "error": f"All {initial_rows} rows had NaN Close prices for '{ticker}'.",
            "hint": "This stock may have been suspended or delisted. Try a different date range.",
            "info": None
        }

    # ── STEP 8: Ensure Volume is numeric and fill zeros ──
    data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce').fillna(0).astype(int)

    # ── STEP 9: Sort by date (safety) ──
    data = data.sort_index()

    # ── SUCCESS ──
    info_parts = [
        f"Fetched {len(data)} candles for '{ticker}'",
        f"from {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}",
        f"({interval} timeframe)"
    ]
    if dropped_rows > 0:
        info_parts.append(f"({dropped_rows} empty rows removed)")

    return {
        "ok": True,
        "data": data,
        "error": None,
        "hint": None,
        "info": " | ".join(info_parts)
    }


# ─────────────────────────────────────────────────────────────
#  HELPER: Search stocks by name
# ─────────────────────────────────────────────────────────────

def search_stocks(query: str) -> dict:
    """
    Searches the STOCK_UNIVERSE dictionary by partial name match.
    Returns a filtered dict of matching stocks.
    """
    if not query or query.strip() == "":
        return STOCK_UNIVERSE

    query = query.strip().lower()
    return {
        name: ticker
        for name, ticker in STOCK_UNIVERSE.items()
        if query in name.lower() or query in ticker.lower()
    }


# ─────────────────────────────────────────────────────────────
#  HELPER: Get stocks by sector
# ─────────────────────────────────────────────────────────────

def get_stocks_by_sector(sector: str) -> dict:
    """
    Returns stocks belonging to a given sector.
    If sector is "All Stocks", returns everything.
    """
    if sector == "All Stocks" or sector not in SECTOR_MAP:
        return STOCK_UNIVERSE

    stock_names = SECTOR_MAP[sector]
    if stock_names is None:
        return STOCK_UNIVERSE

    return {name: STOCK_UNIVERSE[name] for name in stock_names if name in STOCK_UNIVERSE}


# ─────────────────────────────────────────────────────────────
#  HELPER: Validate MA length against data size
# ─────────────────────────────────────────────────────────────

def validate_ma_config(data_length: int, fast_len: int, slow_len: int) -> dict:
    """
    Checks if the MA configuration is valid for the given data size.
    Returns {"ok": True} or {"ok": False, "error": ..., "hint": ...}
    """
    if fast_len >= slow_len:
        return {
            "ok": False,
            "error": f"Fast MA length ({fast_len}) must be LESS than Slow MA length ({slow_len}).",
            "hint": "The Fast MA reacts quickly, the Slow MA confirms the trend. Fast must always be smaller."
        }

    if slow_len >= data_length:
        return {
            "ok": False,
            "error": f"Slow MA length ({slow_len}) is greater than available data ({data_length} candles).",
            "hint": f"Either reduce your Slow MA length to below {data_length}, or increase your date range to get more data."
        }

    min_required = slow_len + 10  # Need at least 10 candles after the MA warms up
    if data_length < min_required:
        return {
            "ok": False,
            "error": f"Not enough data. You have {data_length} candles but need at least {min_required} for Slow MA({slow_len}).",
            "hint": "Increase your date range or reduce the Slow MA length."
        }

    return {"ok": True, "error": None, "hint": None}
