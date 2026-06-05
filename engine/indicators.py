import pandas as pd
import numpy as np

def calculate_sma(series: pd.Series, length: int) -> pd.Series:
    """Simple Moving Average (SMA)"""
    return series.rolling(window=length, min_periods=1).mean()

def calculate_ema(series: pd.Series, length: int) -> pd.Series:
    """Exponential Moving Average (EMA)"""
    return series.ewm(span=length, adjust=False, min_periods=1).mean()

def calculate_dema(series: pd.Series, length: int) -> pd.Series:
    """Double Exponential Moving Average (DEMA)"""
    ema1 = calculate_ema(series, length)
    ema2 = calculate_ema(ema1, length)
    return (2 * ema1) - ema2

def calculate_wma(series: pd.Series, length: int) -> pd.Series:
    """Weighted Moving Average (WMA) using high-speed vectorized convolution"""
    if len(series) < length:
        return pd.Series(np.nan, index=series.index)
        
    weights = np.arange(1, length + 1)
    weights = weights / weights.sum()
    
    # Mode 'valid' means it only calculates where the window fully overlaps
    wma = np.convolve(series.values, weights[::-1], mode='valid')
    
    # Pad the beginning with NaNs to align perfectly with original series
    pad = np.empty(length - 1)
    pad[:] = np.nan
    wma_full = np.concatenate((pad, wma))
    
    return pd.Series(wma_full, index=series.index)

def calculate_hma(series: pd.Series, length: int) -> pd.Series:
    """Hull Moving Average (HMA) - Near Zero Lag"""
    # HMA requires at minimum length=2 (half_length must be >= 1)
    if length < 2:
        return series.copy()
    
    half_length = max(int(length / 2), 1)
    sqrt_length = max(int(np.sqrt(length)), 1)
    
    if len(series) < length + sqrt_length:
         return pd.Series(np.nan, index=series.index)
         
    wmaf = calculate_wma(series, half_length)
    wmas = calculate_wma(series, length)
    
    diff = (2 * wmaf) - wmas
    
    # Fill backwards to prevent cascading NaNs in the final WMA pass
    diff_filled = diff.bfill()
    
    return calculate_wma(diff_filled, sqrt_length)

def calculate_vwap(df: pd.DataFrame, length: int = None) -> pd.Series:
    """Volume Weighted Average Price (VWAP) - Rolling or Cumulative"""
    if not all(col in df.columns for col in ['High', 'Low', 'Close', 'Volume']):
        raise ValueError("VWAP requires 'High', 'Low', 'Close', and 'Volume' columns.")
        
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vp = typical_price * df['Volume']
    
    if length is None or length == 0:
        # Cumulative VWAP (from start of dataset)
        cumulative_vp = vp.cumsum()
        cumulative_vol = df['Volume'].cumsum()
        return cumulative_vp / cumulative_vol
    else:
        # Rolling VWAP
        rolling_vp = vp.rolling(window=length, min_periods=1).sum()
        rolling_vol = df['Volume'].rolling(window=length, min_periods=1).sum()
        return rolling_vp / rolling_vol

def calculate_rsi(series: pd.Series, length: int) -> pd.Series:
    """Relative Strength Index (RSI) using Wilder's Smoothing"""
    if len(series) < length:
        return pd.Series(np.nan, index=series.index)
        
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    # Wilder's smoothing is exactly equivalent to an EMA with alpha = 1/length
    avg_gain = gain.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    
    # Handle divide by zero
    rs = avg_gain / avg_loss
    rsi = np.where(avg_loss == 0, 100.0, 100.0 - (100.0 / (1.0 + rs)))
    
    return pd.Series(rsi, index=series.index)

def calculate_rvol(volume: pd.Series, length: int) -> pd.Series:
    """Relative Volume (RVOL): ratio of current volume to its N-period SMA."""
    if len(volume) < length:
        return pd.Series(1.0, index=volume.index)
        
    avg_vol = volume.rolling(window=length, min_periods=1).mean()
    # Avoid division by zero by replacing 0 with small epsilon, or fillna
    rvol = (volume / avg_vol.replace(0, np.nan)).fillna(0)
    return rvol

def get_indicator(df: pd.DataFrame, ma_type: str, length: int) -> pd.Series:
    """Master Routing Function"""
    ma_type = ma_type.upper()
    close = df['Close']
    
    if ma_type == 'SMA':
        return calculate_sma(close, length)
    elif ma_type == 'EMA':
        return calculate_ema(close, length)
    elif ma_type == 'DEMA':
        return calculate_dema(close, length)
    elif ma_type == 'WMA':
        return calculate_wma(close, length)
    elif ma_type == 'HMA':
        return calculate_hma(close, length)
    elif ma_type == 'VWAP':
        return calculate_vwap(df, length)
    else:
        raise ValueError(f"Unsupported MA Type: {ma_type}")
