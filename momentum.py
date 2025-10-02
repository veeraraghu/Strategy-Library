# -*- coding: utf-8 -*-
"""
Created on Wed May 21 08:12:04 2025

@author: User
"""

import pandas as pd
import numpy as np

#Simple Moving Average CRossover
def sma_crossover(data: pd.DataFrame, short_w = 20, long_w = 50, price_column = 'Close'):
    df = data.copy()
    df[f"SMA{short_w}"] = df[price_col].rolling(short_w).mean()
    df[f"SMA{long_w}"] = df[price_col].rolling(long_w).mean()
    df["signal"] = 0
    df.loc[df[f"SMA{short_w}"] > df[f"SMA{long_w}"], "signal"] = 1
    df.loc[df[f"SMA{short_w}"] < df[f"SMA{long_w}"], "signal"] = -1
    return df

#Exponential Moving Average Crossover
def ema_crossover(data: pd.DataFrame, short_window=20, long_window=50, price_col="close"):
    df = data.copy()
    df[f"EMA{short_window}"] = df[price_col].ewm(span=short_window, adjust=False).mean()
    df[f"EMA{long_window}"] = df[price_col].ewm(span=long_window, adjust=False).mean()
    df["signal"] = 0
    df.loc[df[f"EMA{short_window}"] > df[f"EMA{long_window}"], "signal"] = 1
    df.loc[df[f"EMA{short_window}"] < df[f"EMA{long_window}"], "signal"] = -1
    return df

#Moving Average Slope
def ma_slope(data: pd.DataFrame, window=50, price_col="close"):
    df = data.copy()
    df["SMA"] = df[price_col].rolling(window).mean()
    df["slope"] = df["SMA"].diff()
    df["signal"] = 0
    df.loc[df["slope"] > 0, "signal"] = 1
    df.loc[df["slope"] < 0, "signal"] = -1
    return df

#Rate of Change (ROC) or 
def rate_of_change(data: pd.DataFrame, window=10, price_col="close"):
    df = data.copy()
    df["ROC"] = df[price_col].pct_change(periods=window)
    df["signal"] = 0
    df.loc[df["ROC"] > 0, "signal"] = 1
    df.loc[df["ROC"] < 0, "signal"] = -1
    return df

#RSI Trend-Following
def rsi_trend(data: pd.DataFrame, window=14, overbought=70, oversold=30, price_col="close"):
    df = data.copy()
    delta = df[price_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    df["signal"] = 0
    df.loc[df["RSI"] > overbought, "signal"] = 1
    df.loc[df["RSI"] < oversold, "signal"] = -1
    return df

#MACD Momentum
def macd_momentum(data: pd.DataFrame, short_window=12, long_window=26, signal_window=9, price_col="close"):
    df = data.copy()
    df["EMA_short"] = df[price_col].ewm(span=short_window, adjust=False).mean()
    df["EMA_long"] = df[price_col].ewm(span=long_window, adjust=False).mean()
    df["MACD"] = df["EMA_short"] - df["EMA_long"]
    df["Signal"] = df["MACD"].ewm(span=signal_window, adjust=False).mean()
    df["signal"] = 0
    df.loc[df["MACD"] > df["Signal"], "signal"] = 1
    df.loc[df["MACD"] < df["Signal"], "signal"] = -1
    return df

#ADX Trend Strength Filter
def adx_trend_filter(data: pd.DataFrame, window=14, threshold=25, price_col="close"):
    """
    ADX requires High, Low, Close columns.
    """
    df = data.copy()
    df["TR"] = df[["high","low","close"]].max(axis=1) - df[["high","low","close"]].min(axis=1)
    df["+DM"] = np.where(df["high"].diff() > df["low"].diff(), df["high"].diff().clip(lower=0), 0)
    df["-DM"] = np.where(df["low"].diff() > df["high"].diff(), -df["low"].diff().clip(lower=0), 0)
    df["+DI"] = 100 * (df["+DM"].ewm(alpha=1/window).mean() / df["TR"].ewm(alpha=1/window).mean())
    df["-DI"] = 100 * (df["-DM"].ewm(alpha=1/window).mean() / df["TR"].ewm(alpha=1/window).mean())
    df["DX"] = (abs(df["+DI"] - df["-DI"]) / (df["+DI"] + df["-DI"])) * 100
    df["ADX"] = df["DX"].ewm(alpha=1/window).mean()
    df["signal"] = 0
    df.loc[df["ADX"] > threshold, "signal"] = np.sign(df["+DI"] - df["-DI"])
    return df

#Donchian Channel Breakout
def donchian_breakout(data: pd.DataFrame, window=20, price_col="close"):
    df = data.copy()
    df["upper"] = df[price_col].rolling(window).max()
    df["lower"] = df[price_col].rolling(window).min()
    df["signal"] = 0
    df.loc[df[price_col] > df["upper"], "signal"] = 1
    df.loc[df[price_col] < df["lower"], "signal"] = -1
    return df

#Volatility-Adjusted Momentum
def volatility_adjusted_momentum(data: pd.DataFrame, ma_window=20, atr_window=14, k=2, price_col="close"):
    df = data.copy()
    df["SMA"] = df[price_col].rolling(ma_window).mean()
    df["TR"] = df[["high","low","close"]].max(axis=1) - df[["high","low","close"]].min(axis=1)
    df["ATR"] = df["TR"].rolling(atr_window).mean()
    df["upper"] = df["SMA"] + k * df["ATR"]
    df["lower"] = df["SMA"] - k * df["ATR"]
    df["signal"] = 0
    df.loc[df[price_col] > df["upper"], "signal"] = 1
    df.loc[df[price_col] < df["lower"], "signal"] = -1
    return df

#Cross-Sectional Momentum
def cross_sectional_momentum(data_dict: dict, lookback=60, top_n=3):
    """
    Cross-sectional momentum on a dictionary of DataFrames {asset: df}.
    Returns signals per asset based on relative performance ranking.
    """
    signals = {}
    returns = {asset: (df["close"].iloc[-1] / df["close"].iloc[-lookback] - 1) for asset, df in data_dict.items()}
    ranked = sorted(returns.items(), key=lambda x: x[1], reverse=True)
    longs = [a for a, _ in ranked[:top_n]]
    shorts = [a for a, _ in ranked[-top_n:]]
    for asset, df in data_dict.items():
        sig_df = df.copy()
        sig_df["signal"] = 0
        if asset in longs:
            sig_df["signal"] = 1
        elif asset in shorts:
            sig_df["signal"] = -1
        signals[asset] = sig_df
    return signals

#Relative Momentum (vs Benchmark)
def relative_momentum(data: pd.DataFrame, benchmark: pd.Series, window=60, price_col="close"):
    df = data.copy()
    asset_return = df[price_col].pct_change(window)
    bench_return = benchmark.pct_change(window)
    df["signal"] = 0
    df.loc[asset_return > bench_return, "signal"] = 1
    df.loc[asset_return < bench_return, "signal"] = -1
    return df

#Time-Series Momentum
def time_series_momentum(data: pd.DataFrame, window=60, price_col="close"):
    df = data.copy()
    past_return = df[price_col].pct_change(window)
    df["signal"] = 0
    df.loc[past_return > 0, "signal"] = 1
    df.loc[past_return < 0, "signal"] = -1
    return df

#Moving Average Envelope
def ma_envelope(data: pd.DataFrame, window=20, band=0.02, price_col="close"):
    df = data.copy()
    df["SMA"] = df[price_col].rolling(window).mean()
    df["upper"] = df["SMA"] * (1 + band)
    df["lower"] = df["SMA"] * (1 - band)
    df["signal"] = 0
    df.loc[df[price_col] > df["upper"], "signal"] = 1
    df.loc[df[price_col] < df["lower"], "signal"] = -1
    return df

#RSI Crossover System
def rsi_crossover(data: pd.DataFrame, window=14, fast=5, slow=14, price_col="close"):
    df = data.copy()
    delta = df[price_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI_fast"] = df["RSI"].rolling(fast).mean()
    df["RSI_slow"] = df["RSI"].rolling(slow).mean()
    df["signal"] = 0
    df.loc[df["RSI_fast"] > df["RSI_slow"], "signal"] = 1
    df.loc[df["RSI_fast"] < df["RSI_slow"], "signal"] = -1
    return df