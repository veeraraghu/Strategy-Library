# -*- coding: utf-8 -*-
"""
Created on Mon Jun 02 19:46:28 2025

@author: User
"""
import pandas as pd
import numpy as np

#Bollinger Bands Reversion
def bollinger_reversion(data: pd.DataFrame, window=20, num_std=2, price_col="close"):
    df = data.copy()
    rolling_mean = df[price_col].rolling(window).mean()
    rolling_std = df[price_col].rolling(window).std()
    df["upper_band"] = rolling_mean + num_std * rolling_std
    df["lower_band"] = rolling_mean - num_std * rolling_std
    df["signal"] = 0
    df.loc[df[price_col] < df["lower_band"], "signal"] = 1
    df.loc[df[price_col] > df["upper_band"], "signal"] = -1
    return df

#RSI Mean-Reversion
def rsi_reversion(data: pd.DataFrame, window=14, overbought=70, oversold=30, price_col="close"):
    df = data.copy()
    delta = df[price_col].diff()
    gain = delta.clip(a_min=0)
    loss = -(delta.clip(a_max=0))
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    df["signal"] = 0
    df.loc[df["RSI"] < oversold, "signal"] = 1
    df.loc[df["RSI"] > overbought, "signal"] = -1
    return df

#Moving Average Reversion
def moving_average_reversion(data: pd.DataFrame, window=20, price_col="close"):
    df = data.copy()
    df["SMA"] = df[price_col].rolling(window).mean()
    df["signal"] = 0
    df.loc[df[price_col] < df["SMA"], "signal"] = 1
    df.loc[df[price_col] > df["SMA"], "signal"] = -1
    return df

#Z-Score Reversion
def zscore_reversion(data: pd.DataFrame, window=20, threshold=2.0, price_col="close"):
    df = data.copy()
    rolling_mean = df[price_col].rolling(window).mean()
    rolling_std = df[price_col].rolling(window).std()
    df["zscore"] = (df[price_col] - rolling_mean) / rolling_std
    df["signal"] = 0
    df.loc[df["zscore"] < -threshold, "signal"] = 1
    df.loc[df["zscore"] > threshold, "signal"] = -1
    return df

#Keltner Channel Reversion
def keltner_reversion(data: pd.DataFrame, window=20, atr_window=14, multiplier=2, price_col="close"):
    df = data.copy()
    df["EMA"] = df[price_col].ewm(span=window, adjust=False).mean()
    df["TR"] = df[["high","low","close"]].max(axis=1) - df[["high","low","close"]].min(axis=1)
    df["ATR"] = df["TR"].rolling(atr_window).mean()
    df["upper_band"] = df["EMA"] + multiplier * df["ATR"]
    df["lower_band"] = df["EMA"] - multiplier * df["ATR"]
    df["signal"] = 0
    df.loc[df[price_col] < df["lower_band"], "signal"] = 1
    df.loc[df[price_col] > df["upper_band"], "signal"] = -1
    return df

#VWAP Reversion (intraday only)
def vwap_reversion(data: pd.DataFrame, price_col="close"):
    """
    Intraday strategy: assumes 'volume' column exists.
    """
    df = data.copy()
    df["cum_vol"] = df["volume"].cumsum()
    df["cum_pv"] = (df[price_col] * df["volume"]).cumsum()
    df["VWAP"] = df["cum_pv"] / df["cum_vol"]
    df["signal"] = 0
    df.loc[df[price_col] < df["VWAP"], "signal"] = 1
    df.loc[df[price_col] > df["VWAP"], "signal"] = -1
    return df

#Overnight Gap Reversion
def overnight_gap_reversion(data: pd.DataFrame):
    """
    Requires 'open' and 'close' columns.
    """
    df = data.copy()
    df["gap"] = df["open"] - df["close"].shift(1)
    df["signal"] = 0
    df.loc[df["gap"] > 0, "signal"] = -1  # gap up → short
    df.loc[df["gap"] < 0, "signal"] = 1   # gap down → long
    return df

#Volatility Spike Reversion
def volatility_spike_reversion(data: pd.DataFrame, atr_window=14, multiplier=2, price_col="close"):
    df = data.copy()
    df["TR"] = df[["high","low","close"]].max(axis=1) - df[["high","low","close"]].min(axis=1)
    df["ATR"] = df["TR"].rolling(atr_window).mean()
    df["daily_return"] = df[price_col].pct_change()
    threshold = multiplier * df["ATR"] / df[price_col]
    df["signal"] = 0
    df.loc[df["daily_return"] < -threshold, "signal"] = 1
    df.loc[df["daily_return"] > threshold, "signal"] = -1
    return df

#Polynomial Regression Reversion
def poly_regression_reversion(data: pd.DataFrame, window=20, degree=2, price_col="close"):
    df = data.copy()
    df["signal"] = 0
    for i in range(window, len(df)):
        y = df[price_col].iloc[i-window:i].values
        x = np.arange(window)
        coeffs = np.polyfit(x, y, degree)
        fit_val = np.polyval(coeffs, window-1)
        deviation = df[price_col].iloc[i] - fit_val
        if deviation < -np.std(y):
            df.loc[df.index[i], "signal"] = 1
        elif deviation > np.std(y):
            df.loc[df.index[i], "signal"] = -1
    return df

#Kalman Filter Reversion
def kalman_reversion(data: pd.DataFrame, price_col="close", delta=1e-5, Ve=0.001):
    """
    Simple Kalman filter mean-reversion.
    Estimates dynamic mean and trades on deviation.
    """
    df = data.copy()
    n = len(df)
    price = df[price_col].values
    Q = delta / (1 - delta) * np.var(price)
    xhat = np.zeros(n)
    P = np.zeros(n)
    xhat[0] = price[0]
    P[0] = 1.0
    for t in range(1, n):
        xhatminus = xhat[t-1]
        Pminus = P[t-1] + Q
        K = Pminus / (Pminus + Ve)
        xhat[t] = xhatminus + K * (price[t] - xhatminus)
        P[t] = (1 - K) * Pminus
    df["kalman_mean"] = xhat
    df["signal"] = 0
    df.loc[df[price_col] < df["kalman_mean"], "signal"] = 1
    df.loc[df[price_col] > df["kalman_mean"], "signal"] = -1
    return df