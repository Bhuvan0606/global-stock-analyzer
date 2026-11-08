import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np


st.set_page_config(
    page_title="Global Stock Analyzer",
    page_icon="📈",
    layout="wide"
)


st.title("🌍 Global Stock Analyzer")

st.caption(
    "Live-market trading setup analyzer v5"
)


# --------------------------------------------------
# MARKET DATA
# --------------------------------------------------

def get_market_data(ticker, interval):

    ticker = ticker.strip().upper()

    # Yahoo Finance has different intraday data limits.
    if interval == "5m":
        period = "7d"
    elif interval == "15m":
        period = "30d"
    elif interval == "30m":
        period = "60d"
    else:
        period = "60d"

    data = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False
    )

    if data is None or data.empty:
        return None

    data = data.copy()

    # Handle Yahoo Finance MultiIndex columns.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required = [
        "Close",
        "Volume"
    ]

    for column in required:
        if column not in data.columns:
            raise RuntimeError(
                f"Yahoo Finance did not return {column} data."
            )

    data["Close"] = pd.to_numeric(
        data["Close"],
        errors="coerce"
    )

    data["Volume"] = pd.to_numeric(
        data["Volume"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["Close"]
    )

    if data.empty:
        return None

    return data


# --------------------------------------------------
# TECHNICAL INDICATORS
# --------------------------------------------------

def calculate_indicators(data):

    data = data.copy()

    close = data["Close"].astype(float)

    # Moving averages
    data["SMA_20"] = (
        close
        .rolling(
            20,
            min_periods=20
        )
        .mean()
    )

    data["SMA_50"] = (
        close
        .rolling(
            50,
            min_periods=50
        )
        .mean()
    )

    # RSI
    delta = close.diff()

    gains = delta.clip(
        lower=0
    )

    losses = -delta.clip(
        upper=0
    )

    average_gain = (
        gains
        .rolling(
            14,
            min_periods=14
        )
        .mean()
    )

    average_loss = (
        losses
        .rolling(
            14,
            min_periods=14
        )
        .mean()
    )

    average_loss = average_loss.replace(
        0,
        np.nan
    )

    relative_strength = (
        average_gain /
        average_loss
    )

    data["RSI"] = (
        100 -
        (
            100 /
            (
                1 +
                relative_strength
            )
        )
    )

    # Volume
    data["Volume_SMA_20"] = (
        data["Volume"]
        .rolling(
            20,
            min_periods=20
        )
        .mean()
    )

    data["Volume_Ratio"] = (
        data["Volume"] /
        data["Volume_SMA_20"]
    )

    return data


# --------------------------------------------------
# SIGNAL ENGINE
# --------------------------------------------------

def generate_signal(data):

    valid = data.dropna(
        subset=[
            "SMA_20",
            "SMA_50",
            "RSI",
            "Volume_Ratio"
        ]
    )

    if valid.empty:

        return {
            "signal": "WAIT",
            "score": 0,
            "price": float(
                data["Close"].iloc[-1]
            ),
            "rsi": 0,
            "trend": "UNKNOWN",
            "volume_ratio":
