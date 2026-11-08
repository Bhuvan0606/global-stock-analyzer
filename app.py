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
st.caption("Trading setup analyzer v8")


# =========================
# DATA
# =========================

def get_data(symbol, interval):

    periods = {
        "5m": "7d",
        "15m": "30d",
        "30m": "60d",
        "1h": "60d"
    }

    data = yf.download(
        symbol,
        period=periods[interval],
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False
    )

    if data is None or data.empty:
        return None

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    needed = [
        "Close",
        "High",
        "Low",
        "Volume"
    ]

    if not all(x in data.columns for x in needed):
        return None

    for col in needed:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    return data.dropna(
        subset=needed
    )


# =========================
# ANALYSIS
# =========================

def analyze(data):

    close = data["Close"]
    high = data["High"]
    low = data["Low"]

    # Moving averages
    data["SMA20"] = close.rolling(20).mean()
    data["SMA50"] = close.rolling(50).mean()

    # RSI
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    avg_loss = avg_loss.replace(0, np.nan)

    rs = avg_gain / avg_loss

    data["RSI"] = (
        100 - (100 / (1 + rs))
    )

    data["RSI_prev"] = data["RSI"].shift(3)

    # Volume
    data["VolumeAvg"] = (
        data["Volume"].rolling(20).mean()
    )

    data["VolumeRatio"] = (
