import yfinance as yf
import pandas as pd
import numpy as np


def get_stock_data(ticker, period="30d", interval="15m"):
    try:
        ticker = ticker.strip().upper()

        stock = yf.Ticker(ticker)

        data = stock.history(
            period=period,
            interval=interval,
            auto_adjust=False
        )

        if data is None or data.empty:
            return None

        data = data.copy()

        # Handle possible MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        required = ["Close", "Volume"]

        for column in required:
            if column not in data.columns:
                return None

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

        data = data.dropna(subset=["Close"])

        if data.empty:
            return None

        return data

    except Exception as e:
        raise RuntimeError(
            f"Market data failed for {ticker}: "
            f"{type(e).__name__}: {e}"
        )


def calculate_indicators(data):

    data = data.copy()

    close = data["Close"].astype(float)

    # Moving averages
    data["SMA_20"] = close.rolling(
        window=20,
        min_periods=20
    ).mean()

    data["SMA_50"] = close.rolling(
        window=50,
        min_periods=50
    ).mean()

    # RSI
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(
        window=14,
        min_periods=14
    ).mean()

    avg_loss = loss.rolling(
        window=14,
        min_periods=14
    ).mean()

    avg_loss = avg_loss.replace(
        0,
        np.nan
    )

    rs = avg_gain / avg_loss

    data["RSI"] = 100
