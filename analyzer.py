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

        return data

    except Exception as e:
        print("Market data error:", repr(e))
        return None


def calculate_indicators(data):

    data = data.copy()

    close = data["Close"].astype(float)

    data["SMA_20"] = close.rolling(
        20,
        min_periods=20
    ).mean()

    data["SMA_50"] = close.rolling(
        50,
        min_periods=50
    ).mean()

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(
        14,
        min_periods=14
    ).mean()

    avg_loss = loss.rolling(
        14,
        min_periods=14
    ).mean()

    avg_loss = avg_loss.replace(
        0,
        np.nan
    )

    rs = avg_gain / avg_loss

    data["RSI"] = 100 - (
        100 / (1 + rs)
    )

    data["Volume_SMA_20"] = (
        data["Volume"]
        .rolling(20, min_periods=20)
        .mean()
    )

    data["Volume_Ratio"] = (
        data["Volume"] /
        data["Volume_SMA_20"]
    )

    return data


def get_signal(data):

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
            "price": float(data["Close"].iloc[-1]),
            "rsi": 0,
            "trend": "UNKNOWN",
            "volume_ratio": 0,
            "entry": 0,
            "stop_loss": 0,
            "target1": 0,
            "target2": 0,
            "reasons": [
                "Not enough data to calculate the indicators."
            ]
        }

    latest = valid.iloc[-1]

    close = float(latest["Close"])
    sma20 = float(latest["SMA_20"])
    sma50 = float(latest["SMA_50"])
    rsi = float(latest["RSI"])
    volume_ratio = float(latest["Volume_Ratio"])

    score = 50
    reasons = []

    if close > sma20:
        score += 10
        reasons.append(
            "Price is above the 20-period average."
        )
    else:
        score -= 10
        reasons.append(
            "Price is below the 20-period average."
        )

    if sma20 > sma50:
        score += 10
        reasons.append(
            "Short-term trend is above the medium-term trend."
        )
    else:
        score -= 10
        reasons.append(
            "Short-term trend is below the medium-term trend."
        )

    if 50 <= rsi <= 70:
        score += 10
        reasons.append(
            "Momentum is positive."
        )
    elif rsi < 30:
        score += 5
        reasons.append(
            "RSI indicates potentially oversold conditions."
        )
    elif rsi > 70:
        score -= 10
        reasons.append(
            "RSI indicates potentially overbought conditions."
        )

    if volume_ratio >= 1.2:
        score += 10
        reasons.append(
            "Volume is above its recent average."
        )
    elif volume_ratio < 0.8:
        score -= 5
        reasons.append(
            "Volume is below its recent average."
        )

    score = max(0, min(100, score))

    if score >= 75:
        signal = "BUY SETUP"
    elif score <= 35:
        signal = "EXIT / SELL SETUP"
    else:
        signal = "WAIT"

    if signal == "BUY SETUP":
        stop_loss = close * 0.98
        target1 = close * 1.03
        target2 = close * 1.06
    else:
        stop_loss = close
        target1 = close
        target2 = close

    return {
        "signal": signal,
        "score": score,
        "price": close,
        "rsi": rsi,
        "trend": (
            "BULLISH"
            if sma20 > sma50
            else "BEARISH"
        ),
        "volume_ratio": volume_ratio,
        "entry": close,
        "stop_loss": stop_loss,
        "target1": target1,
        "target2": target2,
        "reasons": reasons
    }
