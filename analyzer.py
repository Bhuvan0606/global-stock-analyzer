import yfinance as yf
import pandas as pd


def get_stock_data(ticker):
    data = yf.download(ticker, period="1y", progress=False)

    if data.empty:
        return None

    return data


def calculate_indicators(data):
    close = data["Close"]

    data["SMA_20"] = close.rolling(20).mean()
    data["SMA_50"] = close.rolling(50).mean()

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    data["RSI"] = 100 - (100 / (1 + rs))

    return data


def get_signal(data):
    latest = data.iloc[-1]

    score = 0

    if latest["Close"] > latest["SMA_20"]:
        score += 1
    else:
        score -= 1

    if latest["Close"] > latest["SMA_50"]:
        score += 1
    else:
        score -= 1

    if latest["RSI"] < 30:
        score += 2
    elif latest["RSI"] > 70:
        score -= 2

    if score >= 2:
        return "BUY"
    elif score <= -2:
        return "SELL"
    else:
        return "HOLD"
