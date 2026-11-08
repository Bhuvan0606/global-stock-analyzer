import yfinance as yf
import pandas as pd


def get_stock_data(ticker):

    data = yf.download(
        ticker,
        period="1y",
        progress=False,
        auto_adjust=False
    )

    if data.empty:
        return None

    # Convert yfinance data to a simple table
    if isinstance(data.columns, pd.MultiIndex):
        data = data.iloc[:, data.columns.get_level_values(1) == ticker]
        data.columns = data.columns.get_level_values(0)

    return data


def calculate_indicators(data):

    close = data["Close"]

    # Make absolutely sure Close is one-dimensional
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = pd.Series(close, index=data.index, dtype=float)

    data["Close"] = close

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

    close = float(latest["Close"])
    sma20 = float(latest["SMA_20"])
    sma50 = float(latest["SMA_50"])
    rsi = float(latest["RSI"])

    score = 0

    if close > sma20:
        score += 1
    else:
        score -= 1

    if close > sma50:
        score += 1
    else:
        score -= 1

    if rsi < 30:
        score += 2
    elif rsi > 70:
        score -= 2

    if score >= 2:
        return "BUY"

    elif score <= -2:
        return "SELL"

    else:
        return "HOLD"
