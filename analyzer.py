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

    # If yfinance returns multiple levels of columns,
    # keep only the requested ticker.
    if isinstance(data.columns, pd.MultiIndex):
        data = data.xs(ticker, axis=1, level=1)

    return data


def calculate_indicators(data):

    close = data["Close"]

    # Force Close to be a simple Series
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = pd.Series(
        close.to_numpy().flatten(),
        index=data.index
    )

    data["Close"] = close

    data["SMA_20"] = (
        close.rolling(window=20).mean()
    )

    data["SMA_50"] = (
        close.rolling(window=50).mean()
    )

    delta = close.diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss

    data["RSI"] = (
        100 - (100 / (1 + rs))
    )

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

    if score <= -2:
        return "SELL"

    return "HOLD"
