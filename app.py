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
st.caption("Live-market trading setup analyzer v6")


def get_data(symbol, interval):

    if interval == "5m":
        period = "7d"
    elif interval == "15m":
        period = "30d"
    else:
        period = "60d"

    data = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False
    )

    if data is None or data.empty:
        return None

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if "Close" not in data.columns:
        return None

    if "Volume" not in data.columns:
        return None

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

    return data


def analyze(data):

    close = data["Close"]

    data["SMA20"] = close.rolling(20).mean()
    data["SMA50"] = close.rolling(50).mean()

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    avg_loss = avg_loss.replace(0, np.nan)

    rs = avg_gain / avg_loss

    data["RSI"] = 100 - (
        100 / (1 + rs)
    )

    data["VolumeAvg"] = (
        data["Volume"]
        .rolling(20)
        .mean()
    )

    data["VolumeRatio"] = (
        data["Volume"] /
        data["VolumeAvg"]
    )

    valid = data.dropna(
        subset=[
            "SMA20",
            "SMA50",
            "RSI",
            "VolumeRatio"
        ]
    )

    if valid.empty:
        return None

    latest = valid.iloc[-1]

    price = float(latest["Close"])
    sma20 = float(latest["SMA20"])
    sma50 = float(latest["SMA50"])
    rsi = float(latest["RSI"])
    volume_ratio = float(latest["VolumeRatio"])

    score = 50
    reasons = []

    if price > sma20:
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
        trend = "BULLISH"
    else:
        score -= 10
        reasons.append(
            "Short-term trend is below the medium-term trend."
        )
        trend = "BEARISH"

    if 50 <= rsi <= 70:
        score += 10
        reasons.append(
            "Momentum is positive."
        )
    elif rsi < 30:
        score += 5
        reasons.append(
            "RSI shows potentially oversold conditions."
        )
    elif rsi > 70:
        score -= 10
        reasons.append(
            "RSI shows potentially overbought conditions."
        )
    else:
        reasons.append(
            "Momentum is neutral."
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
    else:
        reasons.append(
            "Volume is close to its recent average."
        )

    score = max(0, min(100, score))

    if score >= 75:
        signal = "BUY SETUP"
    elif score <= 35:
        signal = "EXIT / SELL SETUP"
    else:
        signal = "WAIT"

    entry = price

    if signal == "BUY SETUP":
        stop = price * 0.98
        target1 = price * 1.03
        target2 = price * 1.06
    else:
        stop = price
        target1 = price
        target2 = price

    return {
        "signal": signal,
        "score": score,
        "price": price,
        "rsi": rsi,
        "trend": trend,
        "volume": volume_ratio,
        "entry": entry,
        "stop": stop,
        "target1": target1,
        "target2": target2,
        "reasons": reasons
    }


symbol = st.text_input(
    "Enter stock symbol",
    placeholder="AAPL, MSFT, RELIANCE.NS, TCS.NS"
)

interval = st.selectbox(
    "Market timeframe",
    ["5m", "15m", "30m", "1h"],
    index=1
)


if st.button("🔎 Analyze Market"):

    if not symbol:
        st.warning("Enter a stock symbol first.")
        st.stop()

    symbol = symbol.strip().upper()

    with st.spinner("Downloading market data..."):

        try:
            data = get_data(
                symbol,
                interval
            )
        except Exception as error:
            st.error("Market data request failed.")
            st.code(
                f"{type(error).__name__}: {error}"
            )
            st.stop()

    if data is None:
        st.error(
            "No market data found. "
            "Check the symbol and timeframe."
        )
        st.stop()

    with st.spinner("Analyzing market..."):

        try:
            result = analyze(data)
        except Exception as error:
            st.error("Analysis failed.")
            st.code(
                f"{type(error).__name__}: {error}"
            )
            st.stop()

    if result is None:
        st.warning(
            "Not enough data to calculate indicators."
        )
        st.stop()

    st.success(
        f"Market analysis completed for {symbol}"
    )

    signal = result["signal"]

    if signal == "BUY SETUP":
        st.success(f"🟢 {signal}")
    elif signal == "EXIT / SELL SETUP":
        st.error(f"🔴 {signal}")
    else:
        st.warning(f"🟡 {signal}")

    st.subheader("Market Overview")

    a, b, c, d = st.columns(4)

    a.metric(
        "Price",
        f"{result['price']:.2f}"
    )

    b.metric(
        "Setup Score",
        f"{result['score']}/100"
    )

    c.metric(
        "RSI",
        f"{result['rsi']:.1f}"
    )

    d.metric(
        "Trend",
        result["trend"]
    )

    st.subheader("Trading Levels")

    a, b, c, d = st.columns(4)

    a.metric(
        "Entry",
        f"{result['entry']:.2f}"
    )

    b.metric(
        "Stop Loss",
        f"{result['stop']:.2f}"
    )

    c.metric(
        "Target 1",
        f"{result['target1']:.2f}"
    )

    d.metric(
        "Target 2",
        f"{result['target2']:.2f}"
    )

    st.subheader("Market Conditions")

    st.write(
        f"**Trend:** {result['trend']}"
    )

    st.write(
        f"**Volume strength:** "
        f"{result['volume']:.2f}× average"
    )

    st.subheader("Why this signal?")

    for reason in result["reasons"]:
        st.write(f"• {reason}")

    st.subheader("Price")

    st.line_chart(
        data["Close"]
    )

    st.subheader("Moving Averages")

    chart = data[
        [
            "Close",
            "SMA20",
            "SMA50"
        ]
    ].dropna()

    st.line_chart(chart)

    st.caption(
        "Research tool only. Signals do not guarantee future returns."
    )
