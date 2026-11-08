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
st.caption("Trading setup analyzer v7")


# -------------------------
# DATA
# -------------------------

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

    required = ["Close", "High", "Low", "Volume"]

    if not all(x in data.columns for x in required):
        return None

    for col in required:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    data = data.dropna(
        subset=required
    )

    return data


# -------------------------
# ANALYSIS
# -------------------------

def analyze(data):

    close = data["Close"]
    high = data["High"]
    low = data["Low"]

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

    data["RSI"] = 100 - (
        100 / (1 + rs)
    )

    # RSI direction
    data["RSI_prev"] = data["RSI"].shift(3)

    # Volume
    data["VolumeAvg"] = (
        data["Volume"]
        .rolling(20)
        .mean()
    )

    data["VolumeRatio"] = (
        data["Volume"] /
        data["VolumeAvg"]
    )

    # Recent support / resistance
    data["Support"] = low.rolling(20).min()
    data["Resistance"] = high.rolling(20).max()

    valid = data.dropna(
        subset=[
            "SMA20",
            "SMA50",
            "RSI",
            "RSI_prev",
            "VolumeRatio",
            "Support",
            "Resistance"
        ]
    )

    if valid.empty:
        return None

    latest = valid.iloc[-1]

    price = float(latest["Close"])
    sma20 = float(latest["SMA20"])
    sma50 = float(latest["SMA50"])
    rsi = float(latest["RSI"])
    rsi_prev = float(latest["RSI_prev"])
    volume = float(latest["VolumeRatio"])
    support = float(latest["Support"])
    resistance = float(latest["Resistance"])

    score = 50
    bullish = []
    bearish = []

    # -------------------------
    # TREND
    # -------------------------

    if price > sma20:
        score += 10
        bullish.append(
            "Price is above the 20-period average."
        )
    else:
        score -= 10
        bearish.append(
            "Price is below the 20-period average."
        )

    if sma20 > sma50:
        score += 10
        bullish.append(
            "Short-term trend is above the medium-term trend."
        )
        trend = "BULLISH"
    else:
        score -= 10
        bearish.append(
            "Short-term trend is below the medium-term trend."
        )
        trend = "BEARISH"

    # -------------------------
    # RSI
    # -------------------------

    if rsi < 30:
        score += 5

        if rsi > rsi_prev:
            score += 5
            bullish.append(
                "RSI is oversold and starting to recover."
            )
        else:
            bullish.append(
                "RSI is oversold, but recovery is not confirmed."
            )

    elif rsi >= 30 and rsi < 45:

        if rsi > rsi_prev:
            score += 5
            bullish.append(
                "RSI is weak but improving."
            )
        else:
            score -= 3
            bearish.append(
                "RSI remains weak."
            )

    elif rsi >= 45 and rsi <= 65:

        score += 5

        if rsi > rsi_prev:
            bullish.append(
                "Momentum is improving."
            )
        else:
            bullish.append(
                "Momentum is in a neutral zone."
            )

    elif rsi > 70:

        score -= 8
        bearish.append(
            "RSI indicates potentially overbought conditions."
        )

    else:

        if rsi > rsi_prev:
            score += 2
            bullish.append(
                "Momentum is recovering."
            )
        else:
            bearish.append(
                "Momentum is weakening."
            )

    # -------------------------
    # VOLUME
    # -------------------------

    if volume >= 1.5:

        score += 10
        bullish.append(
            "Volume is significantly above its recent average."
        )

    elif volume >= 1.2:

        score += 5
        bullish.append(
            "Volume is above its recent average."
        )

    elif volume < 0.8:

        score -= 5
        bearish.append(
            "Volume is below its recent average."
        )

    else:

        bullish.append(
            "Volume is close to its normal range."
        )

    # -------------------------
    # PRICE POSITION
    # -------------------------

    range_size = resistance - support

    if range_size > 0:

        position = (
            (price - support) /
            range_size
        )

    else:
        position = 0.5

    if position < 0.25:

        bullish.append(
            "Price is trading near recent support."
        )

        score += 5

    elif position > 0.75:

        bearish.append(
            "Price is trading near recent resistance."
        )

        score -= 5

    # -------------------------
    # SCORE
    # -------------------------

    score = int(
        max(0, min(100, score))
    )

    # -------------------------
    # SIGNAL
    # -------------------------

    if (
        score >= 75
        and trend == "BULLISH"
    ):
        signal = "BUY SETUP"

    elif (
        score <= 30
        and trend == "BEARISH"
    ):
        signal = "EXIT / SELL SETUP"

    else:
        signal = "WAIT / WATCH"

    # -------------------------
    # CONFIDENCE
    # -------------------------

    confidence = abs(score - 50) * 2

    confidence = int(
        max(0, min(100, confidence))
    )

    # -------------------------
    # TRADING LEVELS
    # -------------------------

    if signal == "BUY SETUP":

        entry_low = max(
            support,
            price * 0.995
        )

        entry_high = price * 1.005

        stop = min(
            support * 0.995,
            price * 0.97
        )

        risk = (
            price - stop
        )

        target1 = price + risk * 1.5
        target2 = price + risk * 2.5

    else:

        entry_low = support
        entry_high = price

        stop = resistance

        target1 = support
        target2 = (
            support -
            (resistance - support) * 0.5
        )

    return {
        "signal": signal,
        "score": score,
        "confidence": confidence,
        "price": price,
        "rsi": rsi,
        "trend": trend,
        "volume": volume,
        "support": support,
        "resistance": resistance,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "stop": stop,
        "target1": target1,
        "target2": target2,
        "bullish": bullish,
        "bearish": bearish
    }


# -------------------------
# UI
# -------------------------

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
        st.warning(
            "Enter a stock symbol first."
        )
        st.stop()

    symbol = symbol.strip().upper()

    with st.spinner(
        "Downloading market data..."
    ):

        try:
            data = get_data(
                symbol,
                interval
            )

        except Exception as error:

            st.error(
                "Market data request failed."
            )

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

    with st.spinner(
        "Analyzing market..."
    ):

        try:
            result = analyze(data)

        except Exception as error:

            st.error(
                "Analysis failed."
            )

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

    # SIGNAL

    signal = result["signal"]

    if signal == "BUY SETUP":

        st.success(
            f"🟢 {signal}"
        )

    elif signal == "EXIT / SELL SETUP":

        st.error(
            f"🔴 {signal}"
        )

    else:

        st.warning(
            f"🟡 {signal}"
        )

    # OVERVIEW

    st.subheader(
        "Market Overview"
    )

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
        "Confidence",
        f"{result['confidence']}%"
    )

    d.metric(
        "Trend",
        result["trend"]
    )

    # MOMENTUM

    st.subheader(
        "Momentum & Volume"
    )

    a, b, c = st.columns(3)

    a.metric(
        "RSI",
        f"{result['rsi']:.1f}"
    )

    b.metric(
        "Volume",
        f"{result['volume']:.2f}×"
    )

    c.metric(
        "Trend",
        result["trend"]
    )

    # LEVELS

    st.subheader(
        "Trading Levels"
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "Entry Zone",
        f"{result['entry_low']:.2f}"
        f" – "
        f"{result['entry_high']:.2f}"
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

    # SUPPORT / RESISTANCE

    st.subheader(
        "Support & Resistance"
    )

    a, b = st.columns(2)

    a.metric(
        "Support",
        f"{result['support']:.2f}"
    )

    b.metric(
        "Resistance",
        f"{result['resistance']:.2f}"
    )

    # REASONS

    st.subheader(
        "Why this signal?"
    )

    if result["bullish"]:

        st.write(
            "**Bullish factors**"
        )

        for reason in result["bullish"]:

            st.write(
                f"🟢 {reason}"
            )

    if result["bearish"]:

        st.write(
            "**Risk / bearish factors**"
        )

        for reason in result["bearish"]:

            st.write(
                f"🔴 {reason}"
            )

    # CHART

    st.subheader(
        "Price"
    )

    st.line_chart(
        data["Close"]
    )

    st.subheader(
        "Moving Averages"
    )

    chart = data[
        [
            "Close",
            "SMA20",
            "SMA50"
        ]
    ].dropna()

    st.line_chart(
        chart
    )

    st.caption(
        "Research tool only. "
        "Signals do not guarantee future returns."
    )
