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
        data["Volume"] /
        data["VolumeAvg"]
    )

    # Support / resistance
    data["Support"] = (
        low.rolling(20).min()
    )

    data["Resistance"] = (
        high.rolling(20).max()
    )

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

    x = valid.iloc[-1]

    price = float(x["Close"])
    sma20 = float(x["SMA20"])
    sma50 = float(x["SMA50"])
    rsi = float(x["RSI"])
    rsi_prev = float(x["RSI_prev"])
    volume = float(x["VolumeRatio"])
    support = float(x["Support"])
    resistance = float(x["Resistance"])

    bullish = 0
    bearish = 0

    bull_reasons = []
    bear_reasons = []

    # =========================
    # TREND
    # =========================

    if price > sma20:

        bullish += 15

        bull_reasons.append(
            "Price is above the 20-period average."
        )

    else:

        bearish += 15

        bear_reasons.append(
            "Price is below the 20-period average."
        )

    if sma20 > sma50:

        bullish += 15

        bull_reasons.append(
            "Short-term trend is above the medium-term trend."
        )

        trend = "BULLISH"

    else:

        bearish += 15

        bear_reasons.append(
            "Short-term trend is below the medium-term trend."
        )

        trend = "BEARISH"

    # =========================
    # RSI
    # =========================

    if rsi < 30:

        if rsi > rsi_prev:

            bullish += 15

            bull_reasons.append(
                "RSI is oversold and starting to recover."
            )

        else:

            bullish += 5

            bear_reasons.append(
                "RSI is deeply oversold but has not confirmed a recovery."
            )

    elif rsi < 40:

        if rsi > rsi_prev:

            bullish += 8

            bull_reasons.append(
                "RSI is weak but improving."
            )

        else:

            bearish += 8

            bear_reasons.append(
                "RSI remains weak."
            )

    elif rsi <= 65:

        if rsi > rsi_prev:

            bullish += 10

            bull_reasons.append(
                "Momentum is improving."
            )

        else:

            bullish += 3

            bear_reasons.append(
                "Momentum is not strongly improving."
            )

    elif rsi > 70:

        bearish += 12

        bear_reasons.append(
            "RSI indicates potentially overbought conditions."
        )

    else:

        bearish += 5

        bear_reasons.append(
            "Momentum is relatively high but weakening."
        )

    # =========================
    # VOLUME
    # =========================

    if volume >= 2:

        bullish += 10

        bull_reasons.append(
            "Volume is unusually high."
        )

    elif volume >= 1.2:

        bullish += 6

        bull_reasons.append(
            "Volume is above its recent average."
        )

    elif volume < 0.8:

        bearish += 5

        bear_reasons.append(
            "Volume is below its recent average."
        )

    # =========================
    # SUPPORT / RESISTANCE
    # =========================

    price_range = resistance - support

    if price_range > 0:

        support_distance = (
            (price - support) /
            price_range
        )

    else:

        support_distance = 0.5

    near_support = (
        support_distance <= 0.25
    )

    near_resistance = (
        support_distance >= 0.75
    )

    if near_support:

        bullish += 10

        bull_reasons.append(
            "Price is near recent support."
        )

    if near_resistance:

        bearish += 10

        bear_reasons.append(
            "Price is near recent resistance."
        )

    # =========================
    # SUPPORT BREAK
    # =========================

    support_break = (
        price < support
    )

    if support_break:

        bearish += 20

        bear_reasons.append(
            "Price has fallen below recent support."
        )

    # =========================
    # SCORES
    # =========================

    total = bullish + bearish

    if total > 0:

        bullish_pct = (
            bullish / total * 100
        )

        bearish_pct = (
            bearish / total * 100
        )

    else:

        bullish_pct = 50
        bearish_pct = 50

    # =========================
    # SIGNAL
    # =========================

    if (
        support_break
        and bearish >= bullish
    ):

        signal = "BREAKDOWN / EXIT"

    elif (
        bullish >= 60
        and bullish > bearish + 15
        and trend == "BULLISH"
    ):

        signal = "BUY SETUP"

    elif (
        trend == "BEARISH"
        and near_support
        and rsi < 40
        and volume >= 1.5
    ):

        signal = "REVERSAL WATCH"

    elif (
        bearish >= 60
        and bearish > bullish + 15
    ):

        signal = "EXIT / SELL WATCH"

    else:

        signal = "WAIT"

    # =========================
    # CONFIDENCE
    # =========================

    difference = abs(
        bullish_pct - bearish_pct
    )

    confidence = int(
        min(95, 40 + difference * 0.7)
    )

    # =========================
    # TRADE LEVELS
    # =========================

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

        target1 = (
            price + risk * 1.5
        )

        target2 = (
            price + risk * 2.5
        )

    elif signal == "REVERSAL WATCH":

        entry_low = support
        entry_high = price

        stop = support * 0.985

        risk = (
            price - stop
        )

        target1 = price + risk * 1.5
        target2 = price + risk * 2.5

    elif signal in [
        "BREAKDOWN / EXIT",
        "EXIT / SELL WATCH"
    ]:

        entry_low = price
        entry_high = resistance

        stop = resistance * 1.01

        risk = (
            stop - price
        )

        target1 = price - risk * 1.5
        target2 = price - risk * 2.5

    else:

        entry_low = support
        entry_high = resistance

        stop = support
        target1 = price
        target2 = resistance

    # Risk / reward
    if signal in [
        "BUY SETUP",
        "REVERSAL WATCH"
    ]:

        risk = price - stop
        reward = target2 - price

    else:

        risk = stop - price
        reward = price - target2

    if risk > 0:

        rr = reward / risk

    else:

        rr = 0

    return {
        "signal": signal,
        "trend": trend,
        "price": price,
        "rsi": rsi,
        "volume": volume,
        "support": support,
        "resistance": resistance,
        "bullish_pct": bullish_pct,
        "bearish_pct": bearish_pct,
        "confidence": confidence,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "stop": stop,
        "target1": target1,
        "target2": target2,
        "rr": rr,
        "bull_reasons": bull_reasons,
        "bear_reasons": bear_reasons
    }


# =========================
# UI
# =========================

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

    symbol = (
        symbol.strip().upper()
    )

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
            "Not enough data."
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

    elif signal == "REVERSAL WATCH":

        st.warning(
            f"🟡 {signal}"
        )

    elif signal == "BREAKDOWN / EXIT":

        st.error(
            f"🔴 {signal}"
        )

    elif signal == "EXIT / SELL WATCH":

        st.error(
            f"🔴 {signal}"
        )

    else:

        st.info(
            f"⚪ {signal}"
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

    a = a

    b.metric(
        "Bullish Score",
        f"{result['bullish_pct']:.0f}%"
    )

    c.metric(
        "Bearish Score",
        f"{result['bearish_pct']:.0f}%"
    )

    d.metric(
        "Confidence",
        f"{result['confidence']}%"
    )

    # CONDITIONS

    st.subheader(
        "Market Conditions"
    )

    a, b, c = st.columns(3)

    a.metric(
        "Trend",
        result["trend"]
    )

    b.metric(
        "RSI",
        f"{result['rsi']:.1f}"
    )

    c.metric(
        "Volume",
        f"{result['volume']:.2f}×"
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

    st.write(
        f"**Risk / Reward:** "
        f"{result['rr']:.2f}R"
    )

    # SUPPORT

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

    if result["bull_reasons"]:

        st.write(
            "**Bullish factors**"
        )

        for reason in result["bull_reasons"]:

            st.write(
                f"🟢 {reason}"
            )

    if result["bear_reasons"]:

        st.write(
            "**Risk / bearish factors**"
        )

        for reason in result["bear_reasons"]:

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
