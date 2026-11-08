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
st.caption("Signal confirmation engine v9")


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

    needed = ["Close", "High", "Low", "Volume"]

    if not all(x in data.columns for x in needed):
        return None

    for col in needed:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    return data.dropna(subset=needed)


# =========================
# ANALYSIS
# =========================

def analyze(data):

    close = data["Close"]
    high = data["High"]
    low = data["Low"]

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

    data["RSI_prev"] = data["RSI"].shift(3)

    data["VolumeAvg"] = (
        data["Volume"].rolling(20).mean()
    )

    data["VolumeRatio"] = (
        data["Volume"] /
        data["VolumeAvg"]
    )

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

    # =========================
    # TREND
    # =========================

    bullish_trend = (
        price > sma20 and
        sma20 > sma50
    )

    bearish_trend = (
        price < sma20 and
        sma20 < sma50
    )

    if bullish_trend:
        trend = "BULLISH"
    elif bearish_trend:
        trend = "BEARISH"
    else:
        trend = "MIXED"

    # =========================
    # MOMENTUM
    # =========================

    rsi_rising = rsi > rsi_prev

    if rsi < 30:
        momentum = "OVERSOLD"
    elif rsi < 40:
        momentum = "WEAK"
    elif rsi <= 65:
        momentum = "HEALTHY"
    elif rsi <= 70:
        momentum = "STRONG"
    else:
        momentum = "OVERBOUGHT"

    # =========================
    # PRICE LOCATION
    # =========================

    price_range = resistance - support

    if price_range > 0:
        position = (
            (price - support) /
            price_range
        )
    else:
        position = 0.5

    near_support = position <= 0.25
    near_resistance = position >= 0.75

    support_broken = price < support

    # =========================
    # SIGNAL CONDITIONS
    # =========================

    buy_confirmations = []
    reversal_conditions = []
    risk_conditions = []

    # BUY confirmation
    if bullish_trend:
        buy_confirmations.append(
            "Trend is bullish."
        )

    if rsi >= 45 and rsi <= 65 and rsi_rising:
        buy_confirmations.append(
            "Momentum is healthy and improving."
        )

    if volume >= 1.2:
        buy_confirmations.append(
            "Volume confirms market participation."
        )

    if price > sma20:
        buy_confirmations.append(
            "Price is holding above the short-term average."
        )

    # Reversal conditions
    if bearish_trend:
        reversal_conditions.append(
            "Trend is currently bearish."
        )

    if rsi < 40:
        reversal_conditions.append(
            "RSI is in a weak/oversold zone."
        )

    if rsi_rising:
        reversal_conditions.append(
            "RSI is beginning to recover."
        )

    if near_support:
        reversal_conditions.append(
            "Price is near recent support."
        )

    if volume >= 1.5:
        reversal_conditions.append(
            "Unusually high volume is present."
        )

    # Risks
    if bearish_trend:
        risk_conditions.append(
            "The broader short-term trend is bearish."
        )

    if rsi < 40:
        risk_conditions.append(
            "Momentum has not yet returned to bullish territory."
        )

    if volume >= 2:
        risk_conditions.append(
            "Very high volume can indicate either accumulation or heavy selling."
        )

    if support_broken:
        risk_conditions.append(
            "Price is below recent support."
        )

    # =========================
    # DECISION ENGINE
    # =========================

    setup = "NO CLEAR SETUP"
    confirmation = "NOT CONFIRMED"
    action = "WAIT"
    trigger = ""
    invalidation = ""

    if support_broken:

        setup = "SUPPORT BREAKDOWN"
        confirmation = "CONFIRMED"
        action = "EXIT / AVOID"

        trigger = (
            "Recovery back above support would weaken the breakdown."
        )

        invalidation = (
            f"Breakdown invalid if price reclaims "
            f"{support:.2f}."
        )

    elif (
        bullish_trend
        and rsi >= 45
        and rsi <= 65
        and rsi_rising
        and volume >= 1.2
    ):

        setup = "BULLISH CONTINUATION"
        confirmation = "CONFIRMED"
        action = "BUY SETUP"

        trigger = (
            "Bullish conditions are currently aligned."
        )

        invalidation = (
            f"Loss of {support:.2f} would weaken the setup."
        )

    elif (
        bearish_trend
        and near_support
        and rsi < 40
    ):

        setup = "POTENTIAL REVERSAL"
        confirmation = "NOT CONFIRMED"
        action = "WATCH"

        trigger = (
            f"BUY confirmation requires price to reclaim "
            f"the short-term trend level near {sma20:.2f}"
            f" while RSI continues improving."
        )

        invalidation = (
            f"A decisive move below {support:.2f} "
            f"invalidates the reversal idea."
        )

    elif bearish_trend:

        setup = "BEARISH CONTINUATION"
        confirmation = "PARTIALLY CONFIRMED"
        action = "AVOID / WAIT"

        trigger = (
            f"Reclaiming {sma20:.2f} would be the first "
            f"sign of improving conditions."
        )

        invalidation = (
            f"Continued trading below {sma20:.2f} "
            f"keeps the bearish bias active."
        )

    else:

        setup = "MIXED MARKET"
        confirmation = "NOT CONFIRMED"
        action = "WAIT"

        trigger = (
            f"Wait for price to establish direction "
            f"above or below {sma20:.2f}."
        )

        invalidation = (
            "No valid directional setup yet."
        )

    # =========================
    # SCORE
    # =========================

    bullish_score = 0
    bearish_score = 0

    if price > sma20:
        bullish_score += 20
    else:
        bearish_score += 20

    if sma20 > sma50:
        bullish_score += 20
    else:
        bearish_score += 20

    if rsi_rising:
        bullish_score += 10
    else:
        bearish_score += 10

    if rsi < 40:
        bearish_score += 10

    if 45 <= rsi <= 65:
        bullish_score += 10

    if volume >= 1.2:
        bullish_score += 10

    if near_support:
        bullish_score += 5

    if near_resistance:
        bearish_score += 5

    if support_broken:
        bearish_score += 20

    total = bullish_score + bearish_score

    if total > 0:
        bullish_pct = (
            bullish_score / total * 100
        )
        bearish_pct = (
            bearish_score / total * 100
        )
    else:
        bullish_pct = 50
        bearish_pct = 50

    confidence = int(
        min(
            95,
            40 +
            abs(
                bullish_pct -
                bearish_pct
            ) * 0.6
        )
    )

    # =========================
    # LEVELS
    # =========================

    if action == "BUY SETUP":

        entry_low = price * 0.995
        entry_high = price * 1.005

        stop = support * 0.995

        risk = price - stop

        target1 = price + risk * 1.5
        target2 = price + risk * 2.5

    elif action == "WATCH":

        entry_low = support
        entry_high = price

        stop = support * 0.985

        risk = price - stop

        target1 = price + risk * 1.5
        target2 = price + risk * 2.5

    elif action == "EXIT / AVOID":

        entry_low = price
        entry_high = resistance

        stop = resistance * 1.01

        risk = stop - price

        target1 = price - risk * 1.5
        target2 = price - risk * 2.5

    else:

        entry_low = support
        entry_high = resistance

        stop = support
        target1 = price
        target2 = resistance

    if action in [
        "BUY SETUP",
        "WATCH"
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
        "setup": setup,
        "confirmation": confirmation,
        "action": action,
        "trigger": trigger,
        "invalidation": invalidation,
        "trend": trend,
        "momentum": momentum,
        "price": price,
        "rsi": rsi,
        "volume": volume,
        "support": support,
        "resistance": resistance,
        "sma20": sma20,
        "bullish_pct": bullish_pct,
        "bearish_pct": bearish_pct,
        "confidence": confidence,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "stop": stop,
        "target1": target1,
        "target2": target2,
        "rr": rr,
        "buy_confirmations": buy_confirmations,
        "reversal_conditions": reversal_conditions,
        "risk_conditions": risk_conditions
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
        "Running confirmation engine..."
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

    # ACTION

    action = result["action"]

    if action == "BUY SETUP":

        st.success(
            f"🟢 {action}"
        )

    elif action == "WATCH":

        st.warning(
            f"🟡 {action}"
        )

    elif action == "EXIT / AVOID":

        st.error(
            f"🔴 {action}"
        )

    else:

        st.info(
            f"⚪ {action}"
        )

    # SETUP

    st.subheader(
        "Setup Decision"
    )

    a, b, c = st.columns(3)

    a.metric(
        "Setup",
        result["setup"]
    )

    b.metric(
        "Confirmation",
        result["confirmation"]
    )

    c.metric(
        "Confidence",
        f"{result['confidence']}%"
    )

    # MARKET

    st.subheader(
        "Market Overview"
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "Price",
        f"{result['price']:.2f}"
    )

    b.metric(
        "Trend",
        result["trend"]
    )

    c.metric(
        "RSI",
        f"{result['rsi']:.1f}"
    )

    d.metric(
        "Volume",
        f"{result['volume']:.2f}×"
    )

    # BIAS

    st.subheader(
        "Market Bias"
    )

    a, b = st.columns(2)

    a.metric(
        "Bullish",
        f"{result['bullish_pct']:.0f}%"
    )

    b.metric(
        "Bearish",
        f"{result['bearish_pct']:.0f}%"
    )

    # ACTION PLAN

    st.subheader(
        "Action Plan"
    )

    st.write(
        f"**Current action:** "
        f"{result['action']}"
    )

    st.write(
        f"**Trigger:** "
        f"{result['trigger']}"
    )

    st.write(
        f"**Invalidation:** "
        f"{result['invalidation']}"
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

    # CONFIRMATION

    st.subheader(
        "Confirmation Evidence"
    )

    if result["buy_confirmations"]:

        st.write(
            "**Bullish confirmation**"
        )

        for item in result["buy_confirmations"]:
            st.write(
                f"🟢 {item}"
            )

    if result["reversal_conditions"]:

        st.write(
            "**Setup conditions**"
        )

        for item in result["reversal_conditions"]:
            st.write(
                f"🟡 {item}"
            )

    if result["risk_conditions"]:

        st.write(
            "**Risks**"
        )

        for item in result["risk_conditions"]:
            st.write(
                f"🔴 {item}"
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

    st.line_chart(chart)

    st.caption(
        "Research tool only. "
        "Signals do not guarantee future returns."
        )
