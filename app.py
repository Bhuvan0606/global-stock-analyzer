import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime


st.set_page_config(
    page_title="Global Stock Analyzer",
    page_icon="📈",
    layout="wide"
)

st.title("🌍 Global Stock Analyzer")
st.caption("Scanner + confirmation engine + signal history v11")


# =====================================================
# SIGNAL HISTORY FILE
# =====================================================

HISTORY_FILE = "signal_history.csv"


def load_history():

    if os.path.exists(HISTORY_FILE):

        try:
            return pd.read_csv(
                HISTORY_FILE
            )
        except Exception:
            pass

    return pd.DataFrame(
        columns=[
            "Time",
            "Symbol",
            "Interval",
            "Action",
            "Setup",
            "Confirmation",
            "Price",
            "RSI",
            "Trend",
            "Confidence",
            "Bullish",
            "Bearish"
        ]
    )


def save_signal(symbol, interval, result):

    history = load_history()

    new_row = pd.DataFrame([{
        "Time": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "Symbol": symbol,
        "Interval": interval,
        "Action": result["action"],
        "Setup": result["setup"],
        "Confirmation": result["confirmation"],
        "Price": result["price"],
        "RSI": result["rsi"],
        "Trend": result["trend"],
        "Confidence": result["confidence"],
        "Bullish": result["bullish_pct"],
        "Bearish": result["bearish_pct"]
    }])

    history = pd.concat(
        [
            history,
            new_row
        ],
        ignore_index=True
    )

    history.to_csv(
        HISTORY_FILE,
        index=False
    )


# =====================================================
# DATA
# =====================================================

def get_data(symbol, interval):

    periods = {
        "5m": "7d",
        "15m": "30d",
        "30m": "60d",
        "1h": "60d"
    }

    try:

        data = yf.download(
            symbol,
            period=periods[interval],
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False
        )

    except Exception:

        return None

    if data is None or data.empty:
        return None

    if isinstance(
        data.columns,
        pd.MultiIndex
    ):

        data.columns = (
            data.columns
            .get_level_values(0)
        )

    needed = [
        "Close",
        "High",
        "Low",
        "Volume"
    ]

    if not all(
        x in data.columns
        for x in needed
    ):

        return None

    for col in needed:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    return data.dropna(
        subset=needed
    )


# =====================================================
# ANALYSIS ENGINE
# =====================================================

def analyze(data):

    close = data["Close"]
    high = data["High"]
    low = data["Low"]

    data["SMA20"] = (
        close.rolling(20).mean()
    )

    data["SMA50"] = (
        close.rolling(50).mean()
    )

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    avg_loss = avg_loss.replace(
        0,
        np.nan
    )

    rs = avg_gain / avg_loss

    data["RSI"] = (
        100 -
        (
            100 /
            (1 + rs)
        )
    )

    data["RSI_prev"] = (
        data["RSI"].shift(3)
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

    price = float(
        x["Close"]
    )

    sma20 = float(
        x["SMA20"]
    )

    sma50 = float(
        x["SMA50"]
    )

    rsi = float(
        x["RSI"]
    )

    rsi_prev = float(
        x["RSI_prev"]
    )

    volume = float(
        x["VolumeRatio"]
    )

    support = float(
        x["Support"]
    )

    resistance = float(
        x["Resistance"]
    )

    # =================================================
    # TREND
    # =================================================

    bullish_trend = (
        price > sma20
        and sma20 > sma50
    )

    bearish_trend = (
        price < sma20
        and sma20 < sma50
    )

    if bullish_trend:
        trend = "BULLISH"

    elif bearish_trend:
        trend = "BEARISH"

    else:
        trend = "MIXED"

    # =================================================
    # MOMENTUM
    # =================================================

    rsi_rising = (
        rsi > rsi_prev
    )

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

    # =================================================
    # PRICE LOCATION
    # =================================================

    price_range = (
        resistance - support
    )

    if price_range > 0:

        position = (
            (price - support)
            /
            price_range
        )

    else:

        position = 0.5

    near_support = (
        position <= 0.25
    )

    near_resistance = (
        position >= 0.75
    )

    support_broken = (
        price < support
    )

    # =================================================
    # BIAS SCORE
    # =================================================

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

    total = (
        bullish_score +
        bearish_score
    )

    if total:

        bullish_pct = (
            bullish_score /
            total *
            100
        )

        bearish_pct = (
            bearish_score /
            total *
            100
        )

    else:

        bullish_pct = 50
        bearish_pct = 50

    # =================================================
    # DECISION
    # =================================================

    if support_broken:

        setup = "SUPPORT BREAKDOWN"
        confirmation = "CONFIRMED"
        action = "EXIT / AVOID"

    elif (
        bullish_trend
        and 45 <= rsi <= 65
        and rsi_rising
        and volume >= 1.2
    ):

        setup = "BULLISH CONTINUATION"
        confirmation = "CONFIRMED"
        action = "BUY SETUP"

    elif (
        bearish_trend
        and near_support
        and rsi < 40
    ):

        setup = "POTENTIAL REVERSAL"
        confirmation = "NOT CONFIRMED"
        action = "WATCH"

    elif bearish_trend:

        setup = "BEARISH CONTINUATION"
        confirmation = "PARTIALLY CONFIRMED"
        action = "AVOID / WAIT"

    else:

        setup = "MIXED MARKET"
        confirmation = "NOT CONFIRMED"
        action = "WAIT"

    # =================================================
    # CONFIDENCE
    # =================================================

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

    # =================================================
    # LEVELS
    # =================================================

    if action in [
        "BUY SETUP",
        "WATCH"
    ]:

        stop = (
            support * 0.985
        )

        risk = (
            price - stop
        )

        target1 = (
            price +
            risk * 1.5
        )

        target2 = (
            price +
            risk * 2.5
        )

    else:

        stop = support

        target1 = price

        target2 = resistance

    # =================================================
    # TRIGGER
    # =================================================

    if action == "BUY SETUP":

        trigger = (
            "Bullish trend, momentum and volume are aligned."
        )

    elif action == "WATCH":

        trigger = (
            f"Watch for price to reclaim "
            f"{sma20:.2f} with improving RSI."
        )

    elif action == "EXIT / AVOID":

        trigger = (
            f"Price must recover above "
            f"{support:.2f} to weaken the breakdown."
        )

    else:

        trigger = (
            f"Wait for clearer direction "
            f"around {sma20:.2f}."
        )

    return {
        "price": price,
        "trend": trend,
        "momentum": momentum,
        "rsi": rsi,
        "volume": volume,
        "support": support,
        "resistance": resistance,
        "setup": setup,
        "confirmation": confirmation,
        "action": action,
        "confidence": confidence,
        "bullish_pct": bullish_pct,
        "bearish_pct": bearish_pct,
        "trigger": trigger,
        "stop": stop,
        "target1": target1,
        "target2": target2
    }


# =====================================================
# SCANNER
# =====================================================

def scan_market(symbols, interval):

    rows = []

    progress = st.progress(0)

    total = len(symbols)

    for i, symbol in enumerate(symbols):

        data = get_data(
            symbol,
            interval
        )

        if data is not None:

            result = analyze(
                data
            )

            if result is not None:

                rows.append({
                    "Symbol": symbol,
                    "Action": result["action"],
                    "Setup": result["setup"],
                    "Confirmation": result["confirmation"],
                    "Trend": result["trend"],
                    "RSI": round(
                        result["rsi"],
                        1
                    ),
                    "Volume": round(
                        result["volume"],
                        2
                    ),
                    "Bullish %": round(
                        result["bullish_pct"]
                    ),
                    "Bearish %": round(
                        result["bearish_pct"]
                    ),
                    "Confidence": result["confidence"],
                    "Price": round(
                        result["price"],
                        2
                    )
                })

        progress.progress(
            (i + 1) / total
        )

    progress.empty()

    return pd.DataFrame(
        rows
    )


# =====================================================
# TABS
# =====================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🔎 Single Stock",
        "🌍 Global Scanner",
        "📊 Signal History"
    ]
)


# =====================================================
# SINGLE STOCK
# =====================================================

with tab1:

    symbol = st.text_input(
        "Enter stock symbol",
        placeholder=(
            "AAPL, MSFT, "
            "RELIANCE.NS, TCS.NS"
        )
    )

    interval = st.selectbox(
        "Market timeframe",
        [
            "5m",
            "15m",
            "30m",
            "1h"
        ],
        index=1
    )

    if st.button(
        "🔎 Analyze Market"
    ):

        if not symbol:

            st.warning(
                "Enter a stock symbol first."
            )

            st.stop()

        symbol = (
            symbol
            .strip()
            .upper()
        )

        with st.spinner(
            "Downloading market data..."
        ):

            data = get_data(
                symbol,
                interval
            )

        if data is None:

            st.error(
                "No market data found."
            )

            st.stop()

        result = analyze(
            data
        )

        if result is None:

            st.warning(
                "Not enough data."
            )

            st.stop()

        save_signal(
            symbol,
            interval,
            result
        )

        st.success(
            f"Analysis completed for {symbol}"
        )

        if result["action"] == "BUY SETUP":

            st.success(
                "🟢 BUY SETUP"
            )

        elif result["action"] == "WATCH":

            st.warning(
                "🟡 WATCH"
            )

        elif result["action"] == "EXIT / AVOID":

            st.error(
                "🔴 EXIT / AVOID"
            )

        else:

            st.info(
                f"⚪ {result['action']}"
            )

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

        st.subheader(
            "Trading Levels"
        )

        a, b, c, d = st.columns(4)

        a.metric(
            "Entry",
            f"{result['price']:.2f}"
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

        st.subheader(
            "Price Chart"
        )

        st.line_chart(
            data["Close"]
        )


# =====================================================
# GLOBAL SCANNER
# =====================================================

with tab2:

    st.subheader(
        "🌍 Global Market Scanner"
    )

    symbols_text = st.text_area(
        "Symbols",
        value=(
            "AAPL, MSFT, NVDA, AMZN, "
            "META, TSLA, GOOGL, AMD, "
            "RELIANCE.NS, TCS.NS, "
            "INFY.NS, HDFCBANK.NS"
        ),
        height=120
    )

    scan_interval = st.selectbox(
        "Scanner timeframe",
        [
            "5m",
            "15m",
            "30m",
            "1h"
        ],
        index=1,
        key="scanner_interval"
    )

    if st.button(
        "🚀 Run Global Scan"
    ):

        symbols = [
            x.strip().upper()
            for x in symbols_text.split(",")
            if x.strip()
        ]

        if not symbols:

            st.warning(
                "Enter at least one symbol."
            )

            st.stop()

        with st.spinner(
            "Scanning markets..."
        ):

            results = scan_market(
                symbols,
                scan_interval
            )

        if results.empty:

            st.error(
                "No market data found."
            )

            st.stop()

        st.session_state[
            "latest_scan"
        ] = results

        st.subheader(
            "Scanner Results"
        )

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )

        buys = len(
            results[
                results["Action"] ==
                "BUY SETUP"
            ]
        )

        watches = len(
            results[
                results["Action"] ==
                "WATCH"
            ]
        )

        a, b, c = st.columns(3)

        a.metric(
            "BUY SETUPS",
            buys
        )

        b.metric(
            "WATCH",
            watches
        )

        c.metric(
            "Markets Scanned",
            len(results)
        )


# =====================================================
# SIGNAL HISTORY
# =====================================================

with tab3:

    st.subheader(
        "📊 Signal History"
    )

    history = load_history()

    if history.empty:

        st.info(
            "No signals recorded yet."
        )

    else:

        st.metric(
            "Signals Recorded",
            len(history)
        )

        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True
        )

        st.subheader(
            "Signal Distribution"
        )

        counts = (
            history["Action"]
            .value_counts()
        )

        st.bar_chart(
            counts
        )

        st.subheader(
            "Recent Signals"
        )

        st.dataframe(
            history.tail(20),
            use_container_width=True,
            hide_index=True
        )

        csv = history.to_csv(
            index=False
        )

        st.download_button(
            "⬇️ Download Signal History",
            csv,
            "signal_history.csv",
            "text/csv"
        )


st.caption(
    "Research tool only. "
    "Signals do not guarantee future returns."
    )
