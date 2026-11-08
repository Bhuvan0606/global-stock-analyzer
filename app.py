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
st.caption("Global scanner + signal confirmation engine v10")


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


# =====================================================
# SIGNAL ENGINE
# =====================================================

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

    avg_loss = avg_loss.replace(
        0,
        np.nan
    )

    rs = avg_gain / avg_loss

    data["RSI"] = (
        100 -
        (100 / (1 + rs))
    )

    data["RSI_prev"] = (
        data["RSI"].shift(3)
    )

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

    # -------------------------
    # TREND
    # -------------------------

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

    # -------------------------
    # MOMENTUM
    # -------------------------

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

    # -------------------------
    # PRICE LOCATION
    # -------------------------

    price_range = (
        resistance - support
    )

    if price_range > 0:

        position = (
            (price - support) /
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

    # -------------------------
    # SCORES
    # -------------------------

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

    if total > 0:

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

    # -------------------------
    # DECISION
    # -------------------------

    if support_broken:

        setup = "SUPPORT BREAKDOWN"
        confirmation = "CONFIRMED"
        action = "EXIT / AVOID"

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

    # -------------------------
    # CONFIDENCE
    # -------------------------

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

    # -------------------------
    # TRIGGER
    # -------------------------

    if action == "BUY SETUP":

        trigger = (
            "Bullish trend, momentum and volume are aligned."
        )

    elif action == "WATCH":

        trigger = (
            f"Watch for a reclaim of "
            f"{sma20:.2f} with improving RSI."
        )

    elif action == "EXIT / AVOID":

        trigger = (
            f"Price must recover above "
            f"{support:.2f} to weaken the breakdown."
        )

    else:

        trigger = (
            f"Wait for clearer direction around "
            f"{sma20:.2f}."
        )

    # -------------------------
    # LEVELS
    # -------------------------

    if action == "BUY SETUP":

        entry = price

        stop = (
            support * 0.995
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

    elif action == "WATCH":

        entry = price

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

        entry = price
        stop = support
        target1 = price
        target2 = resistance

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
        "entry": entry,
        "stop": stop,
        "target1": target1,
        "target2": target2
    }


# =====================================================
# SCANNER
# =====================================================

def scan_symbols(symbols, interval):

    results = []

    progress = st.progress(0)

    total = len(symbols)

    for i, symbol in enumerate(symbols):

        data = get_data(
            symbol,
            interval
        )

        if data is not None:

            result = analyze(data)

            if result is not None:

                row = {
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
                    "Confidence": (
                        f"{result['confidence']}%"
                    ),
                    "Price": round(
                        result["price"],
                        2
                    )
                }

                results.append(row)

        progress.progress(
            (i + 1) / total
        )

    progress.empty()

    return pd.DataFrame(
        results
    )


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header(
    "Scanner Settings"
)

scan_symbols_text = st.sidebar.text_area(
    "Symbols to scan",
    value=(
        "AAPL, MSFT, NVDA, "
        "AMZN, META, TSLA, "
        "GOOGL, AMD"
    ),
    height=150
)

scan_interval = st.sidebar.selectbox(
    "Scanner timeframe",
    [
        "5m",
        "15m",
        "30m",
        "1h"
    ],
    index=1
)


# =====================================================
# TABS
# =====================================================

tab1, tab2 = st.tabs(
    [
        "🔎 Single Stock",
        "🌍 Global Scanner"
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
            symbol.strip().upper()
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

        with st.spinner(
            "Running analysis..."
        ):

            result = analyze(
                data
            )

        if result is None:

            st.warning(
                "Not enough data."
            )

            st.stop()

        st.success(
            f"Market analysis completed for {symbol}"
        )

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


# =====================================================
# GLOBAL SCANNER
# =====================================================

with tab2:

    st.subheader(
        "🌍 Multi-Stock Market Scanner"
    )

    st.write(
        "Scan multiple symbols using the "
        "same signal confirmation engine."
    )

    if st.button(
        "🚀 Run Global Scan"
    ):

        symbols = [
            x.strip().upper()
            for x in scan_symbols_text.split(",")
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

            results = scan_symbols(
                symbols,
                scan_interval
            )

        if results.empty:

            st.error(
                "No valid market data was found."
            )

            st.stop()

        # -------------------------
        # SUMMARY
        # -------------------------

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

        exits = len(
            results[
                results["Action"] ==
                "EXIT / AVOID"
            ]
        )

        waits = len(
            results[
                ~results["Action"].isin(
                    [
                        "BUY SETUP",
                        "WATCH",
                        "EXIT / AVOID"
                    ]
                )
            ]
        )

        a, b, c, d = st.columns(4)

        a.metric(
            "BUY",
            buys
        )

        b.metric(
            "WATCH",
            watches
        )

        c.metric(
            "EXIT",
            exits
        )

        d.metric(
            "WAIT",
            waits
        )

        # -------------------------
        # RANKING
        # -------------------------

        action_order = {
            "BUY SETUP": 0,
            "WATCH": 1,
            "WAIT": 2,
            "AVOID / WAIT": 3,
            "EXIT / AVOID": 4
        }

        results["_rank"] = (
            results["Action"]
            .map(action_order)
            .fillna(9)
        )

        results = (
            results
            .sort_values(
                [
                    "_rank",
                    "Confidence"
                ]
            )
            .drop(
                columns="_rank"
            )
        )

        st.subheader(
            "Scanner Results"
        )

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )

        # -------------------------
        # TOP SETUPS
        # -------------------------

        buy_rows = results[
            results["Action"] ==
            "BUY SETUP"
        ]

        watch_rows = results[
            results["Action"] ==
            "WATCH"
        ]

        if not buy_rows.empty:

            st.subheader(
                "🟢 Strongest Buy Setups"
            )

            st.dataframe(
                buy_rows,
                use_container_width=True,
                hide_index=True
            )

        if not watch_rows.empty:

            st.subheader(
                "🟡 Reversal / Watch Setups"
            )

            st.dataframe(
                watch_rows,
                use_container_width=True,
                hide_index=True
            )

        st.caption(
            "Scanner uses market data available "
            "from the selected timeframe. "
            "Research tool only; signals do not "
            "guarantee future returns."
        )
