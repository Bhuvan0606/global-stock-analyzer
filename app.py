import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime


st.set_page_config(
    page_title="Global Stock Analyzer",
    page_icon="📈",
    layout="wide"
)

st.title("🌍 Global Stock Analyzer")
st.caption("Scanner + signal tracking v12.1")

HISTORY_FILE = "signal_history.csv"


# =====================================================
# HISTORY
# =====================================================

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            return pd.read_csv(HISTORY_FILE)
        except:
            pass

    return pd.DataFrame(columns=[
        "ID", "Time", "Symbol", "Interval",
        "Action", "Setup", "Confirmation",
        "Entry", "Stop", "Target1", "Target2",
        "RSI", "Trend", "Confidence",
        "Bullish", "Bearish", "Status",
        "CurrentPrice", "ReturnPct",
        "TargetHit", "StopHit"
    ])


def save_signal(symbol, interval, r):
    history = load_history()

    row = {
        "ID": len(history) + 1,
        "Time": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "Symbol": symbol,
        "Interval": interval,
        "Action": r["action"],
        "Setup": r["setup"],
        "Confirmation": r["confirmation"],
        "Entry": r["price"],
        "Stop": r["stop"],
        "Target1": r["target1"],
        "Target2": r["target2"],
        "RSI": r["rsi"],
        "Trend": r["trend"],
        "Confidence": r["confidence"],
        "Bullish": r["bullish_pct"],
        "Bearish": r["bearish_pct"],
        "Status": "OPEN",
        "CurrentPrice": r["price"],
        "ReturnPct": 0.0,
        "TargetHit": "NO",
        "StopHit": "NO"
    }

    history = pd.concat(
        [history, pd.DataFrame([row])],
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
    except:
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
# ANALYSIS
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
        100 / (1 + rs)
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

    # -----------------------------
    # SCORE
    # -----------------------------

    bullish = 0
    bearish = 0

    if price > sma20:
        bullish += 20
    else:
        bearish += 20

    if sma20 > sma50:
        bullish += 20
    else:
        bearish += 20

    if rsi_rising:
        bullish += 10
    else:
        bearish += 10

    if rsi < 40:
        bearish += 10

    if 45 <= rsi <= 65:
        bullish += 10

    if volume >= 1.2:
        bullish += 10

    if near_support:
        bullish += 5

    if near_resistance:
        bearish += 5

    if support_broken:
        bearish += 20

    total = bullish + bearish

    if total:
        bullish_pct = (
            bullish / total * 100
        )
        bearish_pct = (
            bearish / total * 100
        )
    else:
        bullish_pct = 50
        bearish_pct = 50

    # -----------------------------
    # DECISION
    # -----------------------------

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

    # -----------------------------
    # LEVELS
    # -----------------------------

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
            price + risk * 1.5
        )

        target2 = (
            price + risk * 2.5
        )

    else:

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
        "stop": stop,
        "target1": target1,
        "target2": target2
    }


# =====================================================
# UPDATE OUTCOMES
# =====================================================

def update_outcomes():

    history = load_history()

    if history.empty:
        return history

    for i in range(len(history)):

        if history.loc[i, "Status"] != "OPEN":
            continue

        symbol = history.loc[
            i,
            "Symbol"
        ]

        interval = history.loc[
            i,
            "Interval"
        ]

        data = get_data(
            symbol,
            interval
        )

        if data is None:
            continue

        current = float(
            data["Close"].iloc[-1]
        )

        entry = float(
            history.loc[
                i,
                "Entry"
            ]
        )

        stop = float(
            history.loc[
                i,
                "Stop"
            ]
        )

        target1 = float(
            history.loc[
                i,
                "Target1"
            ]
        )

        action = history.loc[
            i,
            "Action"
        ]

        if action in [
            "BUY SETUP",
            "WATCH"
        ]:

            return_pct = (
                (current - entry)
                / entry
                * 100
            )

            target_hit = (
                current >= target1
            )

            stop_hit = (
                current <= stop
            )

        else:

            return_pct = (
                (entry - current)
                / entry
                * 100
            )

            target_hit = (
                current <= target1
            )

            stop_hit = (
                current >= stop
            )

        history.loc[
            i,
            "CurrentPrice"
        ] = current

        history.loc[
            i,
            "ReturnPct"
        ] = round(
            return_pct,
            2
        )

        if target_hit:

            history.loc[
                i,
                "TargetHit"
            ] = "YES"

            history.loc[
                i,
                "Status"
            ] = "TARGET HIT"

        elif stop_hit:

            history.loc[
                i,
                "StopHit"
            ] = "YES"

            history.loc[
                i,
                "Status"
            ] = "STOP HIT"

    history.to_csv(
        HISTORY_FILE,
        index=False
    )

    return history


# =====================================================
# SCANNER
# =====================================================

def scan_market(
    symbols,
    interval
):

    rows = []

    progress = st.progress(0)

    total = len(symbols)

    for i, symbol in enumerate(
        symbols
    ):

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

tab1, tab2, tab3 = st.tabs([
    "🔎 Single Stock",
    "🌍 Global Scanner",
    "📊 Performance"
])


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
                "Enter a stock symbol."
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

        action = result["action"]

        if action == "BUY SETUP":

            st.success(
                "🟢 BUY SETUP"
            )

        elif action == "WATCH":

            st.warning(
                "🟡 WATCH"
            )

        elif action == "EXIT / AVOID":

            st.error(
                "🔴 EXIT / AVOID"
            )

        else:

            st.info(
                f"⚪ {action}"
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
            f"{result['volume']:.2f}x"
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
            "Trading Levels"
        )

        a, b, c, d = st.columns(4)

        a.metric(
            "Entry",
            f"{result['price']:.2f}"
        )

        b.metric(
            "Stop",
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
            "AAPL, MSFT, NVDA, "
            "AMZN, META, TSLA, "
            "GOOGL, AMD, "
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
        key="scan_interval"
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
                "Enter symbols."
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

        st.subheader(
            "Scanner Results"
        )

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )


# =====================================================
# PERFORMANCE
# =====================================================

with tab3:

    st.subheader(
        "📊 Signal Performance"
    )

    history = load_history()

    if history.empty:

        st.info(
            "No signals recorded yet."
        )

    else:

        if st.button(
            "🔄 Update Signal Outcomes"
        ):

            with st.spinner(
                "Checking current prices..."
            ):

                history = update_outcomes()

        returns = pd.to_numeric(
            history["ReturnPct"],
            errors="coerce"
        )

        total = len(history)

        target_count = (
            history["Status"]
            == "TARGET HIT"
        ).sum()

        stop_count = (
            history["Status"]
            == "STOP HIT"
        ).sum()

        completed = (
            target_count +
            stop_count
        )

        if completed > 0:

            win_rate = (
                target_count /
                completed *
                100
            )

        else:

            win_rate = 0

        avg_return = (
            returns.mean()
        )

        a, b, c, d = st.columns(4)

        a.metric(
            "Total Signals",
            total
        )

        b.metric(
            "Targets Hit",
            target_count
        )

        c.metric(
            "Win Rate",
            f"{win_rate:.1f}%"
        )

        d.metric(
            "Avg Return",
            f"{avg_return:.2f}%"
        )

        st.subheader(
            "Signal History"
        )

        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True
        )

        st.subheader(
            "Returns"
        )

        st.bar_chart(
            returns
        )

        st.subheader(
            "Performance by Setup"
        )

        setup_stats = (
            history
            .groupby("Setup")
            .agg(
                Signals=("ID", "count"),
                AvgReturn=("ReturnPct", "mean")
            )
            .reset_index()
        )

        setup_stats["AvgReturn"] = (
            setup_stats["AvgReturn"]
            .round(2)
        )

        st.dataframe(
            setup_stats,
            use_container_width=True,
            hide_index=True
        )

        csv = history.to_csv(
            index=False
        )

        st.download_button(
            "⬇️ Download Performance Data",
        
