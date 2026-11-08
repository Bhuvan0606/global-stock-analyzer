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

HISTORY_FILE = "signal_history.csv"


def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            return pd.read_csv(HISTORY_FILE)
        except:
            pass
    return pd.DataFrame()


def save_signal(result):
    row = {
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Symbol": result["symbol"],
        "Interval": result["interval"],
        "Action": result["action"],
        "Setup": result["setup"],
        "Confirmation": result["confirmation"],
        "Price": result["price"],
        "Entry": result["entry"],
        "Stop": result["stop"],
        "Target1": result["target1"],
        "Target2": result["target2"],
        "RSI": result["rsi"],
        "Trend": result["trend"],
        "Confidence": result["confidence"],
        "Bullish": result["bullish"],
        "Bearish": result["bearish"]
    }

    old = load_history()
    new = pd.DataFrame([row])
    history = pd.concat([old, new], ignore_index=True)
    history.to_csv(HISTORY_FILE, index=False)


def get_data(symbol, interval):
    periods = {
        "5m": "7d",
        "15m": "30d",
        "30m": "60d",
        "1h": "60d"
    }

    df = yf.download(
        symbol,
        period=periods[interval],
        interval=interval,
        progress=False,
        auto_adjust=False
    )

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna().copy()

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()

    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    df["VolumeAvg"] = df["Volume"].rolling(20).mean()
    df["VolumeRatio"] = df["Volume"] / df["VolumeAvg"]

    return df.dropna()


def analyze(symbol, interval):
    df = get_data(symbol, interval)

    if df.empty or len(df) < 50:
        return None

    price = float(df["Close"].iloc[-1])
    sma20 = float(df["SMA20"].iloc[-1])
    sma50 = float(df["SMA50"].iloc[-1])
    rsi = float(df["RSI"].iloc[-1])
    volume_ratio = float(df["VolumeRatio"].iloc[-1])

    recent = df.tail(20)

    support = float(recent["Low"].min())
    resistance = float(recent["High"].max())

    bullish = 0
    bearish = 0
    bull_reasons = []
    bear_reasons = []

    if price > sma20:
        bullish += 20
        bull_reasons.append("Price above SMA20")
    else:
        bearish += 20
        bear_reasons.append("Price below SMA20")

    if sma20 > sma50:
        bullish += 20
        bull_reasons.append("SMA20 above SMA50")
    else:
        bearish += 20
        bear_reasons.append("SMA20 below SMA50")

    if rsi > 55:
        bullish += 20
        bull_reasons.append("RSI bullish")
    elif rsi < 45:
        bearish += 20
        bear_reasons.append("RSI weak")

    if volume_ratio >= 1.5:
        if price >= df["Open"].iloc[-1]:
            bullish += 20
            bull_reasons.append("High volume with positive candle")
        else:
            bearish += 20
            bear_reasons.append("High volume with negative candle")

    if price > resistance * 0.995:
        bullish += 10
        bull_reasons.append("Near resistance breakout")
    elif price < support * 1.005:
        bullish += 10
        bull_reasons.append("Near support")

    bullish = min(bullish, 100)
    bearish = min(bearish, 100)

    if bullish >= 60 and bullish > bearish:
        setup = "BULLISH"
    elif bearish >= 60 and bearish > bullish:
        setup = "BEARISH"
    elif bearish >= 50 and rsi < 40 and price <= support * 1.01:
        setup = "POTENTIAL REVERSAL"
    else:
        setup = "NEUTRAL"

    confirmation = "NOT CONFIRMED"

    if setup == "BULLISH" and price > sma20 and rsi > 50:
        confirmation = "CONFIRMED"
    elif setup == "BEARISH" and price < sma20 and rsi < 50:
        confirmation = "CONFIRMED"

    if setup == "BULLISH" and confirmation == "CONFIRMED":
        action = "BUY"
    elif setup == "BEARISH" and confirmation == "CONFIRMED":
        action = "EXIT"
    elif setup == "POTENTIAL REVERSAL":
        action = "WATCH"
    else:
        action = "WAIT"

    confidence = max(bullish, bearish)

    if setup == "POTENTIAL REVERSAL":
        confidence = min(
            90,
            50 + (10 if volume_ratio >= 1.5 else 0)
            + (10 if rsi < 35 else 0)
            + (10 if price <= support * 1.01 else 0)
        )

    if action == "BUY":
        entry_low = price * 0.995
        entry_high = price
        stop = min(support * 0.99, price * 0.97)
        risk = entry_high - stop
        target1 = entry_high + risk * 2
        target2 = entry_high + risk * 3
        rr = 2.0
    elif action == "WATCH":
        entry_low = support
        entry_high = price
        stop = support * 0.985
        risk = entry_high - stop
        target1 = resistance
        target2 = price + risk * 3
        rr = (target1 - entry_high) / risk if risk > 0 else 0
    else:
        entry_low = support
        entry_high = price
        stop = resistance * 1.01
        target1 = support
        target2 = support * 0.99
        rr = 0

    if setup == "POTENTIAL REVERSAL":
        trigger = (
            f"BUY confirmation requires price to reclaim "
            f"the short-term trend level near {sma20:.2f} "
            f"while RSI improves."
        )
        invalidation = f"A decisive move below {support:.2f} invalidates the reversal idea."
    elif action == "BUY":
        trigger = f"Price holding above SMA20 near {sma20:.2f}."
        invalidation = f"Close below support near {support:.2f}."
    else:
        trigger = f"Watch price relative to SMA20 at {sma20:.2f}."
        invalidation = f"Break of support near {support:.2f}."

    if price > sma20 and sma20 > sma50:
        trend = "BULLISH"
    elif price < sma20 and sma20 < sma50:
        trend = "BEARISH"
    else:
        trend = "MIXED"

    return {
        "symbol": symbol,
        "interval": interval,
        "action": action,
        "setup": setup,
        "confirmation": confirmation,
        "confidence": int(confidence),
        "price": price,
        "rsi": rsi,
        "volume": volume_ratio,
        "bullish": bullish,
        "bearish": bearish,
        "trend": trend,
        "support": support,
        "resistance": resistance,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "entry": entry_high,
        "stop": stop,
        "target1": target1,
        "target2": target2,
        "rr": rr,
        "trigger": trigger,
        "invalidation": invalidation,
        "bull_reasons": bull_reasons,
        "bear_reasons": bear_reasons
    }


def scan_market(symbols, interval):
    rows = []

    for symbol in symbols:
        symbol = symbol.strip().upper()

        if not symbol:
            continue

        try:
            r = analyze(symbol, interval)

            if r:
                rows.append({
                    "Symbol": symbol,
                    "Action": r["action"],
                    "Setup": r["setup"],
                    "Confirm": r["confirmation"],
                    "Trend": r["trend"],
                    "RSI": round(r["rsi"], 1),
                    "Volume": round(r["volume"], 2),
                    "Bullish %": r["bullish"],
                    "Bearish %": r["bearish"],
                    "Confidence %": r["confidence"],
                    "Price": round(r["price"], 2)
                })
        except Exception as e:
            rows.append({
                "Symbol": symbol,
                "Action": "ERROR",
                "Setup": str(e),
                "Confirm": "",
                "Trend": "",
                "RSI": 0,
                "Volume": 0,
                "Bullish %": 0,
                "Bearish %": 0,
                "Confidence %": 0,
                "Price": 0
            })

    return pd.DataFrame(rows)


st.title("🌍 Global Stock Analyzer")
st.caption("Multi-market technical analysis and trading setup engine")

tab1, tab2, tab3 = st.tabs(
    ["📊 Single Stock", "🌍 Global Scanner", "📈 Performance"]
)

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        symbol = st.text_input(
            "Stock symbol",
            value="AAPL"
        ).strip().upper()

    with col2:
        interval = st.selectbox(
            "Timeframe",
            ["5m", "15m", "30m", "1h"],
            index=1
        )

    if st.button("🔎 Analyze Stock", use_container_width=True):
        with st.spinner("Analyzing market data..."):
            try:
                result = analyze(symbol, interval)

                if result is None:
                    st.error("No usable market data found.")
                else:
                    save_signal(result)

                    st.success(
                        f"{result['action']} — {result['setup']}"
                    )

                    c1, c2, c3, c4 = st.columns(4)

                    c1.metric(
                        "Price",
                        f"{result['price']:.2f}"
                    )

                    c2.metric(
                        "Confidence",
                        f"{result['confidence']}%"
                    )

                    c3.metric(
                        "RSI",
                        f"{result['rsi']:.1f}"
                    )

                    c4.metric(
                        "Volume",
                        f"{result['volume']:.2f}x"
                    )

                    st.subheader("Trading Decision")

                    d1, d2, d3 = st.columns(3)

                    d1.metric("Action", result["action"])
                    d2.metric("Setup", result["setup"])
                    d3.metric(
                        "Confirmation",
                        result["confirmation"]
                    )

                    st.write(
                        f"**Trend:** {result['trend']}"
                    )

                    st.write(
                        f"**Bullish:** {result['bullish']}%  |  "
                        f"**Bearish:** {result['bearish']}%"
                    )

                    st.subheader("Trade Levels")

                    t1, t2, t3, t4 = st.columns(4)

                    t1.metric(
                        "Entry",
                        f"{result['entry_low']:.2f} - {result['entry_high']:.2f}"
                    )

                    t2.metric(
                        "Stop Loss",
                        f"{result['stop']:.2f}"
                    )

                    t3.metric(
                        "Target 1",
                        f"{result['target1']:.2f}"
                    )

                    t4.metric(
                        "Target 2",
                        f"{result['target2']:.2f}"
                    )

                    st.write(
                        f"**Risk / Reward:** {result['rr']:.2f}R"
                    )

                    st.info(
                        f"**Trigger:** {result['trigger']}"
                    )

                    st.warning(
                        f"**Invalidation:** {result['invalidation']}"
                    )

                    st.subheader("Market Structure")

                    m1, m2 = st.columns(2)

                    m1.metric(
                        "Support",
                        f"{result['support']:.2f}"
                    )

                    m2.metric(
                        "Resistance",
                        f"{result['resistance']:.2f}"
                    )

                    if result["bull_reasons"]:
                        st.write("**Bullish factors**")
                        for x in result["bull_reasons"]:
                            st.write("• " + x)

                    if result["bear_reasons"]:
                        st.write("**Bearish factors**")
                        for x in result["bear_reasons"]:
                            st.write("• " + x)

                    st.subheader("Recent Price Data")

                    data = get_data(symbol, interval)

                    if not data.empty:
                        chart_data = data[["Close"]].tail(100)
                        st.line_chart(chart_data)

            except Exception as e:
                st.error(f"Analysis failed: {e}")


with tab2:
    st.subheader("🌍 Global Market Scanner")

    default_symbols = (
        "AAPL,MSFT,NVDA,AMZN,META,TSLA,GOOGL,AMD,"
        "RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS"
    )

    symbols_text = st.text_area(
        "Symbols",
        value=default_symbols
    )

    scan_interval = st.selectbox(
        "Scanner timeframe",
        ["5m", "15m", "30m", "1h"],
        index=1,
        key="scanner_interval"
    )

    if st.button("🌍 Scan Market", use_container_width=True):
        symbols = symbols_text.split(",")

        with st.spinner("Scanning markets..."):
            df_scan = scan_market(symbols, scan_interval)

        if df_scan.empty:
            st.warning("No results found.")
        else:
            st.dataframe(
                df_scan,
                use_container_width=True,
                hide_index=True
            )

            buy_count = (df_scan["Action"] == "BUY").sum()
            watch_count = (df_scan["Action"] == "WATCH").sum()
            wait_count = (df_scan["Action"] == "WAIT").sum()
            exit_count = (df_scan["Action"] == "EXIT").sum()

            a, b, c, d = st.columns(4)

            a.metric("BUY", int(buy_count))
            b.metric("WATCH", int(watch_count))
            c.metric("WAIT", int(wait_count))
            d.metric("EXIT", int(exit_count))


with tab3:
    st.subheader("📈 Signal Performance")

    history = load_history()

    if history.empty:
        st.info("No signal history yet.")
    else:
        st.metric(
            "Total Signals",
            len(history)
        )

        if "Action" in history.columns:
            st.write("### Action Distribution")
            st.dataframe(
                history["Action"].value_counts().to_frame("Count"),
                use_container_width=True
            )

        st.write("### Recent Signals")

        st.dataframe(
            history.tail(50),
            use_container_width=True,
            hide_index=True
        )

        csv_data = history.to_csv(index=False)

        st.download_button(
            label="⬇️ Download Signal History",
            data=csv_data,
            file_name="signal_history.csv",
            mime="text/csv",
            use_container_width=True
               )
