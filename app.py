import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime

st.set_page_config(page_title="Global Stock Analyzer", layout="wide")

st.title("🌍 Global Stock Analyzer")
st.caption("v15 — Multi-Timeframe Trading Intelligence")


# ---------- DATA ----------

def get_data(symbol, interval):
    periods = {
        "5m": "7d",
        "15m": "30d",
        "30m": "60d",
        "1h": "60d",
        "1d": "2y"
    }

    df = yf.download(
        symbol,
        period=periods[interval],
        interval=interval,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(x).title() for x in df.columns]

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()

    rs = gain / loss.replace(0, np.nan)

    df["RSI"] = 100 - (100 / (1 + rs))
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["VolAvg"] = df["Volume"].rolling(20).mean()
    df["VolRatio"] = df["Volume"] / df["VolAvg"]

    return df.dropna()


# ---------- TREND ----------

def get_trend(df):
    if df.empty:
        return "UNKNOWN"

    x = df.iloc[-1]

    if x["Close"] > x["SMA20"] > x["SMA50"]:
        return "BULLISH"

    if x["Close"] < x["SMA20"] < x["SMA50"]:
        return "BEARISH"

    return "NEUTRAL"


def higher_tf(tf):
    if tf == "5m":
        return "15m"
    if tf in ["15m", "30m"]:
        return "1h"
    return "1d"


def mtf_analysis(symbol, tf):
    high_tf = higher_tf(tf)

    primary = get_data(symbol, tf)
    higher = get_data(symbol, high_tf)
    daily = get_data(symbol, "1d")

    p = get_trend(primary)
    h = get_trend(higher)
    d = get_trend(daily)

    values = {
        "BULLISH": 100,
        "NEUTRAL": 50,
        "BEARISH": 0,
        "UNKNOWN": 50
    }

    score = int(np.mean([
        values[p],
        values[h],
        values[d]
    ]))

    if p == h == d == "BULLISH":
        label = "STRONG BULLISH ALIGNMENT"
    elif p == h == d == "BEARISH":
        label = "STRONG BEARISH ALIGNMENT"
    elif score >= 67:
        label = "BULLISH ALIGNMENT"
    elif score <= 33:
        label = "BEARISH ALIGNMENT"
    else:
        label = "MIXED"

    return {
        "primary": p,
        "higher": h,
        "daily": d,
        "score": score,
        "label": label
    }


# ---------- ANALYSIS ----------

def analyze(df):
    x = df.iloc[-1]

    price = float(x["Close"])
    rsi = float(x["RSI"])
    volume = float(x["VolRatio"])

    trend = get_trend(df)

    bullish = 0
    bearish = 0
    bull_reasons = []
    bear_reasons = []

    if trend == "BULLISH":
        bullish += 30
        bull_reasons.append("Price is above SMA20 and SMA50")

    elif trend == "BEARISH":
        bearish += 30
        bear_reasons.append("Price is below SMA20 and SMA50")

    if rsi < 35:
        bullish += 20
        bull_reasons.append("RSI is oversold")

    elif rsi > 65:
        bearish += 20
        bear_reasons.append("RSI is overbought")

    elif rsi >= 50:
        bullish += 10

    else:
        bearish += 10

    if volume >= 1.5:
        bullish += 10
        bearish += 10
        bull_reasons.append("High volume confirms participation")

    support = float(df["Low"].tail(20).min())
    resistance = float(df["High"].tail(20).max())

    if price <= support * 1.01:
        bullish += 15
        bull_reasons.append("Price is near support")

    if price >= resistance * 0.99:
        bearish += 15
        bear_reasons.append("Price is near resistance")

    if bullish > bearish and bullish >= 60:
        action = "BUY"
    elif bearish > bullish and bearish >= 65:
        action = "EXIT"
    elif bullish >= 45:
        action = "WATCH"
    else:
        action = "WAIT"

    if rsi < 35 and price <= support * 1.01:
        setup = "POTENTIAL REVERSAL"
    elif trend == "BULLISH" and rsi >= 50:
        setup = "TREND MOMENTUM"
    else:
        setup = "NO CLEAR SETUP"

    risk = max(price - support * 0.985, price * 0.01)

    return {
        "price": price,
        "rsi": rsi,
        "volume": volume,
        "trend": trend,
        "bullish": min(bullish, 100),
        "bearish": min(bearish, 100),
        "action": action,
        "setup": setup,
        "support": support,
        "resistance": resistance,
        "entry": price,
        "stop": price - risk,
        "target1": price + risk * 1.5,
        "target2": price + risk * 2.5,
        "bull": bull_reasons,
        "bear": bear_reasons
    }


# ---------- FINAL DECISION ----------

def final_decision(a, m):
    if a["action"] == "BUY" and m["score"] >= 67:
        return "STRONG BUY"

    if a["action"] == "BUY":
        return "BUY"

    if a["action"] == "EXIT" and m["score"] <= 33:
        return "STRONG EXIT"

    if a["action"] == "EXIT":
        return "EXIT"

    if a["action"] == "WATCH":
        return "WATCH"

    return "WAIT"


# ---------- HISTORY ----------

HISTORY = "signal_history.csv"


def save_signal(symbol, tf, a, m, decision):
    row = pd.DataFrame([{
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Symbol": symbol,
        "Timeframe": tf,
        "Decision": decision,
        "Setup": a["setup"],
        "Primary": m["primary"],
        "Higher": m["higher"],
        "Daily": m["daily"],
        "Alignment": m["score"],
        "Price": a["price"],
        "RSI": a["rsi"],
        "Confidence": max(a["bullish"], a["bearish"])
    }])

    if os.path.exists(HISTORY):
        old = pd.read_csv(HISTORY)
        row = pd.concat([old, row], ignore_index=True)

    row.to_csv(HISTORY, index=False)


# ---------- TABS ----------

tab1, tab2, tab3 = st.tabs([
    "📊 Single Stock",
    "🌍 Global Scanner",
    "📜 History"
])


# ---------- SINGLE STOCK ----------

with tab1:

    symbol = st.text_input(
        "Enter stock symbol",
        "AAPL"
    ).upper().strip()

    tf = st.selectbox(
        "Primary timeframe",
        ["5m", "15m", "30m", "1h"],
        index=1
    )

    if st.button("🔍 Analyze Market"):

        with st.spinner("Analyzing market..."):

            df = get_data(symbol, tf)

            if df.empty:
                st.error("No market data found.")
            else:

                a = analyze(df)
                m = mtf_analysis(symbol, tf)
                decision = final_decision(a, m)

                save_signal(
                    symbol,
                    tf,
                    a,
                    m,
                    decision
                )

                st.subheader(
                    f"{symbol} — {tf}"
                )

                c1, c2, c3, c4 = st.columns(4)

                c1.metric("Decision", decision)
                c2.metric("Price", f"{a['price']:.2f}")
                c3.metric("Setup", a["setup"])
                c4.metric(
                    "Confidence",
                    f"{max(a['bullish'], a['bearish'])}%"
                )

                st.markdown("### 🧭 Multi-Timeframe Confirmation")

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(tf, m["primary"])
                c2.metric(higher_tf(tf), m["higher"])
                c3.metric("1D", m["daily"])
                c4.metric(
                    "Alignment",
                    f"{m['score']}/100"
                )

                if m["score"] >= 67:
                    st.success(m["label"])
                elif m["score"] <= 33:
                    st.error(m["label"])
                else:
                    st.warning(m["label"])

                st.markdown("### 📈 Market State")

                c1, c2, c3, c4 = st.columns(4)

                c1.metric("Trend", a["trend"])
                c2.metric("RSI", f"{a['rsi']:.1f}")
                c3.metric("Volume", f"{a['volume']:.2f}×")
                c4.metric(
                    "Bull / Bear",
                    f"{a['bullish']} / {a['bearish']}"
                )

                st.markdown("### 🎯 Trading Levels")

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Entry",
                    f"{a['entry']:.2f}"
                )

                c2.metric(
                    "Stop",
                    f"{a['stop']:.2f}"
                )

                c3.metric(
                    "Target 1",
                    f"{a['target1']:.2f}"
                )

                c4.metric(
                    "Target 2",
                    f"{a['target2']:.2f}"
                )

                st.markdown("### 🟢 Bullish Factors")

                for x in a["bull"]:
                    st.write("• " + x)

                st.markdown("### 🔴 Bearish Factors")

                for x in a["bear"]:
                    st.write("• " + x)

                st.markdown("### 🧠 Analyst Interpretation")

                if decision == "STRONG BUY":
                    st.success(
                        "The setup is bullish and the higher timeframes agree."
                    )
                elif decision == "BUY":
                    st.success(
                        "The primary setup is bullish, but confirmation is not perfect."
                    )
                elif decision == "WATCH":
                    st.warning(
                        "There is an interesting setup, but confirmation is incomplete."
                    )
                elif decision in ["EXIT", "STRONG EXIT"]:
                    st.error(
                        "The current technical structure is bearish."
                    )
                else:
                    st.info(
                        "No sufficiently strong setup is confirmed."
                    )

                st.markdown("### 📊 Price Chart")

                chart = df[[
                    "Close",
                    "SMA20",
                    "SMA50"
                ]].tail(150)

                st.line_chart(chart)


# ---------- SCANNER ----------

with tab2:

    st.subheader("🌍 Global Scanner")

    symbols = st.text_input(
        "Symbols separated by commas",
        "AAPL,MSFT,NVDA,AMZN,META,TSLA,GOOGL,AMD,RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS"
    )

    scan_tf = st.selectbox(
        "Scanner timeframe",
        ["5m", "15m", "30m", "1h"],
        index=1
    )

    if st.button("🚀 Scan Market"):

        result_rows = []

        symbol_list = [
            x.strip().upper()
            for x in symbols.split(",")
            if x.strip()
        ]

        progress = st.progress(0)

        for i, sym in enumerate(symbol_list):

            try:
                df = get_data(sym, scan_tf)

                if not df.empty:

                    a = analyze(df)
                    m = mtf_analysis(sym, scan_tf)
                    decision = final_decision(a, m)

                    result_rows.append({
                        "Symbol": sym,
                        "Decision": decision,
                        "Setup": a["setup"],
                        "Primary": m["primary"],
                        "Higher": m["higher"],
                        "Daily": m["daily"],
                        "Alignment": m["score"],
                        "RSI": round(a["rsi"], 1),
                        "Volume": round(a["volume"], 2),
                        "Confidence": max(
                            a["bullish"],
                            a["bearish"]
                        ),
                        "Price": round(
                            a["price"],
                            2
                        )
                    })

            except Exception:
                pass

            progress.progress(
                (i + 1) / len(symbol_list)
            )

        if result_rows:

            table = pd.DataFrame(
                result_rows
            ).sort_values(
                "Alignment",
                ascending=False
            )

            st.dataframe(
                table,
                use_container_width=True
            )

            st.download_button(
                "⬇️ Download Scanner",
                table.to_csv(index=False),
                "global_scanner.csv",
                "text/csv"
            )

        else:
            st.warning(
                "No usable market data found."
            )


# ---------- HISTORY ----------

with tab3:

    st.subheader("📜 Signal History")

    if os.path.exists(HISTORY):

        history = pd.read_csv(HISTORY)

        st.dataframe(
            history.tail(100),
            use_container_width=True
        )

        st.download_button(
            "⬇️ Download History",
            history.to_csv(index=False),
            "signal_history.csv",
            "text/csv"
        )

    else:
        st.info(
            "No signals recorded yet."
        )

st.caption(
    "Research and analysis tool. Not financial advice."
    )
