import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime

st.set_page_config(
    page_title="Global Stock Analyzer",
    layout="wide"
)

st.title("🌍 Global Stock Analyzer")
st.caption("v16 — Multi-Timeframe + Market Context")


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


# ---------- TRENDS ----------

def trend(df):
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


def mtf(symbol, tf):
    htf = higher_tf(tf)

    p = trend(get_data(symbol, tf))
    h = trend(get_data(symbol, htf))
    d = trend(get_data(symbol, "1d"))

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


# ---------- TECHNICAL ANALYSIS ----------

def analyze(df):
    x = df.iloc[-1]

    price = float(x["Close"])
    rsi = float(x["RSI"])
    volume = float(x["VolRatio"])

    tr = trend(df)

    bull = 0
    bear = 0
    br = []
    sr = []

    if tr == "BULLISH":
        bull += 30
        br.append("Price is above SMA20 and SMA50")
    elif tr == "BEARISH":
        bear += 30
        sr.append("Price is below SMA20 and SMA50")

    if rsi < 35:
        bull += 20
        br.append("RSI is oversold")
    elif rsi > 65:
        bear += 20
        sr.append("RSI is overbought")
    elif rsi >= 50:
        bull += 10
    else:
        bear += 10

    if volume >= 1.5:
        bull += 10
        bear += 10
        br.append("High volume")
        sr.append("High volume increases volatility risk")

    support = float(df["Low"].tail(20).min())
    resistance = float(df["High"].tail(20).max())

    if price <= support * 1.01:
        bull += 15
        br.append("Price is near support")

    if price >= resistance * 0.99:
        bear += 15
        sr.append("Price is near resistance")

    if bull > bear and bull >= 60:
        action = "BUY"
    elif bear > bull and bear >= 65:
        action = "EXIT"
    elif bull >= 45:
        action = "WATCH"
    else:
        action = "WAIT"

    if rsi < 35 and price <= support * 1.01:
        setup = "POTENTIAL REVERSAL"
    elif tr == "BULLISH" and rsi >= 50:
        setup = "TREND MOMENTUM"
    else:
        setup = "NO CLEAR SETUP"

    risk = max(
        price - support * 0.985,
        price * 0.01
    )

    return {
        "price": price,
        "rsi": rsi,
        "volume": volume,
        "trend": tr,
        "bull": min(bull, 100),
        "bear": min(bear, 100),
        "action": action,
        "setup": setup,
        "support": support,
        "resistance": resistance,
        "stop": price - risk,
        "target1": price + risk * 1.5,
        "target2": price + risk * 2.5,
        "bull_reasons": br,
        "bear_reasons": sr
    }


# ---------- NEWS ----------

def get_news(symbol):
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news

        if not news:
            return []

        return news[:8]

    except Exception:
        return []


def news_context(news):
    if not news:
        return {
            "level": "UNKNOWN",
            "text": "No recent news was returned."
        }

    text = " ".join([
        str(n.get("title", ""))
        for n in news
    ]).lower()

    negative = [
        "lawsuit",
        "downgrade",
        "miss",
        "warning",
        "investigation",
        "cut",
        "recall",
        "fraud"
    ]

    positive = [
        "upgrade",
        "beat",
        "growth",
        "record",
        "partnership",
        "approval",
        "strong"
    ]

    neg = sum(word in text for word in negative)
    pos = sum(word in text for word in positive)

    if pos > neg:
        return {
            "level": "POSITIVE",
            "text": "Recent headlines contain more positive signals."
        }

    if neg > pos:
        return {
            "level": "CAUTION",
            "text": "Recent headlines contain potentially negative risk signals."
        }

    return {
        "level": "MIXED",
        "text": "Recent headlines appear mixed."
    }


# ---------- FINAL DECISION ----------

def final_decision(a, m, news):
    score = m["score"]

    if a["action"] == "BUY" and score >= 67:
        decision = "STRONG BUY"
    elif a["action"] == "BUY":
        decision = "BUY"
    elif a["action"] == "EXIT" and score <= 33:
        decision = "STRONG EXIT"
    elif a["action"] == "EXIT":
        decision = "EXIT"
    elif a["action"] == "WATCH":
        decision = "WATCH"
    else:
        decision = "WAIT"

    if news["level"] == "CAUTION" and decision in ["STRONG BUY", "BUY"]:
        decision = "WATCH"

    return decision


# ---------- HISTORY ----------

HISTORY = "signal_history.csv"


def save_signal(symbol, tf, a, m, news, decision):
    row = pd.DataFrame([{
        "Time": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "Symbol": symbol,
        "Timeframe": tf,
        "Decision": decision,
        "Setup": a["setup"],
        "Primary": m["primary"],
        "Higher": m["higher"],
        "Daily": m["daily"],
        "Alignment": m["score"],
        "News": news["level"],
        "Price": a["price"],
        "RSI": a["rsi"]
    }])

    if os.path.exists(HISTORY):
        old = pd.read_csv(HISTORY)
        row = pd.concat(
            [old, row],
            ignore_index=True
        )

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

        with st.spinner(
            "Analyzing market, timeframes and news..."
        ):

            df = get_data(symbol, tf)

            if df.empty:
                st.error("No market data found.")

            else:

                a = analyze(df)
                m = mtf(symbol, tf)
                news = news_context(
                    get_news(symbol)
                )

                decision = final_decision(
                    a,
                    m,
                    news
                )

                save_signal(
                    symbol,
                    tf,
                    a,
                    m,
                    news,
                    decision
                )

                st.subheader(
                    f"{symbol} — {tf}"
                )

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Decision",
                    decision
                )

                c2.metric(
                    "Price",
                    f"{a['price']:.2f}"
                )

                c3.metric(
                    "Setup",
                    a["setup"]
                )

                c4.metric(
                    "Confidence",
                    f"{max(a['bull'], a['bear'])}%"
                )

                st.markdown(
                    "### 🧭 Multi-Timeframe Confirmation"
                )

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    tf,
                    m["primary"]
                )

                c2.metric(
                    higher_tf(tf),
                    m["higher"]
                )

                c3.metric(
                    "1D",
                    m["daily"]
                )

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

                st.markdown(
                    "### 📰 Market Context"
                )

                if news["level"] == "POSITIVE":
                    st.success(
                        "🟢 " + news["text"]
                    )
                elif news["level"] == "CAUTION":
                    st.warning(
                        "⚠️ " + news["text"]
                    )
                else:
                    st.info(
                        "ℹ️ " + news["text"]
                    )

                st.markdown(
                    "### Recent Headlines"
                )

                raw_news = get_news(symbol)

                if raw_news:
                    for item in raw_news:

                        title = item.get(
                            "title",
                            "Headline"
                        )

                        publisher = item.get(
                            "publisher",
                            ""
                        )

                        st.write(
                            f"• **{title}** — {publisher}"
                        )
                else:
                    st.write(
                        "No recent headlines available."
                    )

                st.markdown(
                    "### 📈 Technical State"
                )

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Trend",
                    a["trend"]
                )

                c2.metric(
                    "RSI",
                    f"{a['rsi']:.1f}"
                )

                c3.metric(
                    "Volume",
                    f"{a['volume']:.2f}×"
                )

                c4.metric(
                    "Bull / Bear",
                    f"{a['bull']} / {a['bear']}"
                )

                st.markdown(
                    "### 🎯 Trading Levels"
                )

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Entry",
                    f"{a['price']:.2f}"
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

                st.markdown(
                    "### 🟢 Bullish Factors"
                )

                for x in a["bull_reasons"]:
                    st.write("• " + x)

                st.markdown(
                    "### 🔴 Bearish Factors"
                )

                for x in a["bear_reasons"]:
                    st.write("• " + x)

                st.markdown(
                    "### 🧠 Analyst Interpretation"
                )

                if decision == "STRONG BUY":
                    st.success(
                        "Technical conditions and higher timeframes "
                        "support the setup. News context does not "
                        "currently block the signal."
                    )

                elif decision == "BUY":
                    st.success(
                        "The technical setup is bullish, but "
                        "confirmation is not perfect."
                    )

                elif decision == "WATCH":
                    st.warning(
                        "The setup has potential, but confirmation "
                        "or news conditions require caution."
                    )

                elif decision in [
                    "EXIT",
                    "STRONG EXIT"
                ]:
                    st.error(
                        "Current technical conditions are bearish."
                    )

                else:
                    st.info(
                        "There is not enough confirmation "
                        "for a strong trading decision."
                    )

                st.markdown(
                    "### 📊 Price Chart"
                )

                chart = df[
                    ["Close", "SMA20", "SMA50"]
                ].tail(150)

                st.line_chart(chart)


# ---------- SCANNER ----------

with tab2:

    st.subheader(
        "🌍 Global Scanner"
    )

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

        rows = []

        symbol_list = [
            x.strip().upper()
            for x in symbols.split(",")
            if x.strip()
        ]

        progress = st.progress(0)

        for i, sym in enumerate(
            symbol_list
        ):

            try:

                df = get_data(
                    sym,
                    scan_tf
                )

                if not df.empty:

                    a = analyze(df)
                    m = mtf(sym, scan_tf)

                    news = news_context(
                        get_news(sym)
                    )

                    decision = final_decision(
                        a,
                        m,
                        news
                    )

                    rows.append({
                        "Symbol": sym,
                        "Decision": decision,
                        "Setup": a["setup"],
                        "Primary": m["primary"],
                        "Higher": m["higher"],
                        "Daily": m["daily"],
                        "Alignment": m["score"],
                        "News": news["level"],
                        "RSI": round(
                            a["rsi"],
                            1
                        ),
                        "Confidence": max(
                            a["bull"],
                            a["bear"]
                        ),
                        "Price": round(
                            a["price"],
                            2
                        )
                    })

            except Exception:
                pass

            progress.progress(
                (i + 1) /
                max(len(symbol_list), 1)
            )

        if rows:

            table = pd.DataFrame(
                rows
            ).sort_values(
                ["Alignment", "Confidence"],
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

    st.subheader(
        "📜 Signal History"
    )

    if os.path.exists(HISTORY):

        history = pd.read_csv(
            HISTORY
        )

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
