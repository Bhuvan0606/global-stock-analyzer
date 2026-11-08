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


# =========================
# DATA
# =========================

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
        period=periods.get(interval, "60d"),
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

    df["VolAvg"] = df["Volume"].rolling(20).mean()
    df["VolRatio"] = df["Volume"] / df["VolAvg"]

    return df.dropna()


# =========================
# SIGNAL ENGINE
# =========================

def signal_from_row(df, i):
    r = df.iloc[i]

    price = float(r["Close"])
    sma20 = float(r["SMA20"])
    sma50 = float(r["SMA50"])
    rsi = float(r["RSI"])
    vol = float(r["VolRatio"])

    bullish = 0
    bearish = 0

    if price > sma20:
        bullish += 25
    else:
        bearish += 25

    if sma20 > sma50:
        bullish += 25
    else:
        bearish += 25

    if rsi > 55:
        bullish += 20
    elif rsi < 45:
        bearish += 20

    if vol >= 1.5:
        if price >= float(r["Open"]):
            bullish += 15
        else:
            bearish += 15

    if rsi < 35 and price <= sma20 * 1.01:
        bullish += 15

    bullish = min(bullish, 100)
    bearish = min(bearish, 100)

    if bullish >= 60 and bullish > bearish:
        action = "BUY"
        setup = "BULLISH TREND"
    elif bearish >= 60 and bearish > bullish:
        action = "EXIT"
        setup = "BEARISH TREND"
    elif rsi < 35 and price <= sma20 * 1.01 and vol >= 1.5:
        action = "WATCH"
        setup = "POTENTIAL REVERSAL"
    else:
        action = "WAIT"
        setup = "NEUTRAL"

    return {
        "action": action,
        "setup": setup,
        "price": price,
        "bullish": bullish,
        "bearish": bearish,
        "rsi": rsi,
        "sma20": sma20
    }


# =========================
# CURRENT ANALYSIS
# =========================

def analyze(symbol, interval):
    df = get_data(symbol, interval)

    if df.empty or len(df) < 60:
        return None

    i = len(df) - 1
    s = signal_from_row(df, i)

    price = s["price"]

    recent = df.iloc[max(0, i - 20):i + 1]

    support = float(recent["Low"].min())
    resistance = float(recent["High"].max())

    if s["action"] == "BUY":
        entry = price
        stop = min(support * 0.99, price * 0.97)

        risk = entry - stop

        target1 = entry + risk * 2
        target2 = entry + risk * 3

        rr = 2.0

        confirmation = (
            "CONFIRMED"
            if price > s["sma20"] and s["rsi"] > 50
            else "NOT CONFIRMED"
        )

    elif s["action"] == "WATCH":
        entry = price
        stop = support * 0.985

        risk = entry - stop

        target1 = resistance
        target2 = entry + risk * 3

        rr = (
            (target1 - entry) / risk
            if risk > 0
            else 0
        )

        confirmation = "NOT CONFIRMED"

    else:
        entry = price
        stop = resistance * 1.01
        target1 = support
        target2 = support * 0.99
        rr = 0
        confirmation = "NOT CONFIRMED"

    if price > s["sma20"] and s["sma20"] > float(df["SMA50"].iloc[-1]):
        trend = "BULLISH"
    elif price < s["sma20"] and s["sma20"] < float(df["SMA50"].iloc[-1]):
        trend = "BEARISH"
    else:
        trend = "MIXED"

    return {
        "symbol": symbol,
        "interval": interval,
        "action": s["action"],
        "setup": s["setup"],
        "confirmation": confirmation,
        "price": price,
        "rsi": s["rsi"],
        "volume": float(df["VolRatio"].iloc[-1]),
        "bullish": s["bullish"],
        "bearish": s["bearish"],
        "confidence": max(s["bullish"], s["bearish"]),
        "trend": trend,
        "support": support,
        "resistance": resistance,
        "entry": entry,
        "stop": stop,
        "target1": target1,
        "target2": target2,
        "rr": rr
    }


# =========================
# SIGNAL HISTORY
# =========================

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            return pd.read_csv(HISTORY_FILE)
        except:
            pass

    return pd.DataFrame()


def save_signal(r):
    row = {
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Symbol": r["symbol"],
        "Interval": r["interval"],
        "Action": r["action"],
        "Setup": r["setup"],
        "Price": r["price"],
        "RSI": r["rsi"],
        "Confidence": r["confidence"]
    }

    old = load_history()

    new = pd.DataFrame([row])

    history = pd.concat(
        [old, new],
        ignore_index=True
    )

    history.to_csv(
        HISTORY_FILE,
        index=False
    )


# =========================
# BACKTEST
# =========================

def backtest(symbol, interval, target_r=2.0, stop_pct=0.03):
    df = get_data(symbol, interval)

    if df.empty or len(df) < 100:
        return pd.DataFrame()

    trades = []

    start = 60
    last = len(df) - 1

    i = start

    while i < last - 5:

        s = signal_from_row(df, i)

        if s["action"] != "BUY":
            i += 1
            continue

        entry = float(df["Close"].iloc[i])

        stop = entry * (1 - stop_pct)

        target = entry + (
            entry - stop
        ) * target_r

        outcome = "OPEN"
        exit_price = float(df["Close"].iloc[last])
        bars = 0

        for j in range(i + 1, min(i + 30, last + 1)):

            high = float(df["High"].iloc[j])
            low = float(df["Low"].iloc[j])

            bars += 1

            hit_stop = low <= stop
            hit_target = high >= target

            if hit_stop and hit_target:
                outcome = "LOSS"
                exit_price = stop
                break

            if hit_target:
                outcome = "WIN"
                exit_price = target
                break

            if hit_stop:
                outcome = "LOSS"
                exit_price = stop
                break

        if outcome == "OPEN":
            exit_price = float(df["Close"].iloc[
                min(i + 30, last)
            ])

            if exit_price > entry:
                outcome = "WIN"
            else:
                outcome = "LOSS"

        ret = (
            (exit_price - entry)
            / entry
            * 100
        )

        trades.append({
            "Entry Time": str(df.index[i]),
            "Entry": round(entry, 2),
            "Stop": round(stop, 2),
            "Target": round(target, 2),
            "Exit": round(exit_price, 2),
            "Result": outcome,
            "Return %": round(ret, 2),
            "Bars": bars
        })

        i += 5

    return pd.DataFrame(trades)


# =========================
# UI
# =========================

st.title("🌍 Global Stock Analyzer")

st.caption(
    "Technical analysis + global scanner + historical strategy testing"
)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Single Stock",
        "🌍 Global Scanner",
        "📈 Performance",
        "🧪 Backtest"
    ]
)


# =========================
# SINGLE STOCK
# =========================

with tab1:

    c1, c2 = st.columns(2)

    with c1:
        symbol = st.text_input(
            "Stock symbol",
            "AAPL"
        ).strip().upper()

    with c2:
        interval = st.selectbox(
            "Timeframe",
            ["5m", "15m", "30m", "1h", "1d"],
            index=1
        )

    if st.button(
        "🔎 Analyze Stock",
        use_container_width=True
    ):

        with st.spinner("Analyzing..."):

            try:
                r = analyze(
                    symbol,
                    interval
                )

                if r is None:
                    st.error(
                        "No usable market data found."
                    )

                else:

                    save_signal(r)

                    st.success(
                        f"{r['action']} — {r['setup']}"
                    )

                    a, b, c, d = st.columns(4)

                    a.metric(
                        "Price",
                        f"{r['price']:.2f}"
                    )

                    b.metric(
                        "Confidence",
                        f"{r['confidence']}%"
                    )

                    c.metric(
                        "RSI",
                        f"{r['rsi']:.1f}"
                    )

                    d.metric(
                        "Volume",
                        f"{r['volume']:.2f}x"
                    )

                    st.subheader(
                        "Trading Decision"
                    )

                    a, b, c = st.columns(3)

                    a.metric(
                        "Action",
                        r["action"]
                    )

                    b.metric(
                        "Setup",
                        r["setup"]
                    )

                    c.metric(
                        "Confirmation",
                        r["confirmation"]
                    )

                    st.write(
                        f"**Trend:** {r['trend']}"
                    )

                    st.write(
                        f"**Bullish:** {r['bullish']}%  |  "
                        f"**Bearish:** {r['bearish']}%"
                    )

                    st.subheader(
                        "Trade Levels"
                    )

                    a, b, c, d = st.columns(4)

                    a.metric(
                        "Entry",
                        f"{r['entry']:.2f}"
                    )

                    b.metric(
                        "Stop",
                        f"{r['stop']:.2f}"
                    )

                    c.metric(
                        "Target 1",
                        f"{r['target1']:.2f}"
                    )

                    d.metric(
                        "Target 2",
                        f"{r['target2']:.2f}"
                    )

                    st.write(
                        f"**Risk / Reward:** {r['rr']:.2f}R"
                    )

                    st.subheader(
                        "Market Structure"
                    )

                    a, b = st.columns(2)

                    a.metric(
                        "Support",
                        f"{r['support']:.2f}"
                    )

                    b.metric(
                        "Resistance",
                        f"{r['resistance']:.2f}"
                    )

                    data = get_data(
                        symbol,
                        interval
                    )

                    if not data.empty:
                        st.subheader(
                            "Price Chart"
                        )

                        st.line_chart(
                            data[["Close"]].tail(100)
                        )

            except Exception as e:
                st.error(
                    f"Analysis failed: {e}"
                )


# =========================
# GLOBAL SCANNER
# =========================

with tab2:

    st.subheader(
        "🌍 Global Market Scanner"
    )

    symbols = st.text_area(
        "Symbols",
        (
            "AAPL,MSFT,NVDA,AMZN,META,TSLA,"
            "GOOGL,AMD,RELIANCE.NS,TCS.NS,"
            "INFY.NS,HDFCBANK.NS"
        )
    )

    scan_interval = st.selectbox(
        "Scanner timeframe",
        ["5m", "15m", "30m", "1h"],
        key="scan_tf"
    )

    if st.button(
        "🌍 Scan Market",
        use_container_width=True
    ):

        rows = []

        symbol_list = [
            x.strip().upper()
            for x in symbols.split(",")
            if x.strip()
        ]

        progress = st.progress(0)

        for n, s in enumerate(symbol_list):

            try:
                r = analyze(
                    s,
                    scan_interval
                )

                if r:

                    rows.append({
                        "Symbol": s,
                        "Action": r["action"],
                        "Setup": r["setup"],
                        "Confirm": r["confirmation"],
                        "Trend": r["trend"],
                        "RSI": round(
                            r["rsi"],
                            1
                        ),
                        "Confidence": r["confidence"],
                        "Price": round(
                            r["price"],
                            2
                        )
                    })

            except:
                pass

            progress.progress(
                (n + 1) / len(symbol_list)
            )

        result_df = pd.DataFrame(rows)

        if result_df.empty:
            st.warning(
                "No results."
            )
        else:
            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True
            )


# =========================
# PERFORMANCE
# =========================

with tab3:

    st.subheader(
        "📈 Signal History"
    )

    history = load_history()

    if history.empty:

        st.info(
            "No signal history yet."
        )

    else:

        st.metric(
            "Total Signals",
            len(history)
        )

        st.dataframe(
            history.tail(100),
            use_container_width=True,
            hide_index=True
        )

        csv = history.to_csv(
            index=False
        )

        st.download_button(
            "⬇️ Download History",
            csv,
            "signal_history.csv",
            "text/csv",
            use_container_width=True
        )


# =========================
# BACKTEST
# =========================

with tab4:

    st.subheader(
        "🧪 Historical Strategy Backtest"
    )

    st.write(
        "This tests historical BUY signals and checks "
        "whether price reached the target or stop afterward."
    )

    c1, c2 = st.columns(2)

    with c1:
        bt_symbol = st.text_input(
            "Backtest symbol",
            "AAPL"
        ).strip().upper()

    with c2:
        bt_interval = st.selectbox(
            "Backtest timeframe",
            ["15m", "30m", "1h", "1d"],
            index=0,
            key="bt_tf"
        )

    target_r = st.slider(
        "Target risk/reward",
        1.0,
        5.0,
        2.0,
        0.5
    )

    stop_pct = st.slider(
        "Stop loss %",
        1.0,
        10.0,
        3.0,
        0.5
    ) / 100

    if st.button(
        "🧪 Run Backtest",
        use_container_width=True
    ):

        with st.spinner(
            "Running historical backtest..."
        ):

            trades = backtest(
                bt_symbol,
                bt_interval,
                target_r,
                stop_pct
            )

        if trades.empty:

            st.warning(
                "Not enough historical data "
                "or no BUY signals were found."
            )

        else:

            total = len(trades)

            wins = (
                trades["Result"] == "WIN"
            ).sum()

            losses = (
                trades["Result"] == "LOSS"
            ).sum()

            win_rate = (
                wins / total * 100
            )

            avg_return = (
                trades["Return %"].mean()
            )

            gross_profit = (
                trades.loc[
                    trades["Return %"] > 0,
                    "Return %"
                ].sum()
            )

            gross_loss = abs(
                trades.loc[
                    trades["Return %"] < 0,
                    "Return %"
                ].sum()
            )

            if gross_loss > 0:
                profit_factor = (
                    gross_profit /
                    gross_loss
                )
            else:
                profit_factor = 0

            a, b, c, d = st.columns(4)

            a.metric(
                "Trades",
                total
            )

            b.metric(
                "Win Rate",
                f"{win_rate:.1f}%"
            )

            c.metric(
                "Avg Return",
                f"{avg_return:.2f}%"
            )

            d.metric(
                "Profit Factor",
                f"{profit_factor:.2f}"
            )

            st.subheader(
                "Backtest Trades"
            )

            st.dataframe(
                trades,
                use_container_width=True,
                hide_index=True
            )

            st.subheader(
                "Equity Curve"
            )

            equity = (
                100 *
                (1 + trades["Return %"] / 100)
                .cumprod()
            )

            equity_df = pd.DataFrame(
                {
                    "Equity": equity.values
                }
            )

            st.line_chart(
                equity_df
            )

            csv = trades.to_csv(
                index=False
            )

            st.download_button(
                "⬇️ Download Backtest",
                csv,
                "backtest_results.csv",
                "text/csv",
                use_container_width=True
    )
