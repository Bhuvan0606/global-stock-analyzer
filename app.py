import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import os

st.set_page_config(page_title="Global Stock Analyzer", layout="wide")
st.title("🌍 Global Stock Analyzer")
st.caption("v15 • Multi-Timeframe Trading Intelligence")

# ---------------- DATA ----------------

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
        auto_adjust=True
    )

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.title)
    df = df.dropna(subset=["Close"])

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()

    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    df["VolAvg"] = df["Volume"].rolling(20).mean()
    df["VolRatio"] = df["Volume"] / df["VolAvg"]

    df["PrevHigh"] = df["High"].rolling(20).max().shift(1)

    return df.dropna()


# ---------------- TIMEFRAME ----------------

def higher_timeframe(interval):
    if interval == "5m":
        return "15m"
    if interval == "15m":
        return "1h"
    if interval == "30m":
        return "1h"
    return "1d"


def trend_from_df(df):
    if df.empty:
        return "UNKNOWN"

    last = df.iloc[-1]

    if last["Close"] > last["SMA20"] and last["SMA20"] > last["SMA50"]:
        return "BULLISH"

    if last["Close"] < last["SMA20"] and last["SMA20"] < last["SMA50"]:
        return "BEARISH"

    return "NEUTRAL"


def timeframe_score(trend):
    if trend == "BULLISH":
        return 100
    if trend == "BEARISH":
        return 0
    return 50


# ---------------- MULTI TIMEFRAME ----------------

def multi_timeframe(symbol, primary):
    higher = higher_timeframe(primary)

    primary_df = get_data(symbol, primary)
    higher_df = get_data(symbol, higher)
    daily_df = get_data(symbol, "1d")

    p_trend = trend_from_df(primary_df)
    h_trend = trend_from_df(higher_df)
    d_trend = trend_from_df(daily_df)

    scores = [
        timeframe_score(p_trend),
        timeframe_score(h_trend),
        timeframe_score(d_trend)
    ]

    alignment = int(np.mean(scores))

    bullish_count = sum(x == "BULLISH" for x in [p_trend, h_trend, d_trend])
    bearish_count = sum(x == "BEARISH" for x in [p_trend, h_trend, d_trend])

    if bullish_count == 3:
        label = "STRONG BULLISH ALIGNMENT"
    elif bearish_count == 3:
        label = "STRONG BEARISH ALIGNMENT"
    elif bullish_count >= 2:
        label = "BULLISH ALIGNMENT"
    elif bearish_count >= 2:
        label = "BEARISH ALIGNMENT"
    else:
        label = "MIXED"

    return {
        "primary": p_trend,
        "higher": h_trend,
        "daily": d_trend,
        "alignment": alignment,
        "label": label
    }


# ---------------- SETUP ENGINE ----------------

def analyze(df):
    last = df.iloc[-1]

    price = float(last["Close"])
    rsi = float(last["RSI"])
    vol = float(last["VolRatio"])

    trend = trend_from_df(df)

    bullish = 0
    bearish = 0
    reasons_b = []
    reasons_s = []

    if trend == "BULLISH":
        bullish += 30
        reasons_b.append("Price is above SMA20 and SMA50")
    elif trend == "BEARISH":
        bearish += 30
        reasons_s.append("Price is below SMA20 and SMA50")

    if rsi < 35:
        bullish += 20
        reasons_b.append("RSI is oversold")
    elif rsi > 65:
        bearish += 20
        reasons_s.append("RSI is overbought")

    if rsi > 50:
        bullish += 10
    elif rsi < 50:
        bearish += 10

    if vol >= 1.5:
        bullish += 10
        bearish += 10
        reasons_b.append("High volume confirms participation")
        reasons_s.append("High volume increases selling-risk if trend breaks")

    support = float(df["Low"].tail(20).min())
    resistance = float(df["High"].tail(20).max())

    near_support = price <= support * 1.01
    near_resistance = price >= resistance * 0.99

    if near_support:
        bullish += 15
        reasons_b.append("Price is near support")

    if near_resistance:
        bearish += 15
        reasons_s.append("Price is near resistance")

    breakout = price > float(last["PrevHigh"]) if not pd.isna(last["PrevHigh"]) else False

    if breakout and vol >= 1.2:
        bullish += 20
        reasons_b.append("Price is breaking above recent highs")

    bullish = min(100, bullish)
    bearish = min(100, bearish)

    if trend == "BULLISH" and rsi >= 50 and bullish >= bearish:
        setup = "TREND MOMENTUM"
    elif rsi < 35 and near_support:
        setup = "POTENTIAL REVERSAL"
    elif breakout:
        setup = "BREAKOUT"
    else:
        setup = "NO CLEAR SETUP"

    if bullish >= 65 and bullish > bearish:
        action = "BUY"
    elif bearish >= 70 and bearish > bullish:
        action = "EXIT"
    elif bullish >= 45 and bearish >= 45:
        action = "WATCH"
    else:
        action = "WAIT"

    confidence = int(max(bullish, bearish))

    entry_low = price * 0.998
    entry_high = price * 1.002

    stop = support * 0.985

    if stop >= price:
        stop = price * 0.98

    risk = price - stop

    target1 = price + risk * 1.5
    target2 = price + risk * 2.5

    return {
        "price": price,
        "trend": trend,
        "rsi": rsi,
        "volume": vol,
        "bullish": bullish,
        "bearish": bearish,
        "setup": setup,
        "action": action,
        "confidence": confidence,
        "support": support,
        "resistance": resistance,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "stop": stop,
        "target1": target1,
        "target2": target2,
        "rr": (target1 - price) / risk if risk > 0 else 0,
        "bullish_reasons": reasons_b,
        "bearish_reasons": reasons_s
    }


# ---------------- FINAL DECISION ----------------

def final_decision(result, mtf):
    action = result["action"]
    align = mtf["alignment"]

    if action == "BUY" and align >= 83:
        return "STRONG BUY"

    if action == "BUY" and align >= 50:
        return "BUY"

    if action == "BUY" and align < 50:
        return "WATCH"

    if action == "EXIT" and align <= 17:
        return "STRONG EXIT"

    if action == "EXIT":
        return "EXIT"

    if action == "WATCH":
        return "WATCH"

    return "WAIT"


# ---------------- HISTORY ----------------

HISTORY_FILE = "signal_history.csv"


def save_signal(symbol, interval, result, mtf, decision):
    row = {
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Symbol": symbol,
        "Interval": interval,
        "Decision": decision,
        "Setup": result["setup"],
        "PrimaryTrend": mtf["primary"],
        "HigherTrend": mtf["higher"],
        "DailyTrend": mtf["daily"],
        "Alignment": mtf["alignment"],
        "Price": result["price"],
        "RSI": result["rsi"],
        "Confidence": result["confidence"]
    }

    new = pd.DataFrame([row])

    if os.path.exists(HISTORY_FILE):
        old = pd.read_csv(HISTORY_FILE)
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new

    out.to_csv(HISTORY_FILE, index=False)


# ---------------- BACKTEST ----------------

def strategy_signal(df, i, strategy):
    row = df.iloc[i]

    price = row["Close"]
    sma20 = row["SMA20"]
    sma50 = row["SMA50"]
    rsi = row["RSI"]
    vol = row["VolRatio"]

    if strategy == "Trend Momentum":
        return price > sma20 and sma20 > sma50 and rsi > 50

    if strategy == "Oversold Reversal":
        return rsi < 35 and price <= df["Low"].iloc[max(0, i-20):i+1].min() * 1.01

    if strategy == "Breakout":
        return price > row["PrevHigh"] and vol >= 1.2

    return (
        (price > sma20 and sma20 > sma50 and rsi > 50)
        or
        (rsi < 35 and price <= df["Low"].iloc[max(0, i-20):i+1].min() * 1.01)
    )


def backtest(df, strategy, target_r=1.5, stop_pct=0.02, max_bars=20):
    trades = []

    for i in range(50, len(df) - 2):
        if not strategy_signal(df, i, strategy):
            continue

        entry_i = i + 1
        entry = float(df.iloc[entry_i]["Open"])

        stop = entry * (1 - stop_pct)
        target = entry + (entry - stop) * target_r

        end = min(entry_i + max_bars, len(df) - 1)

        exit_price = None
        result = "OPEN"
        bars = 0

        for j in range(entry_i, end + 1):
            high = float(df.iloc[j]["High"])
            low = float(df.iloc[j]["Low"])

            bars += 1

            if low <= stop:
                exit_price = stop
                result = "LOSS"
                break

            if high >= target:
                exit_price = target
                result = "WIN"
                break

        if exit_price is None:
            exit_price = float(df.iloc[end]["Close"])
            result = "TIME"

        ret = (exit_price - entry) / entry * 100

        trades.append({
            "Signal Time": df.index[i],
            "Entry Time": df.index[entry_i],
            "Entry": entry,
            "Stop": stop,
            "Target": target,
            "Exit": exit_price,
            "Result": result,
            "Return %": ret,
            "Bars": bars
        })

    return pd.DataFrame(trades)


def stats(trades):
    if trades.empty:
        return 0, 0, 0, 0

    wins = trades[trades["Result"] == "WIN"]
    losses = trades[trades["Result"] == "LOSS"]

    win_rate = len(wins) / len(trades) * 100
    avg_return = trades["Return %"].mean()

    gross_win = wins["Return %"].sum()
    gross_loss = abs(losses["Return %"].sum())

    profit_factor = (
        gross_win / gross_loss
        if gross_loss > 0
        else 999
    )

    total_return = trades["Return %"].sum()

    return win_rate, avg_return, profit_factor, total_return


# ---------------- TABS ----------------

tabs = st.tabs([
    "📊 Single Stock",
    "🌍 Global Scanner",
    "📜 Signal History",
    "🧪 Strategy Lab"
])


# ---------------- SINGLE STOCK ----------------

with tabs[0]:

    symbol = st.text_input(
        "Stock / ETF / Index",
        "AAPL"
    ).upper().strip()

    interval = st.selectbox(
        "Primary Timeframe",
        ["5m", "15m", "30m", "1h"],
        index=1
    )

    if st.button("🔍 Analyze Market"):

        with st.spinner("Analyzing multiple timeframes..."):

            df = get_data(symbol, interval)

            if df.empty:
                st.error("No market data found.")
            else:

                result = analyze(df)
                mtf = multi_timeframe(symbol, interval)
                decision = final_decision(result, mtf)

                save_signal(
                    symbol,
                    interval,
                    result,
                    mtf,
                    decision
                )

                st.subheader(f"{symbol} — {interval}")

                c1, c2, c3, c4 = st.columns(4)

                c1.metric("Decision", decision)
                c2.metric("Price", f"{result['price']:.2f}")
                c3.metric("Setup", result["setup"])
                c4.metric("Confidence", f"{result['confidence']}%")

                st.markdown("### 🧭 Multi-Timeframe Confirmation")

                a, b, c, d = st.columns(4)

                a.metric(
                    f"{interval}",
                    mtf["primary"]
                )

                a = b

                b.metric(
                    higher_timeframe(interval),
                    mtf["higher"]
                )

                c.metric(
                    "1D",
                    mtf["daily"]
                )

                d.metric(
                    "Alignment",
                    f"{mtf['alignment']}/100"
                )

                if mtf["alignment"] >= 83:
                    st.success(
                        f"🟢 {mtf['label']}"
                    )
                elif mtf["alignment"] <= 17:
                    st.error(
                        f"🔴 {mtf['label']}"
                    )
                else:
                    st.warning(
                        f"🟡 {mtf['label']}"
                    )

                st.markdown("### 📈 Technical State")

                x1, x2, x3, x4 = st.columns(4)

                x1.metric(
                    "Trend",
                    result["trend"]
                )

                x2.metric(
                    "RSI",
                    f"{result['rsi']:.1f}"
                )

                x3.metric(
                    "Volume",
                    f"{result['volume']:.2f}×"
                )

                x4.metric(
                    "Bull / Bear",
                    f"{result['bullish']} / {result['bearish']}"
                )

                st.markdown("### 🎯 Trading Levels")

                l1, l2, l3, l4 = st.columns(4)

                l1.metric(
                    "Entry",
                    f"{result['entry_low']:.2f} - {result['entry_high']:.2f}"
                )

                l2.metric(
                    "Stop",
                    f"{result['stop']:.2f}"
                )

                l3.metric(
                    "Target 1",
                    f"{result['target1']:.2f}"
                )

                l4.metric(
                    "Target 2",
                    f"{result['target2']:.2f}"
                )

                st.write(
                    f"**Risk / Reward:** {result['rr']:.2f}R"
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 🟢 Bullish Factors")

                    if result["bullish_reasons"]:
                        for x in result["bullish_reasons"]:
                            st.write("• " + x)
                    else:
                        st.write("No strong bullish factors.")

                with col2:
                    st.markdown("### 🔴 Bearish Factors")

                    if result["bearish_reasons"]:
                        for x in result["bearish_reasons"]:
                            st.write("• " + x)
                    else:
                        st.write("No strong bearish factors.")

                st.markdown("### 🧠 Analyst Interpretation")

                if decision == "STRONG BUY":
                    st.success(
                        "Multiple timeframes agree with the setup. "
                        "This is a high-quality candidate, but risk management is still required."
                    )

                elif decision == "BUY":
                    st.success(
                        "The primary setup is bullish and higher-timeframe confirmation "
                        "is supportive, but alignment is not perfect."
                    )

                elif decision == "WATCH":
                    st.warning(
                        "The setup is interesting, but confirmation is incomplete. "
                        "Wait for stronger price/volume confirmation."
                    )

                elif decision in ["EXIT", "STRONG EXIT"]:
                    st.error(
                        "The technical structure is bearish. "
                        "Avoid treating a temporary bounce as confirmation of a new uptrend."
                    )

                else:
                    st.info(
                        "No sufficiently strong trading setup is confirmed yet."
                    )

                st.markdown("### 📊 Price Chart")

                chart = df[["Close", "SMA20", "SMA50"]].tail(150)

                st.line_chart(chart)


# ---------------- SCANNER ----------------

with tabs[1]:

    st.subheader("🌍 Global Scanner")

    symbols_text = st.text_area(
        "Symbols",
        "AAPL,MSFT,NVDA,AMZN,META,TSLA,GOOGL,AMD,RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS"
    )

    scan_interval = st.selectbox(
        "Scanner Timeframe",
        ["5m", "15m", "30m", "1h"],
        index=1
    )

    if st.button("🚀 Scan Global Market"):

        symbols = [
            x.strip().upper()
            for x in symbols_text.split(",")
            if x.strip()
        ]

        rows = []

        progress = st.progress(0)

        for n, sym in enumerate(symbols):

            try:
                df = get_data(sym, scan_interval)

                if not df.empty:

                    result = analyze(df)
                    mtf = multi_timeframe(sym, scan_interval)

                    decision = final_decision(
                        result,
                        mtf
                    )

                    rows.append({
                        "Symbol": sym,
                        "Decision": decision,
                        "Setup": result["setup"],
                        "Primary": mtf["primary"],
                        "Higher": mtf["higher"],
                        "Daily": mtf["daily"],
                        "Alignment": mtf["alignment"],
                        "Trend": result["trend"],
                        "RSI": round(result["rsi"], 1),
                        "Volume": round(result["volume"], 2),
                        "Confidence": result["confidence"],
                        "Price": round(result["price"], 2)
                    })

            except Exception:
                pass

            progress.progress(
                (n + 1) / max(len(symbols), 1)
            )

        if rows:

            scan = pd.DataFrame(rows)

            scan = scan.sort_values(
                ["Alignment", "Confidence"],
                ascending=False
            )

            st.dataframe(
                scan,
                use_container_width=True
            )

            st.download_button(
                "⬇️ Download Scanner Results",
                scan.to_csv(index=False),
                "global_scan.csv",
                "text/csv"
            )

        else:
            st.warning("No symbols returned usable data.")


# ---------------- HISTORY ----------------

with tabs[2]:

    st.subheader("📜 Signal History")

    if os.path.exists(HISTORY_FILE):

        hist = pd.read_csv(HISTORY_FILE)

        st.dataframe(
            hist.tail(100),
            use_container_width=True
        )

        st.download_button(
            "⬇️ Download Signal History",
            hist.to_csv(index=False),
            "signal_history.csv",
            "text/csv"
        )

    else:
        st.info("No signals recorded yet.")


# ---------------- STRATEGY LAB ----------------

with tabs[3]:

    st.subheader("🧪 Strategy Lab")

    bt_symbol = st.text_input(
        "Backtest Symbol",
        "AAPL",
        key="bt_symbol"
    ).upper().strip()

    bt_interval = st.selectbox(
        "Backtest Timeframe",
        ["15m", "30m", "1h", "1d"],
        index=2
    )

    strategies = [
        "Trend Momentum",
        "Oversold Reversal",
        "Breakout",
        "Combined Score"
    ]

    strategy = st.selectbox(
        "Strategy",
        strategies
    )

    target_r = st.slider(
        "Target R",
        1.0,
        4.0,
        1.5,
        0.25
    )

    stop_pct = st.slider(
        "Stop Loss %",
        0.5,
        5.0,
        2.0,
        0.5
    ) / 100

    if st.button("🧪 Run Backtest"):

        with st.spinner("Running historical test..."):

            df = get_data(
                bt_symbol,
                bt_interval
            )

            if df.empty:
                st.error("No historical data.")
            else:

                trades = backtest(
              
