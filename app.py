import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os

st.set_page_config(
    page_title="Global Stock Analyzer",
    page_icon="📈",
    layout="wide"
)

HISTORY_FILE = "signal_history.csv"


def data(symbol, interval):
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

    change = df["Close"].diff()
    gain = change.clip(lower=0).rolling(14).mean()
    loss = -change.clip(upper=0).rolling(14).mean()

    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    df["VolRatio"] = (
        df["Volume"] /
        df["Volume"].rolling(20).mean()
    )

    df["PrevHigh"] = (
        df["High"].rolling(20).max().shift(1)
    )

    return df.dropna()


def signal(df, i, strategy):
    r = df.iloc[i]

    price = float(r["Close"])
    sma20 = float(r["SMA20"])
    sma50 = float(r["SMA50"])
    rsi = float(r["RSI"])
    vol = float(r["VolRatio"])

    if strategy == "Trend Momentum":
        return (
            price > sma20
            and sma20 > sma50
            and rsi > 55
        )

    if strategy == "Oversold Reversal":
        return (
            rsi < 35
            and price <= sma20 * 1.01
            and vol >= 1.2
        )

    if strategy == "Breakout":
        return (
            price > float(r["PrevHigh"])
            and vol >= 1.5
        )

    score = 0

    if price > sma20:
        score += 25

    if sma20 > sma50:
        score += 25

    if rsi > 55:
        score += 20

    if vol >= 1.5:
        score += 15

    if price > float(r["PrevHigh"]):
        score += 15

    return score >= 60


def backtest(df, strategy, target_r, stop_pct, max_bars):
    trades = []
    i = 60

    while i < len(df) - 2:
        if not signal(df, i, strategy):
            i += 1
            continue

        entry_i = i + 1
        entry = float(df["Open"].iloc[entry_i])

        stop = entry * (1 - stop_pct)
        target = entry + (entry - stop) * target_r

        result = "OPEN"
        exit_price = entry
        exit_i = entry_i

        last = min(
            entry_i + max_bars,
            len(df) - 1
        )

        for j in range(entry_i, last + 1):
            high = float(df["High"].iloc[j])
            low = float(df["Low"].iloc[j])

            if low <= stop:
                result = "LOSS"
                exit_price = stop
                exit_i = j
                break

            if high >= target:
                result = "WIN"
                exit_price = target
                exit_i = j
                break

        if result == "OPEN":
            exit_price = float(df["Close"].iloc[last])
            exit_i = last
            result = (
                "WIN"
                if exit_price > entry
                else "LOSS"
            )

        ret = (
            (exit_price - entry)
            / entry
            * 100
        )

        trades.append({
            "Signal": str(df.index[i]),
            "Entry Time": str(df.index[entry_i]),
            "Entry": round(entry, 2),
            "Stop": round(stop, 2),
            "Target": round(target, 2),
            "Exit": round(exit_price, 2),
            "Result": result,
            "Return %": round(ret, 2),
            "Bars": exit_i - entry_i + 1
        })

        i = exit_i + 1

    return pd.DataFrame(trades)


def stats(trades):
    if trades.empty:
        return {
            "Trades": 0,
            "Win Rate": 0,
            "Avg Return": 0,
            "Profit Factor": 0,
            "Total Return": 0
        }

    total = len(trades)

    wins = (
        trades["Result"] == "WIN"
    ).sum()

    win_rate = wins / total * 100
    avg_return = trades["Return %"].mean()
    total_return = trades["Return %"].sum()

    profit = trades.loc[
        trades["Return %"] > 0,
        "Return %"
    ].sum()

    loss = abs(
        trades.loc[
            trades["Return %"] < 0,
            "Return %"
        ].sum()
    )

    pf = (
        profit / loss
        if loss > 0
        else 0
    )

    return {
        "Trades": total,
        "Win Rate": win_rate,
        "Avg Return": avg_return,
        "Profit Factor": pf,
        "Total Return": total_return
    }


def current(symbol, interval):
    df = data(symbol, interval)

    if df.empty:
        return None, df

    r = df.iloc[-1]

    price = float(r["Close"])
    sma20 = float(r["SMA20"])
    sma50 = float(r["SMA50"])
    rsi = float(r["RSI"])
    vol = float(r["VolRatio"])

    bull = 0
    bear = 0

    if price > sma20:
        bull += 25
    else:
        bear += 25

    if sma20 > sma50:
        bull += 25
    else:
        bear += 25

    if rsi > 55:
        bull += 20
    elif rsi < 45:
        bear += 20

    if vol >= 1.5:
        if price >= float(r["Open"]):
            bull += 15
        else:
            bear += 15

    if rsi < 35:
        bull += 15

    if bull >= 60 and bull > bear:
        action = "BUY"
        setup = "BULLISH TREND"
    elif bear >= 60 and bear > bull:
        action = "EXIT"
        setup = "BEARISH TREND"
    elif rsi < 35 and vol >= 1.5:
        action = "WATCH"
        setup = "POTENTIAL REVERSAL"
    else:
        action = "WAIT"
        setup = "NEUTRAL"

    return {
        "price": price,
        "rsi": rsi,
        "volume": vol,
        "bull": bull,
        "bear": bear,
        "action": action,
        "setup": setup
    }, df


def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            return pd.read_csv(HISTORY_FILE)
        except:
            pass
    return pd.DataFrame()


def save_history(symbol, interval, r):
    row = {
        "Time": pd.Timestamp.now(),
        "Symbol": symbol,
        "Interval": interval,
        "Action": r["action"],
        "Setup": r["setup"],
        "Price": r["price"],
        "RSI": r["rsi"],
        "Confidence": max(
            r["bull"],
            r["bear"]
        )
    }

    old = load_history()
    new = pd.DataFrame([row])
    result = pd.concat(
        [old, new],
        ignore_index=True
    )

    result.to_csv(
        HISTORY_FILE,
        index=False
    )


st.title("🌍 Global Stock Analyzer")
st.caption(
    "Market analysis + global scanner + Strategy Lab"
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Single Stock",
    "🌍 Global Scanner",
    "📈 Performance",
    "🧪 Strategy Lab"
])


with tab1:
    st.subheader("Single Stock Analysis")

    a, b = st.columns(2)

    with a:
        symbol = st.text_input(
            "Symbol",
            "AAPL"
        ).strip().upper()

    with b:
        interval = st.selectbox(
            "Timeframe",
            ["5m", "15m", "30m", "1h", "1d"],
            index=1
        )

    if st.button(
        "🔎 Analyze",
        use_container_width=True
    ):
        try:
            r, df = current(
                symbol,
                interval
            )

            if r is None:
                st.error("No market data.")
            else:
                save_history(
                    symbol,
                    interval,
                    r
                )

                st.success(
                    f"{r['action']} — {r['setup']}"
                )

                a, b, c, d = st.columns(4)

                a.metric(
                    "Price",
                    f"{r['price']:.2f}"
                )

                b.metric(
                    "RSI",
                    f"{r['rsi']:.1f}"
                )

                c.metric(
                    "Volume",
                    f"{r['volume']:.2f}x"
                )

                d.metric(
                    "Confidence",
                    f"{max(r['bull'], r['bear'])}%"
                )

                a, b, c = st.columns(3)

                a.metric(
                    "Action",
                    r["action"]
                )

                b.metric(
                    "Bullish",
                    f"{r['bull']}%"
                )

                c.metric(
                    "Bearish",
                    f"{r['bear']}%"
                )

                st.write(
                    f"**Setup:** {r['setup']}"
                )

                st.line_chart(
                    df[
                        ["Close", "SMA20", "SMA50"]
                    ].tail(100)
                )

        except Exception as e:
            st.error(
                f"Analysis failed: {e}"
            )


with tab2:
    st.subheader("🌍 Global Scanner")

    symbols = st.text_area(
        "Symbols",
        "AAPL,MSFT,NVDA,AMZN,META,TSLA,GOOGL,AMD,RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS"
    )

    tf = st.selectbox(
        "Scanner timeframe",
        ["5m", "15m", "30m", "1h"],
        index=1
    )

    if st.button(
        "🌍 Scan Market",
        use_container_width=True
    ):
        rows = []

        items = [
            x.strip().upper()
            for x in symbols.split(",")
            if x.strip()
        ]

        bar = st.progress(0)

        for n, s in enumerate(items):
            try:
                r, _ = current(s, tf)

                if r:
                    rows.append({
                        "Symbol": s,
                        "Action": r["action"],
                        "Setup": r["setup"],
                        "RSI": round(
                            r["rsi"],
                            1
                        ),
                        "Volume": round(
                            r["volume"],
                            2
                        ),
                        "Bullish %": r["bull"],
                        "Bearish %": r["bear"],
                        "Confidence %": max(
                            r["bull"],
                            r["bear"]
                        ),
                        "Price": round(
                            r["price"],
                            2
                        )
                    })
            except:
                pass

            bar.progress(
                (n + 1) / len(items)
            )

        result = pd.DataFrame(rows)

        if result.empty:
            st.warning("No results.")
        else:
            st.dataframe(
                result,
                use_container_width=True,
                hide_index=True
            )


with tab3:
    st.subheader("📈 Signal History")

    history = load_history()

    if history.empty:
        st.info("No signals recorded yet.")
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
            "text/csv"
        )


with tab4:
    st.subheader("🧪 Strategy Lab")

    st.write(
        "Compare four strategies using historical data."
    )

    a, b = st.columns(2)

    with a:
        bt_symbol = st.text_input(
            "Backtest symbol",
            "AAPL"
        ).strip().upper()

    with b:
        bt_tf = st.selectbox(
            "Backtest timeframe",
            ["15m", "30m", "1h", "1d"],
            index=0
        )

    a, b, c = st.columns(3)

    with a:
        target_r = st.slider(
            "Target R",
            1.0,
            5.0,
            2.0,
            0.5
        )

    with b:
        stop_pct = st.slider(
            "Stop Loss %",
            1.0,
            10.0,
            3.0,
            0.5
        ) / 100

    with c:
        max_bars = st.slider(
            "Max Holding Bars",
            5,
            50,
            20,
            5
        )

    if st.button(
        "🧪 Compare Strategies",
        use_container_width=True
    ):
        with st.spinner(
            "Running historical tests..."
        ):
            df = data(
                bt_symbol,
                bt_tf
            )

        if df.empty:
            st.error(
                "No historical data found."
            )
        else:
            strategies = [
                "Trend Momentum",
                "Oversold Reversal",
                "Breakout",
                "Combined Score"
            ]

            results = {}
            summary = []

            for strategy in strategies:
                trades = backtest(
                    df,
                    strategy,
                    target_r,
                    stop_pct,
                    max_bars
                )

                results[strategy] = trades

                m = stats(trades)

                summary.append({
                    "Strategy": strategy,
                    "Trades": m["Trades"],
                    "Win Rate %": round(
                        m["Win Rate"],
                        1
                    ),
                    "Avg Return %": round(
                        m["Avg Return"],
                        2
                    ),
                    "Profit Factor": round(
                        m["Profit Factor"],
                        2
                    ),
                    "Total Return %": round(
                        m["Total Return"],
                        2
                    )
                })

            table = pd.DataFrame(
                summary
            )

            st.subheader(
                "Strategy Comparison"
            )

            st.dataframe(
                table,
                use_container_width=True,
                hide_index=True
            )

            valid = table[
                table["Trades"] > 0
            ]

            if not valid.empty:
                best = valid.sort_values(
                    "Profit Factor",
                    ascending=False
                ).iloc[0]

                st.success(
                    f"Best by Profit Factor: "
                    f"{best['Strategy']} — "
                    f"{best['Profit Factor']:.2f}"
                )

            st.subheader(
                "Equity Curves"
            )

            for strategy in strategies:
                trades = results[strategy]

                if trades.empty:
                    continue

                equity = (
                    100 *
                    (
                        1 +
                        trades["Return %"] / 100
                    ).cumprod()
                )

                st.write(
                    f"**{strategy}**"
                )

                st.line_chart(
                    pd.DataFrame(
                        {
                            "Equity": equity
                        }
                    )
                )

            chosen = st.selectbox(
                "View trades",
                strategies
            )

            trades = results[chosen]

            if trades.empty:
                st.info(
                    "No trades for this strategy."
                )
            else:
                st.dataframe(
                    trades,
                    use_container_width=True,
                    hide_index=True
                )

                csv = trades.to_csv(
                    index=False
                )

                st.download_button(
                    "⬇️ Download Trades",
                    csv,
                    "strategy_trades.csv",
                    "text/csv"
        )
