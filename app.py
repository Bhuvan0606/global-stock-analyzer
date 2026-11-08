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
# MARKET DATA
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

    df["RecentHigh"] = df["High"].rolling(20).max().shift(1)
    df["RecentLow"] = df["Low"].rolling(20).min().shift(1)

    return df.dropna()


# =========================
# STRATEGIES
# =========================

def strategy_signal(df, i, strategy):

    r = df.iloc[i]

    price = float(r["Close"])
    sma20 = float(r["SMA20"])
    sma50 = float(r["SMA50"])
    rsi = float(r["RSI"])
    vol = float(r["VolRatio"])

    if strategy == "Trend Momentum":

        if (
            price > sma20
            and sma20 > sma50
            and rsi > 55
        ):
            return True

    elif strategy == "Oversold Reversal":

        if (
            rsi < 35
            and price <= sma20 * 1.01
            and vol >= 1.2
        ):
            return True

    elif strategy == "Breakout":

        recent_high = float(r["RecentHigh"])

        if (
            price > recent_high
            and vol >= 1.5
        ):
            return True

    elif strategy == "Combined Score":

        bullish = 0

        if price > sma20:
            bullish += 25

        if sma20 > sma50:
            bullish += 25

        if rsi > 55:
            bullish += 20

        if vol >= 1.5:
            bullish += 15

        if price > float(r["RecentHigh"]):
            bullish += 15

        if bullish >= 60:
            return True

    return False


# =========================
# STRATEGY BACKTEST
# =========================

def run_strategy(
    df,
    strategy,
    target_r,
    stop_pct,
    max_bars
):

    trades = []

    i = 60

    while i < len(df) - 2:

        signal = strategy_signal(
            df,
            i,
            strategy
        )

        if not signal:
            i += 1
            continue

        # Entry happens on NEXT candle.
        entry_i = i + 1

        entry = float(
            df["Open"].iloc[entry_i]
        )

        stop = entry * (
            1 - stop_pct
        )

        target = entry + (
            entry - stop
        ) * target_r

        exit_price = entry
        exit_i = entry_i
        result = "OPEN"

        last_i = min(
            entry_i + max_bars,
            len(df) - 1
        )

        for j in range(
            entry_i,
            last_i + 1
        ):

            high = float(
                df["High"].iloc[j]
            )

            low = float(
                df["Low"].iloc[j]
            )

            hit_target = (
                high >= target
            )

            hit_stop = (
                low <= stop
            )

            # Conservative rule:
            # if both are hit in same candle,
            # assume stop was hit first.
            if hit_stop:

                result = "LOSS"
                exit_price = stop
                exit_i = j
                break

            if hit_target:

                result = "WIN"
                exit_price = target
                exit_i = j
                break

        if result == "OPEN":

            exit_price = float(
                df["Close"].iloc[last_i]
            )

            exit_i = last_i

            if exit_price > entry:
                result = "WIN"
            else:
                result = "LOSS"

        ret = (
            (exit_price - entry)
            / entry
            * 100
        )

        trades.append({
            "Signal Time": str(
                df.index[i]
            ),
            "Entry Time": str(
                df.index[entry_i]
            ),
            "Entry": round(
                entry,
                2
            ),
            "Stop": round(
                stop,
                2
            ),
            "Target": round(
                target,
                2
            ),
            "Exit": round(
                exit_price,
                2
            ),
            "Result": result,
            "Return %": round(
                ret,
                2
            ),
            "Bars Held": (
                exit_i - entry_i + 1
            )
        })

        # Avoid taking overlapping
        # trades from the same signal.
        i = exit_i + 1

    return pd.DataFrame(
        trades
    )


# =========================
# METRICS
# =========================

def metrics(trades):

    if trades.empty:
        return {
            "Trades": 0,
            "Win Rate": 0,
            "Avg Return": 0,
            "Profit Factor": 0,
            "Total Return": 0,
            "Max Drawdown": 0
        }

    total = len(trades)

    wins = (
        trades["Result"] == "WIN"
    ).sum()

    win_rate = (
        wins / total * 100
    )

    avg_return = (
        trades["Return %"].mean()
    )

    total_return = (
        trades["Return %"].sum()
    )

    profits = trades.loc[
        trades["Return %"] > 0,
        "Return %"
    ].sum()

    losses = abs(
        trades.loc[
            trades["Return %"] < 0,
            "Return %"
        ].sum()
    )

    if losses > 0:
        pf = profits / losses
    else:
        pf = 0

    equity = (
        100 *
        (1 + trades["Return %"] / 100)
        .cumprod()
    )

    peak = equity.cummax()

    drawdown = (
        (equity - peak)
        / peak
        * 100
    )

    max_dd = drawdown.min()

    return {
        "Trades": total,
        "Win Rate": win_rate,
        "Avg Return": avg_return,
        "Profit Factor": pf,
        "Total Return": total_return,
        "Max Drawdown": max_dd
    }


# =========================
# CURRENT SIGNAL
# =========================

def current_signal(symbol, interval):

    df = get_data(
        symbol,
        interval
    )

    if df.empty:
        return None

    i = len(df) - 1

    price = float(
        df["Close"].iloc[i]
    )

    sma20 = float(
        df["SMA20"].iloc[i]
    )

    sma50 = float(
        df["SMA50"].iloc[i]
    )

    rsi = float(
        df["RSI"].iloc[i]
    )

    vol = float(
        df["VolRatio"].iloc[i]
    )

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
        if price >= float(
            df["Open"].iloc[i]
        ):
            bullish += 15
        else:
            bearish += 15

    if rsi < 35:
        bullish += 15

    if bullish >= 60 and bullish > bearish:
        action = "BUY"
        setup = "BULLISH TREND"
    elif bearish >= 60 and bearish > bullish:
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
        "bullish": bullish,
        "bearish": bearish,
        "action": action,
        "setup": setup
    }


# =========================
# HISTORY
# =========================

def load_history():

    if os.path.exists(
        HISTORY_FILE
    ):

        try:
            return pd.read_csv(
                HISTORY_FILE
            )
        except:
            pass

    return pd.DataFrame()


def save_signal(r):

    row = {
        "Time": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "Symbol": r["symbol"],
        "Interval": r["interval"],
        "Action": r["action"],
        "Setup": r["setup"],
        "Price": r["price"],
        "RSI": r["rsi"],
        "Confidence": max(
            r["bullish"],
            r["bearish"]
        )
    }

    old = load_history()

    new = pd.DataFrame(
        [row]
    )

    history = pd.concat(
        [old, new],
        ignore_index=True
    )

    history.to_csv(
        HISTORY_FILE,
        index=False
    )


# =========================
# UI
# =========================

st.title(
    "🌍 Global Stock Analyzer"
)

st.caption(
    "Live/near-live analysis + scanner + "
    "historical strategy research"
)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Single Stock",
        "🌍 Global Scanner",
        "📈 Performance",
        "🧪 Strategy Lab"
    ]
)


# =========================
# SINGLE STOCK
# =========================

with tab1:

    a, b = st.columns(2)

    with a:

        symbol = st.text_input(
            "Stock symbol",
            "AAPL"
        ).strip().upper()

    with b:

        interval = st.selectbox(
            "Timeframe",
            [
                "5m",
                "15m",
                "30m",
                "1h",
                "1d"
            ],
            index=1
        )

    if st.button(
        "🔎 Analyze Stock",
        use_container_width=True
    ):

        try:

            r = current_signal(
                symbol,
                interval
            )

            if r is None:

                st.error(
                    "No usable market data."
                )

            else:

                r["symbol"] = symbol
                r["interval"] = interval

                save_signal(r)

                st.success(
                    f"{r['action']} — "
                    f"{r['setup']}"
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
                    f"{max(r['bullish'], r['bearish'])}%"
                )

                st.subheader(
                    "Signal"
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
                    "Bullish %",
                    r["bullish"]
                )

                st.write(
                    f"**Bearish:** "
                    f"{r['bearish']}%"
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
                        data[[
                            "Close",
                            "SMA20",
                            "SMA50"
                        ]].tail(100)

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
            "AAPL,MSFT,NVDA,AMZN,META,"
            "TSLA,GOOGL,AMD,RELIANCE.NS,"
            "TCS.NS,INFY.NS,HDFCBANK.NS"
        )
    )

    scan_tf = st.selectbox(
        "Scanner timeframe",
        [
            "5m",
            "15m",
            "30m",
            "1h"
        ],
        index=1,
        key="scanner_tf"
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

        progress = st.progress(
            0
        )

        for n, s in enumerate(
            items
        ):

            try:

                r = current_signal(
                    s,
                    scan_tf
                )

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
                        "Bullish %": r["bullish"],
                        "Bearish %": r["bearish"],
                        "Confidence %":
                            max(
                                r["bullish"],
                                r["bearish"]
                            ),
                        "Price": round(
                            r["price"],
                            2
                        )
                    })

            except:
                pass

            progress.progress(
                (n + 1) / len(items)
            )

        scan = pd.DataFrame(
            rows
        )

        if scan.empty:

            st.warning(
                "No results."
            )

        else:

            st.dataframe(
                scan,
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
            "No signals recorded yet."
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
# STRATEGY LAB
# =========================

with tab4:

    st.subheader(
        "🧪 Strategy Lab"
    )

    st.write(
        "Compare different trading strategies "
        "on the same historical market data."
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
            [
                "15m",
                "30m",
                "1h",
                "1d"
            ],
            index=0,
            key="lab_tf"
        )

    c1, c2, c3 = st.columns(3)

    with c1:

        target_r = st.slider(
            "Target R",
            1.0,
            5.0,
            2.0,
            0.5
        )

    with c2:

        stop_pct = st.slider(
            "Stop Loss %",
            1.0,
            10.0,
            3.0,
            0.5
        )

    with c3:

        max_bars = st.slider(
            "Max Holding Bars",
            5,
            50,
            20,
            5
        )

    strategies = [
        "Trend Momentum",
        "Oversold Reversal",
        "Breakout",
        "Combined Score"
    ]

    if st.button(
        "🧪 Compare Strategies",
        use_container_width=True
    ):

        with st.spinner(
            "Running strategy comparison..."
        ):

            df = get_data(
                bt_symbol,
                bt_tf
            )

        if df.empty:

            st.error(
                "No historical data found."
            )

        else:

            summary = []
            all_trades = {}

            for strategy in strategies:

                trades = run_strategy(
                    df,
                    strategy,
                    target_r,
                    stop_pct,
                    max_bars
                )

                all_trades[
                    strategy
                ] = trades

                m = metrics(
                    trades
                )

                summary.append({
                    "Strategy": strategy,
                    "Trades": m["Trades"],
                    "Win Rate %":
                        round(
                            m["Win Rate"],
                            1
                        ),
                    "Avg Return %":
                        round(
                            m["Avg Return"],
                            2
                        ),
                    "Profit Factor":
                        round(
                            m["Profit Factor"],
                            2
                        ),
                    "Total Return %":
                        round(
                            m["Total Return"],
                            2
                        ),
                    "Max Drawdown %":
                        round(
                            m["Max Drawdown"],
                            2
                        )
                })

            summary_df = pd.DataFrame(
                summary
            )

            st.subheader(
                "Strategy Comparison"
            )

            st.dataframe(
                summary_df,
                use_container_width=True,
                hide_index=True
            )

            valid = summary_df[
                summary_df["Trades"] > 0
            ]

            if not valid.empty:

                best = valid.sort_values(
                    "Profit Factor",
                    ascending=False
                ).iloc[0]

                st.success(
                    f"Best current strategy by "
                    f"Profit Factor: "
                    f"{best['Strategy']} "
                    f"({best['Profit Factor']:.2f})"
                )

            st.subheader(
                "Strategy Equity Curves"
            )

            for strategy in strategies:

                trades = all_trades[
                    strategy
                ]

                if trades.empty:
                    continue

                equity = (
                    100 *
                    (
                        1 +
                        trades[
                            "Return %"
                        ] / 100
                    ).cumprod()
                )

                st.write(
                    f"**{strategy}**"
                )

                st.line_chart(
                    pd.DataFrame(
                        {
                            "Equity":
                                equity.values
                        }
          
