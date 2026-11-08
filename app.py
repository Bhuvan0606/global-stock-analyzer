import streamlit as st
from analyzer import get_stock_data, calculate_indicators, get_signal

st.set_page_config(
    page_title="Global Stock Analyzer",
    page_icon="📈",
    layout="wide"
)

st.title("🌍 Global Stock Analyzer")
st.caption("Live-market trading setup analyzer")


ticker = st.text_input(
    "Enter stock symbol",
    placeholder="Example: AAPL, MSFT, RELIANCE.NS, TCS.NS"
)

interval = st.selectbox(
    "Market timeframe",
    ["5m", "15m", "30m", "1h"],
    index=1
)


if st.button("🔎 Analyze Market"):

    if not ticker:

        st.warning("Enter a stock symbol first.")

    else:

        with st.spinner("Analyzing market..."):

            try:

                data = get_stock_data(
                    ticker.upper(),
                    period="30d",
                    interval=interval
                )

            except Exception as e:

                st.error("Market data request failed.")

                st.code(
                    f"{type(e).__name__}: {e}"
                )

                st.stop()

            if data is None:

                st.error(
                    "No market data found. "
                    "Check the stock symbol and try again."
                )

                st.stop()

            try:

                data = calculate_indicators(data)

                result = get_signal(data)

            except Exception as e:

                st.error("Analysis failed.")

                st.code(
                    f"{type(e).__name__}: {e}"
                )

                st.stop()


            st.success(
                f"Market analysis completed for {ticker.upper()}"
            )


            signal = result["signal"]


            if signal == "BUY SETUP":

                st.success(
                    f"🟢 {signal}"
                )

            elif signal == "EXIT / SELL SETUP":

                st.error(
                    f"🔴 {signal}"
                )

            else:

                st.warning(
                    f"🟡 {signal}"
                )


            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Live Price",
                f"{result['price']:.2f}"
            )

            col2.metric(
                "Setup Score",
                f"{result['score']}/100"
            )

            col3.metric(
                "RSI",
                f"{result['rsi']:.1f}"
            )

            col4.metric(
                "Trend",
                result["trend"]
            )


            st.subheader("Trading Levels")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Entry",
                f"{result['entry']:.2f}"
            )

            c2.metric(
                "Stop Loss",
                f"{result['stop_loss']:.2f}"
            )

            c3.metric(
                "Target 1",
                f"{result['target1']:.2f}"
            )

            c4.metric(
                "Target 2",
                f"{result['target2']:.2f}"
            )


            st.subheader("Market Conditions")

            st.write(
                f"**Trend:** {result['trend']}"
            )

            st.write(
                f"**Volume strength:** "
                f"{result['volume_ratio']:.2f}× average"
            )


            st.subheader("Why this signal?")

            for reason in result["reasons"]:

                st.write(
                    f"• {reason}"
                )


            st.subheader("Price")

            st.line_chart(
                data["Close"]
            )


            st.subheader("Moving Averages")

            chart = data[
                ["Close", "SMA_20", "SMA_50"]
            ].dropna()

            st.line_chart(chart)


            st.caption(
                "This is an analytical research tool, "
                "not a guarantee of future market returns."
            )
