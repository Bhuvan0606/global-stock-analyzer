import streamlit as st

st.set_page_config(
    page_title="Global Stock Analyzer",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Global Stock Analyzer")
st.write("BUY / HOLD / SELL stock analysis")

st.info("🚧 Version 1 is under development.")

ticker = st.text_input(
    "Enter stock symbol",
    placeholder="Example: AAPL"
)

if ticker:
    st.subheader(f"Analysis: {ticker.upper()}")

    st.metric("Signal", "HOLD")
    st.write("Stock analysis will be added in the next version.")
