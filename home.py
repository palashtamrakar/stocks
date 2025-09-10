import streamlit as st

st.set_page_config(
    page_title="Home - Financial Platform",
    layout="wide"
)

# --- CSS to hide the sidebar on the home page ---
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)


st.title("Financial Analysis and Backtesting Platform")

st.markdown("Select a tool below to get started.")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.header("📈 Stock Analysis")
        st.write("An interactive charting tool for live stock analysis with technical indicators and volume metrics.")
        st.page_link("pages/analysis.py", label="Go to Dashboard", icon="➡️")

with col2:
    with st.container(border=True):
        st.header("🛠️ Stock Backtest")
        st.write("Test your trading strategies on historical data to see how they would have performed.")
        st.page_link("pages/backtest.py", label="Go to Backtester", icon="➡️")


# --- NEW: Footer Credit ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>A tool by Palash Tamrakar</p>", unsafe_allow_html=True)
