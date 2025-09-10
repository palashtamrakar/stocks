import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Helper functions (unchanged) ---
def calculate_rsi(data, window=14):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/window, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/window, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    ema_short = data.ewm(span=short_window, adjust=False).mean()
    ema_long = data.ewm(span=long_window, adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return pd.DataFrame({'MACD_12_26_9': macd, 'MACDs_12_26_9': signal})

# --- Streamlit App ---
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

# --- REMOVED: Custom CSS for black theme is gone ---
# Streamlit will now use its default dark gray theme.

# --- Sidebar Controls ---
st.sidebar.title("Controls")
ticker_symbol = st.sidebar.text_input("Ticker", "RELIANCE.NS").upper()
period = st.sidebar.selectbox("Period", ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'ytd', 'max'], index=5)
interval = st.sidebar.selectbox("Interval", ['1d', '5d', '1wk', '1mo'], index=0)

st.sidebar.header("Chart Options")
selected_indicators = st.sidebar.multiselect("Indicators", ["SMA", "EMA", "RSI", "MACD"], default=["SMA", "RSI"])
show_volume = st.sidebar.toggle("Show Volume", True)

with st.sidebar.expander("Indicator Settings"):
    if "SMA" in selected_indicators:
        sma_length = st.slider("SMA Length", 5, 100, 20)
    else:
        sma_length = 20
    if "EMA" in selected_indicators:
        ema_length = st.slider("EMA Length", 5, 100, 20)
    else:
        ema_length = 20

# --- Main Page ---
st.title("📊 Stock Analysis Dashboard")

try:
    df = yf.Ticker(ticker_symbol).history(period=period, interval=interval)
    if df.empty:
        st.error("No data found.")
    else:
        # Calculate indicators
        if "SMA" in selected_indicators: df[f'SMA_{sma_length}'] = df['Close'].rolling(window=sma_length).mean()
        if "EMA" in selected_indicators: df[f'EMA_{ema_length}'] = df['Close'].ewm(span=ema_length, adjust=False).mean()
        if "RSI" in selected_indicators: df['RSI_14'] = calculate_rsi(df['Close'])
        if "MACD" in selected_indicators: df = df.join(calculate_macd(df['Close']))

        st.header(f"{yf.Ticker(ticker_symbol).info.get('longName', ticker_symbol)} ({ticker_symbol})")
        
        # Dynamic Chart Creation
        subplot_count = 1 + ("RSI" in selected_indicators) + ("MACD" in selected_indicators)
        row_heights = [0.7] + [0.3] * (subplot_count - 1)
        specs = [[{"secondary_y": True}]] + [[{"secondary_y": False}]] * (subplot_count - 1)
        fig = make_subplots(rows=subplot_count, cols=1, shared_xaxes=True, 
                              vertical_spacing=0.03, row_heights=row_heights, specs=specs)
        
        # Plot Main Chart and overlays
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
        if "SMA" in selected_indicators: fig.add_trace(go.Scatter(x=df.index, y=df[f'SMA_{sma_length}'], name=f'SMA {sma_length}'), row=1, col=1)
        if "EMA" in selected_indicators: fig.add_trace(go.Scatter(x=df.index, y=df[f'EMA_{ema_length}'], name=f'EMA {ema_length}'), row=1, col=1)
        if show_volume:
            colors = np.where(df['Close'] >= df['Open'], 'green', 'red')
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors, opacity=0.4), 
                          secondary_y=True, row=1, col=1)
        
        # Plot Subplot Indicators
        current_row = 2
        if "RSI" in selected_indicators:
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI_14'], name='RSI'), row=current_row, col=1); current_row += 1
        if "MACD" in selected_indicators and all(c in df.columns for c in ['MACD_12_26_9', 'MACDs_12_26_9']):
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD_12_26_9'], name='MACD'), row=current_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MACDs_12_26_9'], name='Signal'), row=current_row, col=1)
        
        # --- NEW: Updated chart theme to match default gray ---
        fig.update_layout(
            height=800, 
            xaxis_rangeslider_visible=False,
            template='plotly_dark'  # Use the standard dark theme for the chart
        )
        fig.update_yaxes(secondary_y=False)
        fig.update_xaxes()

        st.plotly_chart(fig, use_container_width=True)

        st.header("Latest Data")
        st.dataframe(df.tail())
except Exception as e:
    st.error(f"An error occurred: {e}")