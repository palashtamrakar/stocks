import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def run_backtest(ticker, start_date, short_window, long_window, initial_capital):
    """
    This function takes user inputs, runs the backtest, and returns the results.
    """
    # 1. Fetch Data
    df = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
    
    # --- FINAL FIX: Force the DataFrame to a simple, single-ticker structure ---
    # This handles the case where yfinance returns a multi-column DataFrame
    if isinstance(df.columns, pd.MultiIndex):
        # Select the data for the first ticker in the multi-column DataFrame
        first_ticker = df.columns.levels[1][0]
        st.warning(f"Multiple datasets found for '{ticker}'. Using data for the first ticker: '{first_ticker}'")
        df = df.xs(first_ticker, level=1, axis=1)
    # --- END FIX ---
    
    # Standard cleaning
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df.set_index('Date', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    
    if df.empty:
        return None, None, None

    # 2. Generate Signals (unchanged)
    df['short_ma'] = df['Close'].rolling(window=short_window).mean()
    df['long_ma'] = df['Close'].rolling(window=long_window).mean()
    df['signal'] = 0
    df.loc[df['short_ma'] > df['long_ma'], 'signal'] = 1
    df.loc[df['short_ma'] < df['long_ma'], 'signal'] = -1
    df['positions'] = df['signal'].diff()

    # 3. Simulate Portfolio (unchanged)
    df['shares'] = 0.0
    df['cash'] = float(initial_capital)
    df['holdings'] = 0.0
    df['total'] = float(initial_capital)
    
    buy_points, sell_points = [], []
    close_col, positions_col, shares_col, cash_col, holdings_col, total_col = [df.columns.get_loc(c) for c in ['Close', 'positions', 'shares', 'cash', 'holdings', 'total']]

    for i in range(1, len(df)):
        df.iloc[i, shares_col] = df.iloc[i-1, shares_col]
        df.iloc[i, cash_col] = df.iloc[i-1, cash_col]
        
        position_signal = 0
        if not pd.isna(df.iloc[i, positions_col]):
            position_signal = int(df.iloc[i, positions_col])

        if position_signal == 2 and df.iloc[i-1, cash_col] > 0:
            shares_to_buy = df.iloc[i-1, cash_col] / df.iloc[i, close_col]
            df.iloc[i, shares_col] += shares_to_buy
            df.iloc[i, cash_col] = 0.0
            buy_points.append(df.index[i])
        elif position_signal == -2 and df.iloc[i-1, shares_col] > 0:
            cash_from_sale = df.iloc[i-1, shares_col] * df.iloc[i, close_col]
            df.iloc[i, cash_col] += cash_from_sale
            df.iloc[i, shares_col] = 0.0
            sell_points.append(df.index[i])
        
        current_holdings_value = df.iloc[i, shares_col] * df.iloc[i, close_col]
        df.iloc[i, holdings_col] = current_holdings_value
        df.iloc[i, total_col] = df.iloc[i, cash_col] + current_holdings_value

    results_df = df
    
    # 4. Calculate Metrics & 5. Create Visualization (unchanged)
    final_value = results_df['total'].iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital * 100
    metrics = {"Final Portfolio Value": f"₹{final_value:,.2f}", "Total Return": f"{total_return:.2f}%", "Total Trades": len(buy_points) + len(sell_points)}
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results_df.index, y=results_df['Close'], mode='lines', name='Close Price'))
    fig.add_trace(go.Scatter(x=results_df.index, y=results_df['short_ma'], mode='lines', name=f'SMA {short_window}'))
    fig.add_trace(go.Scatter(x=results_df.index, y=results_df['long_ma'], mode='lines', name=f'SMA {long_window}'))
    fig.add_trace(go.Scatter(x=buy_points, y=results_df.loc[buy_points]['Close'], mode='markers', name='Buy Signal', marker=dict(symbol='triangle-up', color='green', size=12)))
    fig.add_trace(go.Scatter(x=sell_points, y=results_df.loc[sell_points]['Close'], mode='markers', name='Sell Signal', marker=dict(symbol='triangle-down', color='red', size=12)))
    fig.update_layout(title=f'Backtest for {ticker}', xaxis_title='Date', yaxis_title='Price (INR)', template='plotly_dark')
    fig.update_yaxes(tickformat='.2f')

    return metrics, results_df, fig

# --- Streamlit UI (Now with input sanitization) ---
st.set_page_config(layout="wide")
st.title("📈 Interactive Backtesting Environment")

with st.sidebar:
    st.header("Strategy Parameters")
    
    # Sanitize user input to ensure only one ticker is attempted
    user_input = st.text_input("Stock Ticker", "RELIANCE.NS").upper()
    ticker = user_input.split(',')[0].split(' ')[0].strip()
    
    if ticker != user_input:
        st.warning(f"Multiple tickers detected. Backtesting for the first ticker found: **{ticker}**")
    
    start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
    short_window = st.slider("Short-term SMA", 10, 100, 50)
    long_window = st.slider("Long-term SMA", 50, 250, 200)
    initial_capital = st.number_input("Initial Capital", 10000, 1000000, 100000)
    
    run_button = st.button("Run Backtest")

if run_button:
    with st.spinner(f"Running backtest for {ticker}..."):
        metrics, results_df, fig = run_backtest(ticker, start_date, short_window, long_window, initial_capital)

    if metrics:
        st.header("Performance Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Final Portfolio Value", metrics["Final Portfolio Value"])
        col2.metric("Total Return", metrics["Total Return"])
        col3.metric("Total Trades", metrics["Total Trades"])
        st.header("Trade Visualization")
        st.plotly_chart(fig, use_container_width=True)
        st.header("Data & Trades (Excel View)")
        st.info("Scroll through the table to see the data, indicators, and portfolio value for each day.")
        st.dataframe(results_df)
    else:
        st.error(f"Could not retrieve data for {ticker}. Please check the ticker and date range.")